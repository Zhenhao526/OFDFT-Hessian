#!/usr/bin/env python3
"""Merge held-parent results from preregistered geometry-scalar CV folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.qm9_complete_total_geometry_mlp_capacity import _parent_cv_distribution
from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_variant(value: str) -> tuple[str, Path]:
    name, path = value.split("=", maxsplit=1)
    return name, Path(path).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _gate(metrics: dict[str, float]) -> dict[str, bool]:
    result = {
        "median_relative_frobenius_le_0p10": (
            metrics["median_relative_frobenius"] <= 0.10
        ),
        "fraction_relative_frobenius_le_0p15_ge_0p80": (
            metrics["fraction_relative_frobenius_at_or_below_0_15"] >= 0.80
        ),
        "p90_relative_frobenius_le_0p20": (
            metrics["p90_relative_frobenius"] <= 0.20
        ),
        "asymmetry_max_le_0p005": (
            metrics["max_antisymmetric_over_symmetric_frobenius"] <= 0.005
        ),
        "energy_median_no_more_than_5pct_worse_than_source": (
            metrics["energy_median_ratio_to_source"] <= 1.05
        ),
        "force_median_no_more_than_5pct_worse_than_source": (
            metrics["force_median_ratio_to_source"] <= 1.05
        ),
    }
    result["passed"] = all(result.values())
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    vibrational_summary = getattr(args, "vibrational_summary", None)
    protocol = yaml.safe_load(args.protocol.read_text())
    protocol_sha = _sha256(args.protocol)
    registered = {
        str(row["id"]): row for row in protocol["preregistered_variants"]
    }
    baseline, baseline_manifest = _load_parents(args.baseline_manifest)
    baseline_by_id = {parent.molecule_id: parent for parent in baseline}
    source_rows = []
    for parent in baseline:
        metrics = hessian_metrics(parent.hessian, parent.pbe_hessian)
        source_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "relative_frobenius": metrics["relative_frobenius"],
                "antisymmetric_over_symmetric_frobenius": metrics[
                    "antisymmetric_over_symmetric_frobenius"
                ],
                "energy_abs_error_hartree": abs(parent.energy - parent.pbe_energy),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(parent.force - parent.pbe_force))
                ),
            }
        )

    variants = []
    all_rows = []
    result_groups = []
    trajectories: dict[str, dict[int, list[float]]] = {}
    for variant_id, variant_dir in map(_parse_variant, args.variant):
        if variant_id not in registered:
            raise ValueError(f"variant is not preregistered: {variant_id}")
        held_rows = []
        fold_hashes = []
        trajectories[variant_id] = {}
        for fold_index, expected_ids in enumerate(
            protocol["cross_validation"]["fold_assignment"]
        ):
            fold_dir = variant_dir / f"fold_{fold_index}"
            summary_path = fold_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            if summary.get("test100_accessed") is not False:
                raise ValueError(f"fold {fold_index} does not freeze Test100")
            if int(summary.get("test100_evaluations_used", 0)) != 0:
                raise ValueError(f"fold {fold_index} records Test100 access")
            cv = summary.get("parent_cv") or {}
            if cv.get("protocol_sha256") != protocol_sha:
                raise ValueError(f"fold {fold_index} protocol hash differs")
            if cv.get("variant_id") != variant_id or int(cv.get("fold_index", -1)) != fold_index:
                raise ValueError(f"fold {fold_index} identity differs")
            if cv.get("held_parent_hessian_used_for_gradient") is not False:
                raise ValueError(f"fold {fold_index} used held Hessian gradients")
            if cv.get("held_parent_hessian_used_for_checkpoint_selection") is not False:
                raise ValueError(f"fold {fold_index} selected on held Hessians")
            local = [
                row
                for row in summary["per_parent"]
                if row.get("parent_cv_role") == "held"
            ]
            if {str(row["molecule_id"]) for row in local} != set(map(str, expected_ids)):
                raise ValueError(f"fold {fold_index} held parent IDs differ")
            for row in local:
                molecule_id = str(row["molecule_id"])
                parent = baseline_by_id[molecule_id]
                source_metric = hessian_metrics(parent.hessian, parent.pbe_hessian)
                held_rows.append(
                    {
                        "variant_id": variant_id,
                        "fold": fold_index,
                        **row,
                        "source_relative_frobenius": source_metric[
                            "relative_frobenius"
                        ],
                        "summary": summary_path.as_posix(),
                        "hessian_npz": (
                            fold_dir / f"{molecule_id}_result.npz"
                        ).as_posix(),
                    }
                )
            fold_hashes.append(
                {
                    "fold": fold_index,
                    "summary": summary_path.as_posix(),
                    "summary_sha256": _sha256(summary_path),
                    "checkpoint": (fold_dir / "best.ckpt").as_posix(),
                    "checkpoint_sha256": _sha256(fold_dir / "best.ckpt"),
                }
            )
            log_path = fold_dir / "training_metrics.jsonl"
            for line in log_path.read_text().splitlines():
                row = json.loads(line)
                trajectories[variant_id].setdefault(int(row["step"]), []).append(
                    float(row["median_relative_frobenius"])
                )

        expected_parent_ids = {parent.molecule_id for parent in baseline}
        held_parent_ids = [str(row["molecule_id"]) for row in held_rows]
        if len(held_parent_ids) != len(set(held_parent_ids)) or set(held_parent_ids) != expected_parent_ids:
            raise ValueError("held CV rows do not cover every parent exactly once")
        metrics = _parent_cv_distribution(held_rows)
        gate = _gate(metrics)
        win_fraction = float(
            np.mean(
                [
                    float(row["relative_frobenius"])
                    < float(row["source_relative_frobenius"])
                    for row in held_rows
                ]
            )
        )
        role = str(registered[variant_id]["role"])
        variants.append(
            {
                "variant_id": variant_id,
                "role": role,
                "train800_energy_force_replay": bool(
                    registered[variant_id]["train800_energy_force_replay"]
                ),
                **metrics,
                "parent_win_fraction_vs_source": win_fraction,
                "scientific_gate": gate,
                "fold_artifacts": fold_hashes,
            }
        )
        all_rows.extend(held_rows)
        result_groups.extend(
            {
                "run": variant_id,
                "molecule_id": str(row["molecule_id"]),
                "sample_id": 0,
                "success": True,
                "hessian_npz": row["hessian_npz"],
            }
            for row in held_rows
        )

    source_vibrational = None
    if vibrational_summary is not None:
        vibration = json.loads(vibrational_summary.read_text())
        summaries = vibration["summaries"]
        by_run = {str(row["run"]): row for row in summaries}
        source_vibrational = by_run.get("original_A")
        if source_vibrational is None:
            raise ValueError("vibrational summary omits original_A source")
        for variant in variants:
            metrics = by_run.get(str(variant["variant_id"]))
            if metrics is None:
                raise ValueError(
                    f"vibrational summary omits {variant['variant_id']}"
                )
            variant["vibrational_metrics"] = metrics
            source_imaginary_error = abs(
                int(source_vibrational["total_model_imaginary_modes"])
                - int(source_vibrational["total_pbe_imaginary_modes"])
            )
            model_imaginary_error = abs(
                int(metrics["total_model_imaginary_modes"])
                - int(metrics["total_pbe_imaginary_modes"])
            )
            variant["vibrational_gate"] = {
                "frequency_mae_cm1_le_200": float(
                    metrics["mean_frequency_mae_cm-1"]
                )
                <= 200.0,
                "imaginary_mode_error_no_worse_than_source": (
                    model_imaginary_error <= source_imaginary_error
                ),
                "mode_overlap_improves_over_source": float(
                    metrics["mean_mode_overlap"]
                )
                > float(source_vibrational["mean_mode_overlap"]),
            }
            variant["vibrational_gate"]["passed"] = all(
                variant["vibrational_gate"].values()
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "held_parent_metrics.csv", all_rows)
    flat_variants = []
    for row in variants:
        flattened = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "scientific_gate",
                "fold_artifacts",
                "vibrational_metrics",
                "vibrational_gate",
            }
        }
        flattened.update(
            {
                f"gate_{key}": value
                for key, value in row["scientific_gate"].items()
            }
        )
        if "vibrational_metrics" in row:
            flattened.update(
                {
                    f"vibrational_{key}": value
                    for key, value in row["vibrational_metrics"].items()
                }
            )
            flattened.update(
                {
                    f"vibrational_gate_{key}": value
                    for key, value in row["vibrational_gate"].items()
                }
            )
        flat_variants.append(flattened)
    _write_csv(args.output_dir / "variant_summary.csv", flat_variants)
    result_json = args.output_dir / "held_result_rows.json"
    result_json.write_text(
        json.dumps(
            {
                "definition": "held-parent-only geometry scalar CV artifacts",
                "rows": result_groups,
                "test100_accessed": False,
                "test100_evaluations_used": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    source = np.sort([float(row["relative_frobenius"]) for row in source_rows])
    probability = np.arange(1, source.size + 1) / source.size
    axes[0].plot(probability, source, "k--", label="original-A source")
    for variant in variants:
        values = np.sort(
            [
                float(row["relative_frobenius"])
                for row in all_rows
                if row["variant_id"] == variant["variant_id"]
            ]
        )
        axes[0].plot(probability, values, marker=".", label=variant["variant_id"])
    axes[0].axhline(0.10, color="black", linewidth=1)
    axes[0].axhline(0.20, color="gray", linestyle=":", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Empirical CDF")
    axes[0].set_ylabel("Held-parent Hessian relative Frobenius")
    axes[0].legend(fontsize=8)

    for variant_id, by_step in trajectories.items():
        steps = sorted(by_step)
        median = [float(np.median(by_step[step])) for step in steps]
        axes[1].plot(steps, median, label=variant_id)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Median fit-parent relative Frobenius")
    axes[1].legend(fontsize=8)
    plot_path = args.output_dir / "geometry_parent_cv.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    passing_replay_hessian = [
        row["variant_id"]
        for row in variants
        if row["train800_energy_force_replay"] and row["scientific_gate"]["passed"]
    ]
    passing_replay = [
        row["variant_id"]
        for row in variants
        if row["variant_id"] in passing_replay_hessian
        and row.get("vibrational_gate", {}).get("passed") is True
    ]
    result = {
        "definition": (
            "Five-fold train20 Hessian-label-heldout geometry-scalar evaluation; "
            "held Hessians are used only after each fold checkpoint is frozen."
        ),
        "source_hessian_metrics": {
            "median_relative_frobenius": float(np.median(source)),
            "p90_relative_frobenius": float(np.quantile(source, 0.9)),
            "max_relative_frobenius": float(np.max(source)),
        },
        "source_vibrational_metrics": source_vibrational,
        "variants": variants,
        "passing_replay_variants": passing_replay,
        "passing_replay_variants_before_vibrational_check": passing_replay_hessian,
        "advancement_authorized": bool(passing_replay),
        "vibrational_check_complete": vibrational_summary is not None,
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha,
        "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "source_split_sha256": baseline_manifest["source_split_sha256"],
        "held_parent_csv": (args.output_dir / "held_parent_metrics.csv").resolve().as_posix(),
        "variant_summary_csv": (args.output_dir / "variant_summary.csv").resolve().as_posix(),
        "held_result_json": result_json.resolve().as_posix(),
        "plot": plot_path.resolve().as_posix(),
        "plot_sha256": _sha256(plot_path),
        "vibrational_summary": (
            vibrational_summary.resolve().as_posix()
            if vibrational_summary is not None
            else None
        ),
        "vibrational_summary_sha256": (
            _sha256(vibrational_summary)
            if vibrational_summary is not None
            else None
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="ID=DIR")
    parser.add_argument("--vibrational-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
