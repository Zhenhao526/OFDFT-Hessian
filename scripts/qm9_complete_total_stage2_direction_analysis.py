#!/usr/bin/env python3
"""Aggregate frozen Stage-2 train/held-out direction and full-Hessian metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    direction_rows = []
    training_rows = []
    summaries = []
    for item in args.run:
        name, path_text = item.split("=", maxsplit=1)
        path = Path(path_text)
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"{name} does not certify frozen Test100")
        if not summary.get("direction_manifest_sha256"):
            raise ValueError(f"{name} has no frozen direction certificate")
        training_path = path / "training_metrics.jsonl"
        if training_path.is_file():
            for line in training_path.read_text().splitlines():
                source = json.loads(line)
                training_rows.append(
                    {
                        "run": name,
                        "step": source["step"],
                        "energy_loss": source.get(
                            "energy_loss", source.get("sampled_energy_loss")
                        ),
                        "force_loss": source.get(
                            "force_loss", source.get("sampled_force_loss")
                        ),
                        "hessian_loss": source.get(
                            "hessian_loss", source.get("sampled_hvp_loss")
                        ),
                        "spectrum_loss": source.get(
                            "spectrum_loss", source.get("sampled_spectrum_loss")
                        ),
                        "median_train_hvp_relative_frobenius": source.get(
                            "median_train_hvp_relative_frobenius"
                        ),
                        "median_heldout_hvp_relative_frobenius": source.get(
                            "median_heldout_hvp_relative_frobenius"
                        ),
                        "median_full_hessian_relative_frobenius": source.get(
                            "median_relative_frobenius"
                        ),
                        "gradient_norm_energy": source.get("gradient_norm/energy"),
                        "gradient_norm_force": source.get("gradient_norm/force"),
                        "gradient_norm_hvp": source.get("gradient_norm/hvp"),
                        "gradient_norm_spectrum": source.get(
                            "gradient_norm/spectrum"
                        ),
                        "pcgrad_conflict": source.get("pcgrad/conflict"),
                        "pcgrad_cosine_before": source.get("pcgrad/cosine_before"),
                        "wall_time_s": source.get("wall_time_s"),
                    }
                )
        direction_lookup = {}
        direction_manifest_path = Path(summary.get("direction_manifest", ""))
        if direction_manifest_path.is_file():
            direction_manifest = json.loads(direction_manifest_path.read_text())
            direction_lookup = {
                row["molecule_id"]: row for row in direction_manifest["directions"]
            }
        local = []
        for row in summary["per_parent"]:
            normalized = {
                "run": name,
                "molecule_id": row["molecule_id"],
                "natoms": row["natoms"],
                "energy_abs_error_hartree": row["energy_abs_error_hartree"],
                "force_mae_hartree_per_bohr": row["force_mae_hartree_per_bohr"],
                "train_hvp_relative_frobenius": row[
                    "train_hvp_relative_frobenius"
                ],
                "heldout_hvp_relative_frobenius": row[
                    "heldout_hvp_relative_frobenius"
                ],
                "full_hessian_relative_frobenius": row["relative_frobenius"],
                "asymmetry_ratio": row[
                    "antisymmetric_over_symmetric_frobenius"
                ],
            }
            direction_metadata = direction_lookup.get(row["molecule_id"])
            result_path = path / f"{row['molecule_id']}_result.npz"
            if direction_metadata is not None and result_path.is_file():
                with np.load(result_path) as result_payload:
                    difference = np.asarray(result_payload["predicted_hessian"]) - np.asarray(
                        result_payload["pbe_hessian"]
                    )
                    reference_hessian = np.asarray(result_payload["pbe_hessian"])
                with np.load(direction_metadata["direction_path"]) as direction_payload:
                    directions = np.asarray(direction_payload["directions"])
                    roles = np.asarray(direction_payload["roles"]).astype(str)
                    kinds = np.asarray(direction_payload["kinds"]).astype(str)
                    external_dimension = (
                        int(direction_payload["external_basis"].shape[1])
                        if "external_basis" in direction_payload
                        else 0
                    )
                train_directions = directions[roles == "train"]
                train_projector = train_directions.T @ train_directions
                complement = np.eye(train_projector.shape[0]) - train_projector
                unidentified_hessian = (
                    complement @ reference_hessian @ complement
                )
                normalized["train_direction_fraction_of_internal_dimension"] = float(
                    len(train_directions)
                    / (reference_hessian.shape[0] - external_dimension)
                )
                normalized["unidentified_qhq_relative_frobenius"] = float(
                    np.linalg.norm(unidentified_hessian)
                    / max(np.linalg.norm(reference_hessian), np.finfo(float).tiny)
                )
                heldout_directions = directions[roles == "heldout"]
                unidentified_heldout = np.einsum(
                    "ij,dj->di", unidentified_hessian, heldout_directions
                )
                reference_heldout = np.einsum(
                    "ij,dj->di", reference_hessian, heldout_directions
                )
                normalized["minimum_norm_oracle_heldout_hvp_relative_frobenius"] = float(
                    np.linalg.norm(unidentified_heldout)
                    / max(np.linalg.norm(reference_heldout), np.finfo(float).tiny)
                )
                error_hvp = np.einsum("ij,dj->di", difference, directions)
                reference_hvp = np.einsum(
                    "ij,dj->di", reference_hessian, directions
                )
                for direction_index, (role, kind, error, reference) in enumerate(
                    zip(roles, kinds, error_hvp, reference_hvp, strict=True)
                ):
                    direction_rows.append(
                        {
                            "run": name,
                            "molecule_id": row["molecule_id"],
                            "direction_index": direction_index,
                            "role": role,
                            "kind": kind,
                            "hvp_mae": float(np.mean(np.abs(error))),
                            "hvp_rmse": float(np.sqrt(np.mean(error**2))),
                            "hvp_relative_l2": float(
                                np.linalg.norm(error)
                                / max(np.linalg.norm(reference), np.finfo(float).tiny)
                            ),
                            "reference_l2": float(np.linalg.norm(reference)),
                        }
                    )
            rows.append(normalized)
            local.append(normalized)
        energy_errors = np.asarray(
            [row["energy_abs_error_hartree"] for row in local], dtype=float
        )
        force_errors = np.asarray(
            [row["force_mae_hartree_per_bohr"] for row in local], dtype=float
        )
        train_relative = np.asarray(
            [row["train_hvp_relative_frobenius"] for row in local], dtype=float
        )
        heldout_relative = np.asarray(
            [row["heldout_hvp_relative_frobenius"] for row in local], dtype=float
        )
        full_relative = np.asarray(
            [row["full_hessian_relative_frobenius"] for row in local], dtype=float
        )
        asymmetry = np.asarray([row["asymmetry_ratio"] for row in local], dtype=float)
        coverage = np.asarray(
            [
                row["train_direction_fraction_of_internal_dimension"]
                for row in local
                if "train_direction_fraction_of_internal_dimension" in row
            ],
            dtype=float,
        )
        unidentified = np.asarray(
            [
                row["unidentified_qhq_relative_frobenius"]
                for row in local
                if "unidentified_qhq_relative_frobenius" in row
            ],
            dtype=float,
        )
        summaries.append(
            {
                "run": name,
                "parent_count": len(local),
                "best_step": summary.get("best_step"),
                "wall_time_s": summary.get("wall_time_s"),
                "max_rss_mb": summary.get("max_rss_mb"),
                "median_energy_abs_error_hartree": float(np.median(energy_errors)),
                "max_energy_abs_error_hartree": float(np.max(energy_errors)),
                "median_force_mae_hartree_per_bohr": float(
                    np.median(force_errors)
                ),
                "max_force_mae_hartree_per_bohr": float(np.max(force_errors)),
                "median_train_hvp_relative_frobenius": float(
                    np.median(train_relative)
                ),
                "p90_train_hvp_relative_frobenius": float(
                    np.quantile(train_relative, 0.9)
                ),
                "max_train_hvp_relative_frobenius": float(np.max(train_relative)),
                "fraction_train_hvp_below_0p05": float(
                    np.mean(train_relative <= 0.05)
                ),
                "median_heldout_hvp_relative_frobenius": float(
                    np.median(heldout_relative)
                ),
                "p90_heldout_hvp_relative_frobenius": float(
                    np.quantile(heldout_relative, 0.9)
                ),
                "max_heldout_hvp_relative_frobenius": float(
                    np.max(heldout_relative)
                ),
                "fraction_heldout_hvp_below_0p15": float(
                    np.mean(heldout_relative <= 0.15)
                ),
                "mean_full_hessian_relative_frobenius": float(
                    np.mean(full_relative)
                ),
                "median_full_hessian_relative_frobenius": float(
                    np.median(full_relative)
                ),
                "p80_full_hessian_relative_frobenius": float(
                    np.quantile(full_relative, 0.8)
                ),
                "p90_full_hessian_relative_frobenius": float(
                    np.quantile(full_relative, 0.9)
                ),
                "max_full_hessian_relative_frobenius": float(
                    np.max(full_relative)
                ),
                "fraction_full_hessian_below_0p10": float(
                    np.mean(full_relative <= 0.10)
                ),
                "fraction_full_hessian_below_0p15": float(
                    np.mean(full_relative <= 0.15)
                ),
                "fraction_full_hessian_below_0p20": float(
                    np.mean(full_relative <= 0.20)
                ),
                "max_asymmetry_ratio": float(np.max(asymmetry)),
                "median_train_direction_fraction_of_internal_dimension": (
                    float(np.median(coverage)) if coverage.size else None
                ),
                "median_unidentified_qhq_relative_frobenius": (
                    float(np.median(unidentified)) if unidentified.size else None
                ),
                "programmatic_gate_passed": bool(
                    summary.get("stage2_direction_generalization_gate_passed", False)
                ),
            }
        )
    with (args.output_dir / "per_parent.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    direction_kind_summaries = []
    if direction_rows:
        with (args.output_dir / "per_direction.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(direction_rows[0]))
            writer.writeheader()
            writer.writerows(direction_rows)
        group_keys = sorted(
            {(row["run"], row["role"], row["kind"]) for row in direction_rows}
        )
        for run, role, kind in group_keys:
            local_directions = [
                row
                for row in direction_rows
                if row["run"] == run and row["role"] == role and row["kind"] == kind
            ]
            relative = np.asarray(
                [row["hvp_relative_l2"] for row in local_directions], dtype=float
            )
            direction_kind_summaries.append(
                {
                    "run": run,
                    "role": role,
                    "kind": kind,
                    "direction_count": len(local_directions),
                    "median_hvp_relative_l2": float(np.median(relative)),
                    "p90_hvp_relative_l2": float(np.quantile(relative, 0.9)),
                    "max_hvp_relative_l2": float(np.max(relative)),
                    "mean_hvp_mae": float(
                        np.mean([row["hvp_mae"] for row in local_directions])
                    ),
                    "mean_hvp_rmse": float(
                        np.mean([row["hvp_rmse"] for row in local_directions])
                    ),
                }
            )
        with (args.output_dir / "direction_kind_summary.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=sorted(direction_kind_summaries[0])
            )
            writer.writeheader()
            writer.writerows(direction_kind_summaries)
    if training_rows:
        with (args.output_dir / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(training_rows[0]))
            writer.writeheader()
            writer.writerows(training_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name in dict.fromkeys(row["run"] for row in rows):
        local = [row for row in rows if row["run"] == name]
        axes[0].scatter(
            [row["train_hvp_relative_frobenius"] for row in local],
            [row["heldout_hvp_relative_frobenius"] for row in local],
            label=name,
        )
        axes[1].scatter(
            [row["heldout_hvp_relative_frobenius"] for row in local],
            [row["full_hessian_relative_frobenius"] for row in local],
            label=name,
        )
    axes[0].axvline(0.05, color="black", linestyle="--")
    axes[0].axhline(0.15, color="black", linestyle="--")
    axes[0].set(xlabel="train HVP relative Fro", ylabel="held-out HVP relative Fro")
    axes[1].axvline(0.15, color="black", linestyle="--")
    axes[1].axhline(0.15, color="black", linestyle="--")
    axes[1].set(xlabel="held-out HVP relative Fro", ylabel="full Hessian relative Fro")
    axes[0].legend()
    figure.savefig(args.output_dir / "direction_generalization.png", dpi=180)
    plt.close(figure)
    if training_rows:
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for name in dict.fromkeys(row["run"] for row in training_rows):
            local_training = [row for row in training_rows if row["run"] == name]
            steps = [row["step"] for row in local_training]
            for field, label in (
                ("median_train_hvp_relative_frobenius", "train HVP"),
                ("median_heldout_hvp_relative_frobenius", "held-out HVP"),
                ("median_full_hessian_relative_frobenius", "full Hessian"),
            ):
                axes[0, 0].plot(
                    steps, [row[field] for row in local_training], label=f"{name}:{label}"
                )
            for field, label in (
                ("energy_loss", "E"),
                ("force_loss", "F"),
                ("hessian_loss", "HVP"),
                ("spectrum_loss", "spectrum"),
            ):
                axes[0, 1].plot(
                    steps, [row[field] for row in local_training], label=f"{name}:{label}"
                )
            for field, label in (
                ("gradient_norm_energy", "E"),
                ("gradient_norm_force", "F"),
                ("gradient_norm_hvp", "HVP"),
                ("gradient_norm_spectrum", "spectrum"),
            ):
                values = [row[field] for row in local_training]
                if all(value is not None for value in values):
                    axes[1, 0].plot(steps, values, label=f"{name}:{label}")
            conflict_steps = [
                row["step"]
                for row in local_training
                if row["pcgrad_conflict"] is not None
            ]
            conflicts = [
                row["pcgrad_conflict"]
                for row in local_training
                if row["pcgrad_conflict"] is not None
            ]
            if conflicts:
                axes[1, 1].plot(
                    conflict_steps, conflicts, label=f"{name}:conflict"
                )
        axes[0, 0].axhline(0.05, color="black", linestyle="--", linewidth=0.8)
        axes[0, 0].axhline(0.15, color="gray", linestyle="--", linewidth=0.8)
        axes[0, 0].set(xlabel="step", ylabel="relative error")
        axes[0, 1].set(xlabel="step", ylabel="loss", yscale="log")
        axes[1, 0].set(xlabel="step", ylabel="weighted gradient norm", yscale="log")
        axes[1, 1].set(xlabel="step", ylabel="PCGrad conflict", ylim=(-0.05, 1.05))
        for axis in axes.flat:
            axis.legend(fontsize=7, ncol=2)
        figure.savefig(args.output_dir / "training_curves.png", dpi=180)
        plt.close(figure)
    result = {
        "definition": "train-only frozen-direction Stage-2 generalization aggregation",
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "summaries": summaries,
        "direction_kind_summaries": direction_kind_summaries,
        "training_history_row_count": len(training_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
