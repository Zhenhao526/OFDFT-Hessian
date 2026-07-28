#!/usr/bin/env python3
"""Summarize preregistered train-only parent-CV spectral/operator variants."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_run(value: str) -> tuple[str, Path]:
    name, path = value.split("=", maxsplit=1)
    return name, Path(path).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    registered_ids = [
        str(variant["id"]) for variant in protocol["preregistered_variants"]
    ]
    runs = []
    all_rows = []
    parent_ids: list[str] | None = None
    fold_assignment = None
    for name, run_dir in map(_parse_run, args.run):
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"{name} does not certify frozen Test100")
        if int(summary.get("test100_evaluations_used", 0)) != 0:
            raise ValueError(f"{name} records Test100 access")
        if summary.get("protocol_sha256") != _sha256(args.protocol):
            raise ValueError(f"{name} protocol hash differs")
        if summary.get("variant_id") != name:
            raise ValueError(f"{name} directory contains another variant")
        local_ids = [str(row["molecule_id"]) for row in summary["per_parent"]]
        if parent_ids is None:
            parent_ids = local_ids
            fold_assignment = summary["fold_assignment"]
        elif local_ids != parent_ids or summary["fold_assignment"] != fold_assignment:
            raise ValueError("variant parent IDs, order, or folds differ")
        vibration_path = run_dir / "vibrational_metrics" / "summary.json"
        vibration = json.loads(vibration_path.read_text())["summaries"][0]
        metrics = summary["selected_cross_parent_metrics"]
        run_row = {
            "variant_id": name,
            "selected_ridge": summary["selected_ridge"],
            "parent_feature_transform": summary["parent_feature_transform"],
            "parent_feature_transform_scale": summary[
                "parent_feature_transform_scale"
            ],
            "block_parent_conditioning": summary["block_parent_conditioning"],
            "max_correction_to_source": summary["max_correction_to_source"],
            **metrics,
            "parent_win_fraction_vs_source": summary[
                "parent_win_fraction_vs_source"
            ],
            "cross_parent_gate_passed": summary["cross_parent_gate"]["passed"],
            "frequency_mae_cm-1": vibration["mean_frequency_mae_cm-1"],
            "frequency_rmse_cm-1": vibration["mean_frequency_rmse_cm-1"],
            "mean_mode_overlap": vibration["mean_mode_overlap"],
            "model_imaginary_modes": vibration["total_model_imaginary_modes"],
            "pbe_imaginary_modes": vibration["total_pbe_imaginary_modes"],
            "summary": summary_path.as_posix(),
            "summary_sha256": _sha256(summary_path),
            "vibrational_summary": vibration_path.as_posix(),
            "vibrational_summary_sha256": _sha256(vibration_path),
        }
        runs.append((name, run_dir, summary, run_row))
        all_rows.extend({"variant_id": name, **row} for row in summary["per_parent"])
    if [name for name, _, _, _ in runs] != registered_ids:
        raise ValueError("run order or IDs differ from the preregistered variants")

    source_vibration = json.loads(args.source_vibrational_summary.read_text())[
        "summaries"
    ][0]
    passing = [row["variant_id"] for _, _, _, row in runs if row["cross_parent_gate_passed"]]
    diagnostic_best = min(
        [row for _, _, _, row in runs],
        key=lambda row: (
            -row["fraction_relative_frobenius_at_or_below_0_15"],
            row["p90_relative_frobenius"],
            row["median_relative_frobenius"],
            -row["selected_ridge"],
        ),
    )["variant_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [row for _, _, _, row in runs]
    _write_csv(args.output_dir / "variant_summary.csv", run_rows)
    _write_csv(args.output_dir / "per_parent_all_variants.csv", all_rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    source_relative = np.asarray(
        [float(row["source_relative_frobenius"]) for row in runs[0][2]["per_parent"]]
    )
    probabilities = np.arange(1, source_relative.size + 1) / source_relative.size
    axes[0, 0].plot(
        probabilities,
        np.sort(source_relative),
        color="black",
        linestyle="--",
        label="original-A source",
    )
    for name, _, summary, _ in runs:
        relative = np.sort(
            [float(row["relative_frobenius"]) for row in summary["per_parent"]]
        )
        axes[0, 0].plot(probabilities, relative, marker=".", label=name)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(0.10, color="black", linewidth=1)
    axes[0, 0].axhline(0.20, color="gray", linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Empirical CDF")
    axes[0, 0].set_ylabel("Held-parent Hessian relative Frobenius")
    axes[0, 0].legend(fontsize=7)

    labels = [row["variant_id"].split("_")[0] for row in run_rows]
    x = np.arange(len(run_rows))
    axes[0, 1].bar(x - 0.2, [row["median_relative_frobenius"] for row in run_rows], width=0.4, label="median")
    axes[0, 1].bar(x + 0.2, [row["p90_relative_frobenius"] for row in run_rows], width=0.4, label="P90")
    axes[0, 1].axhline(0.10, color="black", linewidth=1)
    axes[0, 1].axhline(0.20, color="gray", linestyle=":", linewidth=1)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Relative Frobenius")
    axes[0, 1].legend()

    axes[1, 0].bar(x, [row["frequency_mae_cm-1"] for row in run_rows])
    axes[1, 0].axhspan(100.0, 200.0, color="green", alpha=0.15)
    axes[1, 0].axhline(
        source_vibration["mean_frequency_mae_cm-1"],
        color="black",
        linestyle="--",
        label="source",
    )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Frequency MAE (cm-1)")
    axes[1, 0].legend()

    axes[1, 1].bar(x, [row["model_imaginary_modes"] for row in run_rows])
    axes[1, 1].axhline(
        source_vibration["total_pbe_imaginary_modes"],
        color="black",
        label="PBE",
    )
    axes[1, 1].axhline(
        source_vibration["total_model_imaginary_modes"],
        color="gray",
        linestyle="--",
        label="source",
    )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Imaginary mode count")
    axes[1, 1].legend()
    plot_path = args.output_dir / "parent_cv_variant_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    result = {
        "definition": (
            "Preregistered train20 parent-held-out comparison; no validation or "
            "Test100 parent is used for fitting, ridge selection, or ranking."
        ),
        "parent_ids": parent_ids,
        "fold_assignment": fold_assignment,
        "source_hessian_metrics": runs[0][2]["source_cross_parent_metrics"],
        "source_vibrational_metrics": source_vibration,
        "variants": run_rows,
        "passing_variants": passing,
        "diagnostic_best_nonpromoted_variant": diagnostic_best,
        "advancement_authorized": bool(passing),
        "decision": (
            "freeze one passing variant before external validation"
            if passing
            else "no variant passes; do not access validation or expand train100"
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_vibrational_summary": args.source_vibrational_summary.resolve().as_posix(),
        "source_vibrational_summary_sha256": _sha256(args.source_vibrational_summary),
        "variant_summary_csv": (args.output_dir / "variant_summary.csv").resolve().as_posix(),
        "per_parent_csv": (args.output_dir / "per_parent_all_variants.csv").resolve().as_posix(),
        "plot": plot_path.resolve().as_posix(),
        "plot_sha256": _sha256(plot_path),
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
    parser.add_argument("--run", action="append", required=True, help="VARIANT_ID=RUN_DIR")
    parser.add_argument("--source-vibrational-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
