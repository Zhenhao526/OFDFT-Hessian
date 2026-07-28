#!/usr/bin/env python3
"""Merge frozen full-Hessian tasks into evaluator and summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _plot_diagnostics(output_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [variant for variant in ("A", "B", "C", "D") if any(
        row["variant"] == variant for row in rows
    )]
    molecules = sorted({row["molecule_id"] for row in rows})
    by_key = {(row["variant"], row["molecule_id"]): row for row in rows}
    x = np.arange(len(molecules), dtype=float)
    width = 0.8 / len(variants)
    colors = {"A": "#666666", "B": "#009E73", "C": "#0072B2", "D": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for index, variant in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2) * width
        selected = [by_key[(variant, molecule)] for molecule in molecules]
        axes[0].bar(
            x + offset,
            [float(row["mae"]) for row in selected],
            width,
            color=colors[variant],
            label=variant,
        )
        axes[1].bar(
            x + offset,
            [float(row["antisymmetric_over_symmetric_fro"]) for row in selected],
            width,
            color=colors[variant],
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Hessian MAE (Ha/Bohr^2)")
    axes[0].set_title("Validation-only complete-total Hessian diagnostic")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_ylabel("||H_asym||F / ||H_sym||F")
    axes[1].set_xticks(x, molecules)
    axes[1].set_xlabel("Validation molecule")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "full_hessian_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.task_manifest.read_text())
    rows, points, resources = [], [], []
    for task in manifest["tasks"]:
        task_dir = args.task_root / (
            f"task_{int(task['task_index']):03d}_{task['run']}_{task['molecule_id']}"
        )
        path = task_dir / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        for row in result["metric_rows"]:
            rows.append({**row, **task})
        for row in result["point_rows"]:
            points.append({**row, "task_index": task["task_index"]})
        resources.append({
            **task,
            "wall_time_s": result["wall_time_s"],
            "max_rss_mb": result["max_rss_mb"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
        })
    grouped = []
    for run in sorted({row["run"] for row in rows}):
        group = [row for row in rows if row["run"] == run]
        grouped.append({
            "run": run,
            "variant": group[0]["variant"],
            "hvp_weight": group[0]["hvp_weight"],
            "seed": group[0]["seed"],
            "molecules": len(group),
            **{
                f"mean_{metric}": float(np.mean([float(row[metric]) for row in group]))
                for metric in (
                    "mae", "rmse", "relative_fro_error", "symmetrized_mae",
                    "model_symmetry_max_abs_error", "antisymmetric_over_symmetric_fro",
                    "wall_time_s", "max_displaced_gradient_norm",
                )
            },
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "full_hessian_per_molecule.csv", rows)
    _write_csv(args.output_dir / "full_hessian_points.csv", points)
    _write_csv(args.output_dir / "full_hessian_resources.csv", resources)
    _write_csv(args.output_dir / "full_hessian_groups.csv", grouped)
    _plot_diagnostics(args.output_dir, rows)
    result = {
        "definition": "Validation-only density-relaxed complete-total Hessian from scalar-derived total forces.",
        "test_accessed": False,
        "task_manifest": manifest,
        "metric_rows": rows,
        "groups": grouped,
        "resources": resources,
    }
    (args.output_dir / "full_hessian_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"groups": grouped, "tasks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
