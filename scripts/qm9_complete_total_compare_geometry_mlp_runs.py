#!/usr/bin/env python3
"""Compare train-only geometry-MLP capacity runs with paired parent metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "relative_frobenius",
    "train_hvp_relative_frobenius",
    "heldout_hvp_relative_frobenius",
    "energy_abs_error_hartree",
    "force_mae_hartree_per_bohr",
    "antisymmetric_over_symmetric_frobenius",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_specification(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR")
    return name, Path(path)


def _load_run(name: str, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "per_parent_metrics.csv"
    log_path = run_dir / "training_metrics.jsonl"
    summary = json.loads(summary_path.read_text())
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"{name} does not certify frozen Test100")
    if int(summary.get("test100_evaluations_used", 0)) != 0:
        raise ValueError(f"{name} records nonzero Test100 evaluations")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{name} has no per-parent metrics")
    parents = {}
    for row in rows:
        molecule_id = str(row["molecule_id"])
        if molecule_id in parents:
            raise ValueError(f"{name} repeats parent {molecule_id}")
        normalized: dict[str, Any] = {
            "molecule_id": molecule_id,
            "natoms": int(row["natoms"]),
        }
        for metric in METRICS:
            if metric in row and row[metric] != "":
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"{name} has non-finite {metric} for {molecule_id}")
                normalized[metric] = value
        parents[molecule_id] = normalized
    trajectory = []
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            trajectory.append(
                {
                    "run": name,
                    **{
                        key: row[key]
                        for key in (
                            "step",
                            "wall_time_s",
                            "median_relative_frobenius",
                            "max_relative_frobenius",
                            "median_train_hvp_relative_frobenius",
                            "max_train_hvp_relative_frobenius",
                            "median_heldout_hvp_relative_frobenius",
                            "max_heldout_hvp_relative_frobenius",
                            "energy_loss",
                            "force_loss",
                            "hessian_loss",
                            "gpu_peak_memory_mb",
                        )
                        if key in row
                    },
                }
            )
    return {
        "name": name,
        "run_dir": run_dir,
        "summary": summary,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "log_path": log_path,
        "parents": parents,
        "trajectory": trajectory,
    }


def _distribution(values: list[float], metric: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "metric": metric,
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p80": float(np.quantile(array, 0.8)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }
    if "relative_frobenius" in metric:
        for threshold in (0.05, 0.10, 0.15, 0.20):
            label = str(threshold).replace(".", "p")
            result[f"count_le_{label}"] = int(np.sum(array <= threshold))
            result[f"fraction_le_{label}"] = float(np.mean(array <= threshold))
    return result


def _plots(
    output_dir: Path,
    runs: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    trajectory_fields = (
        ("median_relative_frobenius", "Full Hessian median rel. Fro"),
        ("median_train_hvp_relative_frobenius", "Train-direction median rel. Fro"),
        ("median_heldout_hvp_relative_frobenius", "Held-direction median rel. Fro"),
    )
    for run in runs:
        trajectory = run["trajectory"]
        for axis, (field, label) in zip(axes, trajectory_fields, strict=True):
            points = [row for row in trajectory if field in row]
            if points:
                axis.plot(
                    [row["step"] for row in points],
                    [row[field] for row in points],
                    label=run["name"],
                )
            axis.set_xlabel("Training step")
            axis.set_ylabel(label)
            axis.axhline(0.15, color="black", linestyle=":", linewidth=1)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    path = output_dir / "capacity_direction_trajectories.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.as_posix())

    run_names = [run["name"] for run in runs]
    molecule_ids = sorted({str(row["molecule_id"]) for row in parent_rows})
    width = 0.8 / max(len(run_names), 1)
    x = np.arange(len(molecule_ids))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for metric, axis in zip(
        ("relative_frobenius", "heldout_hvp_relative_frobenius"), axes, strict=True
    ):
        for run_index, run_name in enumerate(run_names):
            lookup = {
                str(row["molecule_id"]): float(row[metric])
                for row in parent_rows
                if row["run"] == run_name and metric in row
            }
            values = [lookup.get(molecule_id, np.nan) for molecule_id in molecule_ids]
            axis.bar(
                x + (run_index - (len(run_names) - 1) / 2) * width,
                values,
                width,
                label=run_name,
            )
        axis.axhline(0.15, color="black", linestyle=":", linewidth=1)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xticks(x, molecule_ids, rotation=45, ha="right")
    path = output_dir / "per_parent_full_and_heldout.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.as_posix())
    return paths


def compare(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")
    runs = [_load_run(name, path) for name, path in args.run]
    parent_sets = [set(run["parents"]) for run in runs]
    if any(parent_set != parent_sets[0] for parent_set in parent_sets[1:]):
        raise ValueError("all compared runs must contain the same parent IDs")

    parent_rows = []
    distribution_rows = []
    for run in runs:
        for row in run["parents"].values():
            parent_rows.append({"run": run["name"], **row})
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in run["parents"].values()
                if metric in row
            ]
            if values:
                distribution_rows.append(
                    {"run": run["name"], **_distribution(values, metric)}
                )

    pairwise_rows = []
    paired_parent_rows = []
    tiny = np.finfo(np.float64).tiny
    for first, second in combinations(runs, 2):
        for metric in METRICS:
            shared_ids = [
                molecule_id
                for molecule_id in sorted(first["parents"])
                if metric in first["parents"][molecule_id]
                and metric in second["parents"][molecule_id]
            ]
            if not shared_ids:
                continue
            first_values = np.asarray(
                [first["parents"][molecule_id][metric] for molecule_id in shared_ids]
            )
            second_values = np.asarray(
                [second["parents"][molecule_id][metric] for molecule_id in shared_ids]
            )
            for molecule_id, first_value, second_value in zip(
                shared_ids, first_values, second_values, strict=True
            ):
                paired_parent_rows.append(
                    {
                        "first_run": first["name"],
                        "second_run": second["name"],
                        "metric": metric,
                        "molecule_id": molecule_id,
                        "natoms": first["parents"][molecule_id]["natoms"],
                        "first_value": float(first_value),
                        "second_value": float(second_value),
                        "second_minus_first": float(second_value - first_value),
                        "second_over_first": float(
                            second_value / max(first_value, tiny)
                        ),
                        "winner": (
                            second["name"]
                            if second_value < first_value
                            else first["name"]
                            if first_value < second_value
                            else "tie"
                        ),
                    }
                )
            pairwise_rows.append(
                {
                    "first_run": first["name"],
                    "second_run": second["name"],
                    "metric": metric,
                    "parent_count": len(shared_ids),
                    "second_wins": int(np.sum(second_values < first_values)),
                    "first_wins": int(np.sum(first_values < second_values)),
                    "ties": int(np.sum(first_values == second_values)),
                    "median_second_over_first": float(
                        np.median(second_values / np.maximum(first_values, tiny))
                    ),
                    "median_second_minus_first": float(
                        np.median(second_values - first_values)
                    ),
                }
            )

    trajectory_rows = [row for run in runs for row in run["trajectory"]]
    _write_csv(args.output_dir / "per_parent_metrics.csv", parent_rows)
    _write_csv(args.output_dir / "run_distributions.csv", distribution_rows)
    _write_csv(args.output_dir / "pairwise_comparisons.csv", pairwise_rows)
    _write_csv(args.output_dir / "paired_parent_differences.csv", paired_parent_rows)
    _write_csv(args.output_dir / "training_trajectories.csv", trajectory_rows)
    result = {
        "definition": "train-parent-only paired conservative geometry-MLP run comparison",
        "runs": [
            {
                "name": run["name"],
                "run_dir": run["run_dir"].as_posix(),
                "summary": run["summary_path"].as_posix(),
                "summary_sha256": _sha256(run["summary_path"]),
                "per_parent_metrics": run["metrics_path"].as_posix(),
                "per_parent_metrics_sha256": _sha256(run["metrics_path"]),
                "training_log_sha256": (
                    _sha256(run["log_path"]) if run["log_path"].is_file() else None
                ),
                "best_checkpoint": (
                    (run["run_dir"] / "best.ckpt").as_posix()
                    if (run["run_dir"] / "best.ckpt").is_file()
                    else None
                ),
                "best_checkpoint_sha256": (
                    _sha256(run["run_dir"] / "best.ckpt")
                    if (run["run_dir"] / "best.ckpt").is_file()
                    else None
                ),
                "source_checkpoint": run["summary"].get("source_checkpoint"),
                "hidden_size": run["summary"].get("hidden_size"),
                "best_step": run["summary"].get("best_step"),
                "wall_time_s": run["summary"].get("wall_time_s"),
                "max_rss_mb": run["summary"].get("max_rss_mb"),
                "peak_gpu_memory_mb": (
                    max(
                        (
                            float(row["gpu_peak_memory_mb"])
                            for row in run["trajectory"]
                            if "gpu_peak_memory_mb" in row
                        ),
                        default=None,
                    )
                ),
                "parent_count": len(run["parents"]),
            }
            for run in runs
        ],
        "parent_ids": sorted(parent_sets[0]),
        "distributions": distribution_rows,
        "pairwise_comparisons": pairwise_rows,
        "paired_parent_difference_count": len(paired_parent_rows),
        "plots": _plots(args.output_dir, runs, parent_rows),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_specification, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
