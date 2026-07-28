#!/usr/bin/env python3
"""Merge full Hessians, evaluate vibrations, and apply the frozen non-regression gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from qm9_hessian_vibrational_metrics import evaluate as evaluate_vibrations


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostics(output_dir: Path, comparisons: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not comparisons:
        return
    runs = sorted({row["run_name"] for row in comparisons})
    molecules = sorted({row["molecule_id"] for row in comparisons})
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True)
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        selected = {row["molecule_id"]: row for row in comparisons if row["run_name"] == run}
        offset = (index - (len(runs) - 1) / 2) * width
        axes[0].bar(
            np.arange(len(molecules)) + offset,
            [100 * selected[molecule]["mae_relative_to_A"] for molecule in molecules],
            width,
            label=run,
        )
        axes[1].bar(
            np.arange(len(molecules)) + offset,
            [100 * selected[molecule]["frequency_mae_relative_to_A"] for molecule in molecules],
            width,
        )
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Hessian MAE change vs A (%)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Frequency MAE change vs A (%)")
    axes[1].set_xticks(np.arange(len(molecules)), molecules, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "full_hessian_vibration_tradeoff.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict:
    protocol = yaml.safe_load(args.protocol.read_text())
    manifest = json.loads(args.task_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, points, resources, failures = [], [], [], []
    expected_by_run: dict[str, set[str]] = {}
    observed_by_run: dict[str, set[str]] = {}
    failed_runs: set[str] = set()
    for task in manifest["tasks"]:
        expected_by_run.setdefault(task["run_name"], set()).add(task["molecule_id"])
        task_dir = args.task_root / (
            f"task_{int(task['task_index']):03d}_{task['run_name']}_{task['molecule_id']}"
        )
        try:
            summary_path = task_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            result = json.loads(summary_path.read_text())
            if len(result.get("metric_rows", [])) != 1:
                raise ValueError(f"expected one Hessian metric, got {len(result.get('metric_rows', []))}")
            metric = {**result["metric_rows"][0], **task}
            if metric.get("success") is not True:
                raise ValueError("Hessian metric did not report success=true")
            if metric["molecule_id"] != task["molecule_id"]:
                raise ValueError(
                    f"molecule mismatch: {metric['molecule_id']} != {task['molecule_id']}"
                )
            required = [
                "mae",
                "rmse",
                "relative_fro_error",
                "antisymmetric_over_symmetric_fro",
                "force_vs_energy_diagonal_mae",
            ]
            if not np.all(np.isfinite([float(metric[key]) for key in required])):
                raise ValueError("non-finite full-Hessian metric")
            rows.append(metric)
            observed_by_run.setdefault(task["run_name"], set()).add(task["molecule_id"])
            points.extend(
                {**row, "task_index": task["task_index"]}
                for row in result.get("point_rows", [])
            )
            resources.append(
                {
                    **task,
                    "wall_time_s": result["wall_time_s"],
                    "max_rss_mb": result["max_rss_mb"],
                    "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                }
            )
        except Exception as exc:
            failed_runs.add(task["run_name"])
            failures.append(
                {
                    **task,
                    "task_dir": task_dir.as_posix(),
                    "stage": "task_output",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    complete_runs = {
        run
        for run, expected in expected_by_run.items()
        if run not in failed_runs and observed_by_run.get(run, set()) == expected
    }
    _write_csv(args.output_dir / "full_hessian_per_molecule.csv", rows)
    _write_csv(args.output_dir / "full_hessian_points.csv", points)
    _write_csv(args.output_dir / "full_hessian_resources.csv", resources)
    _write_csv(args.output_dir / "full_hessian_failures.csv", failures)
    combined = {
        "definition": "Validation-only strict complete-total Hessian results.",
        "test100_accessed": False,
        "metric_rows": rows,
        "resources": resources,
        "failures": failures,
    }
    combined_path = args.output_dir / "full_hessian_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    if rows:
        vibration_dir = args.output_dir / "vibrations"
        vibration = evaluate_vibrations(
            SimpleNamespace(
                manifest_json=Path(manifest["pbe_manifest"]),
                dataset_dir=args.dataset_dir,
                result_json=[f"full={combined_path}"],
                output_dir=vibration_dir,
                imaginary_threshold_cm=1.0,
            )
        )
    else:
        vibration = {"rows": [], "summaries": []}
    vibration_map = {(row["run"], row["molecule_id"]): row for row in vibration["rows"]}
    baseline = {
        (int(row["seed"]), row["molecule_id"]): row for row in rows if row["variant"] == "A"
    }
    comparisons = []
    for row in rows:
        if row["variant"] == "A" or row["run_name"] not in complete_runs:
            continue
        try:
            reference = baseline[(int(row["seed"]), row["molecule_id"])]
            if reference["run_name"] not in complete_runs:
                raise KeyError(f"matched baseline run incomplete: {reference['run_name']}")
            vib = vibration_map[(row["run_name"], row["molecule_id"])]
            base_vib = vibration_map[(reference["run_name"], row["molecule_id"])]
        except Exception as exc:
            failures.append(
                {
                    "run_name": row["run_name"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "molecule_id": row["molecule_id"],
                    "stage": "matched_comparison",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        floor = np.finfo(float).tiny
        comparisons.append(
            {
                "run_name": row["run_name"],
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "curvature_weight": float(row["curvature_weight"]),
                "reference_floor": float(row["reference_floor"]),
                "molecule_id": row["molecule_id"],
                "mae_relative_to_A": float(row["mae"] / max(reference["mae"], floor) - 1),
                "rmse_relative_to_A": float(row["rmse"] / max(reference["rmse"], floor) - 1),
                "relative_fro_relative_to_A": float(
                    row["relative_fro_error"]
                    / max(reference["relative_fro_error"], floor)
                    - 1
                ),
                "symmetry_relative_to_A": float(
                    row["antisymmetric_over_symmetric_fro"]
                    / max(reference["antisymmetric_over_symmetric_fro"], np.finfo(float).tiny)
                    - 1
                ),
                "frequency_mae_relative_to_A": float(
                    vib["frequency_mae_cm-1"]
                    / max(base_vib["frequency_mae_cm-1"], np.finfo(float).tiny)
                    - 1
                ),
                "mode_overlap_change_from_A": float(
                    vib["mean_mode_overlap"] - base_vib["mean_mode_overlap"]
                ),
                "candidate_imaginary_count_error_abs": abs(vib["imaginary_mode_count_error"]),
                "baseline_imaginary_count_error_abs": abs(
                    base_vib["imaginary_mode_count_error"]
                ),
                "force_energy_diagonal_mae_relative_to_A": float(
                    row["force_vs_energy_diagonal_mae"]
                    / max(reference["force_vs_energy_diagonal_mae"], np.finfo(float).tiny)
                    - 1
                ),
                "candidate_all_points_strict": int(row["strict_points"])
                == int(row["total_displaced_points"]),
                "baseline_all_points_strict": int(reference["strict_points"])
                == int(reference["total_displaced_points"]),
            }
        )
    groups = []
    expected_parent_count = len(manifest.get("molecules", []))
    for run_name in sorted({row["run_name"] for row in comparisons}):
        members = [row for row in comparisons if row["run_name"] == run_name]
        group = {
            "run_name": run_name,
            "variant": members[0]["variant"],
            "seed": members[0]["seed"],
            "molecules": len(members),
            "expected_molecules": expected_parent_count,
            "full_data_complete": len(members) == expected_parent_count,
            "mean_mae_relative_to_A": float(np.mean([row["mae_relative_to_A"] for row in members])),
            "mean_relative_fro_relative_to_A": float(
                np.mean([row["relative_fro_relative_to_A"] for row in members])
            ),
            "mean_rmse_relative_to_A": float(
                np.mean([row["rmse_relative_to_A"] for row in members])
            ),
            "hessian_win_fraction": float(np.mean([row["mae_relative_to_A"] < 0 for row in members])),
            "mean_symmetry_relative_to_A": float(
                np.mean([row["symmetry_relative_to_A"] for row in members])
            ),
            "mean_frequency_mae_relative_to_A": float(
                np.mean([row["frequency_mae_relative_to_A"] for row in members])
            ),
            "mean_mode_overlap_change_from_A": float(
                np.mean([row["mode_overlap_change_from_A"] for row in members])
            ),
            "candidate_imaginary_count_error_abs": int(
                sum(row["candidate_imaginary_count_error_abs"] for row in members)
            ),
            "baseline_imaginary_count_error_abs": int(
                sum(row["baseline_imaginary_count_error_abs"] for row in members)
            ),
            "mean_force_energy_diagonal_mae_relative_to_A": float(
                np.mean(
                    [row["force_energy_diagonal_mae_relative_to_A"] for row in members]
                )
            ),
            "all_points_strict": all(
                row["candidate_all_points_strict"]
                and row["baseline_all_points_strict"]
                for row in members
            ),
        }
        group["full_hessian_vibration_gate"] = bool(
            group["full_data_complete"]
            and group["all_points_strict"]
            and group["mean_mae_relative_to_A"] < 0
            and group["mean_rmse_relative_to_A"] < 0
            and group["mean_relative_fro_relative_to_A"] < 0
            and group["hessian_win_fraction"] > 0.5
            and group["mean_symmetry_relative_to_A"] <= 0.05
            and group["mean_frequency_mae_relative_to_A"] <= 0
            and group["mean_mode_overlap_change_from_A"] >= 0
            and group["mean_force_energy_diagonal_mae_relative_to_A"]
            <= float(
                protocol["full_hessian"][
                    "force_energy_diagonal_relative_degradation_max"
                ]
            )
            and group["candidate_imaginary_count_error_abs"]
            <= group["baseline_imaginary_count_error_abs"]
        )
        groups.append(group)
    promoted = [group for group in groups if group["full_hessian_vibration_gate"]]
    promoted.sort(key=lambda row: (row["mean_mae_relative_to_A"], row["mean_frequency_mae_relative_to_A"]))
    _write_csv(args.output_dir / "candidate_vs_A_per_molecule.csv", comparisons)
    _write_csv(args.output_dir / "candidate_groups.csv", groups)
    _write_csv(args.output_dir / "full_hessian_failures.csv", failures)
    _plot_diagnostics(args.output_dir, comparisons)
    result = {
        "definition": __doc__,
        "protocol": str(args.protocol.resolve()),
        "test100_accessed": False,
        "groups": groups,
        "expected_task_count": len(manifest["tasks"]),
        "completed_task_count": len(rows),
        "failed_task_count": sum(row["stage"] == "task_output" for row in failures),
        "complete_runs": sorted(complete_runs),
        "incomplete_runs": sorted(set(expected_by_run) - complete_runs),
        "failures": failures,
        "validation_promoted_candidates": promoted,
        "test100_allowed": bool(promoted),
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "promotion_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
