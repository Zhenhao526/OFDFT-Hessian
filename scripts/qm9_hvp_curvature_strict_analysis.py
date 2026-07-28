#!/usr/bin/env python3
"""Aggregate stable-only complete-total HVP validation and freeze full-Hessian candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_force(stage1_root: Path, run_name: str) -> dict[str, Any]:
    return json.loads((stage1_root / run_name / "validation_energy_force.json").read_text())


def _task_output(strict_root: Path, task: dict[str, str]) -> Path:
    base = strict_root / task["run_name"] / f"direction_{int(task['direction_index'])}"
    molecule = task["molecules"]
    isolated = base / molecule
    # Keep old grouped fixtures/results readable; newly generated tasks are isolated.
    if isolated.exists() or not (base / "metrics.csv").is_file():
        return isolated
    return base


def _failure(task: dict[str, str], output: Path, stage: str, reason: str) -> dict[str, Any]:
    return {
        "task_index": int(task["task_index"]),
        "run_name": task["run_name"],
        "variant": task["variant"],
        "seed": int(task["seed"]),
        "direction_index": int(task["direction_index"]),
        "molecules": task["molecules"],
        "output_dir": output.as_posix(),
        "stage": stage,
        "reason": reason,
    }


def _plot_diagnostics(output_dir: Path, groups: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compared = [group for group in groups if group["variant"] != "A"]
    if not compared:
        return
    colors = {"B": "#009E73", "C": "#0072B2", "D": "#D55E00", "E": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for group in compared:
        ax.scatter(
            100 * group["mean_force_relative_to_A"],
            100 * group["mean_strict_hvp_mae_relative_to_A"],
            s=75,
            color=colors[group["variant"]],
            label=group["variant"],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation force MAE change vs A (%)")
    ax.set_ylabel("Stable-only strict HVP MAE change vs A (%)")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    fig.tight_layout()
    fig.savefig(output_dir / "strict_force_hvp_tradeoff.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    promotion = protocol["promotion"]
    tasks = _read_tsv(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    task_resource_rows = []
    failures: list[dict[str, Any]] = []
    expected_by_run: dict[str, set[tuple[str, int]]] = defaultdict(set)
    observed_by_run: dict[str, set[tuple[str, int]]] = defaultdict(set)
    failed_runs: set[str] = set()
    for task in tasks:
        expected = set(task["molecules"].split(","))
        direction_index = int(task["direction_index"])
        expected_by_run[task["run_name"]].update(
            (molecule, direction_index) for molecule in expected
        )
        output = _task_output(args.strict_root, task)
        metrics_path = output / "metrics.csv"
        summary_path = output / "summary.json"
        try:
            if not metrics_path.is_file() or not summary_path.is_file():
                missing = [
                    path.name for path in (metrics_path, summary_path) if not path.is_file()
                ]
                raise FileNotFoundError(f"missing {', '.join(missing)}")
            with metrics_path.open() as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(summary_path.read_text())
            observed = {row["molecule_id"] for row in rows}
            if observed != expected:
                raise ValueError(f"molecule mismatch: {observed} != {expected}")
            parsed_rows = []
            for row in rows:
                arrays_path = (
                    output
                    / f"{task['run_name']}_{row['molecule_id']}_0000000_hvp_arrays.npz"
                )
                with np.load(arrays_path) as payload:
                    pbe_hvp = np.asarray(payload["pbe_hvp"], dtype=np.float64)
                reference_hvp_rms = float(np.sqrt(np.mean(pbe_hvp**2)))
                parsed = {
                    "run_name": task["run_name"],
                    "variant": task["variant"],
                    "seed": int(task["seed"]),
                    "curvature_weight": float(task["curvature_weight"]),
                    "reference_floor": float(task["reference_floor"]),
                    "molecule_id": row["molecule_id"],
                    "direction_index": direction_index,
                    "direction_kind": row["direction_kind"],
                    "natoms": int(row["natoms"]),
                    "strict_hvp_mae": float(row["strict_relaxed_vs_pbe_mae"]),
                    "strict_hvp_rmse": float(row["strict_relaxed_vs_pbe_rmse"]),
                    "strict_hvp_relative_frobenius": float(
                        row["strict_relaxed_vs_pbe_relative_frobenius"]
                    ),
                    "reference_hvp_rms": reference_hvp_rms,
                    "small_reference": reference_hvp_rms
                    < float(promotion["small_reference_hvp_rms"]),
                    "max_density_gradient": float(row["strict_relaxed_max_gradient_norm"]),
                    "response_relative_residual": float(row["response_relative_residual"]),
                    "wall_time_s": float(row["wall_time_s"]),
                }
                numeric = [
                    parsed["strict_hvp_mae"],
                    parsed["strict_hvp_rmse"],
                    parsed["strict_hvp_relative_frobenius"],
                    parsed["max_density_gradient"],
                    parsed["response_relative_residual"],
                ]
                if not np.all(np.isfinite(numeric)):
                    raise ValueError(f"non-finite strict metrics for {row['molecule_id']}")
                parsed_rows.append(parsed)
            metric_rows.extend(parsed_rows)
            observed_by_run[task["run_name"]].update(
                (row["molecule_id"], direction_index) for row in parsed_rows
            )
            task_resource_rows.append(
                {
                    "run_name": task["run_name"],
                    "molecules": task["molecules"],
                    "direction_index": direction_index,
                    "wall_time_s": float(summary["wall_time_s"]),
                    "max_rss_mb": float(summary["max_rss_mb"]),
                    "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
                }
            )
        except Exception as exc:
            failed_runs.add(task["run_name"])
            failures.append(_failure(task, output, "task_output", f"{type(exc).__name__}: {exc}"))

    complete_runs = {
        run_name
        for run_name, expected in expected_by_run.items()
        if run_name not in failed_runs and observed_by_run[run_name] == expected
    }

    baseline_runs = {
        int(task["seed"]): task["run_name"]
        for task in tasks
        if task["variant"] == "A" and task["run_name"] in complete_runs
    }
    baseline_values = {
        (row["seed"], row["molecule_id"], row["direction_index"]): row
        for row in metric_rows
        if row["variant"] == "A"
    }
    run_rows = []
    task_metadata = {task["run_name"]: task for task in tasks}
    for run_name in sorted(complete_runs):
        members = [row for row in metric_rows if row["run_name"] == run_name]
        task = task_metadata[run_name]
        first = members[0]
        output = args.strict_root / run_name
        try:
            baseline_run = baseline_runs[first["seed"]]
            force = _load_force(args.stage1_root, run_name)
            baseline_force = _load_force(args.stage1_root, baseline_run)
            baseline_members = [
                baseline_values[
                    (first["seed"], row["molecule_id"], row["direction_index"])
                ]
                for row in members
            ]
        except Exception as exc:
            failures.append(
                _failure(task, output, "run_aggregation", f"{type(exc).__name__}: {exc}")
            )
            continue
        direction_wins = np.mean(
            [
                row["strict_hvp_mae"] < baseline["strict_hvp_mae"]
                for row, baseline in zip(members, baseline_members)
            ]
        )
        parent_values = {}
        baseline_parent_values = {}
        for row, baseline in zip(members, baseline_members):
            parent_values.setdefault(row["molecule_id"], []).append(row["strict_hvp_mae"])
            baseline_parent_values.setdefault(row["molecule_id"], []).append(
                baseline["strict_hvp_mae"]
            )
        parent_wins = np.mean(
            [
                np.mean(parent_values[molecule])
                < np.mean(baseline_parent_values[molecule])
                for molecule in parent_values
            ]
        )
        resources = [row for row in task_resource_rows if row["run_name"] == run_name]
        baseline_mean = float(np.mean([row["strict_hvp_mae"] for row in baseline_members]))
        run_rows.append(
            {
                "run_name": run_name,
                "variant": first["variant"],
                "seed": first["seed"],
                "curvature_weight": first["curvature_weight"],
                "reference_floor": first["reference_floor"],
                "directions": len(members),
                "parents": len(parent_values),
                "energy_mae": float(force["energy_mae"]),
                "force_mae": float(force["force_component_mae"]),
                "energy_relative_to_A": float(force["energy_mae"] / baseline_force["energy_mae"] - 1),
                "force_relative_to_A": float(force["force_component_mae"] / baseline_force["force_component_mae"] - 1),
                "strict_hvp_mae": float(np.mean([row["strict_hvp_mae"] for row in members])),
                "strict_hvp_rmse": float(np.mean([row["strict_hvp_rmse"] for row in members])),
                "strict_hvp_relative_frobenius": float(
                    np.mean([row["strict_hvp_relative_frobenius"] for row in members])
                ),
                "strict_hvp_mae_relative_to_A": float(
                    np.mean([row["strict_hvp_mae"] for row in members]) / baseline_mean - 1
                ),
                "direction_win_fraction": float(direction_wins),
                "parent_win_fraction": float(parent_wins),
                "small_reference_directions": sum(row["small_reference"] for row in members),
                "max_density_gradient": max(row["max_density_gradient"] for row in members),
                "max_response_relative_residual": max(
                    row["response_relative_residual"] for row in members
                ),
                "wall_time_s": sum(row["wall_time_s"] for row in resources),
                "max_rss_mb": max(row["max_rss_mb"] for row in resources),
                "peak_gpu_memory_mb": max(row["peak_gpu_memory_mb"] for row in resources),
            }
        )

    groups = []
    keys = sorted(
        {
            (row["variant"], row["curvature_weight"], row["reference_floor"])
            for row in run_rows
        }
    )
    for key in keys:
        members = [
            row
            for row in run_rows
            if (row["variant"], row["curvature_weight"], row["reference_floor"]) == key
        ]
        group = {
            "variant": key[0],
            "curvature_weight": key[1],
            "reference_floor": key[2],
            "seeds": len(members),
            "energy_gate_passed_seeds": sum(
                row["energy_relative_to_A"]
                <= float(promotion["energy_mae_relative_degradation_max"])
                for row in members
            ),
            "force_improved_seeds": sum(row["force_relative_to_A"] < 0 for row in members),
            "strict_hvp_improved_seeds": sum(
                row["strict_hvp_mae_relative_to_A"] < 0 for row in members
            ),
            "majority_direction_seeds": sum(
                row["direction_win_fraction"] > 0.5 for row in members
            ),
            "majority_parent_seeds": sum(
                row["parent_win_fraction"] > 0.5 for row in members
            ),
            "numerically_converged_seeds": sum(
                row["max_density_gradient"]
                <= float(
                    protocol["strict_complete_total_hvp"]["fallback"][
                        "projected_gradient_threshold"
                    ]
                )
                and row["max_response_relative_residual"]
                <= float(
                    protocol["strict_complete_total_hvp"]["response"][
                        "relative_residual_threshold"
                    ]
                )
                for row in members
            ),
        }
        for metric in (
            "energy_relative_to_A",
            "force_relative_to_A",
            "strict_hvp_mae_relative_to_A",
            "strict_hvp_mae",
            "strict_hvp_relative_frobenius",
            "direction_win_fraction",
            "parent_win_fraction",
        ):
            group[f"mean_{metric}"] = float(np.mean([row[metric] for row in members]))
        group["strict_gate_passed"] = bool(
            group["numerically_converged_seeds"] == len(members)
            and len(members) == int(promotion["energy_gate_required_seeds"])
            and group["energy_gate_passed_seeds"]
            == int(promotion["energy_gate_required_seeds"])
            and group["force_improved_seeds"] >= int(promotion["force_improved_seeds_min"])
            and group["strict_hvp_improved_seeds"]
            >= int(promotion["strict_hvp_improved_seeds_min"])
            and group["majority_direction_seeds"] >= 2
            and group["majority_parent_seeds"] >= 2
            and group["mean_direction_win_fraction"] > 0.5
            and group["mean_parent_win_fraction"] > 0.5
        )
        groups.append(group)

    eligible = [
        group for group in groups if group["variant"] != "A" and group["strict_gate_passed"]
    ]
    eligible.sort(
        key=lambda group: (
            group["mean_strict_hvp_mae_relative_to_A"]
            + group["mean_force_relative_to_A"],
            group["mean_energy_relative_to_A"],
        )
    )
    frozen_candidates = []
    for group in eligible[: int(promotion["full_hessian_candidates_max"])]:
        members = [
            row
            for row in run_rows
            if row["variant"] == group["variant"]
            and row["curvature_weight"] == group["curvature_weight"]
            and row["reference_floor"] == group["reference_floor"]
        ]
        selected = sorted(members, key=lambda row: row["strict_hvp_mae"])[len(members) // 2]
        frozen_candidates.append(
            {
                "variant": group["variant"],
                "curvature_weight": group["curvature_weight"],
                "reference_floor": group["reference_floor"],
                "run_name": selected["run_name"],
                "seed": selected["seed"],
                "seed_selection": "median stable-only strict HVP MAE",
            }
        )

    _write_csv(args.output_dir / "strict_per_direction.csv", metric_rows)
    _write_csv(args.output_dir / "strict_runs.csv", run_rows)
    _write_csv(args.output_dir / "strict_groups.csv", groups)
    _write_csv(args.output_dir / "strict_failures.csv", failures)
    _plot_diagnostics(args.output_dir, groups)
    result = {
        "definition": __doc__,
        "protocol": str(args.protocol.resolve()),
        "test100_accessed": False,
        "runs": run_rows,
        "groups": groups,
        "expected_task_count": len(tasks),
        "completed_task_count": len(tasks) - sum(
            failure["stage"] == "task_output" for failure in failures
        ),
        "failed_task_count": sum(
            failure["stage"] == "task_output" for failure in failures
        ),
        "complete_run_count": len(run_rows),
        "incomplete_runs": sorted(set(expected_by_run) - {row["run_name"] for row in run_rows}),
        "failures": failures,
        "frozen_candidates_for_full_hessian": frozen_candidates,
        "test100_allowed_after_full_hessian_gate": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"groups": groups, "candidates": frozen_candidates}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
