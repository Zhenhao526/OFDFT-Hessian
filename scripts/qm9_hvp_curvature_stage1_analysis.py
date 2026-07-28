#!/usr/bin/env python3
"""Apply frozen Tier-1 gates to A-E validation outputs without consulting Test100."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _load_run(stage1_root: Path, task: dict[str, str]) -> dict[str, Any]:
    root = stage1_root / task["run_name"]
    required = {
        "validation": root / "validation_energy_force.json",
        "train700": root / "train700_forgetting.json",
        "fixed_hvp": root / "fixed_hvp/summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Tier-1 outputs for {task['run_name']}: {missing}")
    return {
        "task": task,
        **{key: json.loads(path.read_text()) for key, path in required.items()},
    }


def _relative(value: float, baseline: float) -> float:
    return float(value / baseline - 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_maps(payload: dict[str, Any]) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    directions = {
        (row["molecule_id"], int(row["direction_index"])): float(
            row["corrected_vs_pbe_mae"]
        )
        for row in payload["rows"]
    }
    parents: dict[str, list[float]] = {}
    for (molecule_id, _), value in directions.items():
        parents.setdefault(molecule_id, []).append(value)
    parent_means = {
        molecule_id: sum(values) / len(values)
        for molecule_id, values in parents.items()
    }
    return directions, parent_means


def _plot_diagnostics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"A": "#666666", "B": "#009E73", "C": "#0072B2", "D": "#D55E00", "E": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for variant in ("A", "B", "C", "D", "E"):
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        ax.scatter(
            [100 * row["validation_force_relative_to_A"] for row in selected],
            [100 * row["fixed_hvp_mae_relative_to_A"] for row in selected],
            s=55,
            color=colors[variant],
            label=variant,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation force MAE change vs matched A (%)")
    ax.set_ylabel("Fixed-density HVP MAE change vs matched A (%)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tier1_force_hvp_tradeoff.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    minimum_stable_parents = int(
        protocol["validation"]["fixed_density_hvp_minimum_stable_parent_count"]
    )
    tasks = _read_tasks(args.tasks)
    runs = [_load_run(args.stage1_root, task) for task in tasks]
    baselines = [run for run in runs if run["task"]["variant"] == "A"]
    baseline_by_seed = {int(run["task"]["seed"]): run for run in baselines}
    if len(baseline_by_seed) != len(baselines):
        raise ValueError("Expected exactly one A baseline per seed")
    task_seeds = {int(run["task"]["seed"]) for run in runs}
    if set(baseline_by_seed) != task_seeds:
        raise ValueError(
            f"A baseline seeds {sorted(baseline_by_seed)} do not match task seeds "
            f"{sorted(task_seeds)}"
        )
    rows = []
    for run in runs:
        task = run["task"]
        seed = int(task["seed"])
        baseline = baseline_by_seed[seed]
        base_directions, base_parents = _fixed_maps(baseline["fixed_hvp"])
        directions, parents = _fixed_maps(run["fixed_hvp"])
        if set(directions) != set(base_directions) or set(parents) != set(base_parents):
            raise ValueError(f"Fixed-HVP key mismatch for {task['run_name']}")
        direction_wins = sum(
            directions[key] < base_directions[key] for key in base_directions
        )
        parent_wins = sum(parents[key] < base_parents[key] for key in base_parents)
        validation = run["validation"]
        train700 = run["train700"]
        row = {
            **task,
            "validation_energy_mae": float(validation["energy_mae"]),
            "validation_force_mae": float(validation["force_component_mae"]),
            "train700_energy_mae": float(train700["energy_mae"]),
            "train700_force_mae": float(train700["force_component_mae"]),
            "fixed_hvp_mae": float(run["fixed_hvp"]["mean_mae"]),
            "fixed_hvp_parent_count": len(parents),
            "fixed_hvp_relative_frobenius": float(
                run["fixed_hvp"]["mean_relative_frobenius"]
            ),
            "direction_win_fraction": direction_wins / len(base_directions),
            "parent_win_fraction": parent_wins / len(base_parents),
        }
        row.update(
            {
                "validation_energy_relative_to_A": _relative(
                    row["validation_energy_mae"], baseline["validation"]["energy_mae"]
                ),
                "validation_force_relative_to_A": _relative(
                    row["validation_force_mae"],
                    baseline["validation"]["force_component_mae"],
                ),
                "train700_energy_relative_to_A": _relative(
                    row["train700_energy_mae"], baseline["train700"]["energy_mae"]
                ),
                "train700_force_relative_to_A": _relative(
                    row["train700_force_mae"],
                    baseline["train700"]["force_component_mae"],
                ),
                "fixed_hvp_mae_relative_to_A": _relative(
                    row["fixed_hvp_mae"], baseline["fixed_hvp"]["mean_mae"]
                ),
            }
        )
        row["energy_gate"] = row["validation_energy_relative_to_A"] <= 0.05
        row["force_gate"] = row["validation_force_relative_to_A"] < 0.0
        row["train700_energy_gate"] = row["train700_energy_relative_to_A"] <= 0.05
        row["train700_force_gate"] = row["train700_force_relative_to_A"] <= 0.05
        row["fixed_hvp_gate"] = row["fixed_hvp_mae_relative_to_A"] < 0.0
        row["fixed_hvp_parent_count_gate"] = (
            row["fixed_hvp_parent_count"] >= minimum_stable_parents
        )
        row["majority_direction_gate"] = row["direction_win_fraction"] > 0.5
        row["majority_parent_gate"] = row["parent_win_fraction"] > 0.5
        row["tier1_eligible"] = all(
            row[key]
            for key in (
                "energy_gate",
                "force_gate",
                "train700_energy_gate",
                "train700_force_gate",
                "fixed_hvp_gate",
                "fixed_hvp_parent_count_gate",
                "majority_direction_gate",
                "majority_parent_gate",
            )
        )
        row["gate_count"] = sum(
            bool(row[key])
            for key in (
                "energy_gate",
                "force_gate",
                "train700_energy_gate",
                "train700_force_gate",
                "fixed_hvp_gate",
                "fixed_hvp_parent_count_gate",
                "majority_direction_gate",
                "majority_parent_gate",
            )
        )
        rows.append(row)

    candidates = []
    for variant in ("B", "C", "D", "E"):
        eligible = [
            row for row in rows if row["variant"] == variant and row["tier1_eligible"]
        ]
        eligible.sort(
            key=lambda row: (
                row["validation_force_relative_to_A"]
                + row["fixed_hvp_mae_relative_to_A"],
                row["validation_energy_relative_to_A"],
                row["run_name"],
            )
        )
        candidates.extend(eligible[: args.max_per_variant])

    selected_config_by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("A", "B", "C", "D", "E"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        configurations: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in variant_rows:
            key = (float(row["curvature_weight"]), float(row["reference_floor"]))
            configurations.setdefault(key, []).append(row)
        configuration_rows = []
        for key, members in configurations.items():
            configuration_rows.append(
                {
                    "variant": variant,
                    "curvature_weight": key[0],
                    "reference_floor": key[1],
                    "seeds": len(members),
                    "all_energy_gates": all(row["energy_gate"] for row in members),
                    "all_train700_energy_gates": all(
                        row["train700_energy_gate"] for row in members
                    ),
                    "all_train700_force_gates": all(
                        row["train700_force_gate"] for row in members
                    ),
                    "all_fixed_hvp_parent_count_gates": all(
                        row["fixed_hvp_parent_count_gate"] for row in members
                    ),
                    "mean_gate_count": sum(row["gate_count"] for row in members)
                    / len(members),
                    "mean_validation_energy_relative_to_A": sum(
                        row["validation_energy_relative_to_A"] for row in members
                    )
                    / len(members),
                    "mean_validation_force_relative_to_A": sum(
                        row["validation_force_relative_to_A"] for row in members
                    )
                    / len(members),
                    "mean_fixed_hvp_mae_relative_to_A": sum(
                        row["fixed_hvp_mae_relative_to_A"] for row in members
                    )
                    / len(members),
                    "tier1_eligible_seed_count": sum(
                        bool(row["tier1_eligible"]) for row in members
                    ),
                    "run_names": [row["run_name"] for row in members],
                }
            )
        if not configuration_rows:
            raise ValueError(f"No configuration found for variant {variant}")
        # Frozen before reading screen metrics: prefer the energy gate, then the
        # number of passed gates, then minimize the worst and summed force/HVP
        # relative changes. This also supplies an explicitly marked diagnostic
        # fallback when no setting passes all Tier-1 gates.
        configuration_rows.sort(
            key=lambda row: (
                not row["all_energy_gates"],
                not row["all_train700_energy_gates"],
                not row["all_train700_force_gates"],
                not row["all_fixed_hvp_parent_count_gates"],
                -row["mean_gate_count"],
                max(
                    row["mean_validation_force_relative_to_A"],
                    row["mean_fixed_hvp_mae_relative_to_A"],
                ),
                row["mean_validation_force_relative_to_A"]
                + row["mean_fixed_hvp_mae_relative_to_A"],
                row["mean_validation_energy_relative_to_A"],
                row["curvature_weight"],
                row["reference_floor"],
            )
        )
        selected = dict(configuration_rows[0])
        selected["selection_status"] = (
            "tier1_eligible"
            if selected["tier1_eligible_seed_count"] == selected["seeds"]
            else "diagnostic_fallback_not_promoted"
        )
        selected_config_by_variant[variant] = selected

    groups = []
    for variant in ("A", "B", "C", "D", "E"):
        keys = sorted(
            {
                (float(row["curvature_weight"]), float(row["reference_floor"]))
                for row in rows
                if row["variant"] == variant
            }
        )
        for weight, floor in keys:
            members = [
                row
                for row in rows
                if row["variant"] == variant
                and float(row["curvature_weight"]) == weight
                and float(row["reference_floor"]) == floor
            ]
            group = {
                "variant": variant,
                "curvature_weight": weight,
                "reference_floor": floor,
                "seeds": len(members),
                "energy_gate_passed_seeds": sum(row["energy_gate"] for row in members),
                "train700_energy_gate_passed_seeds": sum(
                    row["train700_energy_gate"] for row in members
                ),
                "train700_force_gate_passed_seeds": sum(
                    row["train700_force_gate"] for row in members
                ),
                "force_improved_seeds": sum(row["force_gate"] for row in members),
                "fixed_hvp_improved_seeds": sum(
                    row["fixed_hvp_gate"] for row in members
                ),
                "fixed_hvp_parent_count_passed_seeds": sum(
                    row["fixed_hvp_parent_count_gate"] for row in members
                ),
                "majority_direction_seeds": sum(
                    row["majority_direction_gate"] for row in members
                ),
                "majority_parent_seeds": sum(
                    row["majority_parent_gate"] for row in members
                ),
            }
            for metric in (
                "validation_energy_relative_to_A",
                "validation_force_relative_to_A",
                "train700_energy_relative_to_A",
                "train700_force_relative_to_A",
                "fixed_hvp_mae_relative_to_A",
                "direction_win_fraction",
                "parent_win_fraction",
            ):
                group[f"mean_{metric}"] = sum(row[metric] for row in members) / len(
                    members
                )
            required_improved_seeds = 2 if len(members) >= 3 else len(members)
            group["multiseed_tier1_gate"] = bool(
                group["energy_gate_passed_seeds"] == len(members)
                and group["train700_energy_gate_passed_seeds"] == len(members)
                and group["train700_force_gate_passed_seeds"] == len(members)
                and group["fixed_hvp_parent_count_passed_seeds"] == len(members)
                and group["force_improved_seeds"] >= required_improved_seeds
                and group["fixed_hvp_improved_seeds"] >= required_improved_seeds
                and group["majority_direction_seeds"] >= required_improved_seeds
                and group["majority_parent_seeds"] >= required_improved_seeds
            )
            groups.append(group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "stage1_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    group_fields = list(groups[0])
    with (args.output_dir / "stage1_groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(groups)
    _plot_diagnostics(args.output_dir, rows)
    summary = {
        "definition": __doc__,
        "tasks": str(args.tasks.resolve()),
        "protocol": str(args.protocol.resolve()),
        "minimum_stable_validation_parents": minimum_stable_parents,
        "tasks_sha256": _sha256(args.tasks),
        "baseline_runs_by_seed": {
            str(seed): run["task"]["run_name"]
            for seed, run in sorted(baseline_by_seed.items())
        },
        "run_count": len(rows),
        "seed_count": len(task_seeds),
        "tier1_eligible_count": sum(bool(row["tier1_eligible"]) for row in rows),
        "strict_candidate_runs": [row["run_name"] for row in candidates],
        "strict_candidates": candidates,
        "selected_config_by_variant": selected_config_by_variant,
        "groups": groups,
        "test100_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-variant", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
