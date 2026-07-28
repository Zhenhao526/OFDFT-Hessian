#!/usr/bin/env python3
"""Aggregate the frozen random1000 Test100 evaluation without selecting on test data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _mean(values: list[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = np.argsort(np.argsort(np.asarray(left, dtype=float))).astype(float)
    right_rank = np.argsort(np.argsort(np.asarray(right, dtype=float))).astype(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _proxy_diagnostics(
    density_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    model_names: list[str],
    frozen_candidates: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases = {"EGF_lam1": "EGF_w0p1"}
    relaxed = {
        (aliases.get(str(row.get("run")), str(row.get("run"))), str(row.get("molecule_id"))): row
        for row in density_rows
        if row.get("success")
    }
    fixed = {
        (str(row.get("run")), str(row.get("molecule_id"))): row
        for row in fixed_rows
        if row.get("success")
    }
    per_molecule = []
    model_summaries: dict[str, Any] = {}
    for name in model_names:
        molecule_ids = sorted(
            {molecule for run, molecule in relaxed if run == name}
            & {molecule for run, molecule in fixed if run == name}
        )
        fixed_mae, relaxed_mae = [], []
        for molecule_id in molecule_ids:
            fixed_row = fixed[name, molecule_id]
            relaxed_row = relaxed[name, molecule_id]
            fixed_value = float(fixed_row["autograd_vs_pbe"]["mae"])
            relaxed_value = float(relaxed_row["mae"])
            fixed_mae.append(fixed_value)
            relaxed_mae.append(relaxed_value)
            per_molecule.append(
                {
                    "name": name,
                    "molecule_id": molecule_id,
                    "natoms": int(relaxed_row["natoms"]),
                    "fixed_hessian_pbe_mae": fixed_value,
                    "relaxed_hessian_pbe_mae": relaxed_value,
                    "relaxed_over_fixed_mae": (
                        relaxed_value / fixed_value if fixed_value else None
                    ),
                    "absolute_fixed_relaxed_mae_difference": abs(
                        relaxed_value - fixed_value
                    ),
                    "relaxed_hessian_pbe_rmse": relaxed_row.get("rmse"),
                    "relaxed_hessian_pbe_relative_fro": relaxed_row.get(
                        "relative_fro_error"
                    ),
                    "relaxed_hessian_symmetry_max_abs": relaxed_row.get(
                        "model_symmetry_max_abs_error"
                    ),
                }
            )
        model_summaries[name] = {
            "molecules": len(molecule_ids),
            "pearson_fixed_vs_relaxed_mae": (
                float(np.corrcoef(fixed_mae, relaxed_mae)[0, 1])
                if len(molecule_ids) >= 2
                else None
            ),
            "spearman_fixed_vs_relaxed_mae": _rank_correlation(
                fixed_mae, relaxed_mae
            ),
            "fixed_mae_lower_than_relaxed_count": sum(
                fixed_value < relaxed_value
                for fixed_value, relaxed_value in zip(fixed_mae, relaxed_mae)
            ),
            "mean_absolute_fixed_relaxed_mae_difference": _mean(
                [
                    abs(fixed_value - relaxed_value)
                    for fixed_value, relaxed_value in zip(fixed_mae, relaxed_mae)
                ]
            ),
        }

    common_sets = [
        {molecule for run, molecule in relaxed if run == name}
        & {molecule for run, molecule in fixed if run == name}
        for name in model_names
    ]
    common_molecules = sorted(set.intersection(*common_sets)) if common_sets else []
    rank_matches = 0
    for molecule_id in common_molecules:
        fixed_best = min(
            model_names,
            key=lambda name: float(fixed[name, molecule_id]["autograd_vs_pbe"]["mae"]),
        )
        relaxed_best = min(
            model_names,
            key=lambda name: float(relaxed[name, molecule_id]["mae"]),
        )
        rank_matches += fixed_best == relaxed_best

    historical = "EGF_w0p1"
    candidate_summaries = {}
    for candidate in frozen_candidates:
        molecule_ids = sorted(
            {molecule for run, molecule in relaxed if run == candidate}
            & {molecule for run, molecule in relaxed if run == historical}
        )
        ratios = [
            float(relaxed[candidate, molecule]["mae"])
            / float(relaxed[historical, molecule]["mae"])
            for molecule in molecule_ids
        ]
        candidate_summaries[candidate] = {
            "molecules": len(molecule_ids),
            "improved_relaxed_hessian_mae_count": sum(ratio < 1.0 for ratio in ratios),
            "median_relaxed_hessian_mae_ratio": (
                float(np.median(ratios)) if ratios else None
            ),
            "maximum_relaxed_hessian_mae_ratio": max(ratios, default=None),
            "worst_ratio_molecule": (
                molecule_ids[int(np.argmax(ratios))] if ratios else None
            ),
            "largest_mae_molecule": (
                max(
                    molecule_ids,
                    key=lambda molecule: float(relaxed[candidate, molecule]["mae"]),
                )
                if molecule_ids
                else None
            ),
        }
    summary = {
        "models": model_summaries,
        "common_molecules": len(common_molecules),
        "fixed_vs_relaxed_best_model_rank_matches": rank_matches,
        "fixed_vs_relaxed_best_model_rank_match_rate": (
            rank_matches / len(common_molecules) if common_molecules else None
        ),
        "candidates_vs_historical": candidate_summaries,
    }
    return per_molecule, summary


def _plot_candidate_relaxed_scatter(
    per_molecule: list[dict[str, Any]], candidates: list[str], output_dir: Path
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    by_key = {(row["name"], row["molecule_id"]): row for row in per_molecule}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    quantities = (
        ("relaxed_hessian_pbe_mae", "Hessian MAE"),
        ("relaxed_hessian_symmetry_max_abs", "symmetry max abs"),
    )
    for axis, (field, label) in zip(axes, quantities):
        plotted = []
        for candidate in candidates:
            molecule_ids = sorted(
                {molecule for name, molecule in by_key if name == candidate}
                & {molecule for name, molecule in by_key if name == "EGF_w0p1"}
            )
            x = [float(by_key["EGF_w0p1", molecule][field]) for molecule in molecule_ids]
            y = [float(by_key[candidate, molecule][field]) for molecule in molecule_ids]
            axis.scatter(x, y, s=14, alpha=0.65, label=candidate)
            plotted.extend(x + y)
        if plotted:
            low = max(min(plotted) * 0.8, np.finfo(float).tiny)
            high = max(plotted) * 1.2
            axis.plot([low, high], [low, high], "k--", linewidth=1)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
        axis.set_xlabel(f"historical EGF {label}")
        axis.set_ylabel(f"candidate {label}")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "test100_candidate_relaxed_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _density_summary(summary: dict[str, Any], run: str) -> dict[str, Any]:
    row = summary["runs"][run]
    optimization_rows = [
        item for item in summary.get("optimization_rows", []) if item.get("run") == run
    ]
    if optimization_rows:
        strict_points = sum(bool(item.get("converged")) for item in optimization_rows)
        optimization_points = len(optimization_rows)
        cycles = [
            int(item["cycles"])
            for item in optimization_rows
            if item.get("cycles") is not None
        ]
        mean_cycles = float(np.mean(cycles)) if cycles else None
    else:
        strict_points = row.get("strict_converged")
        optimization_points = row.get("optimization_points")
        mean_cycles = row.get("mean_cycles") or row.get("mean_opt_cycles")
    return {
        "test_relaxed_hessian_mae": row.get("mean_mae"),
        "test_relaxed_hessian_rmse": row.get("mean_rmse"),
        "test_relaxed_hessian_relative_fro": row.get("mean_relative_fro_error"),
        "test_relaxed_hessian_symmetry_max_abs": row.get(
            "mean_model_symmetry_max_abs_error"
        ),
        "test_relaxed_hessian_success": row.get("n_success"),
        "test_relaxed_hessian_failed": row.get("n_failed"),
        "test_relaxed_strict_points": strict_points,
        "test_relaxed_optimization_points": optimization_points,
        "test_relaxed_wall_sum_s": row.get("molecule_total_elapsed_s")
        or row.get("total_elapsed_s"),
        "test_relaxed_mean_cycles": mean_cycles,
    }


def _plot_ratios(rows: list[dict[str, Any]], output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    metrics = [
        ("test_energy_mae_vs_historical_ratio", "energy"),
        ("test_force_component_mae_vs_historical_ratio", "force"),
        ("test_fixed_hessian_pbe_mae_vs_historical_ratio", "fixed H"),
        ("test_relaxed_hessian_mae_vs_historical_ratio", "relaxed H"),
        ("test_frequency_mae_cm-1_vs_historical_ratio", "frequency"),
    ]
    width = 0.8 / max(len(rows), 1)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for index, row in enumerate(rows):
        values = [
            float(row[key]) if row.get(key) is not None else np.nan for key, _ in metrics
        ]
        ax.bar(x + (index - (len(rows) - 1) / 2) * width, values, width=width, label=row["name"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Test100 error / historical EGF error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "test100_tradeoff_ratios.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    tier2 = json.loads(args.tier2_summary.read_text())
    frozen = list(tier2["recommended_for_test100"])
    model_rows = {row["name"]: dict(row) for row in tier2["models"]}
    names = []
    for name in ["EG", "EGF_w0p1", *frozen]:
        if name not in names:
            names.append(name)

    baseline = json.loads(args.baseline_density_summary.read_text())
    candidates = (
        json.loads(args.candidate_density_summary.read_text())
        if args.candidate_density_summary.exists()
        else {"runs": {}, "rows": []}
    )
    force_by_name = {
        path.stem: json.loads(path.read_text()) for path in sorted(args.force_dir.glob("*.json"))
    }
    fixed = json.loads(args.fixed_summary.read_text())
    fixed_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed.get("fixed_density_full_hessian", []):
        if row.get("success"):
            fixed_by_run[row["run"]].append(row)
    hvp_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed.get("hvp", []):
        if row.get("success"):
            hvp_by_run[row["run"]].append(row)
    vibration = json.loads(args.vibration_summary.read_text())
    vibration_by_run = {row["run"]: row for row in vibration.get("summaries", [])}
    if "EGF_lam1" in vibration_by_run:
        vibration_by_run["EGF_w0p1"] = vibration_by_run["EGF_lam1"]

    rows = []
    for name in names:
        force = force_by_name[name]
        if force.get("split") != "test" or not force.get("ground_state_only"):
            raise ValueError(f"{name}: final force metrics are not Test100 ground states")
        row = {
            **model_rows[name],
            "frozen_before_test": name in frozen,
            "test_energy_mae": force.get("energy_mae"),
            "test_energy_rmse": force.get("energy_rmse"),
            "test_force_component_mae": force.get("force_component_mae"),
            "test_force_component_rmse": force.get("force_component_rmse"),
            "test_force_vector_mae": force.get("force_vector_mae"),
            "test_force_samples": force.get("samples_evaluated"),
            "test_force_failures": len(force.get("failures", [])),
        }
        if name == "EG":
            row.update(_density_summary(baseline, "EG"))
        elif name == "EGF_w0p1":
            row.update(_density_summary(baseline, "EGF_lam1"))
        else:
            row.update(_density_summary(candidates, name))

        fixed_rows = fixed_by_run.get(name, [])
        hvp_rows = hvp_by_run.get(name, [])
        perturb_hvp_rows = [
            item
            for item in hvp_rows
            if item.get("direction", "").startswith("perturb_unit_s")
        ]
        row.update(
            {
                "test_fixed_hessian_cases": len(fixed_rows),
                "test_fixed_hessian_pbe_mae": _mean(
                    [item.get("autograd_vs_pbe", {}).get("mae") for item in fixed_rows]
                ),
                "test_fixed_hessian_pbe_rmse": _mean(
                    [item.get("autograd_vs_pbe", {}).get("rmse") for item in fixed_rows]
                ),
                "test_fixed_hessian_pbe_relative_fro": _mean(
                    [
                        item.get("autograd_vs_pbe", {}).get("relative_fro_error")
                        for item in fixed_rows
                    ]
                ),
                "test_fixed_hessian_symmetry_max_abs": _mean(
                    [item.get("autograd_stats", {}).get("symmetry_max_abs") for item in fixed_rows]
                ),
                "test_hvp_cases": len(hvp_rows),
                "test_hvp_vs_own_fd_relative_fro": _mean(
                    [
                        item.get("hvp_vs_fd_directional", {}).get("relative_fro_error")
                        for item in hvp_rows
                    ]
                ),
                "test_hvp_vs_pbe_force_secant_mae": _mean(
                    [
                        item.get("hvp_vs_pbe_force_secant_per_bohr", {}).get("mae")
                        for item in perturb_hvp_rows
                    ]
                ),
                "test_hvp_vs_pbe_force_secant_relative_fro": _mean(
                    [
                        item.get("hvp_vs_pbe_force_secant_per_bohr", {}).get(
                            "relative_fro_error"
                        )
                        for item in perturb_hvp_rows
                    ]
                ),
            }
        )
        vib = vibration_by_run.get(name, {})
        row.update(
            {
                "test_frequency_mae_cm-1": vib.get("mean_frequency_mae_cm-1"),
                "test_frequency_rmse_cm-1": vib.get("mean_frequency_rmse_cm-1"),
                "test_mean_mode_overlap": vib.get("mean_mode_overlap"),
                "test_pbe_imaginary_modes": vib.get("total_pbe_imaginary_modes"),
                "test_model_imaginary_modes": vib.get("total_model_imaginary_modes"),
            }
        )
        rows.append(row)

    by_name = {row["name"]: row for row in rows}
    historical = by_name["EGF_w0p1"]
    ratio_metrics = (
        "test_energy_mae",
        "test_force_component_mae",
        "test_fixed_hessian_pbe_mae",
        "test_hvp_vs_pbe_force_secant_mae",
        "test_relaxed_hessian_mae",
        "test_relaxed_hessian_relative_fro",
        "test_frequency_mae_cm-1",
    )
    for row in rows:
        for metric in ratio_metrics:
            value, reference = row.get(metric), historical.get(metric)
            row[f"{metric}_vs_historical_ratio"] = (
                float(value) / float(reference)
                if value is not None and reference not in (None, 0)
                else None
            )
        row["independent_test_force_and_relaxed_hessian_improvement"] = bool(
            row["name"] != "EGF_w0p1"
            and row["name"] in frozen
            and float(row["test_force_component_mae"])
            < float(historical["test_force_component_mae"])
            and float(row["test_relaxed_hessian_mae"])
            < float(historical["test_relaxed_hessian_mae"])
            and float(row["test_energy_mae"]) <= 1.10 * float(historical["test_energy_mae"])
            and row.get("test_relaxed_hessian_failed") == 0
            and row.get("test_relaxed_strict_points")
            == row.get("test_relaxed_optimization_points")
        )

    stable_candidates = [
        row["name"]
        for row in rows
        if row["independent_test_force_and_relaxed_hessian_improvement"]
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = _plot_ratios(rows, output_dir)
    _write_csv(output_dir / "test100_model_summary.csv", rows)
    density_rows = list(baseline.get("rows", [])) + list(candidates.get("rows", []))
    _write_csv(output_dir / "test100_density_relaxed_per_molecule.csv", density_rows)
    _write_csv(
        output_dir / "test100_fixed_hessian_per_molecule.csv",
        fixed.get("fixed_density_full_hessian", []),
    )
    _write_csv(output_dir / "test100_hvp_per_direction.csv", fixed.get("hvp", []))
    _write_csv(
        output_dir / "test100_vibrational_per_molecule.csv", vibration.get("rows", [])
    )
    proxy_rows, proxy_summary = _proxy_diagnostics(
        density_rows,
        fixed.get("fixed_density_full_hessian", []),
        names,
        frozen,
    )
    _write_csv(output_dir / "test100_fixed_vs_relaxed_per_molecule.csv", proxy_rows)
    candidate_scatter = _plot_candidate_relaxed_scatter(
        proxy_rows, frozen, output_dir
    )
    result = {
        "definition": (
            "One-shot Test100 confirmation of candidates frozen by validation Tier 2. "
            "No Test100 metric was used to choose force weight or paired augmentation."
        ),
        "frozen_candidates": frozen,
        "stable_improvement_candidates": stable_candidates,
        "models": rows,
        "plot": plot,
        "candidate_relaxed_scatter_plot": candidate_scatter,
        "proxy_diagnostics": proxy_summary,
    }
    (output_dir / "test100_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier2-summary", type=Path, required=True)
    parser.add_argument("--baseline-density-summary", type=Path, required=True)
    parser.add_argument("--candidate-density-summary", type=Path, required=True)
    parser.add_argument("--force-dir", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--vibration-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
