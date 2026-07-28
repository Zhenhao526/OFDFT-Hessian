#!/usr/bin/env python3
"""Rank validation-only strict density-relaxed Hessian candidates for Test100."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank(rows: list[dict[str, Any]], metric: str) -> None:
    ordered = sorted(
        (row for row in rows if row.get(metric) is not None),
        key=lambda row: float(row[metric]),
    )
    for index, row in enumerate(ordered, start=1):
        row[f"{metric}_rank"] = index


def _optimization_stats(payload: dict[str, Any], run: str) -> dict[str, Any]:
    rows = [row for row in payload.get("optimization_rows", []) if row.get("run") == run]
    if rows:
        cycles = [int(row["cycles"]) for row in rows if row.get("cycles") is not None]
        strict = sum(bool(row.get("converged")) for row in rows)
        return {
            "strict_converged_points": strict,
            "optimization_points": len(rows),
            "strict_convergence_rate": strict / len(rows),
            "fallback_triggers": sum(bool(row.get("used_fallback")) for row in rows),
            "mean_cycles": float(np.mean(cycles)) if cycles else None,
            "max_cycles": max(cycles) if cycles else None,
        }
    summary = payload["runs"][run]
    strict = summary.get("strict_converged")
    total = summary.get("optimization_points")
    return {
        "strict_converged_points": strict,
        "optimization_points": total,
        "strict_convergence_rate": (
            float(strict) / float(total) if strict is not None and total else None
        ),
        "fallback_triggers": summary.get("fallback_triggers"),
        "mean_cycles": summary.get("mean_cycles") or summary.get("mean_opt_cycles"),
        "max_cycles": summary.get("max_cycles"),
    }


def _plot_tradeoff(rows: list[dict[str, Any]], output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for row in rows:
        color = "#4c72b0" if row["name"].startswith("EGF") else "#c44e52"
        axes[0].scatter(row["validation_force_component_mae"], row["validation_relaxed_hessian_mae"], color=color)
        axes[0].annotate(row["name"], (row["validation_force_component_mae"], row["validation_relaxed_hessian_mae"]), fontsize=7)
        axes[1].scatter(row["validation_energy_mae"], row["validation_relaxed_hessian_mae"], color=color)
        axes[1].annotate(row["name"], (row["validation_energy_mae"], row["validation_relaxed_hessian_mae"]), fontsize=7)
    axes[0].set_xlabel("validation force component MAE")
    axes[0].set_ylabel("strict Val8 relaxed-Hessian MAE")
    axes[1].set_xlabel("validation energy MAE")
    axes[1].set_ylabel("strict Val8 relaxed-Hessian MAE")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "tier2_validation_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def analyze(root: Path, output_dir: Path) -> dict[str, Any]:
    model_manifest = json.loads((root / "model_manifest.json").read_text())
    rows = []
    for model in model_manifest:
        name = model["name"]
        force = json.loads((root / "models" / name / "force.json").read_text())
        hessian = json.loads((root / "models" / name / "hessian_summary.json").read_text())
        if force.get("split") != "val" or not force.get("ground_state_only"):
            raise ValueError(f"{name}: Tier 2 force metrics are not validation ground states")
        summary = hessian["runs"][name]
        optimization = _optimization_stats(hessian, name)
        rows.append(
            {
                **model,
                "validation_energy_mae": force.get("energy_mae"),
                "validation_energy_rmse": force.get("energy_rmse"),
                "validation_force_component_mae": force.get("force_component_mae"),
                "validation_force_component_rmse": force.get("force_component_rmse"),
                "validation_force_vector_mae": force.get("force_vector_mae"),
                "force_samples": force.get("samples_evaluated"),
                "force_failures": len(force.get("failures", [])),
                "hessian_success": summary.get("n_success"),
                "hessian_failed": summary.get("n_failed"),
                **optimization,
                "validation_relaxed_hessian_mae": summary.get("mean_mae"),
                "validation_relaxed_hessian_rmse": summary.get("mean_rmse"),
                "validation_relaxed_hessian_relative_fro": summary.get(
                    "mean_relative_fro_error"
                ),
                "validation_relaxed_hessian_symmetry_max_abs": summary.get(
                    "mean_model_symmetry_max_abs_error"
                ),
                "hessian_wall_sum_s": summary.get("total_elapsed_s")
                or summary.get("molecule_total_elapsed_s"),
            }
        )

    by_name = {row["name"]: row for row in rows}
    historical = by_name["EGF_w0p1"]
    energy_limit = 1.10 * float(historical["validation_energy_mae"])
    for metric in (
        "validation_force_component_mae",
        "validation_relaxed_hessian_mae",
        "validation_relaxed_hessian_relative_fro",
    ):
        _rank(rows, metric)

    eligible = []
    for row in rows:
        all_strict = (
            row.get("optimization_points") is not None
            and row.get("strict_converged_points") == row.get("optimization_points")
            and row.get("hessian_failed") == 0
        )
        row["selection_energy_limit"] = energy_limit
        row["selection_energy_eligible"] = bool(
            row["name"].startswith("EGF")
            and float(row["validation_energy_mae"]) <= energy_limit
        )
        row["selection_all_strict"] = all_strict
        row["improves_historical_force"] = (
            float(row["validation_force_component_mae"])
            < float(historical["validation_force_component_mae"])
        )
        row["improves_historical_hessian"] = (
            float(row["validation_relaxed_hessian_mae"])
            < float(historical["validation_relaxed_hessian_mae"])
        )
        ranks = [
            row.get("validation_force_component_mae_rank"),
            row.get("validation_relaxed_hessian_mae_rank"),
            row.get("validation_relaxed_hessian_relative_fro_rank"),
        ]
        row["selection_rank_sum"] = (
            float(sum(ranks)) if all(rank is not None for rank in ranks) else None
        )
        if row["selection_energy_eligible"] and all_strict:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            row["selection_rank_sum"] if row["selection_rank_sum"] is not None else math.inf,
            row["validation_relaxed_hessian_mae"],
            row["validation_force_component_mae"],
        )
    )
    improved = [
        row
        for row in eligible
        if row["improves_historical_force"] and row["improves_historical_hessian"]
    ]
    recommended_rows = improved[:2] if improved else [historical]
    recommended = [row["name"] for row in recommended_rows]
    for row in rows:
        row["recommended_for_test100"] = row["name"] in recommended
    rows.sort(
        key=lambda row: (
            not row["recommended_for_test100"],
            row["selection_rank_sum"] if row["selection_rank_sum"] is not None else math.inf,
            row["name"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tier2_model_summary.csv", rows)
    plot = _plot_tradeoff(rows, output_dir)
    result = {
        "definition": (
            "Validation-only Tier 2: ground-state energy/force plus strict density-relaxed "
            "incomplete-derived-force Hessian at h=1e-3 Bohr and density threshold 1e-4. "
            "No newly trained candidate is evaluated on Test100 until this recommendation is frozen."
        ),
        "energy_eligibility": "validation energy MAE <= 1.10 * historical EGF_w0p1",
        "recommended_for_test100": recommended,
        "models": rows,
        "plot": plot,
    }
    (output_dir / "tier2_model_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output_dir)


if __name__ == "__main__":
    main()
