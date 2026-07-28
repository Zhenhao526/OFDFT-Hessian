#!/usr/bin/env python3
"""Combine random1000 force and fixed-density Hessian/HVP screening metrics."""

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
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_tradeoff(rows: list[dict[str, Any]], output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    selected = [
        row
        for row in rows
        if row.get("force_component_mae") is not None
        and row.get("validation_hvp_pbe_secant_mae") is not None
    ]
    if not selected:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for row in selected:
        color = "#4c72b0" if row["run"].startswith("EGF") else "#c44e52"
        axes[0].scatter(row["force_component_mae"], row["validation_hvp_pbe_secant_mae"], color=color)
        axes[0].annotate(row["run"], (row["force_component_mae"], row["validation_hvp_pbe_secant_mae"]), fontsize=7)
        axes[1].scatter(row["energy_mae"], row["force_component_mae"], color=color)
        axes[1].annotate(row["run"], (row["energy_mae"], row["force_component_mae"]), fontsize=7)
    axes[0].set_xlabel("validation force component MAE")
    axes[0].set_ylabel("validation HVP vs PBE force-secant MAE")
    axes[1].set_xlabel("validation energy MAE")
    axes[1].set_ylabel("validation force component MAE")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "tier1_validation_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def analyze(force_dir: Path, hessian_summary: Path, output_dir: Path) -> dict[str, Any]:
    rows_by_run: dict[str, dict[str, Any]] = {}
    observed_splits = set()
    observed_ground_state_modes = set()
    for path in sorted(force_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        observed_splits.add(payload.get("split", "test"))
        observed_ground_state_modes.add(bool(payload.get("ground_state_only", False)))
        run = path.stem
        rows_by_run[run] = {
            "run": run,
            "energy_mae": payload.get("energy_mae"),
            "energy_rmse": payload.get("energy_rmse"),
            "force_component_mae": payload.get("force_component_mae"),
            "force_component_rmse": payload.get("force_component_rmse"),
            "force_vector_mae": payload.get("force_vector_mae"),
            "force_failures": len(payload.get("failures", [])),
            "force_samples": payload.get("samples_evaluated"),
            "run_dir": payload.get("run_dir"),
            "checkpoint": payload.get("ckpt"),
        }

    payload = json.loads(hessian_summary.read_text())
    hessian_split = payload.get("split", "test")
    hessian_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload.get("fixed_density_full_hessian", []):
        hessian_rows[item["run"]].append(item)
    hvp_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload.get("hvp", []):
        if "run" in item:
            hvp_rows[item["run"]].append(item)

    for run, items in hessian_rows.items():
        row = rows_by_run.setdefault(run, {"run": run})
        successful = [item for item in items if item.get("success")]
        row.update(
            {
                "hessian_cases": len(items),
                "hessian_finite_cases": sum(
                    bool(item.get("autograd_stats", {}).get("finite")) for item in items
                ),
                "fixed_hessian_pbe_mae": _mean(
                    [item.get("autograd_vs_pbe", {}).get("mae") for item in successful]
                ),
                "fixed_hessian_pbe_rmse": _mean(
                    [item.get("autograd_vs_pbe", {}).get("rmse") for item in successful]
                ),
                "fixed_hessian_pbe_relative_fro": _mean(
                    [
                        item.get("autograd_vs_pbe", {}).get("relative_fro_error")
                        for item in successful
                    ]
                ),
                "fixed_hessian_symmetry_max_abs": _mean(
                    [item.get("autograd_stats", {}).get("symmetry_max_abs") for item in successful]
                ),
                "fixed_hessian_elapsed_s": _mean(
                    [item.get("autograd_elapsed_s") for item in successful]
                ),
                "fixed_hessian_peak_cuda_mb": _mean(
                    [item.get("autograd_peak_cuda_mb") for item in successful]
                ),
            }
        )
        hvp = [item for item in hvp_rows.get(run, []) if item.get("success")]
        row.update(
            {
                "hvp_cases": len(hvp_rows.get(run, [])),
                "hvp_finite_cases": sum(
                    bool(item.get("hvp_stats", {}).get("finite"))
                    for item in hvp_rows.get(run, [])
                ),
                "hvp_fd_relative_fro": _mean(
                    [
                        item.get("hvp_vs_fd_directional", {}).get("relative_fro_error")
                        for item in hvp
                    ]
                ),
                "hvp_elapsed_s": _mean([item.get("hvp_elapsed_s") for item in hvp]),
                "validation_hvp_pbe_secant_mae": _mean(
                    [
                        item.get("hvp_vs_pbe_force_secant_per_bohr", {}).get("mae")
                        for item in hvp
                        if item.get("direction", "").startswith("perturb_unit_s")
                    ]
                ),
                "validation_hvp_pbe_secant_rmse": _mean(
                    [
                        item.get("hvp_vs_pbe_force_secant_per_bohr", {}).get("rmse")
                        for item in hvp
                        if item.get("direction", "").startswith("perturb_unit_s")
                    ]
                ),
                "validation_hvp_pbe_secant_relative_fro": _mean(
                    [
                        item.get("hvp_vs_pbe_force_secant_per_bohr", {}).get(
                            "relative_fro_error"
                        )
                        for item in hvp
                        if item.get("direction", "").startswith("perturb_unit_s")
                    ]
                ),
            }
        )

    rows = list(rows_by_run.values())
    if (
        observed_splits != {"val"}
        or observed_ground_state_modes != {True}
        or hessian_split != "val"
        or int(payload.get("scf_iteration", 0)) != -1
    ):
        raise ValueError(
            "Tier-1 model selection must use validation data only: "
            f"force_splits={sorted(observed_splits)} "
            f"ground_state_modes={sorted(observed_ground_state_modes)} "
            f"hessian_split={hessian_split} scf_iteration={payload.get('scf_iteration')}"
        )
    for metric in (
        "force_component_mae",
        "validation_hvp_pbe_secant_mae",
        "energy_mae",
    ):
        ordered = sorted(
            (row for row in rows if row.get(metric) is not None), key=lambda row: row[metric]
        )
        for rank, row in enumerate(ordered, start=1):
            row[f"{metric}_rank"] = rank

    historical = rows_by_run.get("EGF_w0p1")
    energy_limit = (
        1.10 * float(historical["energy_mae"])
        if historical is not None and historical.get("energy_mae") is not None
        else None
    )
    candidates = []
    for row in rows:
        row["selection_energy_limit"] = energy_limit
        row["selection_energy_eligible"] = bool(
            row["run"].startswith("EGF_")
            and energy_limit is not None
            and row.get("energy_mae") is not None
            and float(row["energy_mae"]) <= energy_limit
        )
        force_rank = row.get("force_component_mae_rank")
        hvp_rank = row.get("validation_hvp_pbe_secant_mae_rank")
        row["validation_selection_rank_sum"] = (
            float(force_rank + hvp_rank)
            if force_rank is not None and hvp_rank is not None
            else None
        )
        if row["selection_energy_eligible"] and row["validation_selection_rank_sum"] is not None:
            candidates.append(row)
    candidates.sort(
        key=lambda row: (
            row["validation_selection_rank_sum"],
            row["force_component_mae"],
            row["energy_mae"],
        )
    )
    recommended = [row["run"] for row in candidates[:2]]
    for row in rows:
        row["recommended_for_tier2"] = row["run"] in recommended
    rows.sort(
        key=lambda row: (
            not row["recommended_for_tier2"],
            row.get("validation_selection_rank_sum")
            if row.get("validation_selection_rank_sum") is not None
            else math.inf,
            row["run"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tier1_model_summary.csv", rows)
    plot = _plot_tradeoff(rows, output_dir)
    result = {
        "definition": (
            "Validation-only Tier 1 selection using energy/force and fixed-density autograd HVP "
            "against PBE force secants. Test100 is not used for candidate selection."
        ),
        "selection_split": "val",
        "energy_eligibility": "EGF energy MAE <= 1.10 * historical EGF_w0p1 validation energy MAE",
        "recommended_for_tier2": recommended,
        "force_dir": str(force_dir.resolve()),
        "hessian_summary": str(hessian_summary.resolve()),
        "models": rows,
        "plot": plot,
    }
    (output_dir / "tier1_model_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-dir", type=Path, required=True)
    parser.add_argument("--hessian-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.force_dir, args.hessian_summary, args.output_dir)


if __name__ == "__main__":
    main()
