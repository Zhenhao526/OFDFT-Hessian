#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from mpn_melting.observables import centrosymmetry_parameters


def linear_slope(x: List[float], y: List[float]) -> float:
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        return 0.0
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    ) / denominator


def csp_summary(positions, lattice) -> Dict[str, float]:
    values = centrosymmetry_parameters(positions, lattice)
    return {
        "median_A2": statistics.median(values),
        "mean_A2": statistics.fmean(values),
        "ordered_fraction_CSP_lt_2_5": sum(value < 2.5 for value in values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a stored pair-reference MD trajectory")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected", choices=("solid", "liquid"), required=True)
    parser.add_argument("--structure-model", choices=("fcc", "hcp"), default="fcc")
    parser.add_argument("--minimum-neighbor", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.minimum_neighbor <= 0.0:
        parser.error("--minimum-neighbor must be positive")

    trajectory_path = args.run_dir / "trajectory.jsonl"
    checkpoint_path = args.run_dir / "checkpoint.json"
    summary_path = args.run_dir / "summary.json"
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) < 5:
        raise ValueError("at least five trajectory samples are required")
    if "positions_angstrom" not in rows[0] or "positions_angstrom" not in rows[-1]:
        raise ValueError("trajectory was not written with --store-positions")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lattice = checkpoint["lattice_angstrom"]
    late = rows[len(rows) // 2 :]
    late_steps = [float(row["step"]) for row in late]
    late_msd = [float(row["msd_angstrom2"]) for row in late]
    late_temperature = [float(row["temperature_k"]) for row in late]
    target_temperature = float(checkpoint["target_temperature_k"])
    dt_fs = float(checkpoint["dt_fs"])
    initial_csp = csp_summary(rows[0]["positions_angstrom"], lattice)
    final_csp = csp_summary(rows[-1]["positions_angstrom"], lattice)
    msd_slope = linear_slope(late_steps, late_msd)
    minimum_distance = float(summary["minimum_distance_angstrom"])

    common_gates = {
        "nearest_neighbor_gt_minimum": minimum_distance > args.minimum_neighbor,
        "late_temperature_within_15_percent": (
            0.85 * target_temperature
            <= statistics.fmean(late_temperature)
            <= 1.15 * target_temperature
        ),
    }
    if args.structure_model == "hcp" and args.expected == "solid":
        phase_gates = {
            "final_MSD_lt_0_5_A2": float(rows[-1]["msd_angstrom2"]) < 0.5,
            "late_MSD_slope_lt_0_001_A2_per_step": msd_slope < 0.001,
        }
    elif args.structure_model == "hcp" and args.expected == "liquid":
        phase_gates = {
            "final_MSD_gt_1_A2": float(rows[-1]["msd_angstrom2"]) > 1.0,
            "late_MSD_slope_gt_0_0001_A2_per_step": msd_slope > 0.0001,
        }
    elif args.expected == "solid":
        phase_gates = {
            "final_median_CSP_lt_6_A2": final_csp["median_A2"] < 6.0,
            "final_ordered_fraction_gt_0_1": (
                final_csp["ordered_fraction_CSP_lt_2_5"] > 0.1
            ),
            "late_MSD_slope_lt_0_001_A2_per_step": msd_slope < 0.001,
        }
    else:
        phase_gates = {
            "final_median_CSP_gt_8_A2": final_csp["median_A2"] > 8.0,
            "final_ordered_fraction_lt_0_1": (
                final_csp["ordered_fraction_CSP_lt_2_5"] < 0.1
            ),
            "final_MSD_gt_1_A2": float(rows[-1]["msd_angstrom2"]) > 1.0,
            "late_MSD_slope_gt_0_0001_A2_per_step": msd_slope > 0.0001,
        }
    gates = {**common_gates, **phase_gates}
    verified = all(gates.values())
    result = {
        "schema": "mpn-pair-reference-phase-analysis-v1",
        "run_dir": str(args.run_dir.resolve()),
        "expected_phase": args.expected,
        "structure_model": args.structure_model,
        "status": f"{args.expected}_{'verified' if verified else 'not_verified'}",
        "samples": len(rows),
        "steps": int(rows[-1]["step"]),
        "temperature_K": {
            "late_mean": statistics.fmean(late_temperature),
            "late_sd": statistics.stdev(late_temperature),
            "last": float(rows[-1]["temperature_k"]),
        },
        "minimum_nearest_neighbor_A": minimum_distance,
        "minimum_neighbor_gate_A": args.minimum_neighbor,
        "MSD_A2": {
            "last": float(rows[-1]["msd_angstrom2"]),
            "late_slope_per_step": msd_slope,
            "late_diffusion_A2_per_fs": msd_slope / (6.0 * dt_fs),
        },
        "CSP_initial": initial_csp,
        "CSP_final": final_csp,
        "phase_gate": gates,
    }
    output = args.out or args.run_dir / "phase_analysis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
