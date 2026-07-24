#!/usr/bin/env python3
"""Validate direct-coexistence migration against a WT free-energy melting point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_diagnostics(
    analysis: Dict[str, Any], target_temperature: float, requested_steps: int
) -> Dict[str, Any]:
    md = analysis.get("md", {})
    temperature = md.get("temperature_last_100_steps_K", {})
    migration = analysis.get("interface_migration_indicators", {})
    trajectory = analysis.get("trajectory", {})
    change = migration.get("integrated_ordered_fraction_change")
    initial = migration.get("integrated_ordered_fraction_initial")
    final = migration.get("integrated_ordered_fraction_final")
    mean_temperature = temperature.get("mean")
    nearest_neighbor = float(trajectory.get("nearest_neighbor_A", 0.0))
    minimum_nearest_neighbor = float(
        trajectory.get("minimum_nearest_neighbor_A", nearest_neighbor)
    )
    checks = {
        "requested_steps_reached": int(md.get("max_step", -1)) >= requested_steps,
        "temperature_mean_available": mean_temperature is not None,
        "temperature_mean_within_25_K": mean_temperature is not None
        and abs(float(mean_temperature) - target_temperature) <= 25.0,
        "atom_count_is_1728": int(trajectory.get("atom_count", -1)) == 1728,
        "nearest_neighbor_gt_2_A": nearest_neighbor > 2.0,
        "all_frames_nearest_neighbor_gt_2_A": minimum_nearest_neighbor > 2.0,
        "integrated_order_metrics_available": all(
            value is not None for value in (change, initial, final)
        ),
    }
    return {
        "run_dir": analysis.get("run_dir"),
        "phase_status": analysis.get("status"),
        "target_temperature_k": target_temperature,
        "temperature_last_100_steps_k": temperature,
        "requested_steps": requested_steps,
        "max_step": md.get("max_step"),
        "nearest_neighbor_A": nearest_neighbor,
        "minimum_nearest_neighbor_A": minimum_nearest_neighbor,
        "integrated_ordered_fraction_initial": initial,
        "integrated_ordered_fraction_final": final,
        "integrated_ordered_fraction_change": change,
        "checks": checks,
        "status": "quality_verified" if all(checks.values()) else "quality_failed",
    }


def summarize(
    melting: Dict[str, Any],
    low: Dict[str, Any],
    nominal: Dict[str, Any],
    high: Dict[str, Any],
    *,
    low_temperature: float,
    nominal_temperature: float,
    high_temperature: float,
    requested_steps: int = 500,
    migration_threshold: float = 0.02,
    nominal_stability_tolerance: float = 0.05,
) -> Dict[str, Any]:
    if melting.get("schema") != "wt-gibbs-helmholtz-melting-v2":
        raise ValueError("melting result has the wrong schema")
    if melting.get("status") != "verified" or not melting.get("checks") or not all(
        melting["checks"].values()
    ):
        raise ValueError("melting result is not verified")
    melting_temperature = float(melting["melting_temperature_k"])
    diagnostics = {
        "low": run_diagnostics(low, low_temperature, requested_steps),
        "nominal": run_diagnostics(
            nominal, nominal_temperature, requested_steps
        ),
        "high": run_diagnostics(high, high_temperature, requested_steps),
    }
    low_change = diagnostics["low"]["integrated_ordered_fraction_change"]
    nominal_change = diagnostics["nominal"]["integrated_ordered_fraction_change"]
    high_change = diagnostics["high"]["integrated_ordered_fraction_change"]
    checks = {
        "temperatures_bracket_predicted_melting": low_temperature
        < melting_temperature
        < high_temperature,
        "nominal_temperature_matches_prediction": abs(
            nominal_temperature - melting_temperature
        )
        <= 0.51,
        "all_run_quality_verified": all(
            row["status"] == "quality_verified" for row in diagnostics.values()
        ),
        "nominal_two_phase_verified": nominal.get("status")
        == "two_phase_verified",
        "nominal_interface_nearly_stable": nominal_change is not None
        and abs(float(nominal_change)) <= nominal_stability_tolerance,
        "low_temperature_solid_growth": low_change is not None
        and float(low_change) >= migration_threshold,
        "high_temperature_liquid_growth": high_change is not None
        and float(high_change) <= -migration_threshold,
    }
    return {
        "schema": "wt-direct-coexistence-validation-v1",
        "status": "verified" if all(checks.values()) else "validation_failed",
        "predicted_melting_temperature_k": melting_temperature,
        "validation_temperatures_k": {
            "low": low_temperature,
            "nominal": nominal_temperature,
            "high": high_temperature,
        },
        "migration_threshold": migration_threshold,
        "nominal_stability_tolerance": nominal_stability_tolerance,
        "checks": checks,
        "runs": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--melting", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low-temperature", type=float, required=True)
    parser.add_argument("--nominal-temperature", type=float, required=True)
    parser.add_argument("--high-temperature", type=float, required=True)
    parser.add_argument("--requested-steps", type=int, default=500)
    parser.add_argument("--migration-threshold", type=float, default=0.02)
    parser.add_argument("--nominal-stability-tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "melting": args.melting.resolve(),
        "low": args.low.resolve(),
        "nominal": args.nominal.resolve(),
        "high": args.high.resolve(),
    }
    result = summarize(
        load_json(paths["melting"]),
        load_json(paths["low"]),
        load_json(paths["nominal"]),
        load_json(paths["high"]),
        low_temperature=args.low_temperature,
        nominal_temperature=args.nominal_temperature,
        high_temperature=args.high_temperature,
        requested_steps=args.requested_steps,
        migration_threshold=args.migration_threshold,
        nominal_stability_tolerance=args.nominal_stability_tolerance,
    )
    result["provenance"] = {key: str(value) for key, value in paths.items()}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
