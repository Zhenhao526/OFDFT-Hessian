#!/usr/bin/env python3
"""Apply phase, temperature, pressure, and distance gates to one WT MD run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats


def analyze(
    run: Path,
    *,
    expected_phase: str,
    target_temperature: float,
    target_pressure: float,
    requested_steps: int,
    temperature_tolerance: float,
    pressure_tolerance: float,
) -> dict:
    phase = analyze_phase(run, expected_phase, thermalized_initial=True)
    logs = sorted(run.glob("OUT.*/running_md.log"))
    rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
    late = rows[len(rows) // 2 :]
    temperatures = [row["temperature_K"] for row in late]
    pressures = [row["pressure_kbar"] for row in late]
    temperature = series_stats(temperatures)
    pressure = series_stats(pressures)
    nearest = phase.get("trajectory", {}).get("nearest_neighbor_A", 0.0)
    checks = {
        "reached_requested_step": max_step >= requested_steps,
        "phase_verified": phase.get("status") == f"{expected_phase}_verified",
        "temperature_mean_within_tolerance": abs(
            float(temperature.get("mean", math.inf)) - target_temperature
        )
        <= temperature_tolerance,
        "pressure_mean_within_tolerance": abs(
            float(pressure.get("mean", math.inf)) - target_pressure
        )
        <= pressure_tolerance,
        "nearest_neighbor_gt_2_A": float(nearest) > 2.0,
    }
    return {
        "schema": "wt-phase-confirmation-v1",
        "run": str(run.resolve()),
        "phase": expected_phase,
        "target_temperature_K": target_temperature,
        "target_pressure_kbar": target_pressure,
        "requested_steps": requested_steps,
        "max_step": max_step,
        "temperature_last_half_K": temperature,
        "pressure_last_half_kbar": pressure,
        "nearest_neighbor_A": nearest,
        "phase_status": phase.get("status"),
        "phase_analysis": phase,
        "checks": checks,
        "status": "confirmation_passed" if all(checks.values()) else "confirmation_failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected", choices=("solid", "liquid"), required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--pressure", type=float, default=0.0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--temperature-tolerance", type=float, default=20.0)
    parser.add_argument("--pressure-tolerance", type=float, default=2.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.run.resolve(),
        expected_phase=args.expected,
        target_temperature=args.temperature,
        target_pressure=args.pressure,
        requested_steps=args.steps,
        temperature_tolerance=args.temperature_tolerance,
        pressure_tolerance=args.pressure_tolerance,
    )
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
