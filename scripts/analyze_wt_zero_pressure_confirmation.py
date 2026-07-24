#!/usr/bin/env python3
"""Gate long WT zero-pressure confirmation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--temperature-tolerance", type=float, default=25.0)
    parser.add_argument("--pressure-tolerance", type=float, default=2.5)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "confirmation_manifest.json").read_text())
    phase_results = []
    for item in manifest["phases"]:
        phase = item["phase"]
        run = root / phase
        phase_analysis = analyze_phase(run, phase, thermalized_initial=True)
        (run / "phase_analysis.json").write_text(
            json.dumps(phase_analysis, indent=2, sort_keys=True) + "\n"
        )
        logs = sorted(run.glob("OUT.*/running_md.log"))
        rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
        late = rows[len(rows) // 2 :]
        temperature = series_stats([row["temperature_K"] for row in late])
        pressure = series_stats([row["pressure_kbar"] for row in late if "pressure_kbar" in row])
        temperature_error = abs(
            float(temperature.get("mean", float("inf"))) - manifest["temperature_K"]
        )
        pressure_error = abs(float(pressure.get("mean", float("inf"))))
        nearest = phase_analysis.get("trajectory", {}).get("nearest_neighbor_A", 0.0)
        checks = {
            "reached_requested_step": max_step >= manifest["steps"],
            "phase_verified": phase_analysis.get("status") == f"{phase}_verified",
            "temperature_mean_within_tolerance": temperature_error <= args.temperature_tolerance,
            "pressure_mean_within_tolerance": pressure_error <= args.pressure_tolerance,
            "nearest_neighbor_gt_2_A": nearest > 2.0,
        }
        result = {
            **item,
            "max_step": max_step,
            "temperature_last_half_K": temperature,
            "temperature_target_error_K": temperature_error,
            "pressure_last_half_kbar": pressure,
            "pressure_target_error_kbar": pressure_error,
            "nearest_neighbor_A": nearest,
            "phase_status": phase_analysis.get("status"),
            "checks": checks,
            "status": "confirmation_passed" if all(checks.values()) else "confirmation_failed",
        }
        (run / "confirmation_result.json").write_text(json.dumps(result, indent=2) + "\n")
        phase_results.append(result)
    summary = {
        **manifest,
        "temperature_tolerance_K": args.temperature_tolerance,
        "pressure_tolerance_kbar": args.pressure_tolerance,
        "phase_results": phase_results,
        "status": (
            "all_confirmations_passed"
            if all(item["status"] == "confirmation_passed" for item in phase_results)
            else "confirmation_gate_failed"
        ),
    }
    (root / "confirmation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
