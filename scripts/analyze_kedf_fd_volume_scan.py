#!/usr/bin/env python3
"""Analyze a finite-difference pressure volume scan for solid and liquid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def phase_summary(root: Path) -> dict:
    rows = []
    for result_path in sorted(root.glob("vpa_*/pressure_fd_result.json")):
        result = json.loads(result_path.read_text())
        rows.append(
            {
                "run": str(result_path.parent),
                "volume_per_atom_A3": float(
                    result["base_volume_per_atom_A3"]
                ),
                "static_pressure_kbar": float(
                    result["finite_difference_static_pressure_kbar"]
                ),
                "ideal_ionic_pressure_kbar": float(
                    result["ideal_ionic_pressure_kbar"]
                ),
                "estimated_total_pressure_kbar": float(
                    result["estimated_total_pressure_kbar"]
                ),
            }
        )
    rows.sort(key=lambda row: row["volume_per_atom_A3"])
    bracket = None
    estimate = None
    for left, right in zip(rows, rows[1:]):
        left_pressure = left["estimated_total_pressure_kbar"]
        right_pressure = right["estimated_total_pressure_kbar"]
        if left_pressure * right_pressure <= 0.0:
            bracket = [
                left["volume_per_atom_A3"],
                right["volume_per_atom_A3"],
            ]
            estimate = left["volume_per_atom_A3"] - left_pressure * (
                right["volume_per_atom_A3"] - left["volume_per_atom_A3"]
            ) / (right_pressure - left_pressure)
            break
    monotonic = len(rows) >= 2 and all(
        left["estimated_total_pressure_kbar"]
        > right["estimated_total_pressure_kbar"]
        for left, right in zip(rows, rows[1:])
    )
    checks = {
        "three_volume_points": len(rows) == 3,
        "pressure_strictly_decreases": monotonic,
        "zero_pressure_bracket_found": bracket is not None,
        "zero_pressure_estimate_inside_bracket": (
            bracket is not None
            and estimate is not None
            and bracket[0] <= estimate <= bracket[1]
        ),
    }
    return {
        "rows": rows,
        "zero_pressure_bracket_A3_per_atom": bracket,
        "linear_zero_pressure_estimate_A3_per_atom": estimate,
        "checks": checks,
        "status": (
            "zero_pressure_prescan_verified"
            if all(checks.values())
            else "zero_pressure_prescan_not_verified"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    phases = {
        phase: phase_summary(root / phase) for phase in ("solid", "liquid")
    }
    summary = {
        "schema": "kedf-fd-volume-scan-v1",
        "root": str(root),
        "phases": phases,
        "status": (
            "zero_pressure_prescan_verified"
            if all(
                item["status"] == "zero_pressure_prescan_verified"
                for item in phases.values()
            )
            else "zero_pressure_prescan_not_verified"
        ),
    }
    (root / "fd_volume_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
