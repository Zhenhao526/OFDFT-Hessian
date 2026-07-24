#!/usr/bin/env python3
"""Select a prepared enthalpy grid that brackets the conservative Tm interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


PREPARED_GRIDS_K = (
    (900.0, 975.0, 1050.0),
    (900.0, 975.0, 1050.0, 1100.0),
)


def select_grid(
    center_k: float,
    half_width_k: float,
    grids: Sequence[Sequence[float]] = PREPARED_GRIDS_K,
) -> Dict[str, Any]:
    if half_width_k < 0.0:
        raise ValueError("melting-temperature half width must be non-negative")
    lower = center_k - half_width_k
    upper = center_k + half_width_k
    selected = tuple(float(value) for value in grids[-1])
    status = "grid_incomplete"
    for grid in grids:
        candidate = tuple(float(value) for value in grid)
        if lower >= candidate[0] and upper <= candidate[-1]:
            selected = candidate
            status = "grid_verified"
            break
    return {
        "schema": "wt-multitemperature-grid-decision-v2",
        "status": status,
        "preliminary_melting_temperature_k": center_k,
        "preliminary_conservative_interval_k": [lower, upper],
        "enthalpy_temperature_grid_k": list(selected),
        "checks": {
            "lower_interval_not_below_grid": lower >= selected[0],
            "upper_interval_not_above_grid": upper <= selected[-1],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("combination", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combination_path = args.combination.resolve()
    report = json.loads(combination_path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    if report.get("schema") != "wt-melting-free-energy-combination-v2":
        raise RuntimeError("free-energy combination does not use v2 schema")
    if report.get("status") != "anchor_temperature_free_energy_verified":
        raise RuntimeError("free-energy anchor is not verified")
    if not checks or not all(checks.values()):
        raise RuntimeError("free-energy anchor checks are not all verified")

    decision = select_grid(
        float(report["melting_temperature_linearized_k"]),
        float(report["melting_temperature_conservative_ti_uncertainty_k"]),
    )
    decision["provenance"] = {
        "free_energy_combination": str(combination_path),
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))
    if decision["status"] != "grid_verified":
        raise SystemExit(
            "prepared enthalpy temperature grids do not bracket the conservative interval"
        )


if __name__ == "__main__":
    main()
