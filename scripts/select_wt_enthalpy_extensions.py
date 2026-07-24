#!/usr/bin/env python3
"""Select enthalpy temperatures that are safe to extend for statistics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

ALLOWED_REASONS = {
    "report_gate",
    "discard_sensitivity",
    "block_standard_error",
}
ALLOWED_FAILED_CHECKS = {
    "half_drift_within_tolerance",
    "solid_liquid_temperature_means_match",
}


def select_extensions(summary: Dict[str, Any]) -> Dict[str, Any]:
    if summary.get("schema") != "wt-enthalpy-discard-convergence-summary-v2":
        raise ValueError("enthalpy convergence summary has the wrong schema")
    if summary.get("status") == "verified":
        return {
            "schema": "wt-enthalpy-extension-selection-v1",
            "status": "no_extension_required",
            "critical_temperatures_k": [],
        }
    points = [
        point
        for point in summary.get("points", [])
        if point.get("status") == "extension_required"
    ]
    if not points:
        raise ValueError("extension is required but no critical temperature is identified")
    selected = []
    for point in points:
        temperature = float(point["temperature_k"])
        reasons = set(point.get("extension_reasons", []))
        failed_checks = set(point.get("failed_checks", []))
        if not reasons or not reasons <= ALLOWED_REASONS:
            raise ValueError(
                f"unsupported extension reasons at {temperature:g} K: {sorted(reasons)}"
            )
        if "report_gate" in reasons and not failed_checks:
            raise ValueError(
                f"report gate at {temperature:g} K has no specific failed checks"
            )
        if not failed_checks <= ALLOWED_FAILED_CHECKS:
            raise ValueError(
                f"non-statistical enthalpy failure at {temperature:g} K: "
                f"{sorted(failed_checks)}"
            )
        selected.append(
            {
                "temperature_k": temperature,
                "extension_reasons": sorted(reasons),
                "failed_checks": sorted(failed_checks),
            }
        )
    return {
        "schema": "wt-enthalpy-extension-selection-v1",
        "status": "statistical_extension_required",
        "critical_temperatures_k": sorted(
            point["temperature_k"] for point in selected
        ),
        "points": sorted(selected, key=lambda point: point["temperature_k"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.summary.resolve()
    result = select_extensions(json.loads(source.read_text(encoding="utf-8")))
    result["provenance"] = {"convergence_summary": str(source)}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
