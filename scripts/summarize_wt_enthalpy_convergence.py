#!/usr/bin/env python3
"""Summarize discard-fraction convergence of zero-pressure fusion enthalpies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _temperature_key(value: float) -> str:
    return f"{float(value):.9f}"


def summarize(
    reports: Sequence[Dict[str, Any]],
    *,
    maximum_discard_spread_mev_per_atom: float = 2.0,
    maximum_block_standard_error_mev_per_atom: float = 3.0,
) -> Dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two discard-fraction reports are required")
    if maximum_discard_spread_mev_per_atom < 0.0:
        raise ValueError("maximum discard spread must be non-negative")
    if maximum_block_standard_error_mev_per_atom < 0.0:
        raise ValueError("maximum block standard error must be non-negative")

    discard_fractions = [float(report["discard_fraction"]) for report in reports]
    if len(set(discard_fractions)) != len(discard_fractions):
        raise ValueError("discard fractions must be unique")

    by_report = []
    reference_temperatures = None
    point_rows: Dict[str, list[Dict[str, Any]]] = {}
    for report in reports:
        if report.get("schema") != "wt-zero-pressure-fusion-enthalpy-series-v2":
            raise ValueError("enthalpy reports must use the total-energy v2 schema")
        if (
            report.get("enthalpy_energy_definition")
            != "sampled_total_energy_plus_external_pv"
        ):
            raise ValueError("enthalpy reports use an unsupported energy definition")
        points = sorted(report["points"], key=lambda row: float(row["temperature_k"]))
        temperatures = tuple(_temperature_key(row["temperature_k"]) for row in points)
        if reference_temperatures is None:
            reference_temperatures = temperatures
        elif temperatures != reference_temperatures:
            raise ValueError("enthalpy reports contain different temperature grids")
        by_report.append(
            {
                "discard_fraction": float(report["discard_fraction"]),
                "status": report["status"],
            }
        )
        for point in points:
            key = _temperature_key(point["temperature_k"])
            point_rows.setdefault(key, []).append(
                {
                    "discard_fraction": float(report["discard_fraction"]),
                    "status": point["status"],
                    "delta_h_mev_per_atom": 1000.0
                    * float(point["delta_h_ev_per_atom"]),
                    "block_standard_error_mev_per_atom": float(
                        point["block_standard_error_mev_per_atom"]
                    ),
                    "half_drift_mev_per_atom": float(
                        point["half_drift_mev_per_atom"]
                    ),
                    "liquid_minus_solid_temperature_mean_difference_k": float(
                        point[
                            "liquid_minus_solid_temperature_mean_difference_k"
                        ]
                    ),
                    "failed_checks": sorted(
                        name
                        for name, passed in point["checks"].items()
                        if not passed
                    ),
                }
            )

    critical_temperatures: Dict[str, list[str]] = {}
    points_summary = []
    for key in reference_temperatures or ():
        rows = sorted(point_rows[key], key=lambda row: row["discard_fraction"])
        values = [row["delta_h_mev_per_atom"] for row in rows]
        spread = max(values) - min(values)
        maximum_se = max(
            row["block_standard_error_mev_per_atom"] for row in rows
        )
        failed_checks = sorted(
            {
                check
                for row in rows
                for check in row.get("failed_checks", [])
            }
        )
        reasons = []
        if any(row["status"] != "verified" for row in rows):
            reasons.append("report_gate")
        if spread > maximum_discard_spread_mev_per_atom:
            reasons.append("discard_sensitivity")
        if maximum_se > maximum_block_standard_error_mev_per_atom:
            reasons.append("block_standard_error")
        if reasons:
            critical_temperatures[key] = reasons
        points_summary.append(
            {
                "temperature_k": float(key),
                "reports": rows,
                "delta_h_discard_spread_mev_per_atom": spread,
                "maximum_block_standard_error_mev_per_atom": maximum_se,
                "failed_checks": failed_checks,
                "status": "verified" if not reasons else "extension_required",
                "extension_reasons": reasons,
            }
        )

    checks = {
        "all_reports_verified": all(
            report["status"] == "verified" for report in reports
        ),
        "discard_sensitivity_gate": all(
            point["delta_h_discard_spread_mev_per_atom"]
            <= maximum_discard_spread_mev_per_atom
            for point in points_summary
        ),
        "block_standard_error_gate": all(
            point["maximum_block_standard_error_mev_per_atom"]
            <= maximum_block_standard_error_mev_per_atom
            for point in points_summary
        ),
    }
    return {
        "schema": "wt-enthalpy-discard-convergence-summary-v2",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "status": "verified" if all(checks.values()) else "extension_required",
        "checks": checks,
        "discard_reports": sorted(
            by_report, key=lambda row: row["discard_fraction"]
        ),
        "maximum_discard_spread_mev_per_atom": maximum_discard_spread_mev_per_atom,
        "maximum_block_standard_error_mev_per_atom": (
            maximum_block_standard_error_mev_per_atom
        ),
        "critical_temperatures": critical_temperatures,
        "points": points_summary,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit fusion-enthalpy convergence across discard fractions"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--maximum-discard-spread", type=float, default=2.0)
    parser.add_argument("--maximum-block-standard-error", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report_paths = [path.resolve() for path in args.reports]
    result = summarize(
        [load_json(path) for path in report_paths],
        maximum_discard_spread_mev_per_atom=args.maximum_discard_spread,
        maximum_block_standard_error_mev_per_atom=args.maximum_block_standard_error,
    )
    result["provenance"] = {"reports": [str(path) for path in report_paths]}
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
