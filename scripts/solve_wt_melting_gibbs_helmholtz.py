#!/usr/bin/env python3
"""Solve a WT melting point from an absolute DeltaG anchor and P=0 enthalpies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from mpn_melting.gibbs_helmholtz import (
    gibbs_over_temperature,
    solve_melting_temperature,
)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def perturbed_points(
    points: Sequence[Dict[str, Any]],
    uncertainty_sign: float,
    uncertainty_field: str = "block_standard_error_mev_per_atom",
) -> list[Tuple[float, float]]:
    result = []
    for point in points:
        enthalpy = float(point["delta_h_ev_per_atom"])
        uncertainty = float(point[uncertainty_field]) / 1000.0
        value = enthalpy + uncertainty_sign * uncertainty
        if value <= 0.0:
            raise ValueError("fusion enthalpy minus its uncertainty must remain positive")
        result.append((float(point["temperature_k"]), value))
    return result


def optional_root(
    anchor_temperature_k: float,
    anchor_delta_g_ev: float,
    points: Sequence[Tuple[float, float]],
) -> Optional[float]:
    try:
        return solve_melting_temperature(
            anchor_temperature_k, anchor_delta_g_ev, points
        )
    except ValueError:
        return None


def uncertainty_interval(
    anchor_temperature_k: float,
    anchor_delta_g_ev: float,
    delta_g_uncertainty_mev: float,
    points: Sequence[Dict[str, Any]],
    enthalpy_uncertainty_field: str = "block_standard_error_mev_per_atom",
) -> Dict[str, Any]:
    delta = delta_g_uncertainty_mev / 1000.0
    scenarios = []
    for delta_g_sign in (-1.0, 1.0):
        for enthalpy_sign in (-1.0, 1.0):
            root = optional_root(
                anchor_temperature_k,
                anchor_delta_g_ev + delta_g_sign * delta,
                perturbed_points(
                    points,
                    enthalpy_sign,
                    uncertainty_field=enthalpy_uncertainty_field,
                ),
            )
            scenarios.append(
                {
                    "delta_g_uncertainty_sign": delta_g_sign,
                    "enthalpy_uncertainty_sign": enthalpy_sign,
                    "root_k": root,
                }
            )
    roots = [scenario["root_k"] for scenario in scenarios if scenario["root_k"] is not None]
    lower = min(roots) if len(roots) == len(scenarios) else None
    upper = max(roots) if len(roots) == len(scenarios) else None
    return {
        "lower_k": lower,
        "upper_k": upper,
        "half_width_k": 0.5 * (upper - lower)
        if lower is not None and upper is not None
        else None,
        "scenarios": scenarios,
    }


def convergence_summary_matches_series(
    summary: Dict[str, Any], series: Dict[str, Any]
) -> tuple[bool, list[Dict[str, float]]]:
    if summary.get("status") != "verified" or series.get("status") != "verified":
        return False, []

    discard = float(series["discard_fraction"])
    series_points = {
        float(point["temperature_k"]): point for point in series.get("points", [])
    }
    summary_points = {
        float(point["temperature_k"]): point for point in summary.get("points", [])
    }
    if set(series_points) != set(summary_points):
        return False, []

    uncertainties = []
    for temperature in sorted(series_points):
        point = series_points[temperature]
        convergence = summary_points[temperature]
        if point.get("status") != "verified" or convergence.get("status") != "verified":
            return False, []
        reports = [
            report
            for report in convergence.get("reports", [])
            if math.isclose(
                float(report["discard_fraction"]), discard, abs_tol=1.0e-12
            )
        ]
        if len(reports) != 1 or reports[0].get("status") != "verified":
            return False, []
        report = reports[0]
        delta_h_mev = 1000.0 * float(point["delta_h_ev_per_atom"])
        if not math.isclose(
            float(report["delta_h_mev_per_atom"]),
            delta_h_mev,
            rel_tol=1.0e-11,
            abs_tol=1.0e-11,
        ):
            return False, []
        block = float(point["block_standard_error_mev_per_atom"])
        if not math.isclose(
            float(report["block_standard_error_mev_per_atom"]),
            block,
            rel_tol=1.0e-11,
            abs_tol=1.0e-11,
        ):
            return False, []
        half_drift = float(point["half_drift_mev_per_atom"])
        if not math.isclose(
            float(report["half_drift_mev_per_atom"]),
            half_drift,
            rel_tol=1.0e-11,
            abs_tol=1.0e-11,
        ):
            return False, []
        discard_spread = float(
            convergence["delta_h_discard_spread_mev_per_atom"]
        )
        uncertainties.append(
            {
                "temperature_k": temperature,
                "statistical_mev_per_atom": block,
                "half_drift_mev_per_atom": half_drift,
                "discard_spread_mev_per_atom": discard_spread,
                "conservative_mev_per_atom": block
                + half_drift
                + discard_spread,
            }
        )
    return True, uncertainties


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate DeltaH/T^2 and solve the Gibbs free-energy crossing"
    )
    parser.add_argument("--combination", type=Path, required=True)
    parser.add_argument("--enthalpy-series", type=Path, required=True)
    parser.add_argument("--enthalpy-convergence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    combination = load_json(args.combination.resolve())
    enthalpy_series = load_json(args.enthalpy_series.resolve())
    enthalpy_convergence = load_json(args.enthalpy_convergence.resolve())
    points = sorted(enthalpy_series["points"], key=lambda item: item["temperature_k"])
    convergence_matches, enthalpy_uncertainties = convergence_summary_matches_series(
        enthalpy_convergence, enthalpy_series
    )
    checks = {
        "free_energy_anchor_verified": combination["status"]
        == "anchor_temperature_free_energy_verified"
        and combination.get("schema") == "wt-melting-free-energy-combination-v2",
        "free_energy_anchor_checks_verified": bool(combination.get("checks"))
        and all(combination["checks"].values()),
        "enthalpy_total_energy_definition_verified": (
            enthalpy_series.get("schema")
            == "wt-zero-pressure-fusion-enthalpy-series-v2"
            and enthalpy_series.get("enthalpy_energy_definition")
            == "sampled_total_energy_plus_external_pv"
            and enthalpy_convergence.get("schema")
            == "wt-enthalpy-discard-convergence-summary-v2"
            and enthalpy_convergence.get("enthalpy_energy_definition")
            == "sampled_total_energy_plus_external_pv"
        ),
        "enthalpy_series_verified": enthalpy_series["status"] == "verified",
        "all_enthalpy_points_verified": all(
            point["status"] == "verified" for point in points
        ),
        "enthalpy_discard_convergence_verified": convergence_matches,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"input verification failed: {failed}")

    anchor_temperature = float(combination["temperature_k"])
    delta_g = float(
        combination["free_energy_ev_per_atom"]["liquid_minus_solid"]
    )
    nominal_points = [
        (float(point["temperature_k"]), float(point["delta_h_ev_per_atom"]))
        for point in points
    ]
    uncertainty_by_temperature = {
        row["temperature_k"]: row for row in enthalpy_uncertainties
    }
    points_with_uncertainty = []
    for point in points:
        temperature = float(point["temperature_k"])
        uncertainty = uncertainty_by_temperature[temperature]
        points_with_uncertainty.append(
            {
                **point,
                "enthalpy_statistical_uncertainty_mev_per_atom": uncertainty[
                    "statistical_mev_per_atom"
                ],
                "enthalpy_half_drift_mev_per_atom": uncertainty[
                    "half_drift_mev_per_atom"
                ],
                "enthalpy_discard_spread_mev_per_atom": uncertainty[
                    "discard_spread_mev_per_atom"
                ],
                "enthalpy_conservative_uncertainty_mev_per_atom": uncertainty[
                    "conservative_mev_per_atom"
                ],
            }
        )
    profile = gibbs_over_temperature(anchor_temperature, delta_g, nominal_points)
    melting_temperature = optional_root(anchor_temperature, delta_g, nominal_points)
    statistical = float(
        combination["uncertainty_budget_mev_per_atom"]["statistical_rss"]
    )
    conservative = float(
        combination["uncertainty_budget_mev_per_atom"]["combined_conservative"]
    )
    statistical_interval = uncertainty_interval(
        anchor_temperature,
        delta_g,
        statistical,
        points_with_uncertainty,
        enthalpy_uncertainty_field="enthalpy_statistical_uncertainty_mev_per_atom",
    )
    conservative_interval = uncertainty_interval(
        anchor_temperature,
        delta_g,
        conservative,
        points_with_uncertainty,
        enthalpy_uncertainty_field="enthalpy_conservative_uncertainty_mev_per_atom",
    )
    checks.update(
        {
            "nominal_root_bracketed": melting_temperature is not None,
            "statistical_interval_bracketed": statistical_interval["lower_k"] is not None
            and statistical_interval["upper_k"] is not None,
            "conservative_interval_bracketed": conservative_interval["lower_k"] is not None
            and conservative_interval["upper_k"] is not None,
        }
    )
    result = {
        "schema": "wt-gibbs-helmholtz-melting-v2",
        "status": "verified" if all(checks.values()) else "temperature_bracket_incomplete",
        "checks": checks,
        "anchor_temperature_k": anchor_temperature,
        "anchor_delta_g_ev_per_atom": delta_g,
        "melting_temperature_k": melting_temperature,
        "statistical_interval": statistical_interval,
        "conservative_interval": conservative_interval,
        "gibbs_profile": [
            {
                "temperature_k": temperature,
                "delta_g_over_t_ev_per_atom_k": value,
                "delta_g_ev_per_atom": temperature * value,
            }
            for temperature, value in profile
        ],
        "enthalpy_points": points_with_uncertainty,
        "method": "piecewise-linear DeltaH(T) with exact Gibbs-Helmholtz integration",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "provenance": {
            "combination": str(args.combination.resolve()),
            "enthalpy_series": str(args.enthalpy_series.resolve()),
            "enthalpy_convergence": str(args.enthalpy_convergence.resolve()),
        },
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
