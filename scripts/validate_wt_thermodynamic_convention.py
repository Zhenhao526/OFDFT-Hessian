#!/usr/bin/env python3
"""Validate WT thermodynamic signs, units, anchor values, and uncertainty rules."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mpn_melting.gibbs_helmholtz import (
    gibbs_over_temperature,
    solve_melting_temperature,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convention", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--evidence-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    convention = load_json(args.convention.resolve())
    anchor = load_json(args.anchor.resolve())
    evidence = load_json(args.evidence_audit.resolve())

    policy = convention["central_value_policy"]
    anchor_uncertainty = convention["uncertainty_policy"]["anchor"]
    enthalpy_uncertainty = convention["uncertainty_policy"][
        "enthalpy_per_temperature"
    ]
    point_900 = evidence["temperatures"]["900"]["enthalpy"]

    delta_g = float(anchor["free_energy_ev_per_atom"]["liquid_minus_solid"])
    solid_g = float(anchor["free_energy_ev_per_atom"]["solid"])
    liquid_g = float(anchor["free_energy_ev_per_atom"]["liquid"])
    production_delta_h_mev = float(point_900["d50_mev_per_atom"])
    preliminary_delta_h_mev = 1000.0 * float(anchor["fusion_enthalpy_ev_per_atom"])

    ti_rss = math.sqrt(
        anchor_uncertainty["statistical_mev_per_atom"] ** 2
        + anchor_uncertainty["quadrature_rss_mev_per_atom"] ** 2
        + anchor_uncertainty["temporal_drift_rss_mev_per_atom"] ** 2
    )
    combined_conservative = (
        anchor_uncertainty["ti_combined_conservative_mev_per_atom"]
        + anchor_uncertainty["finite_size_systematic_allowance_mev_per_atom"]
    )
    enthalpy_conservative = (
        enthalpy_uncertainty["900K_statistical_mev_per_atom"]
        + enthalpy_uncertainty["900K_half_drift_mev_per_atom"]
        + enthalpy_uncertainty["900K_discard_spread_mev_per_atom"]
    )

    known_root = solve_melting_temperature(
        900.0, 0.010, [(900.0, 0.100), (1050.0, 0.100)]
    )
    physical_profile = gibbs_over_temperature(
        900.0,
        delta_g,
        [(900.0, production_delta_h_mev / 1000.0), (1050.0, 0.080)],
    )

    checks = {
        "convention_schema": convention.get("schema")
        == "wt-thermodynamic-convention-v1",
        "anchor_schema": anchor.get("schema") == "wt-melting-free-energy-combination-v2",
        "anchor_verified": anchor.get("status")
        == "anchor_temperature_free_energy_verified"
        and all(anchor.get("checks", {}).values()),
        "delta_g_is_liquid_minus_solid": close(delta_g, liquid_g - solid_g),
        "anchor_temperature_matches": close(
            float(anchor["temperature_k"]), float(policy["anchor_temperature_K"])
        ),
        "anchor_delta_g_matches": close(
            1000.0 * delta_g, float(policy["anchor_delta_g_mev_per_atom"])
        ),
        "anchor_delta_g_positive_below_root": delta_g > 0.0,
        "production_enthalpy_matches_audit": close(
            production_delta_h_mev,
            float(policy["fusion_enthalpy_900K_mev_per_atom"]),
        ),
        "production_enthalpy_positive": production_delta_h_mev > 0.0,
        "preliminary_endpoint_enthalpy_is_excluded": close(
            preliminary_delta_h_mev,
            float(
                policy["forbidden_as_gibbs_helmholtz_input"][
                    "value_mev_per_atom"
                ]
            ),
        )
        and not close(preliminary_delta_h_mev, production_delta_h_mev),
        "anchor_ti_rss_matches": close(
            ti_rss,
            float(anchor_uncertainty["ti_combined_conservative_mev_per_atom"]),
        ),
        "anchor_conservative_linear_addition_matches": close(
            combined_conservative,
            float(anchor_uncertainty["combined_conservative_mev_per_atom"]),
        ),
        "enthalpy_conservative_linear_sum_matches": close(
            enthalpy_conservative,
            float(enthalpy_uncertainty["900K_conservative_mev_per_atom"]),
        ),
        "constant_enthalpy_sign_test_root_is_1000K": math.isclose(
            known_root, 1000.0, rel_tol=0.0, abs_tol=1.0e-6
        ),
        "positive_fusion_enthalpy_reduces_delta_g_over_t_with_temperature": (
            physical_profile[1][1] < physical_profile[0][1]
        ),
        "external_pressure_pv_is_zero": close(
            float(
                convention["enthalpy_definition"][
                    "external_pv_at_target_pressure_ev_per_atom"
                ]
            ),
            0.0,
        ),
        "evidence_900_is_thermodynamically_ready": (
            evidence["temperatures"]["900"]["classification"]
            == "thermodynamic_ready"
        ),
        "unready_temperatures_are_not_silently_accepted": (
            evidence["thermodynamically_ready_temperatures_K"] == [900.0]
        ),
    }
    result = {
        "schema": "wt-thermodynamic-convention-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if all(checks.values()) else "failed",
        "checks": checks,
        "values": {
            "anchor_delta_g_mev_per_atom": 1000.0 * delta_g,
            "production_fusion_enthalpy_900K_mev_per_atom": production_delta_h_mev,
            "excluded_preliminary_endpoint_enthalpy_mev_per_atom": (
                preliminary_delta_h_mev
            ),
            "anchor_statistical_mev_per_atom": anchor_uncertainty[
                "statistical_mev_per_atom"
            ],
            "anchor_ti_combined_conservative_mev_per_atom": ti_rss,
            "anchor_combined_conservative_mev_per_atom": combined_conservative,
            "enthalpy_900K_statistical_mev_per_atom": enthalpy_uncertainty[
                "900K_statistical_mev_per_atom"
            ],
            "enthalpy_900K_conservative_mev_per_atom": enthalpy_conservative,
            "constant_enthalpy_sign_test_root_K": known_root,
        },
        "provenance": {
            "convention": str(args.convention.resolve()),
            "anchor": str(args.anchor.resolve()),
            "evidence_audit": str(args.evidence_audit.resolve()),
        },
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
