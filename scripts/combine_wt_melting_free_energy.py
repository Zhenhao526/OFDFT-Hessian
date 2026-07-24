#!/usr/bin/env python3
"""Combine analytic anchors and production TI legs into a WT melting estimate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rss(*values: float) -> float:
    return math.sqrt(sum(value * value for value in values))


def linearized_melting_temperature(
    temperature_k: float, delta_g_ev_per_atom: float, delta_h_ev_per_atom: float
) -> float:
    if delta_h_ev_per_atom <= 0.0:
        raise ValueError("fusion enthalpy must be positive")
    denominator = 1.0 - delta_g_ev_per_atom / delta_h_ev_per_atom
    if denominator <= 0.0:
        raise ValueError("linearized Gibbs-Helmholtz denominator must be positive")
    return temperature_k / denominator


def melting_temperature_uncertainty(
    temperature_k: float,
    delta_g_ev_per_atom: float,
    delta_h_ev_per_atom: float,
    delta_g_uncertainty_ev_per_atom: float,
    delta_h_uncertainty_ev_per_atom: float,
) -> float:
    denominator = 1.0 - delta_g_ev_per_atom / delta_h_ev_per_atom
    derivative_g = temperature_k / (delta_h_ev_per_atom * denominator**2)
    derivative_h = (
        -temperature_k
        * delta_g_ev_per_atom
        / (delta_h_ev_per_atom**2 * denominator**2)
    )
    return rss(
        derivative_g * delta_g_uncertainty_ev_per_atom,
        derivative_h * delta_h_uncertainty_ev_per_atom,
    )


def leg_error_budget(analysis: Dict[str, Any]) -> Dict[str, float]:
    return {
        "block_standard_error_mev_per_atom": float(
            analysis["block_standard_error_mev_per_atom"]
        ),
        "quadrature_difference_mev_per_atom": float(
            analysis["quadrature_difference_mev_per_atom"]
        ),
        "half_drift_mev_per_atom": float(analysis["half_drift_mev_per_atom"]),
    }


def finite_size_allowance_mev_per_atom(anchor_audit: Dict[str, Any]) -> float:
    sensitivity = anchor_audit["finite_size_sensitivity"]
    exact_ideal = abs(
        float(
            sensitivity[
                "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom"
            ]
        )
    )
    polson_scale = abs(float(sensitivity["polson_kbt_ln_n_over_n_mev_per_atom"]))
    return max(exact_ideal, polson_scale)


def convergence_summary_matches_analysis(
    summary: Dict[str, Any], analysis: Dict[str, Any], phase: str
) -> bool:
    if summary.get("status") != "verified" or summary.get("phase") != phase:
        return False
    if int(summary.get("natoms", -1)) != int(analysis.get("natoms", -2)):
        return False
    direct_value = float(analysis["delta_f_wt_minus_pair_simpson_mev_per_atom"])
    if not math.isclose(
        float(summary["integral_consensus_mev_per_atom"]),
        direct_value,
        rel_tol=1.0e-11,
        abs_tol=1.0e-11,
    ):
        return False
    discard = float(analysis["discard_fraction"])
    matching_reports = [
        report
        for report in summary.get("reports", [])
        if math.isclose(float(report["discard_fraction"]), discard, abs_tol=1.0e-12)
    ]
    return len(matching_reports) == 1 and matching_reports[0].get("status") == "verified"


def fusion_enthalpy_from_ti_endpoints(
    solid_wt: dict[str, Any], liquid_wt: dict[str, Any]
) -> dict[str, float]:
    potential = (
        float(liquid_wt["lambda_one_target_potential_ev_per_atom"])
        - float(solid_wt["lambda_one_target_potential_ev_per_atom"])
    )
    kinetic = (
        float(liquid_wt["lambda_one_target_kinetic_ev_per_atom"])
        - float(solid_wt["lambda_one_target_kinetic_ev_per_atom"])
    )
    total = (
        float(liquid_wt["lambda_one_target_total_energy_ev_per_atom"])
        - float(solid_wt["lambda_one_target_total_energy_ev_per_atom"])
    )
    return {
        "potential_difference_ev_per_atom": potential,
        "kinetic_difference_ev_per_atom": kinetic,
        "total_energy_difference_ev_per_atom": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Einstein/sUF anchors and pair-to-WT production TI"
    )
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--solid-anchor", type=Path, required=True)
    parser.add_argument("--liquid-anchor", type=Path, required=True)
    parser.add_argument("--anchor-audit", type=Path, required=True)
    parser.add_argument("--solid-wt", type=Path, required=True)
    parser.add_argument("--liquid-wt", type=Path, required=True)
    parser.add_argument("--solid-wt-convergence", type=Path, required=True)
    parser.add_argument("--liquid-wt-convergence", type=Path, required=True)
    parser.add_argument("--zero-pressure-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analytic = load_json(args.analytic.resolve())
    solid_anchor = load_json(args.solid_anchor.resolve())
    liquid_anchor = load_json(args.liquid_anchor.resolve())
    anchor_audit = load_json(args.anchor_audit.resolve())
    solid_wt = load_json(args.solid_wt.resolve())
    liquid_wt = load_json(args.liquid_wt.resolve())
    solid_wt_convergence = load_json(args.solid_wt_convergence.resolve())
    liquid_wt_convergence = load_json(args.liquid_wt_convergence.resolve())
    zero_pressure = load_json(args.zero_pressure_summary.resolve())

    temperature = float(analytic["temperature_k"])
    natoms = int(analytic["natoms"])
    checks = {
        "solid_anchor_stable": solid_anchor["all_windows_stable"] is True,
        "liquid_anchor_stable": liquid_anchor["all_windows_stable"] is True,
        "reference_anchor_audit_verified": anchor_audit["status"]
        == "verified_with_finite_size_sensitivity",
        "solid_wt_verified": solid_wt["status"] == "verified",
        "liquid_wt_verified": liquid_wt["status"] == "verified",
        "solid_wt_discard_convergence_verified": convergence_summary_matches_analysis(
            solid_wt_convergence, solid_wt, "solid"
        ),
        "liquid_wt_discard_convergence_verified": convergence_summary_matches_analysis(
            liquid_wt_convergence, liquid_wt, "liquid"
        ),
        "zero_pressure_confirmed": zero_pressure["status"] == "all_confirmations_passed",
        "atom_counts_match": int(solid_wt["natoms"]) == natoms
        and int(liquid_wt["natoms"]) == natoms
        and int(anchor_audit["natoms"]) == natoms,
        "temperatures_match": math.isclose(
            float(solid_wt["target_temperature_K"]), temperature
        )
        and math.isclose(float(liquid_wt["target_temperature_K"]), temperature)
        and math.isclose(float(anchor_audit["temperature_k"]), temperature),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"input verification failed: {failed}")

    solid_reference = float(analytic["einstein"]["corrected_free_energy_ev_per_atom"])
    liquid_reference = float(analytic["suf"]["total_free_energy_ev_per_atom"])
    solid_pair_correction = float(
        solid_anchor["delta_f_pair_minus_reference_simpson_ev_per_atom"]
    )
    liquid_pair_correction = float(
        liquid_anchor["delta_f_pair_minus_reference_simpson_ev_per_atom"]
    )
    pair_delta = (
        liquid_reference
        + liquid_pair_correction
        - solid_reference
        - solid_pair_correction
    )
    audited_pair_delta = float(
        anchor_audit["absolute_pair_free_energy_ev_per_atom"][
            "liquid_minus_solid_raw"
        ]
    )
    checks["reference_anchor_delta_matches_audit"] = math.isclose(
        pair_delta, audited_pair_delta, rel_tol=1.0e-11, abs_tol=1.0e-11
    )
    if not checks["reference_anchor_delta_matches_audit"]:
        raise RuntimeError("input verification failed: reference_anchor_delta_matches_audit")
    solid_wt_correction = float(solid_wt["delta_f_wt_minus_pair_simpson_ev_system"]) / natoms
    liquid_wt_correction = (
        float(liquid_wt["delta_f_wt_minus_pair_simpson_ev_system"]) / natoms
    )
    solid_free_energy = solid_reference + solid_pair_correction + solid_wt_correction
    liquid_free_energy = liquid_reference + liquid_pair_correction + liquid_wt_correction
    delta_g = liquid_free_energy - solid_free_energy

    # The confirmed volumes represent P=0, so the external P*DeltaV term is zero.
    enthalpy_components = fusion_enthalpy_from_ti_endpoints(solid_wt, liquid_wt)
    fusion_enthalpy = enthalpy_components["total_energy_difference_ev_per_atom"]
    melting_temperature = linearized_melting_temperature(temperature, delta_g, fusion_enthalpy)

    legs = {
        "solid_reference_to_pair": leg_error_budget(solid_anchor),
        "liquid_reference_to_pair": leg_error_budget(liquid_anchor),
        "solid_pair_to_wt": leg_error_budget(solid_wt),
        "liquid_pair_to_wt": leg_error_budget(liquid_wt),
    }
    statistical = rss(
        *(item["block_standard_error_mev_per_atom"] for item in legs.values())
    )
    quadrature = rss(
        *(item["quadrature_difference_mev_per_atom"] for item in legs.values())
    )
    temporal = rss(*(item["half_drift_mev_per_atom"] for item in legs.values()))
    ti_combined_conservative = rss(statistical, quadrature, temporal)
    finite_size_allowance = finite_size_allowance_mev_per_atom(anchor_audit)
    conservative_delta_g = ti_combined_conservative + finite_size_allowance
    fusion_enthalpy_se = rss(
        float(
            solid_wt[
                "lambda_one_target_total_energy_block_standard_error_mev_per_atom"
            ]
        ),
        float(
            liquid_wt[
                "lambda_one_target_total_energy_block_standard_error_mev_per_atom"
            ]
        ),
    )
    statistical_tm = melting_temperature_uncertainty(
        temperature,
        delta_g,
        fusion_enthalpy,
        statistical / 1000.0,
        fusion_enthalpy_se / 1000.0,
    )
    conservative_tm = melting_temperature_uncertainty(
        temperature,
        delta_g,
        fusion_enthalpy,
        conservative_delta_g / 1000.0,
        fusion_enthalpy_se / 1000.0,
    )

    result = {
        "schema": "wt-melting-free-energy-combination-v2",
        "status": "anchor_temperature_free_energy_verified",
        "temperature_k": temperature,
        "natoms": natoms,
        "checks": checks,
        "free_energy_ev_per_atom": {
            "solid": solid_free_energy,
            "liquid": liquid_free_energy,
            "liquid_minus_solid": delta_g,
        },
        "free_energy_mev_per_atom": {
            "liquid_minus_solid": 1000.0 * delta_g,
        },
        "fusion_enthalpy_ev_per_atom": fusion_enthalpy,
        "fusion_enthalpy_components_ev_per_atom": enthalpy_components,
        "fusion_enthalpy_block_standard_error_mev_per_atom": fusion_enthalpy_se,
        "melting_temperature_linearized_k": melting_temperature,
        "melting_temperature_statistical_uncertainty_k": statistical_tm,
        "melting_temperature_conservative_ti_uncertainty_k": conservative_tm,
        "uncertainty_budget_mev_per_atom": {
            "legs": legs,
            "statistical_rss": statistical,
            "quadrature_rss": quadrature,
            "temporal_drift_rss": temporal,
            "ti_combined_conservative": ti_combined_conservative,
            "finite_size_systematic_allowance": finite_size_allowance,
            "combined_conservative": conservative_delta_g,
        },
        "method_notes": [
            "Melting temperature uses a constant-fusion-enthalpy Gibbs-Helmholtz linearization.",
            "The preliminary fusion enthalpy uses sampled total energies, including the finite-sample kinetic-energy difference.",
            "The external pressure is zero; residual internal-pressure fluctuations are not added as P*DeltaV.",
            "The central DeltaG uses the audited finite-N convention without applying an incomplete one-sided correction.",
            "The larger of the exact ideal N! sensitivity and kBT ln(N)/N scale is added linearly to the conservative DeltaG uncertainty.",
            "A second-temperature free-energy point or direct coexistence check is required before finalizing the melting point.",
        ],
        "provenance": {
            "analytic": str(args.analytic.resolve()),
            "solid_anchor": str(args.solid_anchor.resolve()),
            "liquid_anchor": str(args.liquid_anchor.resolve()),
            "anchor_audit": str(args.anchor_audit.resolve()),
            "solid_wt": str(args.solid_wt.resolve()),
            "liquid_wt": str(args.liquid_wt.resolve()),
            "solid_wt_convergence": str(args.solid_wt_convergence.resolve()),
            "liquid_wt_convergence": str(args.liquid_wt_convergence.resolve()),
            "zero_pressure_summary": str(args.zero_pressure_summary.resolve()),
        },
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
