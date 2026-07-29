#!/usr/bin/env python3
"""Combine independent classical and target TI legs into a KEDF free-energy anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss(*values: float) -> float:
    return math.sqrt(sum(value * value for value in values))


def finite_size_allowance_mev_per_atom(audit: Dict[str, Any]) -> float:
    sensitivity = audit["finite_size_sensitivity"]
    return max(
        abs(
            float(
                sensitivity[
                    "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom"
                ]
            )
        ),
        abs(float(sensitivity["polson_kbt_ln_n_over_n_mev_per_atom"])),
    )


def classical_summary_matches(
    summary: Dict[str, Any], analysis: Dict[str, Any], phase: str
) -> bool:
    if (
        summary.get("status") != "verified"
        or summary.get("expected_phase") != phase
        or analysis.get("status") != "verified"
        or analysis.get("expected_phase") != phase
        or analysis.get("schema")
        != "mpn-classical-reference-pair-ti-analysis-v1"
    ):
        return False
    integrals = [
        float(value) for value in summary.get("discard_integrals_mev_per_atom", [])
    ]
    if len(integrals) != 3:
        return False
    selected = float(
        analysis["delta_f_pair_minus_reference_simpson_mev_per_atom"]
    )
    return math.isclose(selected, integrals[1], rel_tol=1.0e-11, abs_tol=1.0e-11)


def target_summary_matches(
    summary: Dict[str, Any],
    analysis: Dict[str, Any],
    phase: str,
    method: str,
) -> bool:
    if (
        summary.get("status") != "verified"
        or summary.get("phase") != phase
        or summary.get("source_analysis_schema")
        != "kedf-pair-ti-production-analysis-v1"
        or analysis.get("schema") != "kedf-pair-ti-production-analysis-v1"
        or analysis.get("status") != "verified"
        or analysis.get("phase") != phase
        or str(analysis.get("target_kedf", "")).lower() != method
    ):
        return False
    selected = float(
        analysis["delta_f_target_minus_pair_simpson_mev_per_atom"]
    )
    if not math.isclose(
        float(summary["integral_consensus_mev_per_atom"]),
        selected,
        rel_tol=1.0e-11,
        abs_tol=1.0e-11,
    ):
        return False
    discard = float(analysis["discard_fraction"])
    matches = [
        report
        for report in summary.get("reports", [])
        if math.isclose(
            float(report["discard_fraction"]), discard, rel_tol=0.0, abs_tol=1.0e-12
        )
    ]
    return len(matches) == 1 and matches[0].get("status") == "verified"


def leg_errors(analysis: Dict[str, Any]) -> Dict[str, float]:
    return {
        "block_standard_error_mev_per_atom": float(
            analysis["block_standard_error_mev_per_atom"]
        ),
        "quadrature_difference_mev_per_atom": float(
            analysis["quadrature_difference_mev_per_atom"]
        ),
        "half_drift_mev_per_atom": float(analysis["half_drift_mev_per_atom"]),
    }


def combine(
    *,
    method: str,
    analytic: Dict[str, Any],
    reference_audit: Dict[str, Any],
    zero_pressure: Dict[str, Any],
    solid_reference: Dict[str, Any],
    liquid_reference: Dict[str, Any],
    solid_reference_summary: Dict[str, Any],
    liquid_reference_summary: Dict[str, Any],
    solid_target: Dict[str, Any],
    liquid_target: Dict[str, Any],
    solid_target_summary: Dict[str, Any],
    liquid_target_summary: Dict[str, Any],
) -> Dict[str, Any]:
    method = method.lower()
    if method not in {"xwm", "lkt"}:
        raise ValueError(f"unsupported KEDF method {method!r}")
    temperature = float(analytic["temperature_k"])
    natoms = int(analytic["natoms"])
    zero_phases = {
        str(row["phase"]): row for row in zero_pressure.get("phase_results", [])
    }
    checks = {
        "analytic_reference_schema": analytic.get("schema")
        == "mpn-analytic-reference-free-energies-v1",
        "reference_audit_verified": reference_audit.get("status")
        == "verified_with_finite_size_sensitivity"
        and all(reference_audit.get("checks", {}).values()),
        "reference_temperature_matches": math.isclose(
            float(reference_audit.get("temperature_k", math.nan)), temperature
        ),
        "reference_atom_count_matches": int(reference_audit.get("natoms", -1))
        == natoms,
        "zero_pressure_verified": zero_pressure.get("status")
        == "all_confirmations_passed",
        "zero_pressure_method_matches": str(
            zero_pressure.get("target_kedf", "")
        ).lower()
        == method,
        "zero_pressure_temperature_matches": math.isclose(
            float(zero_pressure.get("temperature_K", math.nan)), temperature
        ),
        "zero_pressure_phases_verified": set(zero_phases) == {"solid", "liquid"}
        and all(
            zero_phases[phase].get("status") == "confirmation_passed"
            for phase in ("solid", "liquid")
        ),
        "solid_reference_converged": classical_summary_matches(
            solid_reference_summary, solid_reference, "solid"
        ),
        "liquid_reference_converged": classical_summary_matches(
            liquid_reference_summary, liquid_reference, "liquid"
        ),
        "solid_target_converged": target_summary_matches(
            solid_target_summary, solid_target, "solid", method
        ),
        "liquid_target_converged": target_summary_matches(
            liquid_target_summary, liquid_target, "liquid", method
        ),
        "target_atom_counts_match": int(solid_target.get("natoms", -1)) == natoms
        and int(liquid_target.get("natoms", -1)) == natoms,
        "target_temperatures_match": math.isclose(
            float(solid_target.get("target_temperature_K", math.nan)), temperature
        )
        and math.isclose(
            float(liquid_target.get("target_temperature_K", math.nan)), temperature
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"input verification failed: {failed}")

    solid_pair = float(analytic["einstein"]["corrected_free_energy_ev_per_atom"]) + float(
        solid_reference["delta_f_pair_minus_reference_simpson_ev_per_atom"]
    )
    liquid_pair = float(analytic["suf"]["total_free_energy_ev_per_atom"]) + float(
        liquid_reference["delta_f_pair_minus_reference_simpson_ev_per_atom"]
    )
    audited_pair_delta = float(
        reference_audit["absolute_pair_free_energy_ev_per_atom"][
            "liquid_minus_solid_raw"
        ]
    )
    checks["reference_pair_delta_matches_audit"] = math.isclose(
        liquid_pair - solid_pair,
        audited_pair_delta,
        rel_tol=1.0e-11,
        abs_tol=1.0e-11,
    )
    if not checks["reference_pair_delta_matches_audit"]:
        raise RuntimeError(
            "input verification failed: reference_pair_delta_matches_audit"
        )

    solid_target_correction = (
        float(solid_target["delta_f_target_minus_pair_simpson_mev_per_atom"])
        / 1000.0
    )
    liquid_target_correction = (
        float(liquid_target["delta_f_target_minus_pair_simpson_mev_per_atom"])
        / 1000.0
    )
    solid_free_energy = solid_pair + solid_target_correction
    liquid_free_energy = liquid_pair + liquid_target_correction
    delta_g = liquid_free_energy - solid_free_energy

    legs = {
        "solid_analytic_to_pair": leg_errors(solid_reference),
        "liquid_analytic_to_pair": leg_errors(liquid_reference),
        f"solid_pair_to_{method}": leg_errors(solid_target),
        f"liquid_pair_to_{method}": leg_errors(liquid_target),
    }
    statistical = rss(
        *(row["block_standard_error_mev_per_atom"] for row in legs.values())
    )
    quadrature = rss(
        *(row["quadrature_difference_mev_per_atom"] for row in legs.values())
    )
    temporal = rss(*(row["half_drift_mev_per_atom"] for row in legs.values()))
    discard_spreads = {
        "solid_analytic_to_pair": float(
            solid_reference_summary["discard_spread_mev_per_atom"]
        ),
        "liquid_analytic_to_pair": float(
            liquid_reference_summary["discard_spread_mev_per_atom"]
        ),
        f"solid_pair_to_{method}": float(
            solid_target_summary["integral_discard_spread_mev_per_atom"]
        ),
        f"liquid_pair_to_{method}": float(
            liquid_target_summary["integral_discard_spread_mev_per_atom"]
        ),
    }
    discard_rss = rss(*discard_spreads.values())
    ti_conservative = rss(statistical, quadrature, temporal, discard_rss)
    finite_size = finite_size_allowance_mev_per_atom(reference_audit)

    return {
        "schema": "kedf-melting-free-energy-anchor-v1",
        "status": "anchor_temperature_free_energy_verified",
        "target_kedf": method,
        "temperature_k": temperature,
        "natoms": natoms,
        "checks": checks,
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "root": "delta_g_equals_zero",
        },
        "free_energy_ev_per_atom": {
            "solid_pair": solid_pair,
            "liquid_pair": liquid_pair,
            "solid": solid_free_energy,
            "liquid": liquid_free_energy,
            "liquid_minus_solid": delta_g,
        },
        "free_energy_mev_per_atom": {
            "liquid_minus_solid": 1000.0 * delta_g,
        },
        "target_correction_mev_per_atom": {
            "solid_pair_to_target": 1000.0 * solid_target_correction,
            "liquid_pair_to_target": 1000.0 * liquid_target_correction,
        },
        "uncertainty_budget_mev_per_atom": {
            "legs": legs,
            "discard_spreads": discard_spreads,
            "statistical_rss": statistical,
            "quadrature_rss": quadrature,
            "temporal_drift_rss": temporal,
            "discard_spread_rss": discard_rss,
            "ti_combined_conservative": ti_conservative,
            "finite_size_systematic_allowance": finite_size,
            "combined_conservative": ti_conservative + finite_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a verified XWM or LKT 900 K free-energy anchor"
    )
    parser.add_argument("--method", choices=("xwm", "lkt"), required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--zero-pressure", type=Path, required=True)
    parser.add_argument("--solid-reference", type=Path, required=True)
    parser.add_argument("--liquid-reference", type=Path, required=True)
    parser.add_argument("--solid-reference-summary", type=Path, required=True)
    parser.add_argument("--liquid-reference-summary", type=Path, required=True)
    parser.add_argument("--solid-target", type=Path, required=True)
    parser.add_argument("--liquid-target", type=Path, required=True)
    parser.add_argument("--solid-target-summary", type=Path, required=True)
    parser.add_argument("--liquid-target-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_paths = {
        "analytic": args.analytic.resolve(),
        "reference_audit": args.reference_audit.resolve(),
        "zero_pressure": args.zero_pressure.resolve(),
        "solid_reference": args.solid_reference.resolve(),
        "liquid_reference": args.liquid_reference.resolve(),
        "solid_reference_summary": args.solid_reference_summary.resolve(),
        "liquid_reference_summary": args.liquid_reference_summary.resolve(),
        "solid_target": args.solid_target.resolve(),
        "liquid_target": args.liquid_target.resolve(),
        "solid_target_summary": args.solid_target_summary.resolve(),
        "liquid_target_summary": args.liquid_target_summary.resolve(),
    }
    result = combine(
        method=args.method,
        **{name: load_json(path) for name, path in source_paths.items()},
    )
    result["provenance"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in source_paths.items()
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
