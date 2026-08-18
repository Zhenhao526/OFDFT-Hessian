#!/usr/bin/env python3
"""Combine a KEDF free-energy anchor with multiple classical liquid TI legs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.combine_kedf_free_energy_anchor import (
    finite_size_allowance_mev_per_atom,
    leg_errors,
    load_json,
    rss,
    sha256,
    target_summary_matches,
)


def _declared_phase(document: dict[str, Any]) -> str | None:
    value = document.get("expected_phase", document.get("phase"))
    return None if value is None else str(value)


def classical_leg_matches(
    analysis: dict[str, Any],
    summary: dict[str, Any],
    phase: str,
    *,
    temperature_k: float,
    natoms: int,
) -> bool:
    if (
        analysis.get("schema") != "mpn-classical-reference-pair-ti-analysis-v1"
        or analysis.get("status") != "verified"
        or summary.get("status") != "verified"
    ):
        return False
    phases = {
        value
        for value in (_declared_phase(analysis), _declared_phase(summary))
        if value is not None
    }
    if not phases or phases != {phase}:
        return False
    if int(analysis.get("natoms", -1)) != natoms or not math.isclose(
        float(analysis.get("target_temperature_k", math.nan)), temperature_k
    ):
        return False
    if not all(analysis.get("checks", {}).values()) or not all(
        summary.get("checks", {}).values()
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


def _classical_value_ev_per_atom(analysis: dict[str, Any]) -> float:
    return float(analysis["delta_f_pair_minus_reference_simpson_ev_per_atom"])


def _discard_spread(summary: dict[str, Any]) -> float:
    return float(summary["discard_spread_mev_per_atom"])


def _user_gate_override_matches(
    document: dict[str, Any], *, source_status: str, scope: str
) -> bool:
    override = document.get("gate_override", {})
    return (
        override.get("user_authorized") is True
        and override.get("source_status") == source_status
        and override.get("scope") == scope
    )


def combine_multileg(
    *,
    method: str,
    analytic: dict[str, Any],
    reference_audit: dict[str, Any],
    zero_pressure: dict[str, Any],
    solid_reference_label: str,
    solid_reference: dict[str, Any],
    solid_reference_summary: dict[str, Any],
    liquid_reference_legs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    solid_target: dict[str, Any],
    liquid_target: dict[str, Any],
    solid_target_summary: dict[str, Any],
    liquid_target_summary: dict[str, Any],
) -> dict[str, Any]:
    method = method.lower()
    if method not in {"wt", "xwm", "lkt"}:
        raise ValueError(f"unsupported KEDF method {method!r}")
    if not liquid_reference_legs:
        raise ValueError("at least one liquid classical reference leg is required")
    labels = [label for label, _, _ in liquid_reference_legs]
    if len(labels) != len(set(labels)):
        raise ValueError("liquid classical reference leg labels must be unique")

    temperature = float(analytic["temperature_k"])
    natoms = int(analytic["natoms"])
    zero_phases = {
        str(row["phase"]): row for row in zero_pressure.get("phase_results", [])
    }
    reference_audit_verified = reference_audit.get("status") == (
        "verified_with_finite_size_sensitivity"
    ) and all(reference_audit.get("checks", {}).values())
    reference_audit_overridden = reference_audit.get("status") == (
        "computed_by_user_authorized_gate_override"
    ) and _user_gate_override_matches(
        reference_audit,
        source_status="gate_failed",
        scope="reference_anchor_half_drift_gate_only",
    )
    checks = {
        "analytic_reference_schema": analytic.get("schema")
        == "mpn-analytic-reference-free-energies-v1",
        "reference_audit_verified": reference_audit_verified
        or reference_audit_overridden,
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
        "solid_reference_converged": classical_leg_matches(
            solid_reference,
            solid_reference_summary,
            "solid",
            temperature_k=temperature,
            natoms=natoms,
        ),
        "liquid_reference_legs_converged": all(
            classical_leg_matches(
                analysis,
                summary,
                "liquid",
                temperature_k=temperature,
                natoms=natoms,
            )
            for _, analysis, summary in liquid_reference_legs
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

    solid_classical = _classical_value_ev_per_atom(solid_reference)
    liquid_classical = [
        (label, _classical_value_ev_per_atom(analysis))
        for label, analysis, _ in liquid_reference_legs
    ]
    solid_target_correction = (
        float(solid_target["delta_f_target_minus_pair_simpson_mev_per_atom"])
        / 1000.0
    )
    liquid_target_correction = (
        float(liquid_target["delta_f_target_minus_pair_simpson_mev_per_atom"])
        / 1000.0
    )
    solid_analytic = float(
        analytic["einstein"]["corrected_free_energy_ev_per_atom"]
    )
    liquid_analytic = float(analytic["suf"]["total_free_energy_ev_per_atom"])
    solid_free_energy = solid_analytic + solid_classical + solid_target_correction
    liquid_free_energy = (
        liquid_analytic
        + sum(value for _, value in liquid_classical)
        + liquid_target_correction
    )
    delta_g = liquid_free_energy - solid_free_energy

    leg_documents = [
        (f"solid_{solid_reference_label}", solid_reference),
        *[
            (f"liquid_{label}", analysis)
            for label, analysis, _ in liquid_reference_legs
        ],
        (f"solid_reference_to_{method}", solid_target),
        (f"liquid_reference_to_{method}", liquid_target),
    ]
    legs = {label: leg_errors(analysis) for label, analysis in leg_documents}
    discard_spreads = {
        f"solid_{solid_reference_label}": _discard_spread(
            solid_reference_summary
        ),
        **{
            f"liquid_{label}": _discard_spread(summary)
            for label, _, summary in liquid_reference_legs
        },
        f"solid_reference_to_{method}": float(
            solid_target_summary["integral_discard_spread_mev_per_atom"]
        ),
        f"liquid_reference_to_{method}": float(
            liquid_target_summary["integral_discard_spread_mev_per_atom"]
        ),
    }
    statistical = rss(
        *(row["block_standard_error_mev_per_atom"] for row in legs.values())
    )
    quadrature = rss(
        *(row["quadrature_difference_mev_per_atom"] for row in legs.values())
    )
    temporal = rss(*(row["half_drift_mev_per_atom"] for row in legs.values()))
    discard_rss = rss(*discard_spreads.values())
    ti_conservative = rss(statistical, quadrature, temporal, discard_rss)
    finite_size = finite_size_allowance_mev_per_atom(reference_audit)

    gate_overrides = {}
    if reference_audit_overridden:
        gate_overrides["reference_audit"] = reference_audit["gate_override"]
    for label, summary in (
        ("solid_target", solid_target_summary),
        ("liquid_target", liquid_target_summary),
    ):
        if summary.get("status") == "verified_by_user_authorized_gate_override":
            gate_overrides[label] = summary["gate_override"]

    return {
        "schema": "kedf-melting-multileg-free-energy-anchor-v1",
        "status": (
            "anchor_temperature_free_energy_computed_with_user_authorized_gate_overrides"
            if gate_overrides
            else "anchor_temperature_free_energy_verified"
        ),
        "gate_overrides": gate_overrides,
        "target_kedf": method,
        "temperature_k": temperature,
        "natoms": natoms,
        "checks": checks,
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "root": "delta_g_equals_zero",
        },
        "free_energy_ev_per_atom": {
            "solid": solid_free_energy,
            "liquid": liquid_free_energy,
            "liquid_minus_solid": delta_g,
        },
        "free_energy_mev_per_atom": {
            "liquid_minus_solid": 1000.0 * delta_g,
        },
        "free_energy_path_ev_per_atom": {
            "solid": {
                "analytic_einstein": solid_analytic,
                solid_reference_label: solid_classical,
                f"reference_to_{method}": solid_target_correction,
            },
            "liquid": {
                "analytic_suf": liquid_analytic,
                "classical_legs": {
                    label: value for label, value in liquid_classical
                },
                f"reference_to_{method}": liquid_target_correction,
            },
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


def parse_liquid_leg(values: list[str]) -> tuple[str, Path, Path]:
    label, analysis, summary = values
    if not label:
        raise argparse.ArgumentTypeError("liquid leg label cannot be empty")
    return label, Path(analysis).resolve(), Path(summary).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a verified multileg WT, XWM, or LKT 900 K free-energy anchor"
    )
    parser.add_argument("--method", choices=("wt", "xwm", "lkt"), required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--zero-pressure", type=Path, required=True)
    parser.add_argument("--solid-reference-label", required=True)
    parser.add_argument("--solid-reference", type=Path, required=True)
    parser.add_argument("--solid-reference-summary", type=Path, required=True)
    parser.add_argument(
        "--liquid-reference-leg",
        nargs=3,
        action="append",
        metavar=("LABEL", "ANALYSIS", "SUMMARY"),
        required=True,
    )
    parser.add_argument("--solid-target", type=Path, required=True)
    parser.add_argument("--liquid-target", type=Path, required=True)
    parser.add_argument("--solid-target-summary", type=Path, required=True)
    parser.add_argument("--liquid-target-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fixed_paths = {
        "analytic": args.analytic.resolve(),
        "reference_audit": args.reference_audit.resolve(),
        "zero_pressure": args.zero_pressure.resolve(),
        "solid_reference": args.solid_reference.resolve(),
        "solid_reference_summary": args.solid_reference_summary.resolve(),
        "solid_target": args.solid_target.resolve(),
        "liquid_target": args.liquid_target.resolve(),
        "solid_target_summary": args.solid_target_summary.resolve(),
        "liquid_target_summary": args.liquid_target_summary.resolve(),
    }
    liquid_paths = [parse_liquid_leg(values) for values in args.liquid_reference_leg]
    result = combine_multileg(
        method=args.method,
        analytic=load_json(fixed_paths["analytic"]),
        reference_audit=load_json(fixed_paths["reference_audit"]),
        zero_pressure=load_json(fixed_paths["zero_pressure"]),
        solid_reference_label=args.solid_reference_label,
        solid_reference=load_json(fixed_paths["solid_reference"]),
        solid_reference_summary=load_json(fixed_paths["solid_reference_summary"]),
        liquid_reference_legs=[
            (label, load_json(analysis), load_json(summary))
            for label, analysis, summary in liquid_paths
        ],
        solid_target=load_json(fixed_paths["solid_target"]),
        liquid_target=load_json(fixed_paths["liquid_target"]),
        solid_target_summary=load_json(fixed_paths["solid_target_summary"]),
        liquid_target_summary=load_json(fixed_paths["liquid_target_summary"]),
    )
    result["provenance"] = {
        **{
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in fixed_paths.items()
        },
        "liquid_reference_legs": [
            {
                "label": label,
                "analysis": {
                    "path": str(analysis),
                    "sha256": sha256(analysis),
                },
                "summary": {
                    "path": str(summary),
                    "sha256": sha256(summary),
                },
            }
            for label, analysis, summary in liquid_paths
        ],
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
