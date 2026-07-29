from __future__ import annotations

import pytest

from scripts.combine_kedf_free_energy_anchor import combine


def classical(phase: str, value: float) -> tuple[dict, dict]:
    analysis = {
        "schema": "mpn-classical-reference-pair-ti-analysis-v1",
        "status": "verified",
        "expected_phase": phase,
        "delta_f_pair_minus_reference_simpson_ev_per_atom": value / 1000.0,
        "delta_f_pair_minus_reference_simpson_mev_per_atom": value,
        "block_standard_error_mev_per_atom": 0.2,
        "quadrature_difference_mev_per_atom": 0.3,
        "half_drift_mev_per_atom": 0.4,
    }
    summary = {
        "status": "verified",
        "expected_phase": phase,
        "discard_integrals_mev_per_atom": [value - 0.2, value, value + 0.2],
        "discard_spread_mev_per_atom": 0.4,
    }
    return analysis, summary


def target(method: str, phase: str, value: float) -> tuple[dict, dict]:
    analysis = {
        "schema": "kedf-pair-ti-production-analysis-v1",
        "status": "verified",
        "target_kedf": method,
        "phase": phase,
        "natoms": 108,
        "target_temperature_K": 900.0,
        "discard_fraction": 0.5,
        "delta_f_target_minus_pair_simpson_mev_per_atom": value,
        "block_standard_error_mev_per_atom": 0.5,
        "quadrature_difference_mev_per_atom": 0.6,
        "half_drift_mev_per_atom": 0.7,
    }
    summary = {
        "status": "verified",
        "source_analysis_schema": "kedf-pair-ti-production-analysis-v1",
        "phase": phase,
        "integral_consensus_mev_per_atom": value,
        "integral_discard_spread_mev_per_atom": 0.8,
        "reports": [{"discard_fraction": 0.5, "status": "verified"}],
    }
    return analysis, summary


def inputs(method: str = "xwm") -> dict:
    solid_reference, solid_reference_summary = classical("solid", -4.0)
    liquid_reference, liquid_reference_summary = classical("liquid", 6.0)
    solid_target, solid_target_summary = target(method, "solid", 2.0)
    liquid_target, liquid_target_summary = target(method, "liquid", 5.0)
    analytic = {
        "schema": "mpn-analytic-reference-free-energies-v1",
        "temperature_k": 900.0,
        "natoms": 108,
        "einstein": {"corrected_free_energy_ev_per_atom": -1.000},
        "suf": {"total_free_energy_ev_per_atom": -0.990},
    }
    solid_pair = -1.004
    liquid_pair = -0.984
    audit = {
        "status": "verified_with_finite_size_sensitivity",
        "temperature_k": 900.0,
        "natoms": 108,
        "checks": {"reference": True},
        "absolute_pair_free_energy_ev_per_atom": {
            "liquid_minus_solid_raw": liquid_pair - solid_pair
        },
        "finite_size_sensitivity": {
            "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom": 2.0,
            "polson_kbt_ln_n_over_n_mev_per_atom": 3.0,
        },
    }
    zero_pressure = {
        "status": "all_confirmations_passed",
        "target_kedf": method,
        "temperature_K": 900.0,
        "phase_results": [
            {"phase": "solid", "status": "confirmation_passed"},
            {"phase": "liquid", "status": "confirmation_passed"},
        ],
    }
    return {
        "method": method,
        "analytic": analytic,
        "reference_audit": audit,
        "zero_pressure": zero_pressure,
        "solid_reference": solid_reference,
        "liquid_reference": liquid_reference,
        "solid_reference_summary": solid_reference_summary,
        "liquid_reference_summary": liquid_reference_summary,
        "solid_target": solid_target,
        "liquid_target": liquid_target,
        "solid_target_summary": solid_target_summary,
        "liquid_target_summary": liquid_target_summary,
    }


@pytest.mark.parametrize("method", ["xwm", "lkt"])
def test_combines_independent_kedf_anchor(method: str) -> None:
    result = combine(**inputs(method))

    assert result["status"] == "anchor_temperature_free_energy_verified"
    assert result["target_kedf"] == method
    assert result["free_energy_mev_per_atom"]["liquid_minus_solid"] == pytest.approx(
        23.0
    )
    assert result["target_correction_mev_per_atom"] == {
        "solid_pair_to_target": 2.0,
        "liquid_pair_to_target": 5.0,
    }
    assert result["uncertainty_budget_mev_per_atom"][
        "finite_size_systematic_allowance"
    ] == 3.0


def test_rejects_cross_method_target_ti() -> None:
    documents = inputs("xwm")
    documents["liquid_target"]["target_kedf"] = "lkt"

    with pytest.raises(RuntimeError, match="liquid_target_converged"):
        combine(**documents)


def test_rejects_unverified_zero_pressure_phase() -> None:
    documents = inputs("lkt")
    documents["zero_pressure"]["phase_results"][1]["status"] = "failed"

    with pytest.raises(RuntimeError, match="zero_pressure_phases_verified"):
        combine(**documents)


def test_rejects_reference_audit_mismatch() -> None:
    documents = inputs("xwm")
    documents["reference_audit"]["absolute_pair_free_energy_ev_per_atom"][
        "liquid_minus_solid_raw"
    ] += 0.001

    with pytest.raises(RuntimeError, match="reference_pair_delta_matches_audit"):
        combine(**documents)
