from __future__ import annotations

import pytest

from scripts.combine_kedf_multileg_free_energy_anchor import combine_multileg


def classical(phase: str, value: float) -> tuple[dict, dict]:
    analysis = {
        "schema": "mpn-classical-reference-pair-ti-analysis-v1",
        "status": "verified",
        "expected_phase": phase,
        "natoms": 108,
        "target_temperature_k": 900.0,
        "checks": {"stable": True},
        "delta_f_pair_minus_reference_simpson_ev_per_atom": value / 1000.0,
        "delta_f_pair_minus_reference_simpson_mev_per_atom": value,
        "block_standard_error_mev_per_atom": 0.2,
        "quadrature_difference_mev_per_atom": 0.3,
        "half_drift_mev_per_atom": 0.4,
    }
    summary = {
        "status": "verified",
        "phase": phase,
        "checks": {"three_reports": True},
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
    liquid_proxy, liquid_proxy_summary = classical("liquid", 6.0)
    liquid_bridge, liquid_bridge_summary = classical("liquid", -2.0)
    solid_target, solid_target_summary = target(method, "solid", 2.0)
    liquid_target, liquid_target_summary = target(method, "liquid", 5.0)
    return {
        "method": method,
        "analytic": {
            "schema": "mpn-analytic-reference-free-energies-v1",
            "temperature_k": 900.0,
            "natoms": 108,
            "einstein": {"corrected_free_energy_ev_per_atom": -1.000},
            "suf": {"total_free_energy_ev_per_atom": -0.990},
        },
        "reference_audit": {
            "status": "verified_with_finite_size_sensitivity",
            "temperature_k": 900.0,
            "natoms": 108,
            "checks": {"reference": True},
            "finite_size_sensitivity": {
                "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom": 2.0,
                "polson_kbt_ln_n_over_n_mev_per_atom": 3.0,
            },
        },
        "zero_pressure": {
            "status": "all_confirmations_passed",
            "target_kedf": method,
            "temperature_K": 900.0,
            "phase_results": [
                {"phase": "solid", "status": "confirmation_passed"},
                {"phase": "liquid", "status": "confirmation_passed"},
            ],
        },
        "solid_reference_label": "einstein_to_pair",
        "solid_reference": solid_reference,
        "solid_reference_summary": solid_reference_summary,
        "liquid_reference_legs": [
            ("suf_to_proxy", liquid_proxy, liquid_proxy_summary),
            ("proxy_to_bridge", liquid_bridge, liquid_bridge_summary),
        ],
        "solid_target": solid_target,
        "liquid_target": liquid_target,
        "solid_target_summary": solid_target_summary,
        "liquid_target_summary": liquid_target_summary,
    }


def test_combines_multileg_xwm_anchor() -> None:
    result = combine_multileg(**inputs())

    assert result["status"] == "anchor_temperature_free_energy_verified"
    assert result["free_energy_mev_per_atom"]["liquid_minus_solid"] == pytest.approx(
        21.0
    )
    assert result["free_energy_path_ev_per_atom"]["liquid"]["classical_legs"] == {
        "suf_to_proxy": pytest.approx(0.006),
        "proxy_to_bridge": pytest.approx(-0.002),
    }
    assert result["uncertainty_budget_mev_per_atom"][
        "finite_size_systematic_allowance"
    ] == 3.0


def test_rejects_unverified_middle_leg() -> None:
    documents = inputs()
    documents["liquid_reference_legs"][1][1]["status"] = "failed"

    with pytest.raises(RuntimeError, match="liquid_reference_legs_converged"):
        combine_multileg(**documents)


def test_rejects_duplicate_liquid_leg_labels() -> None:
    documents = inputs()
    label, analysis, summary = documents["liquid_reference_legs"][0]
    documents["liquid_reference_legs"][1] = (label, analysis, summary)

    with pytest.raises(ValueError, match="labels must be unique"):
        combine_multileg(**documents)


def test_rejects_phase_mismatch() -> None:
    documents = inputs()
    documents["liquid_reference_legs"][0][2]["phase"] = "solid"

    with pytest.raises(RuntimeError, match="liquid_reference_legs_converged"):
        combine_multileg(**documents)
