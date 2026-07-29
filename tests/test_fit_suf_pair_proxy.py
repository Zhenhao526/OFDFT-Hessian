from __future__ import annotations

import pytest

pytest.importorskip("torch")

from scripts.fit_suf_pair_proxy import fit_proxy


def test_default_proxy_is_accurate_and_repulsive():
    result = fit_proxy(
        p=50,
        suf_sigma_angstrom=1.28,
        temperature_k=900.0,
        basis_count=81,
        basis_min_angstrom=1.4,
        basis_max_angstrom=6.3,
        basis_width_angstrom=0.10,
        cutoff_angstrom=6.5,
        core_amplitude_ev=100.0,
        core_cutoff_angstrom=2.0,
        core_power=4,
        samples=2000,
        ridge=1.0e-10,
    )

    assert result["reference_gate_passed"] is True
    diagnostics = result["radial_fit_diagnostics"]
    assert diagnostics["energy_max_abs_ev_fit_region"] < 3.0e-5
    assert (
        diagnostics["derivative_max_abs_ev_per_angstrom_fit_region"]
        < 2.0e-3
    )


def test_hard_core_can_be_fit_above_two_angstrom():
    result = fit_proxy(
        p=50,
        suf_sigma_angstrom=1.28,
        temperature_k=900.0,
        basis_count=77,
        basis_min_angstrom=1.8,
        basis_max_angstrom=6.3,
        basis_width_angstrom=0.10,
        cutoff_angstrom=6.5,
        core_amplitude_ev=500.0,
        core_cutoff_angstrom=2.2,
        core_power=4,
        samples=2400,
        ridge=1.0e-10,
        sample_min_angstrom=2.0,
    )

    assert result["reference_gate_passed"] is True
    assert result["fit_parameters"]["sample_min_angstrom"] == 2.0
    short_range = result["short_range_diagnostic"]
    assert short_range["u_1p5_ev"] > 4.0
    assert short_range["du_dr_1p5_ev_per_angstrom"] < -10.0
