from __future__ import annotations

import scripts.analyze_mg_phase_md as module


def phase(final_msd: float, late_slope: float, diffusion_slope: float) -> dict:
    return {
        "trajectory": {
            "minimum_nearest_neighbor_A": 2.4,
            "non_affine_MSD_A2": final_msd,
            "late_MSD_slope_A2_per_step": late_slope,
            "diffusion_MSD_slope_A2_per_step": diffusion_slope,
        }
    }


def test_hcp_solid_gate_does_not_require_fcc_csp() -> None:
    checks = module.hcp_phase_checks(phase(0.2, 0.0002, 0.0003), "solid")
    assert all(checks.values())


def test_hcp_thermalized_solid_uses_late_plateau() -> None:
    checks = module.hcp_phase_checks(
        phase(0.52, 0.0002, 0.004),
        "solid",
        thermalized_initial=True,
    )
    assert all(checks.values())
    assert "diffusion_MSD_slope_lt_0_001_A2_per_step" not in checks


def test_hcp_thermalized_solid_rejects_liquid_diffusion() -> None:
    checks = module.hcp_phase_checks(
        phase(1.02, 0.0028, 0.0045),
        "solid",
        thermalized_initial=True,
    )
    assert not all(checks.values())


def test_hcp_liquid_gate_uses_diffusion() -> None:
    checks = module.hcp_phase_checks(phase(2.0, 0.003, 0.004), "liquid")
    assert all(checks.values())
