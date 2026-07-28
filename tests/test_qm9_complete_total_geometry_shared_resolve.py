import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.qm9_complete_total_geometry_shared_resolve import (
    solve_preconditioned_lsqr,
)


def test_preconditioned_lsqr_handles_bad_column_scales_and_soft_anchors():
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(60, 10))
    design = raw * np.logspace(-6, 6, 10)[None, :]
    anchor = rng.normal(size=(4, 10)) * np.logspace(-6, 6, 10)[None, :]
    source = rng.normal(size=10) / np.logspace(-6, 6, 10)
    target = design @ source
    anchor_target = anchor @ source
    coefficients, diagnostics = solve_preconditioned_lsqr(
        design,
        target,
        anchor_design=anchor,
        anchor_target=anchor_target,
        anchor_weights=np.ones(4),
        tolerance=1.0e-12,
        max_iterations=500,
        condition_limit=1.0e14,
        chunk_rows=13,
    )
    np.testing.assert_allclose(design @ coefficients, target, atol=1.0e-8)
    np.testing.assert_allclose(anchor @ coefficients, anchor_target, atol=1.0e-8)
    assert diagnostics["relative_objective_residual"] < 1.0e-9


def test_preconditioned_lsqr_warm_start_uses_coefficient_coordinates():
    rng = np.random.default_rng(43)
    design = rng.normal(size=(40, 8)) * np.logspace(-4, 4, 8)[None, :]
    source = rng.normal(size=8) / np.logspace(-4, 4, 8)
    target = design @ source
    coefficients, diagnostics = solve_preconditioned_lsqr(
        design,
        target,
        anchor_design=None,
        anchor_target=None,
        anchor_weights=None,
        tolerance=1.0e-12,
        max_iterations=20,
        condition_limit=1.0e14,
        chunk_rows=11,
        initial_coefficients=source,
    )
    np.testing.assert_allclose(coefficients, source, atol=1.0e-10)
    assert diagnostics["warm_started"] is True
    assert diagnostics["relative_objective_residual"] < 1.0e-12
