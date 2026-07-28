import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("scipy")

from scripts.qm9_complete_total_geometry_shared_capacity import (
    FeatureDesign,
    ParentState,
    _fit_shared_stage,
    constrained_lsqr,
)


def test_constrained_lsqr_recovers_shared_reachable_solution():
    rng = np.random.default_rng(23)
    design = rng.normal(size=(30, 12))
    constraints = rng.normal(size=(3, 12))
    source = rng.normal(size=12)
    target = design @ source
    constraint_target = constraints @ source
    coefficients, diagnostics = constrained_lsqr(
        design,
        target,
        constraints,
        constraint_target,
        tolerance=1.0e-11,
        max_iterations=500,
        svd_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(design @ coefficients, target, atol=1.0e-8)
    np.testing.assert_allclose(
        constraints @ coefficients, constraint_target, atol=1.0e-8
    )
    assert diagnostics["constraint_rank"] == 3


def test_constrained_lsqr_preserves_exact_anchors_while_fitting_hessian():
    rng = np.random.default_rng(29)
    design = rng.normal(size=(40, 15))
    constraints = rng.normal(size=(5, 15))
    anchor_source = rng.normal(size=15)
    constraint_target = constraints @ anchor_source
    target = rng.normal(size=40)
    coefficients, diagnostics = constrained_lsqr(
        design,
        target,
        constraints,
        constraint_target,
        tolerance=1.0e-11,
        max_iterations=500,
        svd_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(
        constraints @ coefficients, constraint_target, atol=1.0e-8
    )
    assert diagnostics["constraint_residual_max_abs"] < 1.0e-8


def test_shared_stage_maps_partial_feature_overlap_and_preserves_anchors(tmp_path):
    coefficients = {(1,): 0.3, (2,): -0.2, (3,): 0.4}
    parents = []
    designs = []
    for molecule_id, keys, seed in (
        ("first", [(1,), (2,)], 31),
        ("second", [(2,), (3,)], 37),
    ):
        rng = np.random.default_rng(seed)
        energy_design = rng.normal(size=2)
        gradient_design = rng.normal(size=(2, 2))
        hessian_design = rng.normal(size=(4, 2))
        local = np.asarray([coefficients[key] for key in keys])
        energy = -1.0
        force = np.zeros(2)
        hessian = np.zeros((2, 2))
        parents.append(
            ParentState(
                molecule_id=molecule_id,
                atomic_numbers=np.ones(1, dtype=np.int64),
                positions_bohr=np.zeros((1, 3)),
                pbe_energy=energy + float(energy_design @ local),
                pbe_force=force - gradient_design @ local,
                pbe_hessian=(hessian_design @ local).reshape(2, 2),
                energy=energy,
                force=force,
                hessian=hessian,
            )
        )
        designs.append(
            FeatureDesign(
                molecule_id=molecule_id,
                keys=keys,
                energy=energy_design,
                gradient=gradient_design,
                hessian=hessian_design,
                metadata={},
            )
        )
    args = SimpleNamespace(
        absolute_hessian_scale=0.1,
        relative_loss_fraction=0.5,
        lsqr_tolerance=1.0e-11,
        lsqr_max_iterations=500,
        svd_tolerance=1.0e-12,
    )
    fitted, keys, summary = _fit_shared_stage(
        parents, designs, "three_body", tmp_path, args
    )
    recovered = dict(zip(keys, fitted, strict=True))
    for key, expected in coefficients.items():
        assert recovered[key] == pytest.approx(expected, abs=1.0e-8)
    assert summary["all_below_0p05"] is True
    assert summary["solver"]["constraint_residual_max_abs"] < 1.0e-8
    for parent in parents:
        np.testing.assert_allclose(parent.hessian, parent.pbe_hessian, atol=1.0e-8)
        np.testing.assert_allclose(parent.force, parent.pbe_force, atol=1.0e-8)
        assert parent.energy == pytest.approx(parent.pbe_energy, abs=1.0e-8)
