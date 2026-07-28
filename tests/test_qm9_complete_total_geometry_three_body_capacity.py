import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from scripts.qm9_complete_total_geometry_three_body_capacity import (  # noqa: E402
    build_triplet_groups,
    constrained_row_space_fit,
    hessian_metrics,
    make_three_body_feature_function,
)
from mldft.ml.models.components.three_body_geometry_residual import (  # noqa: E402
    ThreeBodyGeometryResidual,
)


def test_three_body_features_are_translation_and_rotation_invariant():
    atomic_numbers = np.array([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.3, 1.2, 0.0]],
        dtype=torch.float64,
    )
    groups = build_triplet_groups(atomic_numbers)
    feature_function, keys = make_three_body_feature_function(
        groups, np.array([1.0, 2.0]), sigma_bohr=0.5, angular_order=2
    )
    angle = torch.tensor(0.4, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), torch.zeros_like(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle), torch.zeros_like(angle))),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    translated_rotated = positions @ rotation.T + torch.tensor(
        [0.3, -0.8, 1.1], dtype=torch.float64
    )
    assert keys
    torch.testing.assert_close(
        feature_function(positions),
        feature_function(translated_rotated),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_constrained_fit_preserves_constraints_and_fits_reachable_target():
    rng = np.random.default_rng(11)
    constraints = rng.normal(size=(3, 12))
    design = rng.normal(size=(5, 12))
    _, _, constraint_vt = np.linalg.svd(constraints, full_matrices=False)
    row_basis = constraint_vt[:3]
    coefficient_source = rng.normal(size=12)
    coefficient_source -= row_basis.T @ (row_basis @ coefficient_source)
    target = design @ coefficient_source
    coefficients, diagnostics = constrained_row_space_fit(
        design, target, constraints, relative_svd_tolerance=1.0e-12
    )
    np.testing.assert_allclose(constraints @ coefficients, 0.0, atol=1.0e-11)
    np.testing.assert_allclose(design @ coefficients, target, atol=1.0e-11)
    assert diagnostics["reduced_design_rank"] == 5


def test_constrained_fit_supports_nonzero_constraint_target():
    rng = np.random.default_rng(19)
    constraints = rng.normal(size=(3, 12))
    design = rng.normal(size=(5, 12))
    source = rng.normal(size=12)
    constraint_target = constraints @ source
    target = design @ source
    coefficients, _ = constrained_row_space_fit(
        design,
        target,
        constraints,
        relative_svd_tolerance=1.0e-12,
        constraint_target=constraint_target,
    )
    np.testing.assert_allclose(
        constraints @ coefficients, constraint_target, atol=1.0e-10
    )
    np.testing.assert_allclose(design @ coefficients, target, atol=1.0e-10)


def test_trainable_residual_loads_sparse_scalar_coefficients_exactly():
    atomic_numbers = np.array([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.3, 1.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    groups = build_triplet_groups(atomic_numbers)
    direct_function, direct_keys = make_three_body_feature_function(
        groups, np.linspace(0.5, 8.0, 2), sigma_bohr=0.5, angular_order=1
    )
    direct_coefficients = np.linspace(-0.2, 0.3, len(direct_keys))
    residual = ThreeBodyGeometryResidual(center_count=2, angular_order=1)
    residual.load_sparse_coefficients(
        np.asarray(direct_keys, dtype=np.int64), direct_coefficients
    )
    sample = SimpleNamespace(
        pos=positions,
        atomic_numbers=torch.as_tensor(atomic_numbers),
        batch=torch.zeros(atomic_numbers.size, dtype=torch.long),
    )
    expected = torch.dot(
        direct_function(positions), torch.as_tensor(direct_coefficients)
    )
    actual = residual.forward_energy(sample).sum()
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    hessian = torch.func.hessian(lambda pos: residual.forward_energy(
        SimpleNamespace(
            pos=pos,
            atomic_numbers=sample.atomic_numbers,
            batch=sample.batch,
        )
    ).sum())(positions)
    assert torch.isfinite(hessian).all()


def test_hessian_metrics_separate_accuracy_and_symmetry():
    reference = np.eye(2)
    prediction = np.array([[1.0, 0.2], [0.0, 0.8]])
    metrics = hessian_metrics(prediction, reference)
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.02))
    assert metrics["relative_frobenius"] == pytest.approx(0.2)
    assert metrics["symmetry_max_abs"] == pytest.approx(0.2)
    assert metrics["antisymmetric_over_symmetric_frobenius"] > 0
