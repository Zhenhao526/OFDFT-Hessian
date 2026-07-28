import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.qm9_complete_total_geometry_four_body_capacity import (  # noqa: E402
    build_bonded_chain_groups,
    make_four_body_feature_function,
)


def test_four_body_features_are_rigid_motion_and_parity_invariant():
    atomic_numbers = np.array([1, 6, 6, 1])
    positions = np.array(
        [[-1.0, 0.8, 0.4], [0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.1, 0.9, -0.6]],
        dtype=np.float64,
    )
    groups, bonds = build_bonded_chain_groups(
        atomic_numbers, positions, bond_scale=1.35
    )
    assert bonds and groups
    feature_function, keys = make_four_body_feature_function(
        groups, np.array([1.0, 2.5, 4.0]), sigma_bohr=0.5, torsion_order=3
    )
    positions_tensor = torch.as_tensor(positions)
    reflected = positions_tensor.clone()
    reflected[:, 2] *= -1
    translated = reflected + torch.tensor([0.4, -0.2, 0.8])
    assert keys
    torch.testing.assert_close(
        feature_function(positions_tensor),
        feature_function(translated),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_four_body_scalar_has_finite_symmetric_hessian():
    atomic_numbers = np.array([1, 6, 6, 1])
    positions = np.array(
        [[-1.0, 0.8, 0.4], [0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.1, 0.9, -0.6]],
        dtype=np.float64,
    )
    groups, _ = build_bonded_chain_groups(atomic_numbers, positions, bond_scale=1.35)
    feature_function, keys = make_four_body_feature_function(
        groups, np.array([1.0, 2.5, 4.0]), sigma_bohr=0.5, torsion_order=2
    )
    coefficients = torch.linspace(-0.1, 0.1, len(keys), dtype=torch.float64)
    positions_tensor = torch.tensor(positions, dtype=torch.float64)
    hessian = torch.func.hessian(
        lambda candidate: torch.dot(feature_function(candidate), coefficients)
    )(positions_tensor).reshape(12, 12)
    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(hessian, hessian.T, atol=1.0e-11, rtol=1.0e-11)


def test_collinear_four_body_chain_has_finite_second_derivatives():
    atomic_numbers = np.array([1, 6, 6, 1])
    positions = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    groups, _ = build_bonded_chain_groups(atomic_numbers, positions, bond_scale=1.35)
    feature_function, keys = make_four_body_feature_function(
        groups, np.array([1.0, 2.5, 4.0]), sigma_bohr=0.5, torsion_order=2
    )
    coefficients = torch.linspace(-0.1, 0.1, len(keys), dtype=torch.float64)
    positions_tensor = torch.tensor(positions, dtype=torch.float64)
    hessian = torch.func.hessian(
        lambda candidate: torch.dot(feature_function(candidate), coefficients)
    )(positions_tensor).reshape(12, 12)

    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(hessian, hessian.T, atol=1.0e-11, rtol=1.0e-11)
