import numpy as np

from scripts.qm9_complete_total_geometry_residual_capacity import (
    constrained_least_squares,
    pair_rbf_scalar_features,
)


def test_pair_rbf_features_are_conservative_and_translation_invariant():
    energies, forces, hessians, _ = pair_rbf_scalar_features(
        np.asarray([1, 1]),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.2, -0.1]]),
        np.asarray([1.0, 2.0]),
        0.4,
    )

    assert energies.shape == (2,)
    np.testing.assert_allclose(forces.reshape(2, 3, 2).sum(axis=0), 0.0, atol=1e-14)
    for feature in range(hessians.shape[-1]):
        np.testing.assert_allclose(
            hessians[:, :, feature], hessians[:, :, feature].T, atol=1e-14
        )
        np.testing.assert_allclose(
            hessians[:, :, feature].reshape(2, 3, 6).sum(axis=0),
            0.0,
            atol=1e-14,
        )


def test_constrained_least_squares_preserves_constraints():
    design = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    constraints = np.asarray([[1.0, 1.0, 0.0]])
    target = np.asarray([2.0, -2.0])

    coefficients, diagnostics = constrained_least_squares(
        design, target, constraints
    )

    np.testing.assert_allclose(constraints @ coefficients, 0.0, atol=1e-12)
    np.testing.assert_allclose(design @ coefficients, target, atol=1e-12)
    assert diagnostics["null_space_dimension"] == 2
