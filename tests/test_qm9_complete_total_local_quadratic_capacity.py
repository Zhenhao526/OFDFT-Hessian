from __future__ import annotations

import numpy as np

from mldft.ml.models.components.local_quadratic_residual import QuadraticTerm
from scripts.qm9_complete_total_local_quadratic_capacity import Parent, _directional_block


def test_directional_block_matches_explicit_term_hessians() -> None:
    jacobian = np.asarray(
        [[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]], dtype=np.float64
    )
    terms = [
        QuadraticTerm("diagonal", 0, 0, 2.0),
        QuadraticTerm("cross", 0, 1, -0.5),
    ]
    directions = np.asarray([[1.0, 0.5, -0.25], [-0.2, 0.1, 0.8]])
    parent = Parent(
        molecule_id="synthetic",
        natoms=1,
        pbe_energy=0.0,
        pbe_force=np.zeros((1, 3)),
        pbe_hessian=np.zeros((3, 3)),
        source_energy=0.0,
        source_force=np.zeros((1, 3)),
        source_hessian=np.zeros((3, 3)),
        train_directions=directions,
        heldout_directions=directions,
        jacobian=jacobian,
        terms=terms,
        coordinate_kind_counts={},
    )
    actual = _directional_block(
        parent, directions, {"diagonal": 0, "cross": 1}
    ).reshape(2, 3, 2)
    diagonal = 2.0 * np.outer(jacobian[0], jacobian[0])
    cross = -0.5 * (
        np.outer(jacobian[0], jacobian[1])
        + np.outer(jacobian[1], jacobian[0])
    )

    assert np.allclose(actual[:, :, 0], directions @ diagonal.T)
    assert np.allclose(actual[:, :, 1], directions @ cross.T)
