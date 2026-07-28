from __future__ import annotations

import numpy as np
import torch

from scripts.qm9_complete_total_local_scalar_jacobian_range_audit import (
    _split_parent_residuals,
    cgls,
    estimate_right_jacobi_preconditioner,
    is_stable5_protocol,
    select_audit_parents,
    symmetric_frobenius_vector,
)


def test_symmetric_vector_preserves_frobenius_norm() -> None:
    matrix = torch.tensor(
        [[2.0, -0.4, 0.7], [-0.4, 1.3, 0.2], [0.7, 0.2, -0.8]],
        dtype=torch.float64,
    )

    vector = symmetric_frobenius_vector(matrix)

    torch.testing.assert_close(torch.linalg.vector_norm(vector), torch.linalg.matrix_norm(matrix))


def test_split_parent_residuals_recovers_physical_energy_force_scales() -> None:
    class Parent:
        molecule_id = "p"
        natoms = 1

    # One normalized energy value, three normalized force values, then six
    # Frobenius-preserving upper-triangle Hessian values.
    vector = torch.tensor(
        [
            -2.0,
            1.0 / np.sqrt(3.0),
            -2.0 / np.sqrt(3.0),
            3.0 / np.sqrt(3.0),
            1.0,
            2.0,
            2.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=torch.float64,
    )

    rows = _split_parent_residuals(
        vector,
        [Parent()],
        include_energy_force=True,
        energy_scale=0.1,
        force_scale=0.05,
    )

    assert rows[0]["energy_abs_error_hartree"] == 0.2
    np.testing.assert_allclose(rows[0]["force_mae_hartree_per_bohr"], 0.1)
    np.testing.assert_allclose(rows[0]["relative_frobenius"], np.sqrt(10.0))


def test_cgls_matches_dense_least_squares() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(11, 5, dtype=torch.float64, generator=generator)
    right_hand_side = torch.randn(11, dtype=torch.float64, generator=generator)
    expected = torch.linalg.lstsq(matrix, right_hand_side).solution

    actual, residual, rows = cgls(
        lambda value: matrix @ value,
        lambda value: matrix.T @ value,
        right_hand_side,
        matrix.shape[1],
        iterations=10,
        relative_tolerance=1e-12,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(residual, right_hand_side - matrix @ expected)
    assert rows[-1]["normal_gradient_norm"] < 1e-9


def test_cgls_reports_rank_deficient_projection_floor() -> None:
    matrix = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    right_hand_side = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)

    _, residual, rows = cgls(
        lambda value: matrix @ value,
        lambda value: matrix.T @ value,
        right_hand_side,
        matrix.shape[1],
        iterations=4,
        relative_tolerance=1e-12,
    )

    np.testing.assert_allclose(residual.numpy(), [0.0, 0.0, 4.0], atol=1e-12)
    assert rows[-1]["relative_residual_norm"] > 0.0


def test_hutchinson_jacobi_preconditioner_repairs_column_scale() -> None:
    matrix = torch.diag(torch.tensor([1e-4, 1.0, 1e4], dtype=torch.float64))
    right_hand_side = torch.ones(3, dtype=torch.float64)
    scale, metadata = estimate_right_jacobi_preconditioner(
        lambda value: matrix.T @ value,
        right_hand_side,
        matrix.shape[1],
        probes=2,
        seed=17,
    )

    _, residual, _ = cgls(
        lambda value: matrix @ (scale * value),
        lambda value: scale * (matrix.T @ value),
        right_hand_side,
        matrix.shape[1],
        iterations=1,
        relative_tolerance=1e-12,
    )

    assert torch.linalg.vector_norm(residual) < 1e-10
    assert metadata["active_parameter_count"] == 3
    np.testing.assert_allclose(scale.numpy(), [1e4, 1.0, 1e-4], rtol=1e-12)


def test_stable5_protocol_accepts_legacy_scope_and_current_stage() -> None:
    assert is_stable5_protocol({"definitions": {"scope": "stable5; fit only"}})
    assert is_stable5_protocol({"stage": "stable5_fit_only_mature_higher_body_scalar_capacity"})
    assert not is_stable5_protocol({"stage": "train20_direction_generalization"})


def test_parent_selection_is_explicit_and_fail_closed() -> None:
    class Parent:
        def __init__(self, molecule_id: str) -> None:
            self.molecule_id = molecule_id

    parents = [Parent("a"), Parent("b")]
    assert select_audit_parents(parents, None) == parents
    assert select_audit_parents(parents, "b") == [parents[1]]
    with np.testing.assert_raises_regex(ValueError, "absent"):
        select_audit_parents(parents, "c")
