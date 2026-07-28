from __future__ import annotations

import torch

from scripts.qm9_complete_total_local_angular_four_body_linear_ceiling import (
    _cgls_solve,
    _central_resolved_angular_columns,
    _central_resolved_angular_features,
    _normal_equation_solve,
)


def test_central_resolved_angular_feature_values_match_global_columns() -> None:
    atomic = torch.tensor(
        [
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 3.0, 4.0],
            [1.0, 0.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
    )
    element_index = torch.tensor([0, 1, 0])
    local = _central_resolved_angular_features(
        atomic,
        element_index,
        element_count=2,
        central_element_indices=(0, 1),
    )
    columns = _central_resolved_angular_columns(
        atomic_feature_count=4,
        element_count=2,
        central_element_indices=(0, 1),
        device=torch.device("cpu"),
    )

    global_features = torch.zeros(6, dtype=torch.float64).index_copy(
        0, columns, local
    )
    expected = torch.tensor([2.0, 1.0, 6.0, 8.0, 3.0, 4.0])
    assert torch.equal(global_features, expected)


def test_cgls_recovers_overdetermined_linear_solution() -> None:
    generator = torch.Generator().manual_seed(29)
    design = torch.randn(80, 12, dtype=torch.float64, generator=generator)
    expected = torch.randn(12, dtype=torch.float64, generator=generator)
    target = design @ expected

    actual, diagnostics = _cgls_solve(
        design,
        target,
        ridge=0.0,
        column_floor=1e-12,
        max_iterations=100,
        tolerance=1e-12,
        log_interval=100,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert diagnostics["converged"]
    assert diagnostics["design_residual_relative"] < 1e-11


def test_cgls_drops_inactive_columns_and_supports_ridge() -> None:
    design = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)

    actual, diagnostics = _cgls_solve(
        design,
        target,
        ridge=1e-10,
        column_floor=1e-12,
        max_iterations=20,
        tolerance=1e-12,
        log_interval=20,
    )

    assert actual[1] == 0.0
    assert diagnostics["active_feature_count"] == 2
    assert diagnostics["design_residual_relative"] < 1e-9


def test_ridge_gram_cholesky_matches_augmented_lstsq() -> None:
    generator = torch.Generator().manual_seed(31)
    design = torch.randn(50, 9, dtype=torch.float64, generator=generator)
    design[:, 4] *= 1e-4
    target = torch.randn(50, dtype=torch.float64, generator=generator)
    ridge = 1e-6

    actual, diagnostics = _normal_equation_solve(
        design,
        target,
        ridge=ridge,
        column_floor=1e-12,
    )
    scale = torch.linalg.vector_norm(design, dim=0)
    normalized = design / scale[None, :]
    augmented_design = torch.cat(
        (normalized, ridge**0.5 * torch.eye(9, dtype=torch.float64)), dim=0
    )
    augmented_target = torch.cat((target, torch.zeros(9, dtype=torch.float64)))
    expected_normalized = torch.linalg.lstsq(
        augmented_design, augmented_target
    ).solution
    expected = expected_normalized / scale

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert diagnostics["converged"]
    assert diagnostics["final_relative_normal_residual"] < 1e-12
