from __future__ import annotations

import torch

from scripts.qm9_complete_total_local_angular_linear_ceiling import (
    _chunked_vector_jet,
    _solve,
)


def test_chunked_forward_over_reverse_vector_jet_matches_reverse_over_reverse() -> None:
    flat = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)

    def features(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                torch.sin(value[0] * value[1]),
                value[0] ** 3 + value[1] * value[2] ** 2,
                torch.exp(0.1 * torch.sum(value**2)),
            )
        )

    value, jacobian, hessian = _chunked_vector_jet(
        features,
        flat,
        feature_chunk_size=2,
    )

    assert torch.allclose(value, features(flat), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        jacobian,
        torch.func.jacrev(features)(flat),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        hessian,
        torch.func.jacrev(torch.func.jacrev(features))(flat),
        atol=1e-12,
        rtol=1e-12,
    )


def test_column_normalized_linear_ceiling_recovers_synthetic_solution() -> None:
    generator = torch.Generator().manual_seed(19)
    design = torch.randn(20, 5, dtype=torch.float64, generator=generator)
    expected = torch.randn(5, dtype=torch.float64, generator=generator)
    target = design @ expected

    actual, diagnostics = _solve(
        design, target, ridge=0.0, column_floor=1e-12
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert diagnostics["active_feature_count"] == 5
    assert diagnostics["design_residual_relative"] < 1e-12


def test_column_normalized_linear_ceiling_drops_zero_columns() -> None:
    design = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)

    actual, diagnostics = _solve(
        design, target, ridge=1e-8, column_floor=1e-12
    )

    assert actual[1] == 0.0
    assert diagnostics["active_feature_count"] == 2
    assert diagnostics["design_residual_relative"] < 1e-7
