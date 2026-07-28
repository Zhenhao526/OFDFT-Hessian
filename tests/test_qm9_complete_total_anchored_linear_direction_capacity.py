from __future__ import annotations

import numpy as np
import torch

from scripts.qm9_complete_total_anchored_linear_direction_capacity import (
    _calibration_mask,
    conjugate_gradient_normal_equations,
    mixed_hvp_row_weights,
)


def test_mixed_hvp_row_weights_reproduce_loss() -> None:
    reference = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    error = np.asarray([[0.2, -0.1], [0.05, 0.15]], dtype=np.float64)
    weights = mixed_hvp_row_weights(
        reference,
        parent_count=1,
        absolute_scale=0.1,
        relative_fraction=0.4,
        reference_floor=0.5,
    )
    expected = 0.6 * np.mean((error / 0.1) ** 2) + 0.4 * np.mean(
        np.sum(error**2, axis=1)
        / np.maximum(np.sum(reference**2, axis=1), 0.5**2)
    )

    assert np.allclose(np.sum((weights * error) ** 2), expected)


def test_conjugate_gradient_matches_ridge_solution() -> None:
    generator = torch.Generator().manual_seed(17)
    design = torch.randn(30, 8, dtype=torch.float64, generator=generator)
    target = torch.randn(30, dtype=torch.float64, generator=generator)
    ridge = 1e-3
    actual, diagnostics = conjugate_gradient_normal_equations(
        design,
        target,
        ridge=ridge,
        tolerance=1e-12,
        max_iterations=100,
    )
    expected = torch.linalg.solve(
        design.T @ design + ridge * torch.eye(8, dtype=torch.float64),
        design.T @ target,
    )

    assert diagnostics["converged"] is True
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_calibration_mask_holds_out_complete_directions_per_parent() -> None:
    rows = [
        {
            "coordinate_count": 3,
            "train_direction_count": 6,
            "weighted_row_start": 0,
        },
        {
            "coordinate_count": 2,
            "train_direction_count": 4,
            "weighted_row_start": 18,
        },
    ]
    mask = _calibration_mask(
        26,
        rows,
        stride=3,
        offset=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(torch.nonzero(mask).reshape(-1), torch.tensor([6, 7, 8, 15, 16, 17, 22, 23]))
