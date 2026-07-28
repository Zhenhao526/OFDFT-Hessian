from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.qm9_complete_total_relaxed_vector_local_scalar import (
    _hvp_design_blocks,
    _hvp_relative,
)


def test_hvp_relative_uses_vector_norm_floor() -> None:
    assert _hvp_relative(
        np.asarray([0.03, 0.04]), np.zeros(2), floor=0.1
    ) == pytest.approx(0.5)
    assert _hvp_relative(
        np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0]), floor=0.1
    ) == pytest.approx(0.5)


def test_hvp_design_is_scalar_hessian_contraction_and_component_normalized() -> None:
    hessian = torch.zeros((2, 3, 3), dtype=torch.float64)
    hessian[0] = torch.eye(3, dtype=torch.float64)
    hessian[1] = 2.0 * torch.eye(3, dtype=torch.float64)
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
    reference = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    designs, targets = _hvp_design_blocks(
        hessian,
        directions,
        target,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.1,
        absolute_target_cap=10.0,
        relative_target_cap=10.0,
    )
    assert [tuple(value.shape) for value in designs] == [(3, 2), (3, 2)]
    assert [tuple(value.shape) for value in targets] == [(3,), (3,)]
    assert designs[0][0, 1] / designs[0][0, 0] == pytest.approx(2.0)
    assert torch.all(designs[0][1:] == 0.0)
    assert torch.all(targets[0][1:] == 0.0)
