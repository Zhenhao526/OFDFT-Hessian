from __future__ import annotations

import pytest
import torch

from scripts.qm9_complete_total_mace_force_secant_gradient_audit import (
    flatten_task_gradient,
    gradient_cosine,
    parse_direction_groups,
)


def test_flatten_task_gradient_fills_unused_parameters() -> None:
    first = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    unused = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

    gradient = flatten_task_gradient(
        torch.sum(first * first), [first, unused], retain_graph=False
    )

    torch.testing.assert_close(
        gradient, torch.tensor([4.0, -2.0, 0.0], dtype=torch.float64)
    )


def test_gradient_cosine_handles_aligned_opposed_and_zero() -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        torch.tensor(gradient_cosine(first, first)), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        torch.tensor(gradient_cosine(first, -first)), torch.tensor(-1.0)
    )
    assert gradient_cosine(first, torch.zeros_like(first)) == 0.0


def test_parse_direction_groups_preserves_fixed_groups() -> None:
    assert parse_direction_groups("0,7,14;1,8,15") == [[0, 7, 14], [1, 8, 15]]
    with pytest.raises(ValueError, match="unique"):
        parse_direction_groups("0,0,1")
