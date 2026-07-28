from __future__ import annotations

import torch

from scripts.qm9_complete_total_local_quadratic_network_capacity import mixed_hvp_loss


def test_mixed_hvp_loss_is_exact_and_finite_at_reference_floor() -> None:
    reference = torch.tensor([[1.0, -2.0], [0.0, 0.0]], dtype=torch.float64)
    exact = mixed_hvp_loss(
        reference,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.05,
    )
    perturbed = mixed_hvp_loss(
        reference + 0.01,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.05,
    )

    assert float(exact) == 0.0
    assert torch.isfinite(perturbed)
    assert float(perturbed) > 0.0
