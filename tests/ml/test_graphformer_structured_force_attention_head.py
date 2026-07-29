from __future__ import annotations

import torch

from mldft.ml.models.components.graphformer_structured_force_attention_head import (
    GraphformerStructuredForceAttentionHead,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [
            [0.2, -0.1, 0.3],
            [1.5, 0.4, -0.2],
            [-0.6, 1.2, 0.5],
            [0.1, -1.1, 1.4],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(20260729)
    features = torch.randn(
        (4, 24), generator=generator, dtype=torch.float64
    )
    return features, positions


def _model(*, zero_initialize: bool = False) -> GraphformerStructuredForceAttentionHead:
    torch.manual_seed(11)
    return GraphformerStructuredForceAttentionHead(
        node_feature_dim=24,
        attention_dim=16,
        attention_heads=4,
        latent_dim=6,
        pair_hidden_dim=12,
        zero_initialize=zero_initialize,
    ).to(dtype=torch.float64)


def test_force_head_zero_initialization() -> None:
    features, positions = _inputs()
    forces = _model(zero_initialize=True)(features, positions)
    torch.testing.assert_close(forces, torch.zeros_like(forces))


def test_force_head_has_zero_net_force_and_torque() -> None:
    features, positions = _inputs()
    forces = _model()(features, positions)
    torch.testing.assert_close(
        forces.sum(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torque = torch.linalg.cross(positions, forces).sum(dim=0)
    torch.testing.assert_close(
        torque,
        torch.zeros(3, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_force_head_is_rotation_covariant() -> None:
    features, positions = _inputs()
    model = _model()
    forces = model(features, positions)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated = model(features, positions @ rotation.T)
    torch.testing.assert_close(
        rotated, forces @ rotation.T, atol=2e-11, rtol=2e-11
    )


def test_force_head_is_permutation_covariant() -> None:
    features, positions = _inputs()
    model = _model()
    forces = model(features, positions)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted = model(features[permutation], positions[permutation])
    torch.testing.assert_close(
        permuted, forces[permutation], atol=2e-11, rtol=2e-11
    )


def test_batched_force_matches_individual_force() -> None:
    features, positions = _inputs()
    model = _model()
    batched = model.forward_batch(
        torch.stack((features, 0.5 * features)),
        torch.stack((positions, positions + 0.7)),
    )
    expected = torch.stack(
        (
            model(features, positions),
            model(0.5 * features, positions + 0.7),
        )
    )
    torch.testing.assert_close(batched, expected, atol=2e-11, rtol=2e-11)
