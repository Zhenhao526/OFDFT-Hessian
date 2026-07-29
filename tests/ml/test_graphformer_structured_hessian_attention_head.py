from __future__ import annotations

import torch

from mldft.ml.models.components.graphformer_structured_hessian_attention_head import (
    GraphformerStructuredHessianAttentionHead,
)
from mldft.ofdft.internal_directions import rigid_external_basis


def _inputs() -> tuple[torch.Tensor, ...]:
    atomic_numbers = torch.tensor([6, 1, 8, 7], dtype=torch.long)
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
    external = rigid_external_basis(positions.numpy())
    projector = torch.eye(12, dtype=torch.float64) - torch.from_numpy(
        external @ external.T
    )
    return atomic_numbers, positions, features, projector


def _model(*, zero_initialize: bool = False) -> GraphformerStructuredHessianAttentionHead:
    torch.manual_seed(7)
    return GraphformerStructuredHessianAttentionHead(
        node_feature_dim=24,
        attention_dim=16,
        attention_heads=4,
        latent_dim=6,
        structured_hidden_dim=12,
        zero_initialize=zero_initialize,
    ).to(dtype=torch.float64)


def test_attention_head_zero_initialization_preserves_zero_hessian() -> None:
    inputs = _inputs()
    predicted = _model(zero_initialize=True)(*inputs)
    torch.testing.assert_close(predicted, torch.zeros_like(predicted))


def test_attention_head_is_symmetric_and_internal() -> None:
    atomic_numbers, positions, features, projector = _inputs()
    predicted = _model()(
        atomic_numbers, positions, features, projector
    )
    torch.testing.assert_close(predicted, predicted.T, atol=1e-12, rtol=0.0)
    external = torch.eye(12, dtype=torch.float64) - projector
    torch.testing.assert_close(
        predicted @ external,
        torch.zeros_like(predicted),
        atol=1e-11,
        rtol=0.0,
    )


def test_attention_head_is_rotation_covariant() -> None:
    atomic_numbers, positions, features, projector = _inputs()
    model = _model()
    predicted = model(atomic_numbers, positions, features, projector)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated_positions = positions @ rotation.T
    external = rigid_external_basis(rotated_positions.numpy())
    rotated_projector = torch.eye(12, dtype=torch.float64) - torch.from_numpy(
        external @ external.T
    )
    rotated = model(
        atomic_numbers, rotated_positions, features, rotated_projector
    )
    coordinate_rotation = torch.kron(
        torch.eye(4, dtype=torch.float64), rotation.contiguous()
    )
    expected = coordinate_rotation @ predicted @ coordinate_rotation.T
    torch.testing.assert_close(rotated, expected, atol=2e-10, rtol=2e-10)


def test_attention_head_is_permutation_covariant() -> None:
    atomic_numbers, positions, features, projector = _inputs()
    model = _model()
    predicted = model(atomic_numbers, positions, features, projector)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    atom_permutation = torch.zeros((4, 4), dtype=torch.float64)
    atom_permutation[torch.arange(4), permutation] = 1.0
    coordinate_permutation = torch.kron(
        atom_permutation, torch.eye(3, dtype=torch.float64)
    )
    permuted_projector = (
        coordinate_permutation @ projector @ coordinate_permutation.T
    )
    permuted = model(
        atomic_numbers[permutation],
        positions[permutation],
        features[permutation],
        permuted_projector,
    )
    expected = (
        coordinate_permutation @ predicted @ coordinate_permutation.T
    )
    torch.testing.assert_close(permuted, expected, atol=2e-10, rtol=2e-10)
