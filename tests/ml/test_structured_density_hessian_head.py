import numpy as np
import torch

from mldft.ml.models.components.structured_density_hessian_head import (
    StructuredDensityHessianHead,
)
from mldft.ofdft.internal_directions import rigid_external_basis


def _projector(positions: torch.Tensor) -> torch.Tensor:
    external = rigid_external_basis(positions.detach().cpu().numpy())
    return torch.eye(positions.numel(), dtype=positions.dtype) - torch.from_numpy(
        external @ external.T
    ).to(dtype=positions.dtype)


def _rotation() -> torch.Tensor:
    matrix = torch.tensor(
        [[0.2, -0.7, 0.3], [0.6, 0.1, -0.5], [0.4, 0.5, 0.8]],
        dtype=torch.float64,
    )
    return torch.linalg.qr(matrix).Q


def _model(use_density: bool = True) -> StructuredDensityHessianHead:
    torch.manual_seed(7)
    return StructuredDensityHessianHead(
        6,
        hidden_dim=16,
        use_density=use_density,
        zero_initialize=False,
    ).to(dtype=torch.float64)


def _sample():
    atomic_numbers = torch.tensor([6, 1, 8, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [-1.1, 0.4, 0.2], [0.2, 1.5, -0.3]],
        dtype=torch.float64,
    )
    density = torch.arange(24, dtype=torch.float64).reshape(4, 6) / 17.0
    return atomic_numbers, positions, density


def test_structured_head_is_symmetric_and_internal():
    atomic_numbers, positions, density = _sample()
    projector = _projector(positions)
    predicted = _model()(atomic_numbers, positions, density, projector)
    assert torch.max(torch.abs(predicted - predicted.T)) < 1.0e-12
    external = torch.eye(projector.shape[0], dtype=projector.dtype) - projector
    assert torch.linalg.matrix_norm(predicted @ external) < 1.0e-10


def test_structured_head_is_rotation_covariant():
    atomic_numbers, positions, density = _sample()
    model = _model()
    predicted = model(atomic_numbers, positions, density, _projector(positions))
    rotation = _rotation()
    rotated_positions = positions @ rotation.T
    rotated = model(
        atomic_numbers,
        rotated_positions,
        density,
        _projector(rotated_positions),
    )
    coordinate_rotation = torch.kron(
        torch.eye(positions.shape[0], dtype=torch.float64),
        rotation.contiguous(),
    )
    expected = coordinate_rotation @ predicted @ coordinate_rotation.T
    assert torch.allclose(rotated, expected, atol=2.0e-10, rtol=2.0e-10)


def test_structured_head_is_permutation_covariant():
    atomic_numbers, positions, density = _sample()
    model = _model()
    predicted = model(atomic_numbers, positions, density, _projector(positions))
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_positions = positions[permutation]
    permuted = model(
        atomic_numbers[permutation],
        permuted_positions,
        density[permutation],
        _projector(permuted_positions),
    )
    atom_permutation = torch.zeros((4, 4), dtype=torch.float64)
    atom_permutation[torch.arange(4), permutation] = 1.0
    coordinate_permutation = torch.kron(
        atom_permutation, torch.eye(3, dtype=torch.float64)
    )
    expected = coordinate_permutation @ predicted @ coordinate_permutation.T
    assert torch.allclose(permuted, expected, atol=2.0e-10, rtol=2.0e-10)


def test_geometry_ablation_ignores_density_values():
    atomic_numbers, positions, density = _sample()
    model = _model(use_density=False)
    projector = _projector(positions)
    first = model(atomic_numbers, positions, density, projector)
    second = model(atomic_numbers, positions, density + 100.0, projector)
    assert torch.equal(first, second)
