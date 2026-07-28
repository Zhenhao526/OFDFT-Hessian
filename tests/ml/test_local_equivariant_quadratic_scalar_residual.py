from __future__ import annotations

import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from mldft.ml.models.components.local_equivariant_quadratic_scalar_residual import (
    LocalEquivariantQuadraticScalarResidual,
)


def _system() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 8, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.7, 0.2, -0.1],
            [0.3, -0.6, 2.0],
            [-0.4, 1.5, 0.3],
        ],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def _model() -> LocalEquivariantQuadraticScalarResidual:
    model = LocalEquivariantQuadraticScalarResidual(
        hidden_size=12,
        radial_size=6,
        cutoff_bohr=8.0,
        coefficient_scale=0.5,
    )
    with torch.no_grad():
        model.network.pair_readout[-1].weight.add_(0.03)
        model.network.diagonal_readout[-1].weight.add_(0.02)
    return model


def test_quadratic_scalar_hessian_is_covariant_and_rigid_projected() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    angle = torch.tensor(0.63, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), zero)),
            torch.stack((torch.sin(angle), torch.cos(angle), zero)),
            torch.stack((zero, zero, one)),
        )
    )
    transformed = positions @ rotation.T + torch.tensor(
        [1.2, -0.7, 0.4], dtype=torch.float64
    )
    _, _, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=False,
        reference_positions_bohr=positions,
    )
    _, _, transformed_hessian = model.energy_force_hessian(
        transformed,
        atomic_numbers,
        topology,
        create_parameter_graph=False,
        reference_positions_bohr=transformed,
    )
    coordinate_rotation = torch.kron(
        torch.eye(positions.shape[0], dtype=torch.float64), rotation
    )

    torch.testing.assert_close(
        transformed_hessian,
        coordinate_rotation @ hessian @ coordinate_rotation.T,
        atol=2e-9,
        rtol=2e-9,
    )
    translations = []
    for axis in range(3):
        vector = torch.zeros_like(positions)
        vector[:, axis] = 1.0
        translations.append(vector.reshape(-1))
    translation_basis = torch.stack(translations, dim=1)
    assert torch.linalg.matrix_norm(hessian @ translation_basis) < 1e-9


def test_quadratic_scalar_hessian_is_atom_permutation_equivariant() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    _, _, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=False,
        reference_positions_bohr=positions,
    )
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_numbers = atomic_numbers[permutation]
    permuted_positions = positions[permutation]
    permuted_topology = build_body_order_topology(
        permuted_numbers, permuted_positions
    )
    _, _, permuted_hessian = model.energy_force_hessian(
        permuted_positions,
        permuted_numbers,
        permuted_topology,
        create_parameter_graph=False,
        reference_positions_bohr=permuted_positions,
    )
    coordinate_indices = (
        3 * permutation[:, None] + torch.arange(3, dtype=torch.long)[None, :]
    ).reshape(-1)

    torch.testing.assert_close(
        permuted_hessian,
        hessian[coordinate_indices][:, coordinate_indices],
        atol=2e-9,
        rtol=2e-9,
    )


def test_quadratic_scalar_derivatives_are_finite_symmetric_and_trainable() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=True,
        reference_positions_bohr=positions,
    )
    target = torch.eye(hessian.shape[0], dtype=torch.float64) * 0.1
    gradients = torch.autograd.grad(
        torch.mean((hessian - target) ** 2),
        model.trainable_parameters(),
        allow_unused=True,
    )
    expected_hessian = model.hessian_correction(
        positions, atomic_numbers, topology
    )

    assert torch.abs(energy) < 1e-12
    assert torch.max(torch.abs(force)) < 1e-12
    assert torch.isfinite(hessian).all()
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-10
    torch.testing.assert_close(hessian, expected_hessian, atol=1e-11, rtol=1e-11)
    finite = [value for value in gradients if value is not None]
    assert finite
    assert all(torch.isfinite(value).all() for value in finite)
    assert any(torch.linalg.vector_norm(value) > 0 for value in finite)
