from __future__ import annotations

import torch

from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)
from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)


def _system() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 8, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.0],
            [-0.3, 2.0, 0.4],
            [0.4, -0.2, 2.2],
        ],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def test_local_angular_zero_initialization_and_hessian_parameter_gradient() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalAngularScalarResidual(hidden_size=12, radial_size=3, angular_order=2)

    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=True
    )
    target = 0.01 * torch.eye(positions.numel(), dtype=torch.float64)
    gradients = torch.autograd.grad(
        torch.mean((hessian - target) ** 2),
        model.trainable_parameters(),
        allow_unused=True,
    )

    assert float(energy) == 0.0
    assert torch.max(torch.abs(force)) < 1e-14
    assert torch.max(torch.abs(hessian)) < 1e-13
    assert torch.all(torch.isfinite(hessian))
    assert any(
        gradient is not None
        and bool(torch.all(torch.isfinite(gradient)))
        and float(torch.linalg.vector_norm(gradient)) > 0.0
        for gradient in gradients
    )


def test_local_angular_scalar_is_rigid_and_same_element_permutation_invariant() -> None:
    atomic_numbers, positions = _system()
    model = LocalAngularScalarResidual(
        hidden_size=12, radial_size=3, angular_order=2, seed=19
    )
    with torch.no_grad():
        next(model.network.parameters()).add_(0.03)
    topology = build_body_order_topology(atomic_numbers)
    reference = model.forward_energy(positions, atomic_numbers, topology)

    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), angle * 0.0)),
            torch.stack((torch.sin(angle), torch.cos(angle), angle * 0.0)),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    moved = positions @ rotation.T + torch.tensor([0.7, -0.4, 1.1])
    moved_energy = model.forward_energy(moved, atomic_numbers, topology)

    permutation = torch.tensor([0, 3, 2, 1])
    permuted_numbers = atomic_numbers[permutation]
    permuted_positions = positions[permutation]
    permuted_topology = build_body_order_topology(permuted_numbers)
    permuted_energy = model.forward_energy(
        permuted_positions, permuted_numbers, permuted_topology
    )

    assert torch.allclose(reference, moved_energy, atol=1e-11, rtol=1e-11)
    assert torch.allclose(reference, permuted_energy, atol=1e-11, rtol=1e-11)


def test_local_angular_hessian_is_finite_and_symmetric_after_update() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalAngularScalarResidual(hidden_size=12, radial_size=3, angular_order=2)
    with torch.no_grad():
        next(model.network.parameters()).add_(0.02)

    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )

    assert torch.isfinite(energy)
    assert torch.all(torch.isfinite(force))
    assert torch.all(torch.isfinite(hessian))
    assert torch.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10)
