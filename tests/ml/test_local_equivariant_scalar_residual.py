from __future__ import annotations

import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from mldft.ml.models.components.local_equivariant_scalar_residual import (
    LocalEquivariantScalarResidual,
)


def _system() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 1, 8], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.7, 0.2, -0.1],
            [-0.4, 1.5, 0.3],
            [0.3, -0.6, 2.0],
        ],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def test_equivariant_scalar_energy_and_force_transform_correctly() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = LocalEquivariantScalarResidual(hidden_size=12, radial_size=6, interaction_layers=2)
    with torch.no_grad():
        model.network.readout[-1].bias.add_(0.3)
        model.network.readout[-1].weight.add_(0.05)
    angle = torch.tensor(0.63, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), torch.tensor(0.0))),
            torch.stack((torch.sin(angle), torch.cos(angle), torch.tensor(0.0))),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    translation = torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64)
    transformed = positions @ rotation.T + translation
    energy, force, _ = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    transformed_energy, transformed_force, _ = model.energy_force_hessian(
        transformed, atomic_numbers, topology, create_parameter_graph=False
    )

    assert torch.allclose(energy, transformed_energy, atol=2e-11, rtol=2e-11)
    assert torch.allclose(
        transformed_force, force @ rotation.T, atol=2e-10, rtol=2e-10
    )


def test_equivariant_scalar_hessian_is_finite_symmetric_and_trainable() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = LocalEquivariantScalarResidual(hidden_size=10, radial_size=5, interaction_layers=2)
    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=True,
        reference_positions_bohr=positions,
    )
    target = torch.eye(hessian.shape[0], dtype=torch.float64) * 0.1
    loss = torch.mean((hessian - target) ** 2)
    gradients = torch.autograd.grad(loss, model.trainable_parameters(), allow_unused=True)

    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()
    assert torch.isfinite(hessian).all()
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-10
    finite_gradients = [value for value in gradients if value is not None]
    assert finite_gradients
    assert all(torch.isfinite(value).all() for value in finite_gradients)
    assert any(torch.linalg.vector_norm(value) > 0 for value in finite_gradients)
