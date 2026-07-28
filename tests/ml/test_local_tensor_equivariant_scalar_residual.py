from __future__ import annotations

import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from mldft.ml.models.components.local_tensor_equivariant_scalar_residual import (
    LocalTensorEquivariantScalarResidual,
)


def _system() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 8], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.7, 0.2, -0.1], [0.3, -0.6, 2.0]],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def _model() -> LocalTensorEquivariantScalarResidual:
    model = LocalTensorEquivariantScalarResidual(
        scalar_channels=6,
        vector_channels=3,
        tensor_channels=2,
        radial_size=5,
        radial_hidden_size=8,
        interaction_layers=2,
        cutoff_bohr=8.0,
    )
    with torch.no_grad():
        model.network.readout[-1].weight.add_(0.05)
    return model


def test_tensor_equivariant_energy_is_invariant_and_force_is_covariant() -> None:
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

    energy, force, _ = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    transformed_energy, transformed_force, _ = model.energy_force_hessian(
        transformed, atomic_numbers, topology, create_parameter_graph=False
    )

    torch.testing.assert_close(energy, transformed_energy, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(
        transformed_force, force @ rotation.T, atol=2e-9, rtol=2e-9
    )


def test_tensor_equivariant_hessian_is_finite_symmetric_and_trainable() -> None:
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

    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()
    assert torch.isfinite(hessian).all()
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-9
    finite = [value for value in gradients if value is not None]
    assert finite
    assert all(torch.isfinite(value).all() for value in finite)
    assert any(torch.linalg.vector_norm(value) > 0 for value in finite)
