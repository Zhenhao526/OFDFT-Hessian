from __future__ import annotations

import pytest
import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from mldft.ml.models.components.local_message_passing_residual import (
    LocalMessagePassingResidual,
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


def test_initial_function_subtraction_is_exact_through_hessian() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalMessagePassingResidual(hidden_size=16, radial_size=6, message_layers=2)

    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology
    )

    assert float(energy) == 0.0
    assert torch.max(torch.abs(force)) < 1e-15
    assert torch.max(torch.abs(hessian)) < 1e-14
    assert torch.all(torch.isfinite(hessian))


def test_hessian_loss_reaches_trainable_message_parameters() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalMessagePassingResidual(hidden_size=16, radial_size=6, message_layers=2)
    _, _, hessian = model.energy_force_hessian(positions, atomic_numbers, topology)
    loss = torch.mean((hessian - 0.01 * torch.eye(positions.numel())) ** 2)
    loss.backward()

    gradients = [parameter.grad for parameter in model.trainable_parameters()]
    assert any(
        gradient is not None and float(torch.linalg.vector_norm(gradient)) > 0.0
        for gradient in gradients
    )
    assert all(parameter.grad is None for parameter in model.initial_network.parameters())


def test_anchored_scalar_is_invariant_and_has_symmetric_hessian() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalMessagePassingResidual(hidden_size=16, radial_size=6, message_layers=2)
    with torch.no_grad():
        next(model.network.parameters()).add_(0.02)

    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        reference_positions_bohr=positions,
    )
    angle = torch.tensor(0.41, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), angle * 0.0)),
            torch.stack((torch.sin(angle), torch.cos(angle), angle * 0.0)),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    moved = positions @ rotation.T + torch.tensor([0.7, -1.1, 0.3])
    moved_energy = model.forward_energy(moved, atomic_numbers, topology)
    original_energy = model.forward_energy(positions, atomic_numbers, topology)

    assert abs(float(energy)) < 1e-12
    assert torch.max(torch.abs(force)) < 1e-11
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-10
    assert torch.all(torch.isfinite(hessian))
    assert torch.allclose(original_energy, moved_energy, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    ("activation", "softplus_beta"),
    (("softplus", 5.0), ("silu", 1.0), ("tanh", 1.0)),
)
def test_activation_audit_paths_have_finite_trainable_hessians(
    activation: str, softplus_beta: float
) -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalMessagePassingResidual(
        hidden_size=12,
        radial_size=5,
        message_layers=2,
        seed=19,
        activation=activation,
        softplus_beta=softplus_beta,
    )

    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=True,
    )
    loss = torch.mean(hessian.square())
    gradients = torch.autograd.grad(loss, model.trainable_parameters(), allow_unused=True)

    assert torch.isfinite(energy)
    assert torch.all(torch.isfinite(force))
    assert torch.all(torch.isfinite(hessian))
    assert torch.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10)
    assert any(
        gradient is not None
        and bool(torch.all(torch.isfinite(gradient)))
        and float(torch.linalg.vector_norm(gradient)) > 0.0
        for gradient in gradients
    )


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported smooth activation"):
        LocalMessagePassingResidual(activation="relu")
