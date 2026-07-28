from __future__ import annotations

import torch

from mldft.ml.models.components.local_body_order_residual import (
    LocalBodyOrderResidual,
    build_body_order_topology,
)


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 1, 8], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0],
            [-0.4, 1.7, 0.2],
            [0.3, -0.2, 2.1],
        ],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def _split_canceling_units(model: LocalBodyOrderResidual) -> None:
    with torch.no_grad():
        model.pair.weight[:, 0::2].add_(0.03)
        model.triplet.weight[:, 1::2].sub_(0.02)
        model.triplet.bias[:, 0::2].add_(0.01)


def test_topology_has_no_self_interactions() -> None:
    atomic_numbers, _ = _geometry()
    topology = build_body_order_topology(atomic_numbers)

    assert topology.pair_first.numel() == 6
    assert topology.triplet_center.numel() == 12
    assert torch.all(topology.pair_first != topology.pair_second)
    assert torch.all(topology.triplet_center != topology.triplet_first)
    assert torch.all(topology.triplet_center != topology.triplet_second)
    assert torch.all(topology.triplet_first != topology.triplet_second)


def test_bonded_chain_topology_is_unique_and_has_no_self_interactions() -> None:
    atomic_numbers = torch.tensor([1, 6, 6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.8, 0.2, 0.0], [6.5, 1.0, 0.5]],
        dtype=torch.float64,
    )
    topology = build_body_order_topology(atomic_numbers, positions)
    chains = torch.stack(
        (
            topology.torsion_first,
            topology.torsion_second,
            topology.torsion_third,
            topology.torsion_fourth,
        ),
        dim=1,
    )

    assert chains.shape == (1, 4)
    assert torch.unique(chains[0]).numel() == 4


def test_canceling_initialization_preserves_scalar_derivatives() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=7)

    energy, force, hessian = model.energy_force_hessian(positions, topology)

    assert abs(float(energy)) < 1e-14
    assert torch.max(torch.abs(force)) < 1e-14
    assert torch.max(torch.abs(hessian)) < 1e-13
    assert torch.all(torch.isfinite(hessian))


def test_hessian_is_symmetric_and_matches_force_difference() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=11)
    _split_canceling_units(model)

    _, _, hessian = model.energy_force_hessian(positions, topology)
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-11
    assert torch.all(torch.isfinite(hessian))

    step = 1e-4
    columns = []
    for coordinate in range(positions.numel()):
        direction = torch.zeros_like(positions).reshape(-1)
        direction[coordinate] = step
        direction = direction.reshape_as(positions)
        plus = positions + direction
        minus = positions - direction
        plus_energy = model.forward_energy(plus.requires_grad_(True), topology)
        minus_energy = model.forward_energy(minus.requires_grad_(True), topology)
        plus_force = -torch.autograd.grad(plus_energy, plus)[0]
        minus_force = -torch.autograd.grad(minus_energy, minus)[0]
        columns.append((-(plus_force - minus_force) / (2.0 * step)).reshape(-1))
    finite_difference = torch.stack(columns, dim=1)
    assert torch.allclose(hessian, finite_difference, atol=2e-7, rtol=2e-6)


def test_curvature_loss_activates_canceling_branch() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=13)
    _, _, hessian = model.energy_force_hessian(positions, topology)
    target = torch.eye(positions.numel(), dtype=torch.float64) * 0.01
    loss = torch.mean((hessian - target) ** 2)
    loss.backward()

    assert model.triplet.output_magnitude.grad is not None
    assert float(torch.linalg.vector_norm(model.triplet.output_magnitude.grad)) == 0.0
    assert model.triplet.weight.grad is not None
    assert float(torch.linalg.vector_norm(model.triplet.weight.grad)) > 0.0


def test_hvp_matches_full_hessian() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=19)
    _split_canceling_units(model)
    generator = torch.Generator().manual_seed(23)
    directions = torch.randn(
        3, *positions.shape, dtype=torch.float64, generator=generator
    )

    _, _, hessian = model.energy_force_hessian(positions, topology)
    _, _, products = model.energy_force_hvp(positions, topology, directions)
    expected = torch.einsum("ij,dj->di", hessian, directions.reshape(3, -1))
    assert torch.allclose(products.reshape(3, -1), expected, atol=1e-11, rtol=1e-10)


def test_reference_geometry_anchor_preserves_energy_force_but_not_hessian() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=37)
    _split_canceling_units(model)

    raw_energy, raw_force, raw_hessian = model.energy_force_hessian(
        positions, topology
    )
    anchored_energy, anchored_force, anchored_hessian = model.energy_force_hessian(
        positions,
        topology,
        reference_positions_bohr=positions,
    )

    assert abs(float(raw_energy)) > 0.0
    assert torch.linalg.vector_norm(raw_force) > 0.0
    assert abs(float(anchored_energy)) < 1e-14
    assert torch.max(torch.abs(anchored_force)) < 1e-14
    assert torch.allclose(anchored_hessian, raw_hessian, atol=1e-12, rtol=1e-12)
    assert torch.linalg.vector_norm(anchored_hessian) > 0.0


def test_torsion_branch_has_finite_symmetric_second_derivatives() -> None:
    atomic_numbers = torch.tensor([1, 6, 6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.8, 0.2, 0.0], [6.5, 1.0, 0.5]],
        dtype=torch.float64,
    )
    topology = build_body_order_topology(atomic_numbers, positions)
    model = LocalBodyOrderResidual(seed=41)
    with torch.no_grad():
        model.torsion.weight[:, 0::2].add_(0.03)
        model.torsion.bias[:, 1::2].sub_(0.01)

    _, _, hessian = model.energy_force_hessian(positions, topology)

    assert torch.all(torch.isfinite(hessian))
    assert torch.linalg.vector_norm(hessian) > 0.0
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-11


def test_scalar_is_translation_rotation_invariant() -> None:
    atomic_numbers, positions = _geometry()
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=17)
    _split_canceling_units(model)
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), angle * 0.0)),
            torch.stack((torch.sin(angle), torch.cos(angle), angle * 0.0)),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    transformed = positions @ rotation.T + torch.tensor(
        [1.1, -0.7, 0.5], dtype=torch.float64
    )

    original = model.forward_energy(positions, topology)
    moved = model.forward_energy(transformed, topology)
    assert torch.allclose(original, moved, atol=1e-12, rtol=1e-12)


def test_interactions_beyond_cutoff_are_exactly_zero() -> None:
    atomic_numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=torch.float64
    )
    topology = build_body_order_topology(atomic_numbers)
    model = LocalBodyOrderResidual(seed=29, cutoff_bohr=8.0)
    _split_canceling_units(model)

    energy, force, hessian = model.energy_force_hessian(positions, topology)
    assert float(energy) == 0.0
    assert torch.count_nonzero(force) == 0
    assert torch.count_nonzero(hessian) == 0
