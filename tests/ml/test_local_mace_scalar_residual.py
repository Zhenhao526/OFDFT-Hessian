from __future__ import annotations

import importlib.util

import pytest
import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from mldft.ml.models.components.local_mace_scalar_residual import (
    LocalMACEScalarResidual,
)
from mldft.ml.models.components.local_mace_invariant_readout import (
    LocalMACEInvariantReadoutResidual,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mace") is None,
    reason="mace-torch is provided through an isolated vendor path",
)


def _system() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([6, 1, 8], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.7, 0.2, -0.1], [0.3, -0.6, 2.0]],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def _model(*, output_scale: float = 1.0) -> LocalMACEScalarResidual:
    model = LocalMACEScalarResidual(
        hidden_channels=4,
        mlp_channels=4,
        max_ell=2,
        correlation=2,
        num_interactions=2,
        num_bessel=4,
        cutoff_polynomial_order=5,
        cutoff_bohr=8.0,
        avg_num_neighbors=3.0,
        output_scale=output_scale,
        seed=20260801,
    )
    generator = torch.Generator(device="cpu").manual_seed(29)
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.add_(
                1e-3
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
            )
    return model


def _readout_model(
    feature_mode: str = "scalar",
) -> LocalMACEInvariantReadoutResidual:
    return LocalMACEInvariantReadoutResidual(
        hidden_channels=4,
        mlp_channels=4,
        max_ell=2,
        correlation=2,
        num_interactions=2,
        num_bessel=4,
        cutoff_polynomial_order=5,
        cutoff_bohr=8.0,
        avg_num_neighbors=3.0,
        feature_mode=feature_mode,
        random_feature_width=8,
        random_feature_input_scale=10.0 if feature_mode == "power_spectrum" else 1.0,
        output_scale=11.0,
        seed=20260804,
    )


def test_mace_scalar_hessian_is_finite_symmetric_and_trainable() -> None:
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

    assert torch.abs(energy) < 1e-10
    assert torch.max(torch.abs(force)) < 1e-9
    assert torch.isfinite(hessian).all()
    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-8
    finite = [value for value in gradients if value is not None]
    assert finite
    assert all(torch.isfinite(value).all() for value in finite)
    assert any(torch.linalg.vector_norm(value) > 0 for value in finite)


def test_mace_scalar_energy_force_hessian_are_rigid_covariant() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    angle = torch.tensor(0.57, dtype=torch.float64)
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
        [0.8, -1.1, 0.4], dtype=torch.float64
    )
    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=False,
    )
    transformed_energy, transformed_force, transformed_hessian = (
        model.energy_force_hessian(
            transformed,
            atomic_numbers,
            topology,
            create_parameter_graph=False,
        )
    )
    coordinate_rotation = torch.kron(
        torch.eye(positions.shape[0], dtype=torch.float64), rotation
    )

    torch.testing.assert_close(transformed_energy, energy, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        transformed_force, force @ rotation.T, atol=1e-8, rtol=1e-7
    )
    torch.testing.assert_close(
        transformed_hessian,
        coordinate_rotation @ hessian @ coordinate_rotation.T,
        atol=2e-8,
        rtol=2e-7,
    )


def test_mace_scalar_output_scale_rescales_all_energy_derivatives() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    unit_model = _model()
    scaled_model = _model(output_scale=37.0)
    energy, force, hessian = unit_model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    scaled_energy, scaled_force, scaled_hessian = scaled_model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )

    torch.testing.assert_close(scaled_energy, 37.0 * energy, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(scaled_force, 37.0 * force, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        scaled_hessian, 37.0 * hessian, atol=1e-8, rtol=1e-7
    )


def test_mace_scalar_energy_force_hessian_are_atom_permutation_equivariant() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_numbers = atomic_numbers[permutation]
    permuted_positions = positions[permutation]
    permuted_topology = build_body_order_topology(
        permuted_numbers, permuted_positions
    )
    permuted_energy, permuted_force, permuted_hessian = model.energy_force_hessian(
        permuted_positions,
        permuted_numbers,
        permuted_topology,
        create_parameter_graph=False,
    )
    coordinate_indices = (
        3 * permutation[:, None] + torch.arange(3, dtype=torch.long)[None, :]
    ).reshape(-1)

    torch.testing.assert_close(permuted_energy, energy, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        permuted_force, force[permutation], atol=1e-8, rtol=1e-7
    )
    torch.testing.assert_close(
        permuted_hessian,
        hessian[coordinate_indices][:, coordinate_indices],
        atol=2e-8,
        rtol=2e-7,
    )


def test_mace_node_scalar_features_are_rigid_invariant() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _model()
    angle = torch.tensor(0.41, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    transformed = positions @ rotation.T + torch.tensor(
        [0.4, -0.3, 0.8], dtype=torch.float64
    )

    features = model.forward_node_scalar_features(
        positions, atomic_numbers, topology
    )
    transformed_features = model.forward_node_scalar_features(
        transformed, atomic_numbers, topology
    )
    invariant_features = model.forward_node_invariant_features(
        positions, atomic_numbers, topology
    )
    transformed_invariant_features = model.forward_node_invariant_features(
        transformed, atomic_numbers, topology
    )

    assert features.shape == (3, 8)
    assert invariant_features.shape == (3, 28)
    torch.testing.assert_close(
        transformed_features, features, atol=1e-9, rtol=1e-8
    )
    torch.testing.assert_close(
        transformed_invariant_features,
        invariant_features,
        atol=1e-9,
        rtol=1e-8,
    )


def test_frozen_mace_power_spectrum_readout_has_finite_scalar_jet() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _readout_model("power_spectrum")
    generator = torch.Generator(device="cpu").manual_seed(43)
    with torch.no_grad():
        model.readout.copy_(
            1e-4
            * torch.randn(
                model.readout.shape, dtype=torch.float64, generator=generator
            )
        )

    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )

    assert model.readout.shape == (5, 36)
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()
    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(hessian, hessian.T, atol=1e-9, rtol=1e-9)


def test_frozen_mace_readout_has_zero_initial_jet_and_trainable_hessian() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _readout_model()

    energy, force, hessian = model.energy_force_hessian(
        positions,
        atomic_numbers,
        topology,
        create_parameter_graph=True,
        reference_positions_bohr=positions,
    )
    target = torch.eye(hessian.shape[0], dtype=torch.float64) * 0.1
    gradient = torch.autograd.grad(torch.mean((hessian - target) ** 2), model.readout)[0]

    assert torch.abs(energy) < 1e-12
    assert torch.max(torch.abs(force)) < 1e-12
    assert torch.max(torch.abs(hessian)) < 1e-12
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0.0
    trainable = model.trainable_parameters()
    assert len(trainable) == 1
    assert trainable[0] is model.readout


def test_frozen_mace_readout_jet_is_linear_in_trainable_parameters() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _readout_model()
    generator = torch.Generator(device="cpu").manual_seed(41)
    coefficients = torch.randn(
        model.readout.shape, dtype=torch.float64, generator=generator
    )
    scale = 0.37

    with torch.no_grad():
        model.readout.copy_(coefficients)
    jet = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    with torch.no_grad():
        model.readout.copy_(scale * coefficients)
    scaled_jet = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )

    for scaled_value, value in zip(scaled_jet, jet, strict=True):
        torch.testing.assert_close(scaled_value, scale * value, atol=1e-9, rtol=1e-9)


def test_frozen_mace_readout_scalar_is_rigid_covariant() -> None:
    atomic_numbers, positions = _system()
    topology = build_body_order_topology(atomic_numbers, positions)
    model = _readout_model()
    generator = torch.Generator(device="cpu").manual_seed(37)
    with torch.no_grad():
        model.readout.copy_(
            1e-3
            * torch.randn(
                model.readout.shape, dtype=torch.float64, generator=generator
            )
        )
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    transformed = positions @ rotation.T + torch.tensor(
        [-0.2, 0.7, 0.3], dtype=torch.float64
    )
    energy, force, hessian = model.energy_force_hessian(
        positions, atomic_numbers, topology, create_parameter_graph=False
    )
    transformed_energy, transformed_force, transformed_hessian = (
        model.energy_force_hessian(
            transformed,
            atomic_numbers,
            topology,
            create_parameter_graph=False,
        )
    )
    coordinate_rotation = torch.kron(
        torch.eye(positions.shape[0], dtype=torch.float64), rotation
    )

    torch.testing.assert_close(transformed_energy, energy, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        transformed_force, force @ rotation.T, atol=1e-8, rtol=1e-7
    )
    torch.testing.assert_close(
        transformed_hessian,
        coordinate_rotation @ hessian @ coordinate_rotation.T,
        atol=2e-8,
        rtol=2e-7,
    )
