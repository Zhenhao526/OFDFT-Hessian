from __future__ import annotations

import torch

from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)
from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from scripts.qm9_complete_total_local_random_feature_kernel_ceiling import (
    _central_random_features,
    _random_feature_columns,
    _random_kernel_parameters,
)


def test_central_random_features_are_permutation_invariant() -> None:
    environment = torch.tensor(
        [[1.0, 0.2], [0.3, -0.1], [0.8, 0.4]], dtype=torch.float64
    )
    element_index = torch.tensor([0, 1, 0])
    projection = torch.tensor(
        [[0.5, -0.2, 0.7], [0.1, 0.4, -0.3]], dtype=torch.float64
    )
    kwargs = {
        "inverse_rms": torch.tensor([2.0, 0.5], dtype=torch.float64),
        "projection": projection,
        "bias": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        "activation_scale": torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64),
        "central_element_indices": (0, 1),
    }
    reference = _central_random_features(environment, element_index, **kwargs)
    permutation = torch.tensor([2, 0, 1])
    actual = _central_random_features(
        environment[permutation], element_index[permutation], **kwargs
    )
    assert torch.allclose(reference, actual, atol=1e-14, rtol=1e-14)


def test_random_feature_columns_are_element_resolved_and_disjoint() -> None:
    columns = _random_feature_columns(
        base_feature_count=11,
        width=3,
        central_element_indices=(0, 2),
        device=torch.device("cpu"),
    )
    assert torch.equal(columns, torch.tensor([11, 12, 13, 17, 18, 19]))


def test_random_kernel_scalar_has_finite_rigid_invariant_hessian() -> None:
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
    topology = build_body_order_topology(atomic_numbers)
    model = LocalAngularScalarResidual(
        hidden_size=8, radial_size=3, angular_order=2, seed=17
    )
    element_index = model._element_index(atomic_numbers)
    with torch.no_grad():
        atomic = model.network.atomic_features(positions, element_index, topology)
    environment_size = atomic.shape[1] - model.network.element_count
    projection, bias, activation_scale = _random_kernel_parameters(
        input_size=environment_size,
        active_input_count=environment_size,
        width_per_scale=3,
        scales=(0.5, 1.0),
        seed=23,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    inverse_rms = torch.ones(environment_size, dtype=torch.float64)
    coefficients = torch.linspace(-0.2, 0.3, 18, dtype=torch.float64)

    def energy(candidate: torch.Tensor) -> torch.Tensor:
        candidate_atomic = model.network.atomic_features(
            candidate, element_index, topology
        )
        features = _central_random_features(
            candidate_atomic[:, model.network.element_count :],
            element_index,
            inverse_rms=inverse_rms,
            projection=projection,
            bias=bias,
            activation_scale=activation_scale,
            central_element_indices=(0, 1, 3),
        )
        return torch.dot(features, coefficients)

    reference = energy(positions)
    angle = torch.tensor(0.41, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), angle * 0.0)),
            torch.stack((torch.sin(angle), torch.cos(angle), angle * 0.0)),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    moved = positions @ rotation.T + torch.tensor([0.2, -0.7, 0.4])
    hessian = torch.func.hessian(energy)(positions).reshape(
        positions.numel(), positions.numel()
    )

    assert torch.allclose(reference, energy(moved), atol=1e-11, rtol=1e-11)
    assert torch.all(torch.isfinite(hessian))
    assert torch.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10)
