from __future__ import annotations

import torch

from mldft.ml.models.components.local_quadratic_residual import (
    LocalQuadraticCoefficientNetwork,
    anchored_quadratic_energy,
    assemble_network_hessian,
    assemble_quadratic_hessian,
    build_cross_coordinate_pairs,
    build_internal_coordinate_specs,
    build_quadratic_terms,
    coordinate_feature_matrix,
    cross_coordinate_feature_matrix,
    global_cross_coordinate_feature_matrix,
    internal_coordinate_values_and_jacobian,
    network_anchored_quadratic_energy,
    network_hvp,
)


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([1, 6, 6, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [4.6, 0.3, 0.2], [6.2, 1.0, 0.8]],
        dtype=torch.float64,
    )
    return atomic_numbers, positions


def test_assembled_quadratic_hessian_matches_scalar_autograd() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions, bond_scale=1.25)
    values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
    terms = build_quadratic_terms(specs, values, include_cross_terms=True)
    coefficients = {
        term.key: torch.tensor((index % 7 - 3) * 0.002, dtype=torch.float64)
        for index, term in enumerate(terms)
    }
    expected = assemble_quadratic_hessian(jacobian, terms, coefficients)
    actual = torch.func.hessian(
        lambda geometry: anchored_quadratic_energy(
            geometry, values, specs, terms, coefficients
        )
    )(positions).reshape(positions.numel(), positions.numel())

    assert torch.all(torch.isfinite(actual))
    assert torch.max(torch.abs(actual - actual.T)) < 1e-11
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-10)


def test_quadratic_scalar_is_exactly_energy_force_anchored() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, _ = internal_coordinate_values_and_jacobian(positions, specs)
    terms = build_quadratic_terms(specs, values, include_cross_terms=False)
    coefficients = {term.key: 0.01 for term in terms}
    geometry = positions.detach().requires_grad_(True)
    energy = anchored_quadratic_energy(
        geometry, values, specs, terms, coefficients
    )
    force = -torch.autograd.grad(energy, geometry)[0]

    assert float(energy) == 0.0
    assert torch.count_nonzero(force) == 0


def test_environment_types_and_full_cross_expand_the_feature_space() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, _ = internal_coordinate_values_and_jacobian(positions, specs)
    local = build_quadratic_terms(
        specs,
        values,
        include_cross_terms=True,
        use_environment_types=False,
    )
    contextual = build_quadratic_terms(
        specs,
        values,
        include_cross_terms=True,
        use_environment_types=True,
    )
    full = build_quadratic_terms(
        specs,
        values,
        include_cross_terms=True,
        cross_all_pairs=True,
        use_environment_types=True,
    )

    assert {term.key for term in local} != {term.key for term in contextual}
    assert len(full) >= len(contextual)


def test_network_hvp_matches_assembled_symmetric_hessian() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
    coordinate_features = coordinate_feature_matrix(specs, values, jacobian)
    first, second, overlap = build_cross_coordinate_pairs(specs)
    cross_features = cross_coordinate_feature_matrix(
        coordinate_features, jacobian, first, second, overlap, specs
    )
    model = LocalQuadraticCoefficientNetwork(
        coordinate_features.shape[1], cross_features.shape[1], seed=43
    )
    with torch.no_grad():
        model.diagonal[-1].weight.fill_(0.002)
        model.cross[-1].weight.fill_(-0.001)
    diagonal, cross = model(coordinate_features, cross_features)
    hessian = assemble_network_hessian(jacobian, diagonal, cross, first, second)
    directions = torch.randn(3, positions.numel(), dtype=torch.float64)
    products = network_hvp(
        jacobian, directions, diagonal, cross, first, second
    )

    assert torch.max(torch.abs(hessian - hessian.T)) < 1e-11
    assert torch.allclose(products, directions @ hessian.T, atol=1e-11, rtol=1e-10)


def test_initial_function_subtraction_preserves_zero_but_activates_hidden_gradients() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
    coordinate_features = coordinate_feature_matrix(specs, values, jacobian)
    first, second, overlap = build_cross_coordinate_pairs(specs)
    cross_features = cross_coordinate_feature_matrix(
        coordinate_features, jacobian, first, second, overlap, specs
    )
    model = LocalQuadraticCoefficientNetwork(
        coordinate_features.shape[1],
        cross_features.shape[1],
        seed=47,
        zero_output=False,
    )
    initial_diagonal, initial_cross = model(coordinate_features, cross_features)
    diagonal, cross = model(coordinate_features, cross_features)
    diagonal = diagonal - initial_diagonal.detach()
    cross = cross - initial_cross.detach()
    assert torch.count_nonzero(diagonal) == 0
    assert torch.count_nonzero(cross) == 0

    target = torch.ones_like(diagonal) * 0.01
    loss = torch.mean((diagonal - target) ** 2) + torch.mean(cross.square())
    loss.backward()

    assert model.diagonal[0].weight.grad is not None
    assert torch.linalg.vector_norm(model.diagonal[0].weight.grad) > 0.0


def test_global_cross_features_are_rigid_motion_invariant_and_cover_all_pairs() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
    coordinate_features = coordinate_feature_matrix(specs, values, jacobian)
    local_first, _, _ = build_cross_coordinate_pairs(specs, local_only=True)
    first, second, overlap = build_cross_coordinate_pairs(specs, local_only=False)
    features = global_cross_coordinate_feature_matrix(
        coordinate_features,
        jacobian,
        first,
        second,
        overlap,
        specs,
        positions,
    )

    angle = torch.tensor(0.47, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), angle * 0.0)),
            torch.stack((torch.sin(angle), torch.cos(angle), angle * 0.0)),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    transformed = positions @ rotation.T + torch.tensor(
        [1.7, -0.4, 2.1], dtype=torch.float64
    )
    transformed_specs = build_internal_coordinate_specs(atomic_numbers, transformed)
    transformed_values, transformed_jacobian = internal_coordinate_values_and_jacobian(
        transformed, transformed_specs
    )
    transformed_coordinate_features = coordinate_feature_matrix(
        transformed_specs, transformed_values, transformed_jacobian
    )
    transformed_features = global_cross_coordinate_feature_matrix(
        transformed_coordinate_features,
        transformed_jacobian,
        first,
        second,
        overlap,
        transformed_specs,
        transformed,
    )

    assert first.numel() > local_first.numel()
    assert torch.all(torch.isfinite(features))
    assert torch.allclose(features, transformed_features, atol=1e-11, rtol=1e-10)


def test_network_assembled_hessian_is_exact_scalar_second_derivative() -> None:
    atomic_numbers, positions = _geometry()
    specs = build_internal_coordinate_specs(atomic_numbers, positions)
    values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
    first, second, _ = build_cross_coordinate_pairs(specs, local_only=False)
    generator = torch.Generator().manual_seed(53)
    diagonal = torch.randn(len(specs), dtype=torch.float64, generator=generator) * 0.01
    cross = torch.randn(first.numel(), dtype=torch.float64, generator=generator) * 0.001
    expected = assemble_network_hessian(
        jacobian, diagonal, cross, first, second
    )
    actual = torch.func.hessian(
        lambda geometry: network_anchored_quadratic_energy(
            geometry,
            values,
            specs,
            diagonal,
            cross,
            first,
            second,
        )
    )(positions).reshape(positions.numel(), positions.numel())
    geometry = positions.detach().requires_grad_(True)
    energy = network_anchored_quadratic_energy(
        geometry, values, specs, diagonal, cross, first, second
    )
    force = -torch.autograd.grad(energy, geometry)[0]

    assert float(energy) == 0.0
    assert torch.count_nonzero(force) == 0
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-10)
