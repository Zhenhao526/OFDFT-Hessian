"""Reference-anchored scalar residuals built from equivariant Hessian operators."""

from __future__ import annotations

import torch


def internal_projector(
    coordinate_count: int,
    external_basis: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    external = torch.as_tensor(external_basis, dtype=dtype)
    if external.ndim != 2 or external.shape[0] != coordinate_count:
        raise ValueError("external_basis must have shape (coordinates, external_modes)")
    gram = external.T @ external
    identity = torch.eye(gram.shape[0], dtype=dtype, device=external.device)
    if not torch.allclose(gram, identity, atol=1e-9, rtol=1e-9):
        external = torch.linalg.qr(external, mode="reduced").Q
    return torch.eye(coordinate_count, dtype=dtype, device=external.device) - external @ external.T


def spectral_operator_basis(
    source_hessian: torch.Tensor,
    external_basis: torch.Tensor,
    *,
    polynomial_degree: int = 5,
    rbf_centers: tuple[float, ...] = (-0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5),
    rbf_width: float = 0.5,
) -> tuple[torch.Tensor, list[str], torch.Tensor, float]:
    """Build matrix-function bases that transform equivariantly with the source Hessian."""
    source = torch.as_tensor(source_hessian, dtype=torch.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("source_hessian must be square")
    if polynomial_degree < 0 or rbf_width <= 0.0:
        raise ValueError("invalid spectral basis controls")
    projector = internal_projector(source.shape[0], external_basis, dtype=source.dtype)
    symmetric = projector @ (0.5 * (source + source.T)) @ projector
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    internal_rank = max(source.shape[0] - external_basis.shape[1], 1)
    scale_tensor = torch.linalg.matrix_norm(symmetric) / internal_rank**0.5
    scale = max(float(scale_tensor), torch.finfo(source.dtype).tiny)
    normalized = eigenvalues / scale

    values = []
    names = []
    for degree in range(polynomial_degree + 1):
        values.append(scale * normalized.pow(degree))
        names.append(f"power_{degree}")
    for center in rbf_centers:
        values.append(scale * torch.exp(-0.5 * ((normalized - center) / rbf_width) ** 2))
        names.append(f"rbf_{center:g}")
    spectral_values = torch.stack(values)
    matrices = torch.einsum(
        "ik,mk,jk->mij", eigenvectors, spectral_values, eigenvectors
    )
    matrices = torch.einsum("ij,mjk,kl->mil", projector, matrices, projector)
    return matrices, names, projector, scale


def equivariant_block_operator_basis(
    source_hessian: torch.Tensor,
    atomic_numbers: torch.Tensor,
    positions_bohr: torch.Tensor,
    external_basis: torch.Tensor,
    *,
    elements: tuple[int, ...] = (1, 6, 7, 8, 9),
    radial_centers: tuple[float, ...] = (1.5, 3.0, 5.0, 8.0),
    radial_width: float = 1.5,
) -> tuple[torch.Tensor, list[str]]:
    """Build shared equivariant atom-block operators that can rotate source modes."""
    source = torch.as_tensor(source_hessian, dtype=torch.float64)
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
    positions = torch.as_tensor(positions_bohr, dtype=source.dtype)
    atom_count = int(atomic_numbers.numel())
    if source.shape != (3 * atom_count, 3 * atom_count):
        raise ValueError("source Hessian and atom count are inconsistent")
    if positions.shape != (atom_count, 3) or radial_width <= 0.0:
        raise ValueError("invalid positions or radial width")
    element_values = tuple(sorted(int(value) for value in elements))
    if not set(atomic_numbers.tolist()).issubset(element_values):
        raise ValueError("unsupported atomic number")
    pair_types = [
        (first, second)
        for first_index, first in enumerate(element_values)
        for second in element_values[first_index:]
    ]
    pair_lookup = {pair: index for index, pair in enumerate(pair_types)}
    pair_transform_names = (
        "source_sym",
        "source_trace_I",
        "source_longitudinal_I",
        "source_trace_uu",
        "source_longitudinal_uu",
        "source_projected_sym",
        "scale_I",
        "scale_uu",
    )
    radial = torch.as_tensor(radial_centers, dtype=source.dtype)
    pair_feature_count = len(pair_types) * radial.numel() * len(pair_transform_names)
    diagonal_transform_names = (
        "source_block",
        "source_trace_I",
        "source_trace_environment",
        "source_environment_sym",
        "scale_I",
        "scale_environment",
    )
    diagonal_feature_count = len(element_values) * len(diagonal_transform_names)
    matrices = torch.zeros(
        pair_feature_count + diagonal_feature_count,
        source.shape[0],
        source.shape[1],
        dtype=source.dtype,
    )
    names = [
        f"pair_{first}_{second}_r{float(center):g}_{transform}"
        for first, second in pair_types
        for center in radial
        for transform in pair_transform_names
    ] + [
        f"atom_{element}_{transform}"
        for element in element_values
        for transform in diagonal_transform_names
    ]
    symmetric_source = 0.5 * (source + source.T)
    projector = internal_projector(source.shape[0], external_basis, dtype=source.dtype)
    internal_rank = max(source.shape[0] - external_basis.shape[1], 1)
    source_scale = max(
        float(torch.linalg.matrix_norm(projector @ symmetric_source @ projector))
        / internal_rank**0.5,
        torch.finfo(source.dtype).tiny,
    )
    identity3 = torch.eye(3, dtype=source.dtype)
    environment = torch.zeros(atom_count, 3, 3, dtype=source.dtype)

    for first in range(atom_count):
        for second in range(first + 1, atom_count):
            vector = positions[second] - positions[first]
            distance = torch.linalg.vector_norm(vector)
            unit = vector / distance.clamp_min(torch.finfo(source.dtype).eps)
            longitudinal_projector = torch.outer(unit, unit)
            environment[first] += torch.exp(-distance / 4.0) * longitudinal_projector
            environment[second] += torch.exp(-distance / 4.0) * longitudinal_projector
            first_slice = slice(3 * first, 3 * first + 3)
            second_slice = slice(3 * second, 3 * second + 3)
            block = symmetric_source[first_slice, second_slice]
            block_symmetric = 0.5 * (block + block.T)
            trace = torch.trace(block)
            longitudinal = unit @ block @ unit
            projected = 0.5 * (
                longitudinal_projector @ block + block.T @ longitudinal_projector
            )
            transforms = (
                block_symmetric,
                trace * identity3,
                longitudinal * identity3,
                trace * longitudinal_projector,
                longitudinal * longitudinal_projector,
                projected,
                source_scale * identity3,
                source_scale * longitudinal_projector,
            )
            pair_type = tuple(sorted((int(atomic_numbers[first]), int(atomic_numbers[second]))))
            pair_index = pair_lookup[pair_type]
            radial_values = torch.exp(-0.5 * ((distance - radial) / radial_width) ** 2)
            for radial_index, radial_value in enumerate(radial_values):
                for transform_index, transform in enumerate(transforms):
                    feature_index = (
                        (pair_index * radial.numel() + radial_index)
                        * len(pair_transform_names)
                        + transform_index
                    )
                    weighted = radial_value * transform
                    matrices[feature_index, first_slice, first_slice] += weighted
                    matrices[feature_index, second_slice, second_slice] += weighted
                    matrices[feature_index, first_slice, second_slice] -= weighted
                    matrices[feature_index, second_slice, first_slice] -= weighted

    element_lookup = {value: index for index, value in enumerate(element_values)}
    diagonal_offset = pair_feature_count
    for atom in range(atom_count):
        atom_slice = slice(3 * atom, 3 * atom + 3)
        block = symmetric_source[atom_slice, atom_slice]
        moment = environment[atom]
        moment = moment / torch.linalg.matrix_norm(moment).clamp_min(
            torch.finfo(source.dtype).eps
        )
        trace = torch.trace(block)
        transforms = (
            block,
            trace * identity3,
            trace * moment,
            0.5 * (moment @ block + block @ moment),
            source_scale * identity3,
            source_scale * moment,
        )
        element_index = element_lookup[int(atomic_numbers[atom])]
        for transform_index, transform in enumerate(transforms):
            feature_index = (
                diagonal_offset
                + element_index * len(diagonal_transform_names)
                + transform_index
            )
            matrices[feature_index, atom_slice, atom_slice] += transform

    matrices = torch.einsum("ij,mjk,kl->mil", projector, matrices, projector)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    return matrices, names


def anchored_cartesian_quadratic_energy(
    positions_bohr: torch.Tensor,
    reference_positions_bohr: torch.Tensor,
    hessian_correction: torch.Tensor,
) -> torch.Tensor:
    """Scalar Taylor residual with exactly zero reference energy and force."""
    displacement = (positions_bohr - reference_positions_bohr).reshape(-1)
    symmetric = 0.5 * (hessian_correction + hessian_correction.T)
    return 0.5 * displacement @ symmetric @ displacement


def symmetric_hvp_interpolant(
    directions: torch.Tensor,
    target_hvp: torch.Tensor,
) -> torch.Tensor:
    """Minimum-rank symmetric operator that exactly matches orthonormal HVP labels."""
    directions = torch.as_tensor(directions, dtype=torch.float64)
    target_hvp = torch.as_tensor(target_hvp, dtype=directions.dtype)
    if directions.ndim != 2 or target_hvp.shape != directions.shape:
        raise ValueError("directions and target_hvp must share shape (directions, coordinates)")
    gram = directions @ directions.T
    identity = torch.eye(gram.shape[0], dtype=directions.dtype, device=directions.device)
    if not torch.allclose(gram, identity, atol=1e-8, rtol=1e-8):
        raise ValueError("directions must be orthonormal")
    vectors = directions.T
    actions = target_hvp.T
    consistency = vectors.T @ actions
    consistency_asymmetry = torch.max(torch.abs(consistency - consistency.T))
    consistency_scale = torch.max(torch.abs(consistency)).clamp_min(1.0)
    if consistency_asymmetry > 1e-7 * consistency_scale:
        raise ValueError("target HVPs are inconsistent with a symmetric operator")
    consistency = 0.5 * (consistency + consistency.T)
    result = (
        actions @ vectors.T
        + vectors @ actions.T
        - vectors @ consistency @ vectors.T
    )
    return 0.5 * (result + result.T)
