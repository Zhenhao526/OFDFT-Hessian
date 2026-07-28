#!/usr/bin/env python3
"""Chemistry-aware coefficient subspaces for conservative local scalar features."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def _orthonormal_dct_basis(size: int, mode_count: int) -> np.ndarray:
    if size <= 0 or not 0 < mode_count <= size:
        raise ValueError("DCT mode count must be in [1, size]")
    positions = np.arange(size, dtype=np.float64)[:, None]
    modes = np.arange(mode_count, dtype=np.float64)[None, :]
    basis = np.cos(math.pi * (positions + 0.5) * modes / size)
    basis[:, 0] *= math.sqrt(1.0 / size)
    if mode_count > 1:
        basis[:, 1:] *= math.sqrt(2.0 / size)
    return basis


def _environment_coefficient_basis(
    *,
    element_count: int,
    radial_size: int,
    angular_order: int,
    radial_mode_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Map local-environment feature coefficients to typed smooth radial modes."""
    if element_count <= 0 or angular_order < 0:
        raise ValueError("invalid environment dimensions")
    radial_basis = _orthonormal_dct_basis(radial_size, radial_mode_count)
    pair_count = element_count * (element_count + 1) // 2
    radial_parameter_count = element_count * radial_mode_count
    angular_parameter_count = (
        pair_count
        * radial_mode_count
        * radial_mode_count
        * (angular_order + 1)
    )
    feature_count = element_count * radial_size + (
        pair_count * radial_size * radial_size * (angular_order + 1)
    )
    parameter_count = radial_parameter_count + angular_parameter_count
    mapping = np.zeros((feature_count, parameter_count), dtype=np.float64)

    for neighbor in range(element_count):
        row_start = neighbor * radial_size
        column_start = neighbor * radial_mode_count
        mapping[
            row_start : row_start + radial_size,
            column_start : column_start + radial_mode_count,
        ] = radial_basis

    feature_offset = element_count * radial_size
    parameter_offset = radial_parameter_count
    for pair in range(pair_count):
        for first_radial in range(radial_size):
            for second_radial in range(radial_size):
                for order in range(angular_order + 1):
                    row = feature_offset + (
                        ((pair * radial_size + first_radial) * radial_size + second_radial)
                        * (angular_order + 1)
                        + order
                    )
                    for first_mode in range(radial_mode_count):
                        for second_mode in range(radial_mode_count):
                            column = parameter_offset + (
                                (
                                    (pair * radial_mode_count + first_mode)
                                    * radial_mode_count
                                    + second_mode
                                )
                                * (angular_order + 1)
                                + order
                            )
                            mapping[row, column] = (
                                radial_basis[first_radial, first_mode]
                                * radial_basis[second_radial, second_mode]
                            )
    return mapping, {
        "feature_count": feature_count,
        "parameter_count": parameter_count,
        "radial_parameter_count": radial_parameter_count,
        "angular_parameter_count": angular_parameter_count,
    }


def _append_dense_block(
    row_parts: list[np.ndarray],
    column_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    zero_tolerance: float = 0.0,
) -> None:
    if values.shape != (rows.size, columns.size):
        raise ValueError("structured mapping block shape mismatch")
    row_grid = np.repeat(rows, columns.size)
    column_grid = np.tile(columns, rows.size)
    flat_values = values.reshape(-1)
    selected = np.abs(flat_values) > zero_tolerance
    row_parts.append(row_grid[selected])
    column_parts.append(column_grid[selected])
    value_parts.append(flat_values[selected])


def build_structured_scalar_subspace_mapping(
    *,
    element_count: int,
    radial_size: int,
    angular_order: int,
    angular_feature_count: int,
    four_body_keys: tuple[tuple[int, ...], ...],
    four_body_center_count: int,
    angular_radial_modes: int,
    four_body_radial_modes: int,
    random_environment_radial_modes: int,
    random_bias_order: int,
    inverse_rms: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    activation_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build ``c = P beta`` using typed radial/angular/torsion coefficient fields."""
    tensors = (inverse_rms, projection, bias, activation_scale)
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("structured mapping metadata must be on CPU")
    inverse_rms_np = inverse_rms.detach().to(torch.float64).numpy()
    projection_np = projection.detach().to(torch.float64).numpy()
    bias_np = bias.detach().to(torch.float64).numpy()
    scale_np = activation_scale.detach().to(torch.float64).numpy()
    if projection_np.shape != (inverse_rms_np.size, bias_np.size):
        raise ValueError("random-kernel tensor shape mismatch")
    if scale_np.shape != bias_np.shape:
        raise ValueError("random activation-scale shape mismatch")
    if random_bias_order < 0:
        raise ValueError("random bias order must be non-negative")

    explicit_environment, explicit_metadata = _environment_coefficient_basis(
        element_count=element_count,
        radial_size=radial_size,
        angular_order=angular_order,
        radial_mode_count=angular_radial_modes,
    )
    expected_angular_count = element_count + (
        element_count * explicit_environment.shape[0]
    )
    if angular_feature_count != expected_angular_count:
        raise ValueError(
            f"angular feature schema mismatch: {angular_feature_count} != "
            f"{expected_angular_count}"
        )
    random_environment, random_environment_metadata = _environment_coefficient_basis(
        element_count=element_count,
        radial_size=radial_size,
        angular_order=angular_order,
        radial_mode_count=random_environment_radial_modes,
    )
    if random_environment.shape[0] != inverse_rms_np.size:
        raise ValueError("random environment feature schema mismatch")

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    parameter_cursor = 0

    count_rows = np.arange(element_count, dtype=np.int64)
    count_columns = np.arange(parameter_cursor, parameter_cursor + element_count)
    _append_dense_block(
        row_parts,
        column_parts,
        value_parts,
        rows=count_rows,
        columns=count_columns,
        values=np.eye(element_count, dtype=np.float64),
    )
    parameter_cursor += element_count

    explicit_environment_parameter_count = explicit_environment.shape[1]
    for central in range(element_count):
        row_start = element_count + central * explicit_environment.shape[0]
        column_start = parameter_cursor
        _append_dense_block(
            row_parts,
            column_parts,
            value_parts,
            rows=np.arange(
                row_start,
                row_start + explicit_environment.shape[0],
                dtype=np.int64,
            ),
            columns=np.arange(
                column_start,
                column_start + explicit_environment_parameter_count,
                dtype=np.int64,
            ),
            values=explicit_environment,
        )
        parameter_cursor += explicit_environment_parameter_count
    angular_parameter_count = parameter_cursor

    four_body_basis = _orthonormal_dct_basis(
        four_body_center_count, four_body_radial_modes
    )
    four_body_groups: dict[tuple[int, ...], list[tuple[int, tuple[int, int, int]]]] = (
        defaultdict(list)
    )
    for local_index, key in enumerate(four_body_keys):
        if len(key) != 8:
            raise ValueError("four-body key must contain four elements, three centers, order")
        centers = tuple(int(value) for value in key[4:7])
        if any(not 0 <= value < four_body_center_count for value in centers):
            raise ValueError("four-body center index is outside the frozen grid")
        group = (*key[:4], key[7])
        four_body_groups[group].append((local_index, centers))
    four_body_parameter_start = parameter_cursor
    four_body_block_size = four_body_radial_modes**3
    for group in sorted(four_body_groups):
        entries = four_body_groups[group]
        values = np.empty((len(entries), four_body_block_size), dtype=np.float64)
        for row_index, (_, centers) in enumerate(entries):
            first, second, third = centers
            values[row_index] = np.einsum(
                "a,b,c->abc",
                four_body_basis[first],
                four_body_basis[second],
                four_body_basis[third],
            ).reshape(-1)
        _append_dense_block(
            row_parts,
            column_parts,
            value_parts,
            rows=np.asarray(
                [angular_feature_count + index for index, _ in entries],
                dtype=np.int64,
            ),
            columns=np.arange(
                parameter_cursor,
                parameter_cursor + four_body_block_size,
                dtype=np.int64,
            ),
            values=values,
        )
        parameter_cursor += four_body_block_size
    four_body_parameter_count = parameter_cursor - four_body_parameter_start

    weighted_projection = inverse_rms_np[:, None] * projection_np
    unique_scales, scale_inverse = np.unique(scale_np, return_inverse=True)
    random_width = projection_np.shape[1]
    random_feature_start = angular_feature_count + len(four_body_keys)
    bias_coordinate = 2.0 * bias_np
    bias_basis = np.polynomial.chebyshev.chebvander(
        bias_coordinate, random_bias_order
    )
    random_descriptor_count = random_environment.shape[1] + bias_basis.shape[1]
    random_parameter_start = parameter_cursor
    random_blocks = []
    for central in range(element_count):
        for scale_index, scale in enumerate(unique_scales):
            neuron_indices = np.flatnonzero(scale_inverse == scale_index)
            descriptor = np.concatenate(
                (
                    weighted_projection[:, neuron_indices].T @ random_environment,
                    bias_basis[neuron_indices],
                ),
                axis=1,
            )
            _append_dense_block(
                row_parts,
                column_parts,
                value_parts,
                rows=(
                    random_feature_start
                    + central * random_width
                    + neuron_indices
                ).astype(np.int64),
                columns=np.arange(
                    parameter_cursor,
                    parameter_cursor + random_descriptor_count,
                    dtype=np.int64,
                ),
                values=descriptor,
                zero_tolerance=np.finfo(np.float64).tiny,
            )
            random_blocks.append(
                {
                    "central_element_index": central,
                    "activation_scale": float(scale),
                    "neuron_count": int(neuron_indices.size),
                    "parameter_count": random_descriptor_count,
                }
            )
            parameter_cursor += random_descriptor_count
    random_parameter_count = parameter_cursor - random_parameter_start

    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    columns = np.concatenate(column_parts).astype(np.int64, copy=False)
    values = np.concatenate(value_parts).astype(np.float64, copy=False)
    feature_count = random_feature_start + element_count * random_width
    indices = torch.from_numpy(np.stack((rows, columns)))
    value_tensor = torch.from_numpy(values)
    mapping = torch.sparse_coo_tensor(
        indices,
        value_tensor,
        size=(feature_count, parameter_cursor),
        dtype=torch.float64,
    ).coalesce()
    digest = hashlib.sha256()
    digest.update(mapping.indices().numpy().tobytes(order="C"))
    digest.update(mapping.values().numpy().tobytes(order="C"))
    digest.update(np.asarray(mapping.shape, dtype=np.int64).tobytes())
    metadata = {
        "definition": (
            "label-independent chemistry-aware typed DCT coefficient field with "
            "random-kernel projection metadata"
        ),
        "feature_count": feature_count,
        "subspace_dimension": parameter_cursor,
        "mapping_nonzero_count": int(mapping._nnz()),
        "mapping_sha256": digest.hexdigest(),
        "element_count": element_count,
        "radial_size": radial_size,
        "angular_order": angular_order,
        "angular_radial_modes": angular_radial_modes,
        "angular_parameter_count": angular_parameter_count,
        "four_body_center_count": four_body_center_count,
        "four_body_radial_modes": four_body_radial_modes,
        "four_body_group_count": len(four_body_groups),
        "four_body_parameter_count": four_body_parameter_count,
        "random_environment_radial_modes": random_environment_radial_modes,
        "random_environment_parameter_count": random_environment.shape[1],
        "random_bias_order": random_bias_order,
        "random_descriptor_count_per_block": random_descriptor_count,
        "random_parameter_count": random_parameter_count,
        "random_blocks": random_blocks,
        "explicit_environment": explicit_metadata,
        "random_environment": random_environment_metadata,
    }
    return mapping, metadata


def project_design_to_structured_subspace(
    design: torch.Tensor,
    mapping: torch.Tensor,
    *,
    row_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if design.device.type != "cpu" or design.dtype != torch.float64:
        raise ValueError("structured projection expects a float64 CPU design")
    if not mapping.is_sparse or mapping.dtype != torch.float64:
        raise ValueError("structured coefficient mapping must be sparse float64")
    if mapping.shape[0] != design.shape[1]:
        raise ValueError("structured mapping feature count mismatch")
    if row_chunk_size <= 0:
        raise ValueError("structured projection row chunk must be positive")
    transposed = mapping.transpose(0, 1).coalesce().to(device)
    projected = torch.empty(
        (design.shape[0], mapping.shape[1]), dtype=torch.float64
    )
    for start in range(0, design.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, design.shape[0])
        source = design[start:stop].to(device)
        output = torch.sparse.mm(transposed, source.T).T
        projected[start:stop] = output.cpu()
        del source, output
    return projected


def expand_structured_subspace_coefficients(
    subspace_coefficients: torch.Tensor,
    mapping: torch.Tensor,
) -> torch.Tensor:
    if not mapping.is_sparse or mapping.shape[1] != subspace_coefficients.numel():
        raise ValueError("structured coefficient expansion shape mismatch")
    return torch.sparse.mm(
        mapping,
        subspace_coefficients.to(dtype=torch.float64, device="cpu")[:, None],
    ).squeeze(1)
