#!/usr/bin/env python3
"""Audit whether stable5 E/F/H corrections are jets of a rigid-invariant scalar."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
    _load_protocol,
    _load_selected_parents,
    _sha256,
)


_ROTATION_GENERATORS = np.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    dtype=np.float64,
)


def _symmetric_basis(dimension: int) -> np.ndarray:
    """Return a Frobenius-orthonormal basis for symmetric matrices."""
    basis = []
    for row in range(dimension):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        matrix[row, row] = 1.0
        basis.append(matrix)
        for column in range(row + 1, dimension):
            matrix = np.zeros((dimension, dimension), dtype=np.float64)
            matrix[row, column] = 1.0 / np.sqrt(2.0)
            matrix[column, row] = 1.0 / np.sqrt(2.0)
            basis.append(matrix)
    return np.stack(basis)


def _rigid_vectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = positions - positions.mean(axis=0, keepdims=True)
    atom_count = positions.shape[0]
    translations = []
    for axis in range(3):
        vector = np.zeros((atom_count, 3), dtype=np.float64)
        vector[:, axis] = 1.0
        translations.append(vector.reshape(-1))
    rotations = [
        (centered @ generator.T).reshape(-1)
        for generator in _ROTATION_GENERATORS
    ]
    return np.stack(translations, axis=1), np.stack(rotations, axis=1)


def _rotation_action(generator: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return (vector.reshape(-1, 3) @ generator.T).reshape(-1)


def _constraint_matrix(
    positions: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    translations, rotations = _rigid_vectors(positions)
    hessian_blocks = []
    gradient_blocks = []
    dimension = positions.size
    for vector in translations.T:
        hessian_blocks.append(np.einsum("pij,j->ip", basis, vector))
        gradient_blocks.append(np.zeros((dimension, dimension), dtype=np.float64))
    for generator, vector in zip(_ROTATION_GENERATORS, rotations.T, strict=True):
        hessian_blocks.append(np.einsum("pij,j->ip", basis, vector))
        action = np.kron(np.eye(positions.shape[0]), generator)
        gradient_blocks.append(-action)
    rigid = np.concatenate((translations, rotations), axis=1)
    hessian_constraints = np.concatenate(hessian_blocks, axis=0)
    gradient_constraints = np.concatenate(gradient_blocks, axis=0)
    gradient_invariance = rigid.T
    full = np.block(
        [
            [hessian_constraints, gradient_constraints],
            [np.zeros((rigid.shape[1], basis.shape[0])), gradient_invariance],
        ]
    )
    return hessian_constraints, gradient_blocks, full


def _symmetric_coefficients(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("pij,ij->p", basis, 0.5 * (matrix + matrix.T))


def _matrix_from_coefficients(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("p,pij->ij", coefficients, basis)


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def audit_scalar_jet(
    positions: np.ndarray,
    force: np.ndarray,
    hessian: np.ndarray,
    *,
    normalization_hessian: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure first/second-order rigid-invariance residuals for one scalar jet."""
    positions = np.asarray(positions, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64).reshape(-1, 3)
    hessian = np.asarray(hessian, dtype=np.float64)
    dimension = positions.size
    if hessian.shape != (dimension, dimension):
        raise ValueError("Hessian shape does not match positions")
    basis = _symmetric_basis(dimension)
    coefficients = _symmetric_coefficients(hessian, basis)
    symmetric_hessian = _matrix_from_coefficients(coefficients, basis)
    gradient = -force.reshape(-1)
    translations, rotations = _rigid_vectors(positions)
    rigid = np.concatenate((translations, rotations), axis=1)
    q, singular, _ = np.linalg.svd(rigid, full_matrices=False)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    rigid_basis = q[:, :rank]
    invariant_gradient = gradient - rigid_basis @ (rigid_basis.T @ gradient)

    hessian_constraints, gradient_blocks, full_constraints = _constraint_matrix(
        positions, basis
    )
    rotation_rhs = [
        _rotation_action(generator, invariant_gradient)
        for generator in _ROTATION_GENERATORS
    ]
    exact_force_rhs = np.concatenate(
        [
            np.zeros(3 * dimension, dtype=np.float64),
            *rotation_rhs,
        ]
    )
    correction, *_ = np.linalg.lstsq(
        hessian_constraints,
        exact_force_rhs - hessian_constraints @ coefficients,
        rcond=1e-12,
    )
    exact_force_coefficients = coefficients + correction
    exact_force_constraint_residual = (
        hessian_constraints @ exact_force_coefficients - exact_force_rhs
    )

    _, singular_full, vh = np.linalg.svd(full_constraints, full_matrices=True)
    full_rank = int(
        np.sum(singular_full > 1e-10 * max(singular_full[0], np.finfo(float).tiny))
    )
    null = vh[full_rank:].T
    hessian_null = null[: basis.shape[0]]
    if hessian_null.size:
        u, singular_hessian, _ = np.linalg.svd(hessian_null, full_matrices=False)
        hessian_rank = int(
            np.sum(
                singular_hessian
                > 1e-10 * max(singular_hessian[0], np.finfo(float).tiny)
            )
        )
        admissible_basis = u[:, :hessian_rank]
        free_gradient_coefficients = admissible_basis @ (
            admissible_basis.T @ coefficients
        )
    else:
        hessian_rank = 0
        free_gradient_coefficients = np.zeros_like(coefficients)

    normalizer_matrix = (
        symmetric_hessian
        if normalization_hessian is None
        else np.asarray(normalization_hessian, dtype=np.float64)
    )
    normalizer = max(np.linalg.norm(normalizer_matrix), np.finfo(float).tiny)
    exact_force_projection = _matrix_from_coefficients(
        exact_force_coefficients, basis
    )
    free_gradient_projection = _matrix_from_coefficients(
        free_gradient_coefficients, basis
    )
    translation_response = symmetric_hessian @ translations
    expected_rotation_response = np.stack(
        [
            _rotation_action(generator, gradient)
            for generator in _ROTATION_GENERATORS
        ],
        axis=1,
    )
    actual_rotation_response = symmetric_hessian @ rotations
    response_denominator = max(
        np.linalg.norm(actual_rotation_response),
        np.linalg.norm(expected_rotation_response),
        np.finfo(float).tiny,
    )
    force_denominator = max(np.linalg.norm(gradient), np.finfo(float).tiny)
    centered = positions - positions.mean(axis=0, keepdims=True)
    net_force = np.sum(force, axis=0)
    net_torque = np.sum(np.cross(centered, force), axis=0)
    position_force_scale = max(
        np.linalg.norm(centered) * np.linalg.norm(force), np.finfo(float).tiny
    )
    return {
        "dimension": float(dimension),
        "rigid_rank": float(rank),
        "admissible_hessian_rank": float(hessian_rank),
        "force_rigid_projection_relative": float(
            np.linalg.norm(gradient - invariant_gradient) / force_denominator
        ),
        "net_force_relative": float(np.linalg.norm(net_force) / force_denominator),
        "net_torque_relative": float(np.linalg.norm(net_torque) / position_force_scale),
        "translation_hessian_response_relative": float(
            np.linalg.norm(translation_response) / normalizer
        ),
        "rotation_jet_response_relative": float(
            np.linalg.norm(actual_rotation_response - expected_rotation_response)
            / response_denominator
        ),
        "exact_invariant_force_hessian_floor_relative_frobenius": float(
            np.linalg.norm(symmetric_hessian - exact_force_projection) / normalizer
        ),
        "exact_invariant_force_constraint_residual": float(
            np.linalg.norm(exact_force_constraint_residual)
        ),
        "free_invariant_force_hessian_floor_relative_frobenius": float(
            np.linalg.norm(symmetric_hessian - free_gradient_projection) / normalizer
        ),
        "antisymmetric_over_symmetric_frobenius": float(
            np.linalg.norm(0.5 * (hessian - hessian.T))
            / max(np.linalg.norm(symmetric_hessian), np.finfo(float).tiny)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, _ = _load_protocol(args.protocol, args.arm_id, "smoke")
    parents, provenance = _load_selected_parents(protocol, torch.device("cpu"))
    rows: list[dict[str, Any]] = []
    for parent in parents:
        positions = parent.positions.detach().cpu().numpy()
        correction_force = parent.pbe_force - parent.source_force
        correction_hessian = parent.pbe_hessian - parent.source_hessian_symmetric
        row: dict[str, Any] = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
        }
        for prefix, force, hessian in (
            ("pbe", parent.pbe_force, parent.pbe_hessian),
            ("source", parent.source_force, parent.source_hessian_symmetric),
            ("correction", correction_force, correction_hessian),
        ):
            metrics = audit_scalar_jet(
                positions,
                force,
                hessian,
                normalization_hessian=parent.pbe_hessian,
            )
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        rows.append(row)

    summary_keys = [
        key
        for key in rows[0]
        if key not in {"molecule_id", "natoms", "dimension", "rigid_rank"}
        and isinstance(rows[0][key], (int, float))
    ]
    result = {
        "definition": {
            "force": "F=-dE/dR",
            "translation": "H*T=0",
            "rotation": "H*(Omega*R)=Omega*dE/dR=-Omega*F",
            "exact_force_floor": "nearest symmetric Hessian satisfying rigid invariance for the rigid-projected supplied force",
            "free_force_floor": "nearest Hessian that is compatible with some rigid-invariant scalar gradient",
        },
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id_for_loading_only": args.arm_id,
        "parents": [parent.molecule_id for parent in parents],
        "per_parent": rows,
        "distributions": {key: _distribution(rows, key) for key in summary_keys},
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    rows_path = args.output_dir / "per_parent_metrics.csv"
    _write_csv(rows_path, rows)
    result["artifacts"] = {
        "per_parent_metrics_csv": rows_path.resolve().as_posix(),
        "per_parent_metrics_csv_sha256": _sha256(rows_path),
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
