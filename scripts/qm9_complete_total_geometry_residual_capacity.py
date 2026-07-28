#!/usr/bin/env python3
"""Audit scalar pair-RBF residual capacity without reading validation or Test100 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def pair_rbf_scalar_features(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    centers_bohr: np.ndarray,
    sigma_bohr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """Return scalar energy, force, and Hessian features for element-pair RBFs."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions_bohr = np.asarray(positions_bohr, dtype=np.float64)
    centers_bohr = np.asarray(centers_bohr, dtype=np.float64)
    if positions_bohr.shape != (atomic_numbers.size, 3):
        raise ValueError("positions must have shape (natoms, 3)")
    if sigma_bohr <= 0 or centers_bohr.ndim != 1 or centers_bohr.size == 0:
        raise ValueError("sigma and centers must define a non-empty positive RBF grid")

    pair_types = sorted(
        {
            tuple(sorted((int(atomic_numbers[i]), int(atomic_numbers[j]))))
            for i in range(atomic_numbers.size)
            for j in range(i + 1, atomic_numbers.size)
        }
    )
    feature_keys = [
        (first, second, float(center))
        for first, second in pair_types
        for center in centers_bohr
    ]
    key_to_index = {key: index for index, key in enumerate(feature_keys)}
    ncoordinate = 3 * atomic_numbers.size
    energies = np.zeros(len(feature_keys), dtype=np.float64)
    forces = np.zeros((ncoordinate, len(feature_keys)), dtype=np.float64)
    hessians = np.zeros(
        (ncoordinate, ncoordinate, len(feature_keys)), dtype=np.float64
    )
    identity = np.eye(3, dtype=np.float64)
    inverse_sigma_squared = 1.0 / sigma_bohr**2

    for i in range(atomic_numbers.size):
        for j in range(i + 1, atomic_numbers.size):
            pair_type = tuple(
                sorted((int(atomic_numbers[i]), int(atomic_numbers[j])))
            )
            displacement = positions_bohr[i] - positions_bohr[j]
            distance = float(np.linalg.norm(displacement))
            if distance <= np.finfo(float).eps:
                raise ValueError("coincident atoms are unsupported")
            unit = displacement / distance
            outer = np.outer(unit, unit)
            for center in centers_bohr:
                feature = key_to_index[(pair_type[0], pair_type[1], float(center))]
                offset = distance - center
                value = np.exp(-0.5 * offset**2 * inverse_sigma_squared)
                first_derivative = -offset * inverse_sigma_squared * value
                second_derivative = (
                    offset**2 * inverse_sigma_squared**2 - inverse_sigma_squared
                ) * value
                pair_hessian = (
                    (second_derivative - first_derivative / distance) * outer
                    + (first_derivative / distance) * identity
                )
                energies[feature] += value
                force = -first_derivative * unit
                forces[3 * i : 3 * i + 3, feature] += force
                forces[3 * j : 3 * j + 3, feature] -= force
                i_slice = slice(3 * i, 3 * i + 3)
                j_slice = slice(3 * j, 3 * j + 3)
                hessians[i_slice, i_slice, feature] += pair_hessian
                hessians[j_slice, j_slice, feature] += pair_hessian
                hessians[i_slice, j_slice, feature] -= pair_hessian
                hessians[j_slice, i_slice, feature] -= pair_hessian
    return energies, forces, hessians, feature_keys


def constrained_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    constraints: np.ndarray,
    relative_svd_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize ``||design @ x - target||`` subject to ``constraints @ x = 0``."""
    _, singular_values, right_vectors = np.linalg.svd(
        constraints, full_matrices=True
    )
    threshold = relative_svd_tolerance * max(
        constraints.shape
    ) * (singular_values[0] if singular_values.size else 1.0)
    rank = int(np.sum(singular_values > threshold))
    null_space = right_vectors[rank:].T
    if null_space.shape[1] == 0:
        raise ValueError("energy/force constraints leave no residual-kernel freedom")
    reduced_design = design @ null_space
    reduced_coefficients, _, reduced_rank, reduced_singular_values = np.linalg.lstsq(
        reduced_design, target, rcond=relative_svd_tolerance
    )
    coefficients = null_space @ reduced_coefficients
    return coefficients, {
        "constraint_rank": rank,
        "null_space_dimension": int(null_space.shape[1]),
        "reduced_design_rank": int(reduced_rank),
        "reduced_design_condition": (
            float(reduced_singular_values[0] / reduced_singular_values[-1])
            if reduced_singular_values.size and reduced_singular_values[-1] > 0
            else float("inf")
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("manifest does not certify frozen Test100")
    entries = {
        str(row["molecule_id"]): row for row in manifest["parents"]
    }
    if args.molecule_id not in entries:
        raise ValueError("molecule is absent from frozen Stage-1 manifest")
    entry = entries[args.molecule_id]
    label = zarr.open(entry["label_path"], mode="r")
    atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
    positions_bohr = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
    with np.load(args.capacity_array) as payload:
        prediction = np.asarray(payload["predicted_hessian"], dtype=np.float64)
        reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    reference = reference[:, : prediction.shape[1]]
    centers = np.linspace(args.center_min, args.center_max, args.center_count)
    energies, forces, hessians, feature_keys = pair_rbf_scalar_features(
        atomic_numbers,
        positions_bohr,
        centers,
        args.sigma,
    )
    design = hessians[:, : prediction.shape[1], :].reshape(
        -1, hessians.shape[-1]
    )
    target = (reference - prediction).reshape(-1)
    constraints = np.concatenate((energies[None, :], forces), axis=0)
    coefficients, solver = constrained_least_squares(
        design,
        target,
        constraints,
        relative_svd_tolerance=args.svd_tolerance,
    )
    correction = (design @ coefficients).reshape(prediction.shape)
    corrected = prediction + correction
    reference_norm = max(np.linalg.norm(reference), np.finfo(float).tiny)
    result = {
        "definition": (
            "train-only scalar pair-RBF geometry residual, constrained to zero anchor "
            "energy and force"
        ),
        "molecule_id": args.molecule_id,
        "natoms": int(atomic_numbers.size),
        "column_count": int(prediction.shape[1]),
        "feature_count": len(feature_keys),
        "pair_types": sorted({f"{a}-{b}" for a, b, _ in feature_keys}),
        "center_count": args.center_count,
        "center_range_bohr": [args.center_min, args.center_max],
        "sigma_bohr": args.sigma,
        "source_capacity_array": str(args.capacity_array.resolve()),
        "baseline_relative_frobenius": float(
            np.linalg.norm(prediction - reference) / reference_norm
        ),
        "corrected_relative_frobenius": float(
            np.linalg.norm(corrected - reference) / reference_norm
        ),
        "adapter_energy_abs": float(abs(energies @ coefficients)),
        "adapter_force_max_abs": float(np.max(np.abs(forces @ coefficients))),
        "adapter_hessian_symmetry_max_abs": float(
            np.max(
                np.abs(
                    np.tensordot(hessians, coefficients, axes=(2, 0))
                    - np.tensordot(hessians, coefficients, axes=(2, 0)).T
                )
            )
        ),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **solver,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "geometry_residual_capacity.npz",
        coefficients=coefficients,
        predicted_hessian=prediction,
        correction_hessian_columns=correction,
        corrected_hessian=corrected,
        pbe_hessian_columns=reference,
        centers_bohr=centers,
    )
    (args.output_dir / "geometry_residual_capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument("--capacity-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-min", type=float, default=0.5)
    parser.add_argument("--center-max", type=float, default=8.0)
    parser.add_argument("--center-count", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--svd-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
