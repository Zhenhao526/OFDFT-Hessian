#!/usr/bin/env python3
"""Freeze train-only Stage-2 parents and structured train/held-out HVP directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

try:
    from scripts.prepare_qm9_complete_total_capacity_stage1 import (
        _load_stability,
        _parent_ids,
        _read_reference,
        _sha256,
    )
except ModuleNotFoundError:
    from prepare_qm9_complete_total_capacity_stage1 import (
        _load_stability,
        _parent_ids,
        _read_reference,
        _sha256,
    )


BOHR_TO_ANGSTROM = 0.529177210903
QM9_COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}


def _external_basis(positions: np.ndarray) -> np.ndarray:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    vectors = []
    for axis in range(3):
        vector = np.zeros_like(positions)
        vector[:, axis] = 1.0
        vectors.append(vector.reshape(-1))
    for axis in np.eye(3):
        vectors.append(np.cross(axis[None, :], centered).reshape(-1))
    matrix = np.stack(vectors, axis=1)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return u[:, singular > threshold]


def _bond_graph(atomic_numbers: np.ndarray, positions_bohr: np.ndarray):
    positions = positions_bohr * BOHR_TO_ANGSTROM
    bonds = []
    for first in range(len(atomic_numbers)):
        for second in range(first + 1, len(atomic_numbers)):
            cutoff = 1.25 * (
                QM9_COVALENT_RADII_ANGSTROM[int(atomic_numbers[first])]
                + QM9_COVALENT_RADII_ANGSTROM[int(atomic_numbers[second])]
            )
            if np.linalg.norm(positions[first] - positions[second]) <= cutoff:
                bonds.append((first, second))
    adjacency = {index: [] for index in range(len(atomic_numbers))}
    for first, second in bonds:
        adjacency[first].append(second)
        adjacency[second].append(first)
    return bonds, adjacency


def _structured_candidates(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    pbe_hessian: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    bonds, adjacency = _bond_graph(atomic_numbers, positions)
    result: dict[str, list[np.ndarray]] = {
        "bond": [],
        "angle": [],
        "torsion": [],
        "low_mode": [],
        "random_internal": [],
    }
    for first, second in bonds:
        axis = positions[second] - positions[first]
        axis /= np.linalg.norm(axis)
        vector = np.zeros_like(positions)
        vector[first] = -axis
        vector[second] = axis
        result["bond"].append(vector.reshape(-1))
    for center, neighbors in adjacency.items():
        for left_index, left in enumerate(neighbors):
            for right in neighbors[left_index + 1 :]:
                u = positions[left] - positions[center]
                v = positions[right] - positions[center]
                u /= np.linalg.norm(u)
                v /= np.linalg.norm(v)
                left_tangent = v - np.dot(v, u) * u
                right_tangent = u - np.dot(u, v) * v
                vector = np.zeros_like(positions)
                vector[left] = left_tangent
                vector[right] = right_tangent
                vector[center] = -(left_tangent + right_tangent)
                result["angle"].append(vector.reshape(-1))
    seen_chains = set()
    for middle_left, middle_right in bonds:
        for left in adjacency[middle_left]:
            if left == middle_right:
                continue
            for right in adjacency[middle_right]:
                if right in (middle_left, left):
                    continue
                chain = (left, middle_left, middle_right, right)
                reverse = tuple(reversed(chain))
                canonical = min(chain, reverse)
                if canonical in seen_chains:
                    continue
                seen_chains.add(canonical)
                axis = positions[middle_right] - positions[middle_left]
                axis /= np.linalg.norm(axis)
                vector = np.zeros_like(positions)
                vector[left] = np.cross(axis, positions[left] - positions[middle_left])
                vector[right] = -np.cross(
                    axis, positions[right] - positions[middle_right]
                )
                result["torsion"].append(vector.reshape(-1))

    external = _external_basis(positions)
    projector = np.eye(positions.size) - external @ external.T
    reduced_hessian = projector @ (0.5 * (pbe_hessian + pbe_hessian.T)) @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
    external_overlap = np.linalg.norm(external.T @ eigenvectors, axis=0)
    internal = np.flatnonzero(external_overlap < 1.0e-7)
    internal = internal[np.argsort(np.abs(eigenvalues[internal]))]
    result["low_mode"] = [eigenvectors[:, index] for index in internal]
    return result


def build_direction_bank(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    pbe_hessian: np.ndarray,
    *,
    seed: int,
    maximum: int,
    structured_per_kind: int,
    heldout_fraction: float,
    minimum_random_fraction: float = 0.0,
) -> dict[str, Any]:
    if not 0.0 <= minimum_random_fraction < 1.0:
        raise ValueError("minimum_random_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    external = _external_basis(positions)
    internal_dimension = positions.size - external.shape[1]
    target_count = min(maximum, internal_dimension)
    minimum_random_count = int(math.ceil(minimum_random_fraction * target_count))
    structured_target = target_count - minimum_random_count
    candidates = _structured_candidates(atomic_numbers, positions, pbe_hessian)
    accepted: list[np.ndarray] = []
    kinds: list[str] = []

    def accept(vector: np.ndarray, kind: str) -> bool:
        projected = np.asarray(vector, dtype=np.float64).copy()
        # Near a complete internal basis, one modified Gram-Schmidt pass loses
        # orthogonality. A second pass keeps the frozen directions audit-grade.
        for _ in range(2):
            projected -= external @ (external.T @ projected)
            for previous in accepted:
                projected -= np.dot(previous, projected) * previous
        norm = np.linalg.norm(projected)
        if norm < 1.0e-9:
            return False
        accepted.append(projected / norm)
        kinds.append(kind)
        return True

    for kind in ("bond", "angle", "torsion", "low_mode"):
        local = candidates[kind]
        for index in rng.permutation(len(local))[:structured_per_kind]:
            if len(accepted) >= structured_target:
                break
            accept(local[int(index)], kind)
    attempts = 0
    while len(accepted) < target_count:
        if not accept(rng.normal(size=positions.size), "random_internal"):
            attempts += 1
            if attempts > 10 * positions.size:
                raise RuntimeError("could not complete internal direction bank")

    roles = np.full(len(accepted), "train", dtype="U7")
    for kind in sorted(set(kinds)):
        indices = np.flatnonzero(np.asarray(kinds) == kind)
        shuffled = rng.permutation(indices)
        heldout_count = int(round(heldout_fraction * len(indices)))
        if len(indices) >= 2:
            heldout_count = min(len(indices) - 1, max(1, heldout_count))
        else:
            heldout_count = 0
        roles[shuffled[:heldout_count]] = "heldout"
    if not np.any(roles == "heldout"):
        roles[rng.integers(len(roles))] = "heldout"
    directions = np.stack(accepted)
    return {
        "directions": directions,
        "roles": roles,
        "kinds": np.asarray(kinds, dtype="U16"),
        "external_basis": external,
        "orthonormality_max_abs": float(
            np.max(np.abs(directions @ directions.T - np.eye(len(directions))))
        ),
        "external_overlap_max_abs": float(np.max(np.abs(directions @ external))),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Protocol must explicitly freeze Test100")
    split_path = args.dataset_dir / "split.pkl"
    if _sha256(split_path) != str(protocol["data"]["source_split_sha256"]):
        raise ValueError("source split hash mismatch")
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    memberships = {
        name: _parent_ids(split[name]) for name in ("train", "val", "test")
    }
    stability = _load_stability(args.stability_csv)
    direction_config = protocol["directions"]
    sample_id = int(protocol["data"]["sample_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction_dir = args.output_dir / "directions"
    direction_dir.mkdir(exist_ok=True)
    candidates = []
    direction_rows = []
    for order, entry in enumerate(protocol["data"]["parent_candidates"]):
        molecule_id = str(entry["molecule_id"])
        parent_membership = [name for name, ids in memberships.items() if molecule_id in ids]
        if parent_membership != ["train"]:
            raise ValueError(f"candidate {molecule_id} is not train-only: {parent_membership}")
        stability_row = stability.get(molecule_id)
        if stability_row is None or stability_row["parent_stable"].lower() != "true":
            raise ValueError(f"candidate {molecule_id} is not parent-stable")
        reference = _read_reference(
            args.dataset_dir, args.reference_dir, molecule_id, sample_id
        )
        if reference["natoms"] != int(entry["natoms"]):
            raise ValueError(f"natoms mismatch for {molecule_id}")
        label = zarr.open(reference["label_path"], mode="r")
        positions = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
        atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
        with np.load(reference["pbe_hessian_path"]) as payload:
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        bank = build_direction_bank(
            atomic_numbers,
            positions,
            pbe_hessian,
            seed=int(direction_config["seed"]) + order,
            maximum=int(direction_config["maximum_per_parent"]),
            structured_per_kind=int(direction_config["structured_per_kind"]),
            heldout_fraction=float(direction_config["heldout_fraction"]),
            minimum_random_fraction=float(
                direction_config.get("minimum_random_fraction", 0.0)
            ),
        )
        train_directions = bank["directions"][bank["roles"] == "train"]
        heldout_directions = bank["directions"][bank["roles"] == "heldout"]
        train_projector = train_directions.T @ train_directions
        complement = np.eye(train_projector.shape[0]) - train_projector
        unidentified_hessian = complement @ pbe_hessian @ complement
        unidentified_heldout_hvp = np.einsum(
            "ij,dj->di", unidentified_hessian, heldout_directions
        )
        reference_heldout_hvp = np.einsum(
            "ij,dj->di", pbe_hessian, heldout_directions
        )
        external_dimension = int(bank["external_basis"].shape[1])
        internal_dimension = int(positions.size - external_dimension)
        direction_path = direction_dir / f"{molecule_id}.npz"
        np.savez_compressed(
            direction_path,
            directions=bank["directions"],
            roles=bank["roles"],
            kinds=bank["kinds"],
            external_basis=bank["external_basis"],
            pbe_hvp=np.einsum("ij,dj->di", pbe_hessian, bank["directions"]),
        )
        candidates.append(
            {
                **reference,
                "candidate_order": order,
                "stability_direction_count": int(stability_row["direction_count"]),
                "stability_passing_direction_count": int(
                    stability_row["stable_direction_count"]
                ),
            }
        )
        direction_rows.append(
            {
                "molecule_id": molecule_id,
                "direction_path": direction_path.resolve().as_posix(),
                "direction_sha256": _sha256(direction_path),
                "direction_count": int(len(bank["directions"])),
                "train_direction_count": int(np.sum(bank["roles"] == "train")),
                "heldout_direction_count": int(np.sum(bank["roles"] == "heldout")),
                "random_direction_count": int(
                    np.sum(bank["kinds"] == "random_internal")
                ),
                "external_dimension": external_dimension,
                "internal_dimension": internal_dimension,
                "train_fraction_of_internal_dimension": float(
                    len(train_directions) / internal_dimension
                ),
                "unidentified_qhq_relative_frobenius": float(
                    np.linalg.norm(unidentified_hessian)
                    / max(np.linalg.norm(pbe_hessian), np.finfo(float).tiny)
                ),
                "minimum_norm_oracle_heldout_hvp_relative_frobenius": float(
                    np.linalg.norm(unidentified_heldout_hvp)
                    / max(
                        np.linalg.norm(reference_heldout_hvp), np.finfo(float).tiny
                    )
                ),
                "kind_counts": {
                    kind: int(np.sum(bank["kinds"] == kind))
                    for kind in sorted(set(bank["kinds"].tolist()))
                },
                "orthonormality_max_abs": bank["orthonormality_max_abs"],
                "external_overlap_max_abs": bank["external_overlap_max_abs"],
            }
        )
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_split_sha256": _sha256(split_path),
        "stability_csv": args.stability_csv.resolve().as_posix(),
        "stability_sha256": _sha256(args.stability_csv),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "candidate_count": len(candidates),
        "parent_count": len(candidates),
        "final_parent_count": int(protocol["data"]["final_parent_count"]),
        "numerical_gate": protocol["data"]["numerical_gate"],
        "candidates": candidates,
        "parents": candidates,
        "directions": direction_rows,
    }
    output = args.output_dir / "candidate_direction_manifest.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    output.with_suffix(".json.sha256").write_text(f"{digest}  {output.name}\n")
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "manifest": output.as_posix(),
                "manifest_sha256": digest,
                "test100_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
