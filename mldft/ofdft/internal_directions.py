"""Deterministic structured Cartesian bases with rigid motions removed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
QM9_COVALENT_RADII_ANGSTROM = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
}


@dataclass(frozen=True)
class InternalDirectionBank:
    """One complete orthonormal internal Cartesian basis."""

    directions: np.ndarray
    kinds: np.ndarray
    partial_roles: np.ndarray
    external_basis: np.ndarray
    projector: np.ndarray
    external_rank: int
    internal_dimension: int
    orthonormality_max_abs: float
    external_overlap_max_abs: float
    projector_idempotence_max_abs: float


@dataclass(frozen=True)
class InternalHutchinsonSample:
    """One unnormalized Rademacher probe in a complete internal basis.

    If the rows of ``B`` are an orthonormal basis for the internal Cartesian
    subspace and ``z`` contains independent Rademacher signs, ``vector`` is
    ``B.T @ z``.  It deliberately has norm ``sqrt(internal_dimension)``:

    ``E_z ||(H_pred - H_ref) B.T z||^2 = ||(H_pred - H_ref) B.T||_F^2``.
    """

    vector: np.ndarray
    signs: np.ndarray
    internal_dimension: int
    seed: int
    squared_norm: float
    external_overlap_max_abs: float


def sample_internal_rademacher_direction(
    bank: InternalDirectionBank,
    *,
    seed: int,
) -> InternalHutchinsonSample:
    """Sample one deterministic Hutchinson probe from a complete internal basis."""
    if bank.directions.shape[0] != bank.internal_dimension:
        raise ValueError(
            "Hutchinson sampling requires the complete internal direction bank"
        )
    if bank.directions.ndim != 2:
        raise ValueError("bank directions must be a two-dimensional matrix")
    generator = np.random.default_rng(seed)
    signs = generator.integers(
        0, 2, size=bank.internal_dimension, dtype=np.int8
    )
    signs = (2 * signs - 1).astype(np.float64)
    vector = signs @ bank.directions
    squared_norm = float(np.dot(vector, vector))
    expected_squared_norm = float(bank.internal_dimension)
    tolerance = 1.0e-10 * max(1.0, expected_squared_norm)
    if abs(squared_norm - expected_squared_norm) > tolerance:
        raise RuntimeError(
            "internal Hutchinson direction norm drift: "
            f"{squared_norm:.16g} != {expected_squared_norm:.16g}"
        )
    external_overlap = (
        np.asarray(vector) @ np.asarray(bank.external_basis)
    )
    external_overlap_max_abs = float(np.max(np.abs(external_overlap)))
    if external_overlap_max_abs > 1.0e-10:
        raise RuntimeError(
            "internal Hutchinson direction contains rigid-motion leakage: "
            f"{external_overlap_max_abs:.3e}"
        )
    return InternalHutchinsonSample(
        vector=vector,
        signs=signs,
        internal_dimension=bank.internal_dimension,
        seed=int(seed),
        squared_norm=squared_norm,
        external_overlap_max_abs=external_overlap_max_abs,
    )


def rigid_external_basis(
    positions_bohr: np.ndarray,
    *,
    rank_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return an orthonormal basis for Cartesian translations and rotations."""
    positions = np.asarray(positions_bohr, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_bohr must have shape (natoms, 3)")
    if positions.shape[0] < 2:
        raise ValueError("at least two atoms are required")
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    vectors: list[np.ndarray] = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = 1.0
        vectors.append(translation.reshape(-1))
    for axis in np.eye(3, dtype=np.float64):
        vectors.append(np.cross(axis[None, :], centered).reshape(-1))
    matrix = np.stack(vectors, axis=1)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = rank_tolerance * max(float(singular_values[0]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank not in (5, 6):
        raise ValueError(f"unexpected rigid-motion rank {rank}")
    return u[:, :rank]


def _bond_graph(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    *,
    bond_scale: float,
) -> tuple[list[tuple[int, int]], dict[int, list[int]]]:
    if bond_scale <= 0.0:
        raise ValueError("bond_scale must be positive")
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions_angstrom = (
        np.asarray(positions_bohr, dtype=np.float64) * BOHR_TO_ANGSTROM
    )
    unsupported = sorted(
        set(int(value) for value in atomic_numbers).difference(
            QM9_COVALENT_RADII_ANGSTROM
        )
    )
    if unsupported:
        raise ValueError(f"unsupported elements for QM9 direction bank: {unsupported}")
    bonds: list[tuple[int, int]] = []
    for first in range(atomic_numbers.size):
        for second in range(first + 1, atomic_numbers.size):
            cutoff = bond_scale * (
                QM9_COVALENT_RADII_ANGSTROM[int(atomic_numbers[first])]
                + QM9_COVALENT_RADII_ANGSTROM[int(atomic_numbers[second])]
            )
            if (
                np.linalg.norm(
                    positions_angstrom[first] - positions_angstrom[second]
                )
                <= cutoff
            ):
                bonds.append((first, second))
    if not bonds:
        raise ValueError("bond graph is empty")
    adjacency = {index: [] for index in range(atomic_numbers.size)}
    for first, second in bonds:
        adjacency[first].append(second)
        adjacency[second].append(first)
    for neighbors in adjacency.values():
        neighbors.sort()
    return bonds, adjacency


def structured_internal_candidates(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    *,
    bond_scale: float = 1.25,
) -> dict[str, list[np.ndarray]]:
    """Return geometry-only bond, angle, and torsion displacement candidates."""
    positions = np.asarray(positions_bohr, dtype=np.float64)
    bonds, adjacency = _bond_graph(
        atomic_numbers, positions, bond_scale=bond_scale
    )
    result: dict[str, list[np.ndarray]] = {
        "bond": [],
        "angle": [],
        "torsion": [],
    }
    for first, second in bonds:
        axis = positions[second] - positions[first]
        axis /= np.linalg.norm(axis)
        vector = np.zeros_like(positions)
        vector[first] = -axis
        vector[second] = axis
        result["bond"].append(vector.reshape(-1))

    for center, neighbors in adjacency.items():
        for left_offset, left in enumerate(neighbors):
            for right in neighbors[left_offset + 1 :]:
                left_axis = positions[left] - positions[center]
                right_axis = positions[right] - positions[center]
                left_axis /= np.linalg.norm(left_axis)
                right_axis /= np.linalg.norm(right_axis)
                left_tangent = right_axis - np.dot(
                    right_axis, left_axis
                ) * left_axis
                right_tangent = left_axis - np.dot(
                    left_axis, right_axis
                ) * right_axis
                vector = np.zeros_like(positions)
                vector[left] = left_tangent
                vector[right] = right_tangent
                vector[center] = -(left_tangent + right_tangent)
                result["angle"].append(vector.reshape(-1))

    seen: set[tuple[int, int, int, int]] = set()
    for middle_left, middle_right in bonds:
        for left in adjacency[middle_left]:
            if left == middle_right:
                continue
            for right in adjacency[middle_right]:
                if right in (middle_left, left):
                    continue
                chain = (left, middle_left, middle_right, right)
                canonical = min(chain, tuple(reversed(chain)))
                if canonical in seen:
                    continue
                seen.add(canonical)
                axis = positions[middle_right] - positions[middle_left]
                axis /= np.linalg.norm(axis)
                vector = np.zeros_like(positions)
                vector[left] = np.cross(
                    axis, positions[left] - positions[middle_left]
                )
                vector[right] = -np.cross(
                    axis, positions[right] - positions[middle_right]
                )
                result["torsion"].append(vector.reshape(-1))
    return result


def _partial_roles(
    kinds: np.ndarray,
    heldout_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    count = int(kinds.size)
    heldout_count = min(count - 1, max(1, int(round(heldout_fraction * count))))
    by_kind = {
        kind: np.flatnonzero(kinds == kind) for kind in sorted(set(kinds.tolist()))
    }
    heldout: list[int] = []
    for kind in ("bond", "angle", "torsion", "random_internal"):
        indices = by_kind.get(kind)
        if indices is not None and indices.size and len(heldout) < heldout_count:
            heldout.append(int(rng.choice(indices)))
    remaining = np.asarray(
        [index for index in range(count) if index not in set(heldout)],
        dtype=np.int64,
    )
    if len(heldout) < heldout_count:
        heldout.extend(
            int(index)
            for index in rng.permutation(remaining)[: heldout_count - len(heldout)]
        )
    roles = np.full(count, "train", dtype="U7")
    roles[np.asarray(sorted(set(heldout)), dtype=np.int64)] = "heldout"
    if int(np.count_nonzero(roles == "heldout")) != heldout_count:
        raise RuntimeError("partial direction split count drift")
    return roles


def build_structured_internal_direction_bank(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    *,
    seed: int,
    heldout_fraction: float = 0.2,
    structured_per_kind: int = 12,
    bond_scale: float = 1.25,
    acceptance_tolerance: float = 1.0e-10,
) -> InternalDirectionBank:
    """Build a full ``3N-rigid_rank`` basis and an immutable partial split."""
    if structured_per_kind <= 0:
        raise ValueError("structured_per_kind must be positive")
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions = np.asarray(positions_bohr, dtype=np.float64)
    if positions.shape != (atomic_numbers.size, 3):
        raise ValueError("atomic number and position shapes disagree")
    external = rigid_external_basis(positions)
    coordinate_count = int(positions.size)
    internal_dimension = coordinate_count - int(external.shape[1])
    projector = np.eye(coordinate_count, dtype=np.float64) - external @ external.T
    candidates = structured_internal_candidates(
        atomic_numbers, positions, bond_scale=bond_scale
    )
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    kinds: list[str] = []

    def accept(vector: np.ndarray, kind: str) -> bool:
        projected = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
        for _ in range(3):
            projected -= external @ (external.T @ projected)
            for previous in accepted:
                projected -= np.dot(previous, projected) * previous
        norm = float(np.linalg.norm(projected))
        if norm <= acceptance_tolerance:
            return False
        accepted.append(projected / norm)
        kinds.append(kind)
        return True

    shuffled = {
        kind: [
            candidates[kind][int(index)]
            for index in rng.permutation(len(candidates[kind]))
        ]
        for kind in ("bond", "angle", "torsion")
    }
    for offset in range(structured_per_kind):
        for kind in ("bond", "angle", "torsion"):
            if len(accepted) >= internal_dimension:
                break
            if offset < len(shuffled[kind]):
                accept(shuffled[kind][offset], kind)
    attempts = 0
    while len(accepted) < internal_dimension:
        attempts += 1
        if attempts > 100 * coordinate_count:
            raise RuntimeError("could not complete the random internal complement")
        accept(rng.normal(size=coordinate_count), "random_internal")

    directions = np.stack(accepted)
    kind_array = np.asarray(kinds, dtype="U16")
    roles = _partial_roles(kind_array, heldout_fraction, rng)
    orthonormality = float(
        np.max(
            np.abs(
                directions @ directions.T
                - np.eye(internal_dimension, dtype=np.float64)
            )
        )
    )
    external_overlap = float(np.max(np.abs(directions @ external)))
    projector_idempotence = float(
        np.max(np.abs(projector @ projector - projector))
    )
    if orthonormality > 1.0e-10 or external_overlap > 1.0e-10:
        raise RuntimeError(
            "internal direction basis failed numerical gates: "
            f"orthonormality={orthonormality:.3e}, "
            f"external_overlap={external_overlap:.3e}"
        )
    return InternalDirectionBank(
        directions=directions,
        kinds=kind_array,
        partial_roles=roles,
        external_basis=external,
        projector=projector,
        external_rank=int(external.shape[1]),
        internal_dimension=internal_dimension,
        orthonormality_max_abs=orthonormality,
        external_overlap_max_abs=external_overlap,
        projector_idempotence_max_abs=projector_idempotence,
    )
