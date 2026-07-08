from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random
from typing import Iterable, List, Sequence, Tuple

Vector = Tuple[float, float, float]


@dataclass(frozen=True)
class AtomSet:
    symbols: List[str]
    scaled_positions: List[Vector]
    lattice_vectors: List[Vector]
    velocities: List[Vector] | None = None

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def species(self) -> List[str]:
        seen = []
        for symbol in self.symbols:
            if symbol not in seen:
                seen.append(symbol)
        return seen


def _validate_size(size: Sequence[int]) -> Tuple[int, int, int]:
    if len(size) != 3:
        raise ValueError("size must contain exactly three integers")
    nx, ny, nz = (int(x) for x in size)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("all size entries must be positive")
    return nx, ny, nz


def build_fcc(symbol: str, a: float, size: Sequence[int]) -> AtomSet:
    nx, ny, nz = _validate_size(size)
    basis = [
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ]
    symbols: List[str] = []
    positions: List[Vector] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for bx, by, bz in basis:
                    symbols.append(symbol)
                    positions.append(((i + bx) / nx, (j + by) / ny, (k + bz) / nz))
    lattice = [(a * nx, 0.0, 0.0), (0.0, a * ny, 0.0), (0.0, 0.0, a * nz)]
    return AtomSet(symbols=symbols, scaled_positions=positions, lattice_vectors=lattice)


def build_hcp(symbol: str, a: float, c: float, size: Sequence[int]) -> AtomSet:
    nx, ny, nz = _validate_size(size)
    basis = [(0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 0.5)]
    symbols: List[str] = []
    positions: List[Vector] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for bx, by, bz in basis:
                    symbols.append(symbol)
                    positions.append(((i + bx) / nx, (j + by) / ny, (k + bz) / nz))
    lattice = [
        (a * nx, 0.0, 0.0),
        (-0.5 * a * ny, 0.5 * sqrt(3.0) * a * ny, 0.0),
        (0.0, 0.0, c * nz),
    ]
    return AtomSet(symbols=symbols, scaled_positions=positions, lattice_vectors=lattice)


def select_structure(element_config: dict, size: Sequence[int]) -> AtomSet:
    element = element_config["element"]
    structure = element_config["structure"].lower()
    if structure == "fcc":
        return build_fcc(element, float(element_config["lattice_a_angstrom"]), size)
    if structure == "hcp":
        return build_hcp(
            element,
            float(element_config["lattice_a_angstrom"]),
            float(element_config["lattice_c_angstrom"]),
            size,
        )
    raise ValueError(f"unsupported structure: {structure}")


def build_coexistence_seed(
    element_config: dict,
    size: Sequence[int],
    split_axis: int = 2,
    split: float = 0.5,
    liquid_scaled_positions: Sequence[Vector] | None = None,
    liquid_shift: Sequence[float] | None = None,
    liquid_displacement_angstrom: float = 0.35,
    seed: int | None = None,
) -> tuple[AtomSet, List[str]]:
    if split_axis not in (0, 1, 2):
        raise ValueError("split_axis must be 0, 1, or 2")
    if not 0.0 < split < 1.0:
        raise ValueError("split must lie between 0 and 1")
    atoms = select_structure(element_config, size)
    labels = repeat_scaled_regions(atoms, split_axis=split_axis, split=split)
    liquid_count = labels.count("liquid_seed")
    if liquid_scaled_positions is not None and len(liquid_scaled_positions) < liquid_count:
        raise ValueError(
            f"liquid source has {len(liquid_scaled_positions)} atoms but coexistence liquid half needs {liquid_count}"
        )

    rng = random.Random(seed)
    liquid_iter = iter(liquid_scaled_positions or [])
    shift = tuple(float(value) for value in (liquid_shift or (0.0, 0.0, 0.0)))
    if len(shift) != 3:
        raise ValueError("liquid_shift must contain exactly three values")
    new_positions: List[Vector] = []
    inv_lattice = invert_orthogonal_lattice(atoms.lattice_vectors)
    for pos, label in zip(atoms.scaled_positions, labels):
        if label == "solid_seed":
            new_positions.append(pos)
            continue
        if liquid_scaled_positions is not None:
            source = next(liquid_iter)
            mapped = [(source[idx] + shift[idx]) % 1.0 for idx in range(3)]
            source_axis = mapped[split_axis]
            source_axis = min(1.0 - 1e-6, max(1e-6, source_axis))
            mapped[split_axis] = split + source_axis * (1.0 - split)
            new_positions.append(tuple(mapped))
            continue
        cart_delta = (
            rng.uniform(-liquid_displacement_angstrom, liquid_displacement_angstrom),
            rng.uniform(-liquid_displacement_angstrom, liquid_displacement_angstrom),
            rng.uniform(-liquid_displacement_angstrom, liquid_displacement_angstrom),
        )
        frac_delta = matmul_row(cart_delta, inv_lattice)
        moved = [pos[i] + frac_delta[i] for i in range(3)]
        for idx in range(3):
            if idx == split_axis:
                moved[idx] = min(1.0 - 1e-6, max(split + 1e-6, moved[idx]))
            else:
                moved[idx] %= 1.0
        new_positions.append(tuple(moved))
    return AtomSet(atoms.symbols, new_positions, atoms.lattice_vectors), labels


def repeat_scaled_regions(atoms: AtomSet, split_axis: int = 2, split: float = 0.5) -> List[str]:
    labels = []
    for pos in atoms.scaled_positions:
        labels.append("solid_seed" if pos[split_axis] < split else "liquid_seed")
    return labels


def format_vector(vec: Iterable[float]) -> str:
    return " ".join(f"{value:.12f}" for value in vec)


def matmul_row(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> Vector:
    return (
        vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0],
        vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1],
        vector[0] * matrix[0][2] + vector[1] * matrix[1][2] + vector[2] * matrix[2][2],
    )


def invert_orthogonal_lattice(lattice: Sequence[Vector]) -> tuple[Vector, Vector, Vector]:
    if any(abs(lattice[row][col]) > 1e-12 for row in range(3) for col in range(3) if row != col):
        raise ValueError("random coexistence displacements currently require an orthogonal lattice")
    return (
        (1.0 / lattice[0][0], 0.0, 0.0),
        (0.0, 1.0 / lattice[1][1], 0.0),
        (0.0, 0.0, 1.0 / lattice[2][2]),
    )
