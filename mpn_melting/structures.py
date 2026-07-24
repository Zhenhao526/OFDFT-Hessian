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
    movements: List[Tuple[int, int, int]] | None = None

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


def build_bcc(symbol: str, a: float, size: Sequence[int]) -> AtomSet:
    nx, ny, nz = _validate_size(size)
    basis = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
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


def build_sc(symbol: str, a: float, size: Sequence[int]) -> AtomSet:
    nx, ny, nz = _validate_size(size)
    symbols: List[str] = []
    positions: List[Vector] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                symbols.append(symbol)
                positions.append((i / nx, j / ny, k / nz))
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
    if structure == "bcc":
        return build_bcc(element, float(element_config["lattice_a_angstrom"]), size)
    if structure == "sc":
        return build_sc(element, float(element_config["lattice_a_angstrom"]), size)
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


def tile_coexistence_template(
    atoms: AtomSet,
    labels: Sequence[str],
    repeat: Sequence[int],
    split_axis: int = 2,
    split: float = 0.5,
) -> tuple[AtomSet, List[str]]:
    """Tile the solid and liquid halves of an existing two-phase template."""
    rx, ry, rz = _validate_size(repeat)
    repeats = (rx, ry, rz)
    if split_axis not in (0, 1, 2):
        raise ValueError("split_axis must be 0, 1, or 2")
    if not 0.0 < split < 1.0:
        raise ValueError("split must lie between 0 and 1")
    if len(labels) != atoms.natoms:
        raise ValueError("region label count must match atom count")
    if any(label not in {"solid_seed", "liquid_seed"} for label in labels):
        raise ValueError("template labels must be solid_seed or liquid_seed")

    new_symbols: List[str] = []
    new_positions: List[Vector] = []
    new_labels: List[str] = []
    for symbol, position, label in zip(atoms.symbols, atoms.scaled_positions, labels):
        if label == "solid_seed":
            coordinate = position[split_axis] % 1.0
            if coordinate < split:
                local_axis = coordinate / split
            elif coordinate - split <= 1.0 - coordinate:
                local_axis = 1.0 - (coordinate - split) / split
            else:
                local_axis = (1.0 - coordinate) / split
            region_start = 0.0
            region_width = split
        else:
            coordinate = position[split_axis] % 1.0
            liquid_width = 1.0 - split
            if coordinate >= split:
                local_axis = (coordinate - split) / liquid_width
            elif split - coordinate <= coordinate:
                local_axis = (split - coordinate) / liquid_width
            else:
                local_axis = 1.0 - coordinate / liquid_width
            region_start = split
            region_width = 1.0 - split
        local_axis = min(1.0 - 1e-9, max(1e-9, local_axis))
        for ix in range(rx):
            for iy in range(ry):
                for iz in range(rz):
                    tile = (ix, iy, iz)
                    tiled = [
                        (position[axis] + tile[axis]) / repeats[axis]
                        for axis in range(3)
                    ]
                    tiled[split_axis] = region_start + region_width * (
                        local_axis + tile[split_axis]
                    ) / repeats[split_axis]
                    new_symbols.append(symbol)
                    new_positions.append(tuple(tiled))
                    new_labels.append(label)

    lattice = [
        tuple(component * repeats[axis] for component in vector)
        for axis, vector in enumerate(atoms.lattice_vectors)
    ]
    return AtomSet(new_symbols, new_positions, lattice), new_labels


def join_phase_sources(
    solid: AtomSet,
    liquid: AtomSet,
    liquid_shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[AtomSet, List[str]]:
    """Join independently equilibrated phases while preserving each phase volume."""
    if solid.species != liquid.species:
        raise ValueError("solid and liquid sources must contain the same species")
    if not is_orthogonal_lattice(solid.lattice_vectors) or not is_orthogonal_lattice(liquid.lattice_vectors):
        raise ValueError("phase joining currently requires orthogonal source cells")
    if len(liquid_shift) != 3:
        raise ValueError("liquid shift must contain exactly three values")
    shift_x, shift_y, shift_z = (float(value) for value in liquid_shift)

    common_x = solid.lattice_vectors[0][0]
    common_y = solid.lattice_vectors[1][1]
    common_area = common_x * common_y
    if common_area <= 0:
        raise ValueError("solid source must have positive x-y cross-sectional area")

    solid_z = lattice_volume(solid.lattice_vectors) / common_area
    liquid_z = lattice_volume(liquid.lattice_vectors) / common_area
    total_z = solid_z + liquid_z
    solid_fraction = solid_z / total_z

    symbols: List[str] = []
    positions: List[Vector] = []
    labels: List[str] = []
    for symbol, position in zip(solid.symbols, solid.scaled_positions):
        symbols.append(symbol)
        positions.append((position[0] % 1.0, position[1] % 1.0, (position[2] % 1.0) * solid_fraction))
        labels.append("solid_seed")
    for symbol, position in zip(liquid.symbols, liquid.scaled_positions):
        shifted = (
            (position[0] + shift_x) % 1.0,
            (position[1] + shift_y) % 1.0,
            (position[2] + shift_z) % 1.0,
        )
        symbols.append(symbol)
        positions.append(
            (
                shifted[0],
                shifted[1],
                solid_fraction + shifted[2] * (1.0 - solid_fraction),
            )
        )
        labels.append("liquid_seed")

    lattice = [
        (common_x, 0.0, 0.0),
        (0.0, common_y, 0.0),
        (0.0, 0.0, total_z),
    ]
    return AtomSet(symbols, positions, lattice), labels


def reshape_orthorhombic_cell(atoms: AtomSet, lengths: Sequence[float]) -> AtomSet:
    """Change an orthorhombic cell while preserving fractional state data."""
    if len(lengths) != 3:
        raise ValueError("cell lengths must contain exactly three values")
    lx, ly, lz = (float(value) for value in lengths)
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise ValueError("cell lengths must be positive")
    return AtomSet(
        symbols=list(atoms.symbols),
        scaled_positions=list(atoms.scaled_positions),
        lattice_vectors=[(lx, 0.0, 0.0), (0.0, ly, 0.0), (0.0, 0.0, lz)],
        velocities=list(atoms.velocities) if atoms.velocities is not None else None,
        movements=list(atoms.movements) if atoms.movements is not None else None,
    )


def reshape_coexistence_z_lengths(
    atoms: AtomSet,
    labels: Sequence[str],
    solid_length: float,
    liquid_length: float,
    split: float = 0.5,
) -> AtomSet:
    """Set independent slab lengths while keeping a continuous two-phase cell."""
    if len(labels) != atoms.natoms:
        raise ValueError("region label count must match atom count")
    if solid_length <= 0.0 or liquid_length <= 0.0:
        raise ValueError("phase lengths must be positive")
    if not 0.0 < split < 1.0:
        raise ValueError("split must lie between 0 and 1")
    total_length = float(solid_length) + float(liquid_length)
    new_positions: List[Vector] = []
    for position, label in zip(atoms.scaled_positions, labels):
        if label == "solid_seed":
            local_z = (position[2] / split) % 1.0
            z = local_z * solid_length / total_length
        elif label == "liquid_seed":
            local_z = ((position[2] - split) / (1.0 - split)) % 1.0
            z = (solid_length + local_z * liquid_length) / total_length
        else:
            raise ValueError(f"unsupported coexistence region: {label}")
        new_positions.append((position[0] % 1.0, position[1] % 1.0, z))
    lattice = list(atoms.lattice_vectors)
    lattice[2] = (0.0, 0.0, total_length)
    return AtomSet(
        symbols=list(atoms.symbols),
        scaled_positions=new_positions,
        lattice_vectors=lattice,
        velocities=list(atoms.velocities) if atoms.velocities is not None else None,
        movements=list(atoms.movements) if atoms.movements is not None else None,
    )


def constrain_regions(
    atoms: AtomSet,
    labels: Sequence[str],
    fixed_labels: Sequence[str],
) -> AtomSet:
    """Return an atom set whose selected regions are fixed in all directions."""
    if len(labels) != atoms.natoms:
        raise ValueError("region label count must match atom count")
    fixed = set(fixed_labels)
    movements = [(0, 0, 0) if label in fixed else (1, 1, 1) for label in labels]
    return AtomSet(
        symbols=atoms.symbols,
        scaled_positions=atoms.scaled_positions,
        lattice_vectors=atoms.lattice_vectors,
        velocities=atoms.velocities,
        movements=movements,
    )


def lattice_volume(lattice: Sequence[Vector]) -> float:
    a, b, c = lattice
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def is_orthogonal_lattice(lattice: Sequence[Vector], tolerance: float = 1e-10) -> bool:
    return all(
        abs(lattice[row][column]) <= tolerance
        for row in range(3)
        for column in range(3)
        if row != column
    )


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
