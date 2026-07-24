from __future__ import annotations

from pathlib import Path

from .abacus_input import ANGSTROM_TO_BOHR
from .structures import AtomSet


def read_stru(path: Path) -> AtomSet:
    """Read a Direct-coordinate ABACUS STRU file written by this project."""
    lines = path.read_text(encoding="utf-8").splitlines()

    def next_nonempty(index: int) -> tuple[str, int]:
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            raise ValueError(f"unexpected end of STRU: {path}")
        return lines[index].strip(), index + 1

    try:
        lc_index = next(i for i, line in enumerate(lines) if line.strip() == "LATTICE_CONSTANT")
        vec_index = next(i for i, line in enumerate(lines) if line.strip() == "LATTICE_VECTORS")
        pos_index = next(i for i, line in enumerate(lines) if line.strip() == "ATOMIC_POSITIONS")
    except StopIteration as exc:
        raise ValueError(f"missing required STRU section: {path}") from exc

    lc_text, _ = next_nonempty(lc_index + 1)
    scale = float(lc_text.split()[0]) / ANGSTROM_TO_BOHR
    lattice = []
    cursor = vec_index + 1
    for _ in range(3):
        row, cursor = next_nonempty(cursor)
        values = tuple(float(value) * scale for value in row.split()[:3])
        if len(values) != 3:
            raise ValueError(f"invalid lattice vector in {path}: {row}")
        lattice.append(values)

    coordinate_type, cursor = next_nonempty(pos_index + 1)
    if coordinate_type.lower() != "direct":
        raise ValueError(f"only Direct STRU coordinates are supported: {path}")

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float] | None] = []
    movements: list[tuple[int, int, int] | None] = []
    while True:
        try:
            symbol, cursor = next_nonempty(cursor)
        except ValueError:
            break
        _, cursor = next_nonempty(cursor)  # species magnetization
        count_text, cursor = next_nonempty(cursor)
        count = int(count_text.split()[0])
        for _ in range(count):
            atom_line, cursor = next_nonempty(cursor)
            fields = atom_line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid atom line in {path}: {atom_line}")
            symbols.append(symbol)
            positions.append(tuple(float(value) % 1.0 for value in fields[:3]))
            if len(fields) >= 6:
                try:
                    movements.append(tuple(int(value) for value in fields[3:6]))
                except ValueError:
                    movements.append(None)
            else:
                movements.append(None)
            if "v" in fields:
                velocity_index = fields.index("v") + 1
                velocities.append(tuple(float(value) for value in fields[velocity_index : velocity_index + 3]))
            else:
                velocities.append(None)

    if not symbols:
        raise ValueError(f"no atoms found in STRU: {path}")
    complete_velocities = None
    if all(velocity is not None for velocity in velocities):
        complete_velocities = [velocity for velocity in velocities if velocity is not None]
    complete_movements = None
    if all(movement is not None for movement in movements):
        complete_movements = [movement for movement in movements if movement is not None]
    return AtomSet(
        symbols=symbols,
        scaled_positions=positions,
        lattice_vectors=lattice,
        velocities=complete_velocities,
        movements=complete_movements,
    )
