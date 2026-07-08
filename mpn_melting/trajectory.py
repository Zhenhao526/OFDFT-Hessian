from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

Vector = Tuple[float, float, float]
Matrix = Tuple[Vector, Vector, Vector]


@dataclass
class MdFrame:
    step: int
    lattice: Matrix
    positions: List[Vector]
    velocities: List[Vector] | None = None


@dataclass
class TrajectorySummary:
    frames: int
    natoms: int
    first_step: int
    last_step: int
    rms_displacement: float
    max_displacement: float
    nearest_neighbor: float
    final_nearest_neighbor: float
    mean_nearest_neighbor: float
    final_mean_nearest_neighbor: float
    lindemann_ratio: float


def parse_md_dump(path: Path) -> List[MdFrame]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: List[MdFrame] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("MDSTEP:"):
            idx += 1
            continue
        step = int(line.split(":", 1)[1])
        idx += 1
        while idx < len(lines) and lines[idx].strip() != "LATTICE_VECTORS":
            idx += 1
        if idx >= len(lines):
            break
        lattice = tuple(parse_vector(lines[idx + offset]) for offset in (1, 2, 3))
        idx += 4
        while idx < len(lines) and not lines[idx].strip().startswith("INDEX"):
            idx += 1
        idx += 1
        positions: List[Vector] = []
        velocities: List[Vector] = []
        while idx < len(lines) and lines[idx].strip():
            parts = lines[idx].split()
            if len(parts) >= 5:
                positions.append((float(parts[2]), float(parts[3]), float(parts[4])))
                if len(parts) >= 11:
                    velocities.append((float(parts[8]), float(parts[9]), float(parts[10])))
            idx += 1
        frames.append(MdFrame(step=step, lattice=lattice, positions=positions, velocities=velocities or None))
    return frames


def summarize_md_dump(path: Path) -> TrajectorySummary | None:
    frames = parse_md_dump(path)
    if len(frames) < 2 or not frames[0].positions:
        return None
    first = frames[0]
    last = frames[-1]
    natoms = min(len(first.positions), len(last.positions))
    inv_lattice = invert_3x3(first.lattice)
    squared_displacements = []
    for start, end in zip(first.positions[:natoms], last.positions[:natoms]):
        delta = tuple(end[i] - start[i] for i in range(3))
        delta_frac = matmul_row(delta, inv_lattice)
        wrapped_frac = tuple(component - round(component) for component in delta_frac)
        wrapped_cart = matmul_row(wrapped_frac, first.lattice)
        squared_displacements.append(sum(component * component for component in wrapped_cart))
    mean_square = sum(squared_displacements) / len(squared_displacements)
    nearest_neighbor = nearest_neighbor_distance(first.positions[:natoms], first.lattice)
    final_nearest_neighbor = nearest_neighbor_distance(last.positions[:natoms], last.lattice)
    mean_nearest_neighbor = mean_nearest_neighbor_distance(first.positions[:natoms], first.lattice)
    final_mean_nearest_neighbor = mean_nearest_neighbor_distance(last.positions[:natoms], last.lattice)
    rms_displacement = math.sqrt(mean_square)
    return TrajectorySummary(
        frames=len(frames),
        natoms=natoms,
        first_step=first.step,
        last_step=last.step,
        rms_displacement=rms_displacement,
        max_displacement=math.sqrt(max(squared_displacements)),
        nearest_neighbor=nearest_neighbor,
        final_nearest_neighbor=final_nearest_neighbor,
        mean_nearest_neighbor=mean_nearest_neighbor,
        final_mean_nearest_neighbor=final_mean_nearest_neighbor,
        lindemann_ratio=rms_displacement / nearest_neighbor if nearest_neighbor else 0.0,
    )


def summarize_trajectories(run_dir: Path) -> str:
    lines: List[str] = []
    for path in sorted(run_dir.rglob("MD_dump"), key=lambda item: item.relative_to(run_dir).as_posix()):
        summary = summarize_md_dump(path)
        name = path.relative_to(run_dir).as_posix()
        if not summary:
            lines.append(f"{name}: no usable trajectory frames")
            continue
        lines.append(
            f"{name}: frames={summary.frames}, natoms={summary.natoms}, "
            f"steps={summary.first_step}->{summary.last_step}"
        )
        lines.append(f"  nearest_neighbor_start={summary.nearest_neighbor:.6g} Angstrom")
        lines.append(f"  nearest_neighbor_end={summary.final_nearest_neighbor:.6g} Angstrom")
        lines.append(f"  mean_nearest_neighbor_start={summary.mean_nearest_neighbor:.6g} Angstrom")
        lines.append(f"  mean_nearest_neighbor_end={summary.final_mean_nearest_neighbor:.6g} Angstrom")
        lines.append(f"  rms_displacement={summary.rms_displacement:.6g} Angstrom")
        lines.append(f"  max_displacement={summary.max_displacement:.6g} Angstrom")
        lines.append(f"  lindemann_ratio={summary.lindemann_ratio:.6g}")
    return "\n".join(lines)


def parse_vector(text: str) -> Vector:
    parts = text.split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def nearest_neighbor_distance(positions: Sequence[Vector], lattice: Matrix) -> float:
    distances = nearest_neighbor_distances(positions, lattice)
    return min(distances) if distances else 0.0


def mean_nearest_neighbor_distance(positions: Sequence[Vector], lattice: Matrix) -> float:
    distances = nearest_neighbor_distances(positions, lattice)
    return sum(distances) / len(distances) if distances else 0.0


def nearest_neighbor_distances(positions: Sequence[Vector], lattice: Matrix) -> List[float]:
    if len(positions) < 2:
        return []
    inv_lattice = invert_3x3(lattice)
    shortest_by_atom = [math.inf] * len(positions)
    for idx, first in enumerate(positions[:-1]):
        for other_idx, second in enumerate(positions[idx + 1 :], start=idx + 1):
            delta = tuple(second[i] - first[i] for i in range(3))
            delta_frac = matmul_row(delta, inv_lattice)
            wrapped_frac = tuple(component - round(component) for component in delta_frac)
            wrapped_cart = matmul_row(wrapped_frac, lattice)
            distance = math.sqrt(sum(component * component for component in wrapped_cart))
            shortest_by_atom[idx] = min(shortest_by_atom[idx], distance)
            shortest_by_atom[other_idx] = min(shortest_by_atom[other_idx], distance)
    return [distance for distance in shortest_by_atom if math.isfinite(distance)]


def matmul_row(vector: Sequence[float], matrix: Matrix) -> Vector:
    return (
        vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0],
        vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1],
        vector[0] * matrix[0][2] + vector[1] * matrix[1][2] + vector[2] * matrix[2][2],
    )


def invert_3x3(matrix: Matrix) -> Matrix:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    cofactor00 = e * i - f * h
    cofactor01 = -(d * i - f * g)
    cofactor02 = d * h - e * g
    cofactor10 = -(b * i - c * h)
    cofactor11 = a * i - c * g
    cofactor12 = -(a * h - b * g)
    cofactor20 = b * f - c * e
    cofactor21 = -(a * f - c * d)
    cofactor22 = a * e - b * d
    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    if abs(determinant) < 1e-14:
        raise ValueError("lattice matrix is singular")
    inv_det = 1.0 / determinant
    return (
        (cofactor00 * inv_det, cofactor10 * inv_det, cofactor20 * inv_det),
        (cofactor01 * inv_det, cofactor11 * inv_det, cofactor21 * inv_det),
        (cofactor02 * inv_det, cofactor12 * inv_det, cofactor22 * inv_det),
    )
