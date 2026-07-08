from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .trajectory import (
    MdFrame,
    invert_3x3,
    matmul_row,
    mean_nearest_neighbor_distance,
    nearest_neighbor_distance,
    parse_md_dump,
)


@dataclass
class RegionDisplacement:
    label: str
    natoms: int
    rms_displacement: float
    max_displacement: float
    nearest_neighbor: float
    final_nearest_neighbor: float
    mean_nearest_neighbor: float
    final_mean_nearest_neighbor: float
    lindemann_ratio: float


def summarize_coexistence(run_dir: Path, threshold: float = 0.1) -> str:
    region_path = run_dir / "regions.csv"
    if not region_path.exists():
        return ""
    dump_path = first_md_dump(run_dir)
    if dump_path is None:
        return "regions.csv exists, but no MD_dump was found."
    frames = parse_md_dump(dump_path)
    if len(frames) < 2:
        return "regions.csv exists, but MD_dump has fewer than two frames."
    labels = load_region_labels(region_path)
    summaries = region_displacements(frames[0], frames[-1], labels)
    lines = [
        f"{dump_path.relative_to(run_dir).as_posix()}: coexistence regions, frames={len(frames)}, "
        f"steps={frames[0].step}->{frames[-1].step}",
    ]
    for item in summaries:
        lines.append(
            f"  {item.label}: natoms={item.natoms}, "
            f"nearest_neighbor_start={item.nearest_neighbor:.6g} Angstrom, "
            f"nearest_neighbor_end={item.final_nearest_neighbor:.6g} Angstrom, "
            f"mean_nearest_neighbor_start={item.mean_nearest_neighbor:.6g} Angstrom, "
            f"mean_nearest_neighbor_end={item.final_mean_nearest_neighbor:.6g} Angstrom, "
            f"rms_displacement={item.rms_displacement:.6g} Angstrom, "
            f"max_displacement={item.max_displacement:.6g} Angstrom, "
            f"lindemann_ratio={item.lindemann_ratio:.6g}"
        )
    lines.append(f"  trend={coexistence_trend(summaries, threshold)}")
    return "\n".join(lines)


def first_md_dump(run_dir: Path) -> Path | None:
    for path in sorted(run_dir.rglob("MD_dump"), key=lambda item: item.relative_to(run_dir).as_posix()):
        return path
    return None


def load_region_labels(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["region"] for row in reader]


def region_displacements(first: MdFrame, last: MdFrame, labels: Sequence[str]) -> List[RegionDisplacement]:
    natoms = min(len(first.positions), len(last.positions), len(labels))
    grouped: Dict[str, List[int]] = {}
    for index, label in enumerate(labels[:natoms]):
        grouped.setdefault(label, []).append(index)
    return [
        displacement_for_indices(label, indices, first, last)
        for label, indices in sorted(grouped.items(), key=lambda item: item[0])
    ]


def displacement_for_indices(label: str, indices: Sequence[int], first: MdFrame, last: MdFrame) -> RegionDisplacement:
    inv_lattice = invert_3x3(first.lattice)
    squared = []
    first_positions = []
    last_positions = []
    for index in indices:
        start = first.positions[index]
        end = last.positions[index]
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        delta_frac = matmul_row(delta, inv_lattice)
        wrapped_frac = tuple(component - round(component) for component in delta_frac)
        wrapped_cart = matmul_row(wrapped_frac, first.lattice)
        squared.append(sum(component * component for component in wrapped_cart))
        first_positions.append(start)
        last_positions.append(end)
    mean_square = sum(squared) / len(squared) if squared else 0.0
    nearest_neighbor = nearest_neighbor_distance(first_positions, first.lattice)
    final_nearest_neighbor = nearest_neighbor_distance(last_positions, last.lattice)
    mean_nearest_neighbor = mean_nearest_neighbor_distance(first_positions, first.lattice)
    final_mean_nearest_neighbor = mean_nearest_neighbor_distance(last_positions, last.lattice)
    rms = math.sqrt(mean_square)
    return RegionDisplacement(
        label=label,
        natoms=len(indices),
        rms_displacement=rms,
        max_displacement=math.sqrt(max(squared)) if squared else 0.0,
        nearest_neighbor=nearest_neighbor,
        final_nearest_neighbor=final_nearest_neighbor,
        mean_nearest_neighbor=mean_nearest_neighbor,
        final_mean_nearest_neighbor=final_mean_nearest_neighbor,
        lindemann_ratio=rms / nearest_neighbor if nearest_neighbor else 0.0,
    )


def coexistence_trend(summaries: Sequence[RegionDisplacement], threshold: float = 0.1) -> str:
    by_label = {item.label: item for item in summaries}
    solid = by_label.get("solid_seed")
    liquid = by_label.get("liquid_seed")
    if solid is None or liquid is None:
        return "missing solid_seed or liquid_seed region"
    solid_mobile = solid.lindemann_ratio >= threshold
    liquid_mobile = liquid.lindemann_ratio >= threshold
    if solid_mobile and liquid_mobile:
        return "liquid_growth_or_complete_melting"
    if not solid_mobile and not liquid_mobile:
        return "solid_growth_or_complete_freezing"
    if not solid_mobile and liquid_mobile:
        return "two_phase_persisting_or_near_stationary_interface"
    return "interface_unstable_or_mixed_response"
