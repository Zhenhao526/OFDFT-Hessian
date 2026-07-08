from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .parser import parse_directory
from .trajectory import summarize_md_dump


@dataclass
class ScanPoint:
    run_dir: Path
    target_temperature: float
    thermostat: str
    mean_temperature: float | None
    last_temperature: float | None
    lindemann_ratio: float | None
    rms_displacement: float | None
    nearest_neighbor: float | None
    md_energy_drift_per_atom: float | None


def collect_scan_points(run_dirs: Iterable[Path]) -> List[ScanPoint]:
    points = [collect_scan_point(run_dir) for run_dir in run_dirs]
    return sorted(points, key=lambda point: point.target_temperature)


def collect_scan_point(run_dir: Path) -> ScanPoint:
    metadata = load_metadata(run_dir)
    parsed = parse_directory(run_dir)
    temperatures = []
    md_energies = []
    for series in parsed.values():
        temperatures.extend(series.temperatures)
        md_energies.extend(series.md_energies)
    trajectory = first_trajectory_summary(run_dir)
    natoms = metadata.get("natoms")
    drift_per_atom = None
    if md_energies and natoms:
        drift_per_atom = (md_energies[-1] - md_energies[0]) / natoms
    return ScanPoint(
        run_dir=run_dir,
        target_temperature=float(metadata.get("target_temperature_k", 0.0)),
        thermostat=str(metadata.get("thermostat", "")),
        mean_temperature=sum(temperatures) / len(temperatures) if temperatures else None,
        last_temperature=temperatures[-1] if temperatures else None,
        lindemann_ratio=trajectory.lindemann_ratio if trajectory else None,
        rms_displacement=trajectory.rms_displacement if trajectory else None,
        nearest_neighbor=trajectory.nearest_neighbor if trajectory else None,
        md_energy_drift_per_atom=drift_per_atom,
    )


def summarize_scan(points: List[ScanPoint], threshold: float = 0.1) -> str:
    if not points:
        return "No scan points."
    lines = [
        "# OFDFT Melting Scan Summary",
        "",
        f"Lindemann threshold: {threshold:.6g}",
        "",
        "| run | target_K | thermostat | mean_K | last_K | lindemann | rms_A | nn_A | drift_eV_atom |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in points:
        lines.append(
            "| "
            f"{point.run_dir.as_posix()} | "
            f"{point.target_temperature:.6g} | "
            f"{point.thermostat or '-'} | "
            f"{format_optional(point.mean_temperature)} | "
            f"{format_optional(point.last_temperature)} | "
            f"{format_optional(point.lindemann_ratio)} | "
            f"{format_optional(point.rms_displacement)} | "
            f"{format_optional(point.nearest_neighbor)} | "
            f"{format_optional(point.md_energy_drift_per_atom)} |"
        )
    estimate = estimate_threshold_temperature(points, threshold)
    lines.extend(["", "## Estimate", ""])
    lines.append(estimate)
    return "\n".join(lines) + "\n"


def estimate_threshold_temperature(points: List[ScanPoint], threshold: float) -> str:
    usable = [point for point in points if point.lindemann_ratio is not None]
    if len(usable) < 2:
        return "Not enough trajectory points to estimate a threshold crossing."
    for lower, upper in zip(usable[:-1], usable[1:]):
        if lower.lindemann_ratio is None or upper.lindemann_ratio is None:
            continue
        if lower.lindemann_ratio <= threshold <= upper.lindemann_ratio:
            slope = (upper.lindemann_ratio - lower.lindemann_ratio) / (
                upper.target_temperature - lower.target_temperature
            )
            if slope == 0:
                return "The threshold lies between two equal Lindemann ratios; no interpolation possible."
            estimate = lower.target_temperature + (threshold - lower.lindemann_ratio) / slope
            return (
                f"Estimated threshold crossing: {estimate:.6g} K "
                f"between {lower.target_temperature:.6g} K and {upper.target_temperature:.6g} K."
            )
    ratios = [point.lindemann_ratio for point in usable if point.lindemann_ratio is not None]
    if max(ratios) < threshold:
        return (
            f"No crossing observed. Highest Lindemann ratio is {max(ratios):.6g}; "
            "extend the scan to higher temperatures or longer trajectories."
        )
    if min(ratios) > threshold:
        return (
            f"All observed Lindemann ratios exceed {threshold:.6g}; "
            "extend the scan to lower temperatures."
        )
    return "No monotonic threshold crossing found; inspect trajectories before assigning a melting point."


def first_trajectory_summary(run_dir: Path):
    for path in sorted(run_dir.rglob("MD_dump"), key=lambda item: item.relative_to(run_dir).as_posix()):
        return summarize_md_dump(path)
    return None


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"
