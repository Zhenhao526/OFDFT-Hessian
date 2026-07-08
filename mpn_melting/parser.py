from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class ParsedSeries:
    energies: List[float]
    temperatures: List[float]
    pressures: List[float]
    md_energies: List[float] = field(default_factory=list)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RY_TO_EV = 13.605693122994
MD_OBSERVABLE_ROW = re.compile(
    rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$"
)

ENERGY_PATTERNS = [
    re.compile(rf"!\s*FINAL_ETOT_IS\s+({NUMBER})"),
    re.compile(rf"^\s*TN\d+\s+({NUMBER})\s+"),
    re.compile(rf"^\s*E_Total\s+{NUMBER}\s+({NUMBER})\s*$", re.IGNORECASE),
    re.compile(rf"\bETOT\b[^-+0-9]*({NUMBER})"),
    re.compile(rf"\benergy\b[^-+0-9]*({NUMBER})", re.IGNORECASE),
]
TEMP_PATTERNS = [
    re.compile(rf"\btemperature\b[^-+0-9]*({NUMBER})", re.IGNORECASE),
    re.compile(rf"(?:^|[\s,;])T\s*(?:=|:)\s*({NUMBER})"),
]
PRESSURE_PATTERN = re.compile(rf"\bpressure\b[^-+0-9]*({NUMBER})", re.IGNORECASE)


def parse_text(text: str) -> ParsedSeries:
    energies: List[float] = []
    md_energies: List[float] = []
    temperatures: List[float] = []
    pressures: List[float] = []
    expect_md_observable_row = False
    for line in text.splitlines():
        if expect_md_observable_row:
            md_match = MD_OBSERVABLE_ROW.search(line)
            if md_match:
                md_energy = float(md_match.group(1)) * RY_TO_EV
                energies.append(md_energy)
                md_energies.append(md_energy)
                temperatures.append(float(md_match.group(4)))
                expect_md_observable_row = False
                continue
            if line.strip() and not set(line.strip()) <= {"-"}:
                expect_md_observable_row = False
        if "Energy (Ry)" in line and "Temperature (K)" in line:
            expect_md_observable_row = True
            continue
        for pattern in ENERGY_PATTERNS:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
                break
        for pattern in TEMP_PATTERNS:
            temp_match = pattern.search(line)
            if temp_match:
                temperatures.append(float(temp_match.group(1)))
                break
        pressure_match = PRESSURE_PATTERN.search(line)
        if pressure_match:
            pressures.append(float(pressure_match.group(1)))
    return ParsedSeries(
        energies=energies,
        temperatures=temperatures,
        pressures=pressures,
        md_energies=md_energies,
    )


def parse_directory(run_dir: Path) -> Dict[str, ParsedSeries]:
    files = find_output_files(run_dir)
    return {
        path.relative_to(run_dir).as_posix(): parse_text(path.read_text(encoding="utf-8", errors="ignore"))
        for path in files
    }


def find_output_files(run_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for pattern in ("*.out", "*.log", "*.aimd", "running_*.log"):
        candidates.extend(run_dir.rglob(pattern))
    return sorted(set(candidates), key=lambda path: path.relative_to(run_dir).as_posix())


def summarize(parsed: Dict[str, ParsedSeries], natoms: int | None = None) -> str:
    if not parsed:
        return "No ABACUS output files found."
    lines = []
    for name, series in parsed.items():
        lines.append(
            f"{name}: energies={len(series.energies)}, "
            f"temperatures={len(series.temperatures)}, pressures={len(series.pressures)}"
        )
        if series.energies:
            lines.append(f"  last_energy={series.energies[-1]:.12g}")
        if series.temperatures:
            lines.append(f"  last_temperature={series.temperatures[-1]:.6g}")
            lines.append(
                "  temperature_stats="
                f"mean {sum(series.temperatures) / len(series.temperatures):.6g} K, "
                f"min {min(series.temperatures):.6g} K, "
                f"max {max(series.temperatures):.6g} K"
            )
        if series.md_energies:
            drift = series.md_energies[-1] - series.md_energies[0]
            lines.append(f"  md_steps={len(series.md_energies)}")
            lines.append(f"  md_energy_drift={drift:.6g} eV")
            if natoms:
                lines.append(f"  md_energy_drift_per_atom={drift / natoms:.6g} eV/atom")
        if series.pressures:
            lines.append(f"  last_pressure={series.pressures[-1]:.6g}")
    return "\n".join(lines)
