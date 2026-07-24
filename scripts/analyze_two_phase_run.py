#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpn_melting.coexistence import load_region_labels, region_displacements
from mpn_melting.observables import centrosymmetry_parameters
from mpn_melting.trajectory import (
    MdFrame,
    invert_3x3,
    matmul_row,
    nearest_neighbor_distance,
    non_affine_msd_series,
    parse_md_dump,
)

RY_TO_EV = 13.605693122994
MIN_NEIGHBOR_A = 2.0
CSP_PHASE_THRESHOLD_A2 = 2.5
MIN_CSP_MEDIAN_CONTRAST_A2 = 0.5
MIN_ORDERED_FRACTION_CONTRAST = 0.15
LINDEMANN_MELTING_THRESHOLD = 0.15
MIN_LINDEMANN_CONTRAST = 0.05
PROFILE_BINS = 24
CORE_MARGIN_FRACTION = 0.25
STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")
ROW_RE = re.compile(
    r"^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)"
    r"(?:\s+([-+\d.eE]+))?\s*$"
)


def parse_md_log(path: Path) -> tuple[list[dict], int]:
    rows = []
    step = None
    max_step = 0
    expect_row = False
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            step_match = STEP_RE.search(line)
            if step_match:
                step = int(step_match.group(1))
                max_step = max(max_step, step)
            if expect_row:
                match = ROW_RE.match(line)
                if match and step is not None:
                    total, potential, kinetic, temperature = map(float, match.groups()[:4])
                    row = {
                        "step": step,
                        "total_Ry": total,
                        "potential_Ry": potential,
                        "kinetic_Ry": kinetic,
                        "temperature_K": temperature,
                    }
                    if match.group(5) is not None:
                        row["pressure_kbar"] = float(match.group(5))
                    rows.append(row)
                    expect_row = False
                    continue
                if line.strip() and set(line.strip()) != {"-"}:
                    expect_row = False
            if "Energy (Ry)" in line and "Temperature (K)" in line:
                expect_row = True
    deduplicated = {row["step"]: row for row in rows}
    return [deduplicated[step] for step in sorted(deduplicated)], max_step


def parse_md_rows(path: Path) -> list[dict]:
    return parse_md_log(path)[0]


def series_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def energy_drift_ev_per_atom(rows: list[dict], natoms: int) -> float | None:
    if not rows:
        return None
    if natoms <= 0:
        raise ValueError("atom count must be positive")
    return (rows[-1]["total_Ry"] - rows[0]["total_Ry"]) * RY_TO_EV / natoms


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    pick = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return {
        "n": len(values),
        "mean_A2": statistics.mean(values),
        "median_A2": statistics.median(values),
        "p10_A2": pick(0.10),
        "p90_A2": pick(0.90),
        "fraction_CSP_lt_2_5": sum(value < 2.5 for value in values) / len(values),
    }


def late_series_slope(series: list[tuple[int, float]]) -> float | None:
    """Return a least-squares slope over the latter half of a sampled series."""
    tail = series[len(series) // 2 :]
    if len(tail) < 2:
        return None
    mean_step = statistics.mean(step for step, _ in tail)
    mean_value = statistics.mean(value for _, value in tail)
    denominator = sum((step - mean_step) ** 2 for step, _ in tail)
    if denominator == 0.0:
        return None
    return sum(
        (step - mean_step) * (value - mean_value)
        for step, value in tail
    ) / denominator


def phase_regions(fraction: float, split: float) -> tuple[str, str | None]:
    if fraction < split:
        half = "solid_half"
        core_low = split * CORE_MARGIN_FRACTION
        core_high = split * (1.0 - CORE_MARGIN_FRACTION)
        core = "solid_core" if core_low <= fraction < core_high else None
    else:
        half = "liquid_half"
        width = 1.0 - split
        core_low = split + width * CORE_MARGIN_FRACTION
        core_high = split + width * (1.0 - CORE_MARGIN_FRACTION)
        core = "liquid_core" if core_low <= fraction < core_high else None
    return half, core


def structure_snapshot(frame: MdFrame, split_axis: int, split: float) -> tuple[dict, dict]:
    inv_lattice = invert_3x3(frame.lattice)
    spatial_labels = []
    core_labels = []
    axis_fractions = []
    for position in frame.positions:
        fractional = matmul_row(position, inv_lattice)
        axis_fraction = fractional[split_axis] % 1.0
        half, core = phase_regions(axis_fraction, split)
        axis_fractions.append(axis_fraction)
        spatial_labels.append(half)
        core_labels.append(core)

    csp = centrosymmetry_parameters(frame.positions, frame.lattice)
    structure = {
        label: distribution([value for value, region in zip(csp, spatial_labels) if region == label])
        for label in ("solid_half", "liquid_half")
    }
    structure.update(
        {
            label: distribution([value for value, region in zip(csp, core_labels) if region == label])
            for label in ("solid_core", "liquid_core")
        }
    )

    profile = []
    for index in range(PROFILE_BINS):
        low = index / PROFILE_BINS
        high = (index + 1) / PROFILE_BINS
        values = [
            value
            for value, fraction in zip(csp, axis_fractions)
            if low <= fraction < high
        ]
        profile.append(
            {
                "bin": index,
                "fraction_low": low,
                "fraction_high": high,
                **distribution(values),
            }
        )
    return structure, {
        "axis": split_axis,
        "bins": profile,
        "core_margin_fraction_of_half": CORE_MARGIN_FRACTION,
    }


def integrated_ordered_fraction(profile: dict) -> float:
    bins = profile.get("bins", [])
    total = sum(int(row.get("n", 0)) for row in bins)
    if total <= 0:
        raise ValueError("structure profile contains no atoms")
    ordered = sum(
        int(row.get("n", 0)) * float(row.get("fraction_CSP_lt_2_5", 0.0))
        for row in bins
    )
    return ordered / total


def region_msd_diagnostics(frames: list[MdFrame], labels: list[str]) -> dict:
    diagnostics = {}
    for label in ("solid_seed", "liquid_seed"):
        indices = [index for index, item in enumerate(labels) if item == label]
        if not indices:
            continue
        region_frames = [
            MdFrame(
                step=frame.step,
                lattice=frame.lattice,
                positions=[frame.positions[index] for index in indices],
            )
            for frame in frames
        ]
        series = non_affine_msd_series(region_frames)
        diagnostics[label] = {
            "natoms": len(indices),
            "last_A2": series[-1][1] if series else None,
            "late_slope_A2_per_step": late_series_slope(series),
            "series": [
                {"step": step, "msd_A2": value}
                for step, value in series
            ],
        }
    return diagnostics


def complete_trajectory_frames(frames: list[MdFrame]) -> tuple[list[MdFrame], int]:
    """Discard a partially written trailing frame from a live MD dump."""
    if not frames:
        return [], 0
    expected_atom_count = max(len(frame.positions) for frame in frames)
    if expected_atom_count == 0:
        return [], len(frames)
    complete = [
        frame for frame in frames if len(frame.positions) == expected_atom_count
    ]
    return complete, len(frames) - len(complete)


def evaluate_two_phase(result: dict) -> tuple[bool, dict]:
    solid = result["structure"]["solid_core"]
    liquid = result["structure"]["liquid_core"]
    solid_fraction = solid["fraction_CSP_lt_2_5"]
    liquid_fraction = liquid["fraction_CSP_lt_2_5"]
    checks = {
        "nearest_neighbor_gt_2_A": result["trajectory"]["nearest_neighbor_A"]
        > MIN_NEIGHBOR_A,
        "all_frames_nearest_neighbor_gt_2_A": result["trajectory"].get(
            "minimum_nearest_neighbor_A",
            result["trajectory"]["nearest_neighbor_A"],
        )
        > MIN_NEIGHBOR_A,
        "liquid_core_median_CSP_gt_2_5_A2": liquid["median_A2"] > CSP_PHASE_THRESHOLD_A2,
        "core_median_CSP_contrast_gt_0_5_A2": (
            liquid["median_A2"] - solid["median_A2"] > MIN_CSP_MEDIAN_CONTRAST_A2
        ),
        "liquid_core_ordered_fraction_lt_0_5": liquid_fraction < 0.5,
        "core_ordered_fraction_contrast_gt_0_15": (
            solid_fraction - liquid_fraction > MIN_ORDERED_FRACTION_CONTRAST
        ),
    }

    displacement = result.get("identity_region_displacement", {})
    solid_motion = displacement.get("solid_seed")
    liquid_motion = displacement.get("liquid_seed")
    if solid_motion and liquid_motion:
        solid_lindemann = solid_motion["lindemann_ratio"]
        liquid_lindemann = liquid_motion["lindemann_ratio"]
        checks.update(
            {
                "solid_lindemann_lt_0_15": solid_lindemann < LINDEMANN_MELTING_THRESHOLD,
                "liquid_lindemann_gt_0_15": liquid_lindemann > LINDEMANN_MELTING_THRESHOLD,
                "lindemann_contrast_gt_0_05": (
                    liquid_lindemann - solid_lindemann > MIN_LINDEMANN_CONTRAST
                ),
            }
        )
    else:
        checks["identity_region_displacement_available"] = False

    return all(checks.values()), checks


def analyze(run_dir: Path, split_axis: int, split: float) -> dict:
    logs = sorted(run_dir.glob("OUT.*/running_md.log"))
    dumps = sorted(run_dir.glob("OUT.*/MD_dump"))
    rows, max_step = parse_md_log(logs[0]) if logs else ([], 0)
    result = {
        "run_dir": run_dir.as_posix(),
        "md": {
            "completed_steps": max_step,
            "max_step": max_step,
            "observable_samples": len(rows),
            "last_observable_step": rows[-1]["step"] if rows else 0,
        },
    }
    if rows:
        temperatures = [row["temperature_K"] for row in rows]
        result["md"]["temperature_all_K"] = series_stats(temperatures)
        recent_rows = [row for row in rows if row["step"] >= max_step - 99]
        result["md"]["temperature_last_100_steps_K"] = series_stats(
            [row["temperature_K"] for row in recent_rows]
        )
        pressure_rows = [row["pressure_kbar"] for row in rows if "pressure_kbar" in row]
        if pressure_rows:
            result["md"]["pressure_all_kbar"] = series_stats(pressure_rows)
            result["md"]["pressure_last_100_steps_kbar"] = series_stats(
                [row["pressure_kbar"] for row in recent_rows if "pressure_kbar" in row]
            )
    if not dumps:
        result["status"] = "waiting_for_trajectory"
        return result

    parsed_frames = parse_md_dump(dumps[0])
    frames, dropped_frames = complete_trajectory_frames(parsed_frames)
    if not frames:
        result["status"] = "waiting_for_complete_frame"
        return result
    final = frames[-1]
    nearest_neighbors = [
        (frame.step, nearest_neighbor_distance(frame.positions, frame.lattice))
        for frame in frames
    ]
    minimum_nearest_neighbor_step, minimum_nearest_neighbor = min(
        nearest_neighbors, key=lambda item: item[1]
    )
    result["trajectory"] = {
        "frames": len(frames),
        "parsed_frames": len(parsed_frames),
        "dropped_incomplete_frames": dropped_frames,
        "atom_count": len(final.positions),
        "first_step": frames[0].step,
        "last_step": final.step,
        "initial_nearest_neighbor_A": nearest_neighbors[0][1],
        "nearest_neighbor_A": nearest_neighbors[-1][1],
        "minimum_nearest_neighbor_A": minimum_nearest_neighbor,
        "minimum_nearest_neighbor_step": minimum_nearest_neighbor_step,
    }
    if rows:
        result["md"]["energy_drift_eV_per_atom"] = energy_drift_ev_per_atom(
            rows, len(final.positions)
        )

    initial_structure, initial_profile = structure_snapshot(frames[0], split_axis, split)
    final_structure, final_profile = structure_snapshot(final, split_axis, split)
    result["initial_structure"] = initial_structure
    result["structure"] = final_structure
    result["initial_structure_profile"] = initial_profile
    result["structure_profile"] = final_profile
    initial_integrated_order = integrated_ordered_fraction(initial_profile)
    final_integrated_order = integrated_ordered_fraction(final_profile)
    result["interface_migration_indicators"] = {
        "integrated_ordered_fraction_initial": initial_integrated_order,
        "integrated_ordered_fraction_final": final_integrated_order,
        "integrated_ordered_fraction_change": (
            final_integrated_order - initial_integrated_order
        ),
        "solid_core_ordered_fraction_change": (
            final_structure["solid_core"]["fraction_CSP_lt_2_5"]
            - initial_structure["solid_core"]["fraction_CSP_lt_2_5"]
        ),
        "solid_core_median_CSP_change_A2": (
            final_structure["solid_core"]["median_A2"]
            - initial_structure["solid_core"]["median_A2"]
        ),
        "liquid_core_ordered_fraction_change": (
            final_structure["liquid_core"]["fraction_CSP_lt_2_5"]
            - initial_structure["liquid_core"]["fraction_CSP_lt_2_5"]
        ),
        "liquid_core_median_CSP_change_A2": (
            final_structure["liquid_core"]["median_A2"]
            - initial_structure["liquid_core"]["median_A2"]
        ),
    }

    region_path = run_dir / "regions.csv"
    if len(frames) >= 2 and region_path.exists():
        labels = load_region_labels(region_path)
        result["identity_region_displacement"] = {
            item.label: {
                "natoms": item.natoms,
                "rms_displacement_A": item.rms_displacement,
                "lindemann_ratio": item.lindemann_ratio,
            }
            for item in region_displacements(frames[0], final, labels)
        }
        result["identity_region_msd"] = region_msd_diagnostics(frames, labels)

    two_phase, checks = evaluate_two_phase(result)
    result["two_phase_gate"] = checks
    result["status"] = "two_phase_verified" if two_phase else "two_phase_not_verified"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split-axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.split_axis, args.split)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
