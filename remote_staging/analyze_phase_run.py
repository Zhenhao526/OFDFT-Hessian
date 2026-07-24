#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpn_melting.observables import centrosymmetry_parameters
from mpn_melting.trajectory import (
    lattice_volume,
    nearest_neighbor_distance,
    non_affine_msd_series,
    parse_md_dump,
)
from scripts.analyze_two_phase_run import distribution, parse_md_log, series_stats

MIN_NEIGHBOR_A = 2.0
CSP_ORDERED_THRESHOLD_A2 = 2.5
LINDEMANN_THRESHOLD = 0.15
SOLID_MAX_LATE_MSD_SLOPE_A2_PER_FS = 1.0e-3
LIQUID_MIN_LATE_MSD_SLOPE_A2_PER_FS = 1.0e-3
SOLID_MAX_FINAL_MSD_A2 = 0.5
LIQUID_MIN_FINAL_MSD_A2 = 0.3


def linear_slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def late_msd_window(
    msd_series: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Use enough of the trajectory to avoid a noisy endpoint-only slope."""
    window_size = min(len(msd_series), max(6, len(msd_series) // 2))
    return msd_series[-window_size:]


def solid_initial_checks(
    initial_ordered_fraction: float,
    initial_median_csp_A2: float,
    thermalized_initial: bool,
) -> dict[str, bool]:
    if thermalized_initial:
        return {
            "initial_thermalized_ordered_fraction_gt_0_1": initial_ordered_fraction > 0.1,
            "initial_thermalized_median_CSP_lt_6_A2": initial_median_csp_A2 < 6.0,
        }
    return {"initial_ordered_fraction_gt_0_9": initial_ordered_fraction > 0.9}


def analyze(run_dir: Path, expected: str, thermalized_initial: bool = False) -> dict:
    logs = sorted(run_dir.glob("OUT.*/running_md.log"))
    dumps = sorted(run_dir.glob("OUT.*/MD_dump"))
    rows, max_step = parse_md_log(logs[0]) if logs else ([], 0)
    result = {
        "run_dir": run_dir.as_posix(),
        "expected_phase": expected,
        "initial_state_mode": "thermalized_continuation" if thermalized_initial else "fresh_preparation",
        "md": {"max_step": max_step, "observable_samples": len(rows)},
    }
    if rows:
        result["md"]["temperature_K"] = series_stats([row["temperature_K"] for row in rows])
        pressures = [row["pressure_kbar"] for row in rows if "pressure_kbar" in row]
        if pressures:
            result["md"]["pressure_kbar"] = series_stats(pressures)
    if not dumps:
        result["status"] = "waiting_for_trajectory"
        return result

    frames = parse_md_dump(dumps[0])
    if not frames or not frames[-1].positions:
        result["status"] = "waiting_for_complete_frame"
        return result
    final = frames[-1]
    initial = frames[0]
    nearest = nearest_neighbor_distance(final.positions, final.lattice)
    initial_nearest = nearest_neighbor_distance(initial.positions, initial.lattice)
    initial_csp = centrosymmetry_parameters(initial.positions, initial.lattice)
    csp = centrosymmetry_parameters(final.positions, final.lattice)
    initial_csp_stats = distribution(initial_csp)
    csp_stats = distribution(csp)
    initial_ordered_fraction = sum(value < CSP_ORDERED_THRESHOLD_A2 for value in initial_csp) / len(initial_csp)
    ordered_fraction = sum(value < CSP_ORDERED_THRESHOLD_A2 for value in csp) / len(csp)
    msd_series = non_affine_msd_series(frames)
    final_msd = msd_series[-1][1] if msd_series else 0.0
    late_msd = late_msd_window(msd_series)
    late_msd_slope = linear_slope([(float(step), value) for step, value in late_msd])
    rms = math.sqrt(final_msd)
    lindemann = rms / nearest if nearest else math.inf
    volumes = [lattice_volume(frame.lattice) for frame in frames]
    result["trajectory"] = {
        "frames": len(frames),
        "first_step": frames[0].step,
        "last_step": final.step,
        "natoms": len(final.positions),
        "initial_nearest_neighbor_A": initial_nearest,
        "nearest_neighbor_A": nearest,
        "volume_A3": series_stats(volumes),
        "non_affine_MSD_A2": final_msd,
        "late_MSD_slope_A2_per_step": late_msd_slope,
        "late_MSD_window_frames": len(late_msd),
        "late_MSD_window_steps": late_msd[-1][0] - late_msd[0][0] if late_msd else 0,
        "non_affine_RMS_displacement_A": rms,
        "lindemann_proxy": lindemann,
        "msd_series": [{"step": step, "MSD_A2": value} for step, value in msd_series],
    }
    result["structure_initial"] = {
        **initial_csp_stats,
        "ordered_fraction_CSP_lt_2_5": initial_ordered_fraction,
    }
    result["structure"] = {**csp_stats, "ordered_fraction_CSP_lt_2_5": ordered_fraction}
    common = {"nearest_neighbor_gt_2_A": nearest > MIN_NEIGHBOR_A}
    if expected == "solid":
        checks = {
            **common,
            **solid_initial_checks(
                initial_ordered_fraction,
                initial_csp_stats["median_A2"],
                thermalized_initial,
            ),
            "late_MSD_slope_lt_0_001_A2_per_step": (
                late_msd_slope < SOLID_MAX_LATE_MSD_SLOPE_A2_PER_FS
            ),
            "final_MSD_lt_0_5_A2": final_msd < SOLID_MAX_FINAL_MSD_A2,
            "median_CSP_lt_6_A2": csp_stats["median_A2"] < 6.0,
            "ordered_fraction_gt_0_1": ordered_fraction > 0.1,
        }
    else:
        checks = {
            **common,
            "late_MSD_slope_gt_0_001_A2_per_step": (
                late_msd_slope > LIQUID_MIN_LATE_MSD_SLOPE_A2_PER_FS
            ),
            "final_MSD_gt_0_3_A2": final_msd > LIQUID_MIN_FINAL_MSD_A2,
            "median_CSP_gt_6_A2": csp_stats["median_A2"] > 6.0,
            "ordered_fraction_lt_0_1": ordered_fraction < 0.1,
        }
    result["phase_gate"] = checks
    result["status"] = f"{expected}_verified" if all(checks.values()) else f"{expected}_not_verified"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected", choices=("solid", "liquid"), required=True)
    parser.add_argument(
        "--thermalized-initial",
        action="store_true",
        help="validate a velocity-preserving continuation from an already thermalized phase",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.expected, thermalized_initial=args.thermalized_initial)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
