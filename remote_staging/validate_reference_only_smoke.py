#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mpn_melting.trajectory import parse_md_dump
from scripts.analyze_two_phase_run import parse_md_log


def max_vector_difference(left_frames, right_frames, attribute: str) -> float:
    difference = 0.0
    for left, right in zip(left_frames, right_frames):
        for left_vector, right_vector in zip(
            getattr(left, attribute), getattr(right, attribute)
        ):
            difference = max(
                difference,
                max(abs(left_value - right_value) for left_value, right_value in zip(left_vector, right_vector)),
            )
    return difference


def total_seconds(run_dir: Path) -> int | None:
    match = re.search(r"TOTAL\s+Time\s+:\s+(\d+)", (run_dir / "run.stdout").read_text())
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--reference-only", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [args.normal.resolve(), args.reference_only.resolve()]
    logs = [next(run.glob("OUT.*/running_md.log")) for run in runs]
    rows_and_steps = [parse_md_log(log) for log in logs]
    rows = [item[0] for item in rows_and_steps]
    max_steps = [item[1] for item in rows_and_steps]
    frames = [parse_md_dump(next(run.glob("OUT.*/MD_dump"))) for run in runs]

    scalar_differences = {
        key: max(abs(left[key] - right[key]) for left, right in zip(*rows))
        for key in ("total_Ry", "potential_Ry", "kinetic_Ry", "temperature_K")
    }
    vector_differences = {
        attribute: max_vector_difference(frames[0], frames[1], attribute)
        for attribute in ("positions", "forces", "velocities")
    }
    all_differences = list(scalar_differences.values()) + list(vector_differences.values())
    gate = {
        "both_reached_five_steps": min(max_steps) >= 5,
        "matching_record_counts": len(rows[0]) == len(rows[1]) and len(frames[0]) == len(frames[1]),
        "max_absolute_difference_le_1e-12": max(all_differences) <= 1.0e-12,
        "reference_only_marker_present": "PAIR_REFERENCE_COMPONENTS"
        in logs[1].read_text(errors="replace"),
    }
    result = {
        "schema": "wt-ti-reference-only-validation-v1",
        "status": "reference_only_verified" if all(gate.values()) else "reference_only_not_verified",
        "normal_run": str(runs[0]),
        "reference_only_run": str(runs[1]),
        "max_steps": max_steps,
        "record_counts": [len(value) for value in rows],
        "frame_counts": [len(value) for value in frames],
        "max_absolute_scalar_differences": scalar_differences,
        "max_absolute_vector_component_differences": vector_differences,
        "wall_time_seconds_from_abacus": {
            "normal": total_seconds(runs[0]),
            "reference_only": total_seconds(runs[1]),
        },
        "gate": gate,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "reference_only_verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
