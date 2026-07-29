#!/usr/bin/env python3
"""Cross-check the KEDF-to-pair ABACUS TI force mixer at both endpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from mpn_melting.trajectory import parse_md_dump
from scripts.prepare_al108_ti_windows import SUPPORTED_KEDFS, parse_components
from scripts.run_pair_reference_md import evaluate_model
from scripts.analyze_two_phase_run import parse_md_log

RY_TO_EV = 13.605693122994
TARGET_ENERGY_TOLERANCE_EV = 1.0e-4
REFERENCE_ENERGY_TOLERANCE_EV = 1.0e-4
REFERENCE_FORCE_TOLERANCE_EV_PER_A = 1.0e-4


def baseline_energy_by_component_step(rows: list[dict]) -> dict[int, float]:
    """Map ABACUS MD steps 1..N onto TI component steps 0..N-1."""
    return {
        int(row["step"]) - 1: float(row["potential_Ry"]) * RY_TO_EV
        for row in rows
    }


def component_run_reached_steps(rows: list[dict], expected_steps: int) -> bool:
    return bool(rows) and int(rows[-1]["step"]) + 1 >= expected_steps


def centered_max_abs(values: list[float]) -> float:
    if not values:
        return math.inf
    center = sum(values) / len(values)
    return max(abs(value - center) for value in values)


def only_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def max_frame_difference(left, right, field: str) -> float:
    maximum = 0.0
    for left_frame, right_frame in zip(left, right):
        left_values = getattr(left_frame, field)
        right_values = getattr(right_frame, field)
        if left_values is None or right_values is None:
            raise ValueError(f"missing {field}")
        for left_vector, right_vector in zip(left_values, right_values):
            maximum = max(maximum, *(abs(a - b) for a, b in zip(left_vector, right_vector)))
    return maximum


def validate(args: argparse.Namespace) -> dict:
    endpoint_manifest = json.loads((args.root.parent / "endpoint_manifest.json").read_text())
    target_kedf = endpoint_manifest.get("target_kedf")
    baseline = args.root / "baseline" / "lambda_1p000"
    ti_one = args.root / "ti" / "lambda_1p000"
    ti_zero = args.root / "ti" / "lambda_0p000"
    baseline_log = only_file(baseline, "OUT.*/running_md.log")
    one_log = only_file(ti_one, "OUT.*/running_md.log")
    zero_log = only_file(ti_zero, "OUT.*/running_md.log")
    baseline_md, baseline_max_step = parse_md_log(baseline_log)
    one_components = parse_components(one_log)
    zero_components = parse_components(zero_log)
    baseline_energy = baseline_energy_by_component_step(baseline_md)
    energy_errors = [
        row["U_target_eV"] - baseline_energy[row["step"]]
        for row in one_components
        if row["step"] in baseline_energy
    ]
    target_energy_offset = (
        sum(energy_errors) / len(energy_errors)
        if energy_errors
        else None
    )
    target_energy_centered_error = centered_max_abs(energy_errors)

    baseline_frames = parse_md_dump(only_file(baseline, "OUT.*/MD_dump"))
    one_frames = parse_md_dump(only_file(ti_one, "OUT.*/MD_dump"))
    zero_frames = parse_md_dump(only_file(ti_zero, "OUT.*/MD_dump"))
    model_document = json.loads(args.pair_model.read_text())
    model = model_document["model"]
    zero_by_step = {row["step"]: row for row in zero_components}
    reference_energy_errors = []
    reference_force_errors = []
    for frame in zero_frames:
        positions = torch.tensor(frame.positions, dtype=torch.float64)
        lattice = torch.tensor(frame.lattice, dtype=torch.float64)
        energy, forces, _ = evaluate_model(positions, lattice, model)
        reference_energy_errors.append(
            float(energy) - zero_by_step[frame.step]["U_REF_eV"]
        )
        if frame.forces is None:
            raise ValueError("lambda=0 dump is missing forces")
        for actual, expected in zip(frame.forces, forces.tolist()):
            reference_force_errors.extend(abs(a - b) for a, b in zip(actual, expected))

    baseline_metadata = json.loads((baseline / "metadata.json").read_text())
    one_metadata = json.loads((ti_one / "metadata.json").read_text())
    zero_metadata = json.loads((ti_zero / "metadata.json").read_text())
    checks = {
        "target_kedf_supported": target_kedf in SUPPORTED_KEDFS,
        "target_kedf_matches_all_runs": all(
            metadata.get("target_kedf") == target_kedf
            and str(metadata.get("abacus", {}).get("of_kinetic", "")).lower()
            == target_kedf
            for metadata in (baseline_metadata, one_metadata, zero_metadata)
        ),
        "all_runs_reached_requested_step": baseline_max_step
        >= args.expected_steps
        and component_run_reached_steps(one_components, args.expected_steps)
        and component_run_reached_steps(zero_components, args.expected_steps),
        "lambda1_target_energy_offset_is_constant": (
            target_energy_centered_error
            < TARGET_ENERGY_TOLERANCE_EV
        ),
        "lambda1_positions_match_baseline": max_frame_difference(baseline_frames, one_frames, "positions")
        < 1.0e-10,
        "lambda1_forces_match_baseline": max_frame_difference(baseline_frames, one_frames, "forces")
        < 1.0e-8,
        "lambda0_reference_energy_matches_python": (
            centered_max_abs(reference_energy_errors)
            < REFERENCE_ENERGY_TOLERANCE_EV
        ),
        "lambda0_reference_forces_match_python": (
            max(reference_force_errors, default=math.inf)
            < REFERENCE_FORCE_TOLERANCE_EV_PER_A
        ),
        "lambda0_minimum_distance_gt_2_A": min(
            (row["nearest_neighbor_A"] for row in zero_components), default=0.0
        )
        > 2.0,
    }
    result = {
        "root": str(args.root.resolve()),
        "target_kedf": target_kedf,
        "lambda1_target_minus_md_potential_offset_eV": target_energy_offset,
        "max_lambda1_target_energy_centered_error_eV": (
            target_energy_centered_error
            if math.isfinite(target_energy_centered_error)
            else None
        ),
        "lambda0_python_minus_reference_energy_offset_eV": (
            sum(reference_energy_errors) / len(reference_energy_errors)
            if reference_energy_errors
            else None
        ),
        "max_lambda0_reference_energy_error_eV": (
            max((abs(value) for value in reference_energy_errors), default=None)
        ),
        "max_lambda0_reference_energy_centered_error_eV": (
            centered_max_abs(reference_energy_errors)
            if reference_energy_errors
            else None
        ),
        "max_lambda0_reference_force_error_eV_per_A": max(reference_force_errors, default=None),
        "max_lambda1_position_error_A": max_frame_difference(baseline_frames, one_frames, "positions"),
        "max_lambda1_force_error_eV_per_A": max_frame_difference(baseline_frames, one_frames, "forces"),
        "tolerances": {
            "target_energy_eV_per_system": TARGET_ENERGY_TOLERANCE_EV,
            "reference_energy_eV_per_system": REFERENCE_ENERGY_TOLERANCE_EV,
            "reference_force_eV_per_A": REFERENCE_FORCE_TOLERANCE_EV_PER_A,
        },
        "checks": checks,
        "status": "endpoint_validation_passed" if all(checks.values()) else "endpoint_validation_failed",
    }
    (args.root / "endpoint_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(validate(args), indent=2))


if __name__ == "__main__":
    main()
