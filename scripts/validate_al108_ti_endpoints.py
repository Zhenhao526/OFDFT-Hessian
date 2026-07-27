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
    baseline_energy = {row["step"]: row["potential_Ry"] * RY_TO_EV for row in baseline_md}
    energy_errors = [
        abs(row["U_target_eV"] - baseline_energy[row["step"]])
        for row in one_components
        if row["step"] in baseline_energy
    ]

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
        reference_energy_errors.append(abs(float(energy) - zero_by_step[frame.step]["U_REF_eV"]))
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
        "all_runs_reached_requested_step": min(
            baseline_max_step,
            one_components[-1]["step"] if one_components else -1,
            zero_components[-1]["step"] if zero_components else -1,
        )
        >= args.expected_steps,
        "lambda1_target_energy_matches_baseline": max(energy_errors, default=math.inf) < 1.0e-5,
        "lambda1_positions_match_baseline": max_frame_difference(baseline_frames, one_frames, "positions")
        < 1.0e-10,
        "lambda1_forces_match_baseline": max_frame_difference(baseline_frames, one_frames, "forces")
        < 1.0e-8,
        "lambda0_reference_energy_matches_python": max(reference_energy_errors, default=math.inf)
        < 1.0e-6,
        "lambda0_reference_forces_match_python": max(reference_force_errors, default=math.inf) < 1.0e-6,
        "lambda0_minimum_distance_gt_2_A": min(
            (row["nearest_neighbor_A"] for row in zero_components), default=0.0
        )
        > 2.0,
    }
    result = {
        "root": str(args.root.resolve()),
        "target_kedf": target_kedf,
        "max_lambda1_target_energy_error_eV": max(energy_errors, default=None),
        "max_lambda0_reference_energy_error_eV": max(reference_energy_errors, default=None),
        "max_lambda0_reference_force_error_eV_per_A": max(reference_force_errors, default=None),
        "max_lambda1_position_error_A": max_frame_difference(baseline_frames, one_frames, "positions"),
        "max_lambda1_force_error_eV_per_A": max_frame_difference(baseline_frames, one_frames, "forces"),
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
