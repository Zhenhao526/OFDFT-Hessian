#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    run_dir: Path,
    *,
    expected_steps: int,
    target_temperature_k: float,
    minimum_neighbor_angstrom: float = 1.5,
    temperature_tolerance_k: float = 25.0,
    minimum_final_msd_angstrom2: float = 1.0,
    minimum_late_msd_slope_angstrom2_per_step: float = 1.0e-4,
    minimum_final_csp_median_angstrom2: float = 8.0,
    maximum_final_ordered_fraction: float = 0.1,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    phase_path = run_dir / "phase_analysis.json"
    required_paths = {
        "summary": summary_path,
        "phase_analysis": phase_path,
        "trajectory": run_dir / "trajectory.jsonl",
        "checkpoint": run_dir / "checkpoint.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing required validation files: {missing}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    final_csp = phase["CSP_final"]
    msd = phase["MSD_A2"]

    checks = {
        "trajectory_complete": (
            int(summary["steps"]) == expected_steps
            and int(phase["steps"]) == expected_steps
        ),
        "sampler_stable": bool(summary["stable"]),
        "expected_liquid_analysis": phase["expected_phase"] == "liquid",
        "auxiliary_nearest_neighbor_stable": (
            float(summary["minimum_distance_angstrom"])
            >= minimum_neighbor_angstrom
            and float(phase["minimum_nearest_neighbor_A"])
            >= minimum_neighbor_angstrom
        ),
        "temperature_control": (
            abs(float(summary["temperature_mean_k"]) - target_temperature_k)
            <= temperature_tolerance_k
            and abs(float(phase["temperature_K"]["late_mean"]) - target_temperature_k)
            <= temperature_tolerance_k
        ),
        "liquid_final_msd": (
            float(msd["last"]) >= minimum_final_msd_angstrom2
        ),
        "liquid_late_msd_slope": (
            float(msd["late_slope_per_step"])
            >= minimum_late_msd_slope_angstrom2_per_step
        ),
        "liquid_csp_median": (
            float(final_csp["median_A2"])
            >= minimum_final_csp_median_angstrom2
        ),
        "liquid_ordered_fraction": (
            float(final_csp["ordered_fraction_CSP_lt_2_5"])
            <= maximum_final_ordered_fraction
        ),
    }
    verified = all(checks.values())
    return {
        "schema": "mpn-suf-reference-dynamics-validation-v1",
        "status": "verified" if verified else "validation_failed",
        "run_dir": str(run_dir.resolve()),
        "role": "auxiliary_liquid_reference",
        "note": (
            "The auxiliary sUF numerical-stability distance is intentionally "
            "separate from the >2 A target pair-potential endpoint gate."
        ),
        "observed": {
            "steps": int(summary["steps"]),
            "samples": int(summary["samples"]),
            "temperature_mean_k": float(summary["temperature_mean_k"]),
            "temperature_late_mean_k": float(phase["temperature_K"]["late_mean"]),
            "minimum_nearest_neighbor_angstrom": min(
                float(summary["minimum_distance_angstrom"]),
                float(phase["minimum_nearest_neighbor_A"]),
            ),
            "final_msd_angstrom2": float(msd["last"]),
            "late_msd_slope_angstrom2_per_step": float(
                msd["late_slope_per_step"]
            ),
            "final_csp_median_angstrom2": float(final_csp["median_A2"]),
            "final_ordered_fraction_csp_lt_2_5": float(
                final_csp["ordered_fraction_CSP_lt_2_5"]
            ),
            "generic_pair_phase_status": phase["status"],
        },
        "gates": {
            "expected_steps": expected_steps,
            "target_temperature_k": target_temperature_k,
            "temperature_tolerance_k": temperature_tolerance_k,
            "minimum_auxiliary_neighbor_angstrom": minimum_neighbor_angstrom,
            "minimum_final_msd_angstrom2": minimum_final_msd_angstrom2,
            "minimum_late_msd_slope_angstrom2_per_step": (
                minimum_late_msd_slope_angstrom2_per_step
            ),
            "minimum_final_csp_median_angstrom2": (
                minimum_final_csp_median_angstrom2
            ),
            "maximum_final_ordered_fraction": maximum_final_ordered_fraction,
        },
        "checks": checks,
        "input_sha256": {
            label: sha256(path) for label, path in required_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an auxiliary liquid sUF reference trajectory"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--target-temperature-k", type=float, required=True)
    parser.add_argument("--minimum-neighbor", type=float, default=1.5)
    parser.add_argument("--temperature-tolerance-k", type=float, default=25.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = validate(
        args.run_dir,
        expected_steps=args.expected_steps,
        target_temperature_k=args.target_temperature_k,
        minimum_neighbor_angstrom=args.minimum_neighbor,
        temperature_tolerance_k=args.temperature_tolerance_k,
    )
    output = args.out or args.run_dir / "suf_reference_validation.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
