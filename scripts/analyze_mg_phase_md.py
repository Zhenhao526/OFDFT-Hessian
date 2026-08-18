#!/usr/bin/env python3
"""Apply physical and optional zero-pressure gates to one Mg MD phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats


FATAL_MARKERS = ("nan", "convergence failed", "non-zero status", "segmentation fault")


def hcp_phase_checks(
    phase: dict[str, object],
    expected: str,
    *,
    thermalized_initial: bool = False,
) -> dict[str, bool]:
    trajectory = phase.get("trajectory", {})
    minimum_nearest = float(trajectory.get("minimum_nearest_neighbor_A", 0.0))
    final_msd = float(trajectory.get("non_affine_MSD_A2", math.inf))
    late_slope = float(
        trajectory.get("late_MSD_slope_A2_per_step", math.inf)
    )
    diffusion_slope = float(
        trajectory.get("diffusion_MSD_slope_A2_per_step", -math.inf)
    )
    common = {"all_frames_nearest_neighbor_gt_2_A": minimum_nearest > 2.0}
    if expected == "solid":
        checks = {
            **common,
            "late_MSD_slope_lt_0_001_A2_per_step": late_slope < 1.0e-3,
        }
        if thermalized_initial:
            # A continuation can show an initial vibrational dephasing when its
            # velocities are regenerated.  Judge diffusion from the late-time
            # plateau while retaining a conservative displacement bound midway
            # between the validated Mg solid and liquid pilot populations.
            return {
                **checks,
                "thermalized_non_affine_MSD_lt_0_75_A2": final_msd < 0.75,
            }
        return {
            **checks,
            "non_affine_MSD_lt_0_5_A2": final_msd < 0.5,
            "diffusion_MSD_slope_lt_0_001_A2_per_step": (
                diffusion_slope < 1.0e-3
            ),
        }
    return {
        **common,
        "non_affine_MSD_gt_0_3_A2": final_msd > 0.3,
        "diffusion_MSD_slope_gt_0_001_A2_per_step": (
            diffusion_slope > 1.0e-3
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(
    root: Path,
    expected: str,
    require_zero_pressure: bool = False,
    thermalized_initial: bool = False,
) -> dict[str, object]:
    manifest_path = root / "mg_phase_md_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    method = str(manifest["method"])
    logs = sorted(root.glob("OUT.*/running_md.log"))
    rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
    late = rows[len(rows) // 2 :]
    temperature = series_stats([row["temperature_K"] for row in late])
    pressure_values = [row["pressure_kbar"] for row in late if "pressure_kbar" in row]
    pressure = series_stats(pressure_values)
    phase = analyze_phase(root, expected, thermalized_initial=thermalized_initial)
    hcp_checks = hcp_phase_checks(
        phase,
        expected,
        thermalized_initial=thermalized_initial,
    )
    hcp_phase_status = (
        f"{expected}_verified" if all(hcp_checks.values()) else f"{expected}_not_verified"
    )
    stdout = (root / "run.stdout").read_text(encoding="utf-8", errors="ignore")
    fatal = [marker for marker in FATAL_MARKERS if marker in stdout.lower()]
    nearest = float(phase.get("trajectory", {}).get("minimum_nearest_neighbor_A", 0.0))
    temperature_error = abs(
        float(temperature.get("mean", math.inf)) - float(manifest["temperature_K"])
    )
    pressure_error = abs(float(pressure.get("mean", math.inf)))
    analytic_pressure_available = method in {"ksdft", "wt"}
    checks = {
        "reached_requested_step": max_step >= int(manifest["steps"]),
        "phase_verified": hcp_phase_status == f"{expected}_verified",
        "temperature_last_half_within_20_K": temperature_error <= 20.0,
        "minimum_nearest_neighbor_gt_2_A": nearest > 2.0,
        "no_fatal_markers": not fatal,
    }
    if require_zero_pressure:
        checks["pressure_last_half_within_2_5_kbar_if_analytic"] = (
            pressure_error <= 2.5 if analytic_pressure_available else True
        )
    status = "physical_verified" if all(checks.values()) else "failed"
    if require_zero_pressure and not analytic_pressure_available and status == "physical_verified":
        status = "physical_verified_requires_posthoc_pressure"
    result = {
        "schema": "mg-phase-md-analysis-v1",
        "status": status,
        "method": method,
        "expected_phase": expected,
        "require_zero_pressure": require_zero_pressure,
        "manifest_sha256": sha256(manifest_path),
        "max_step": max_step,
        "temperature_last_half_K": temperature,
        "pressure_last_half_kbar": pressure,
        "analytic_pressure_available": analytic_pressure_available,
        "requires_posthoc_pressure": require_zero_pressure and not analytic_pressure_available,
        "minimum_nearest_neighbor_A": nearest,
        "phase_analysis": phase,
        "phase_model": "hcp_diffusion_and_non_affine_msd_v2",
        "hcp_phase_status": hcp_phase_status,
        "hcp_phase_gate": hcp_checks,
        "fatal_markers": fatal,
        "checks": checks,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", choices=("solid", "liquid"), required=True)
    parser.add_argument("--require-zero-pressure", action="store_true")
    parser.add_argument("--thermalized-initial", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.root.resolve(),
        args.expected,
        require_zero_pressure=args.require_zero_pressure,
        thermalized_initial=args.thermalized_initial,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
