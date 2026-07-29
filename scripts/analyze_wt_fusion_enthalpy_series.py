#!/usr/bin/env python3
"""Analyze verified zero-pressure WT phase trajectories into DeltaH(T)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import RY_TO_EV, parse_md_log, series_stats

KBAR_A3_TO_EV = 0.000624150907446


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def block_standard_error(values: List[float], blocks: int) -> float:
    if blocks < 2 or len(values) < blocks:
        raise ValueError("each phase needs at least as many samples as blocks")
    block_means = [
        mean(
            values[
                index * len(values) // blocks : (index + 1)
                * len(values)
                // blocks
            ]
        )
        for index in range(blocks)
    ]
    average = mean(block_means)
    sample_variance = sum(
        (value - average) ** 2 for value in block_means
    ) / (blocks - 1)
    return math.sqrt(sample_variance / blocks)


def fusion_enthalpy_components(
    solid: Dict[str, Any],
    liquid: Dict[str, Any],
    pv_difference: float,
) -> Dict[str, float]:
    potential_difference = (
        float(liquid["potential_mean_ev_per_atom"])
        - float(solid["potential_mean_ev_per_atom"])
    )
    kinetic_difference = (
        float(liquid["kinetic_mean_ev_per_atom"])
        - float(solid["kinetic_mean_ev_per_atom"])
    )
    return {
        "potential_difference_ev_per_atom": potential_difference,
        "kinetic_difference_ev_per_atom": kinetic_difference,
        "delta_h_ev_per_atom": (
            float(liquid["total_mean_ev_per_atom"])
            - float(solid["total_mean_ev_per_atom"])
            + pv_difference
        ),
    }


def concatenate_segment_rows(
    segments: Sequence[tuple[List[Dict[str, Any]], int]],
) -> tuple[List[Dict[str, Any]], int, List[int]]:
    """Join restart segments while dropping a duplicated step-zero boundary row."""
    combined: List[Dict[str, Any]] = []
    segment_max_steps = []
    for index, (rows, max_step) in enumerate(segments):
        selected = rows
        if index > 0 and selected and int(selected[0].get("step", -1)) == 0:
            selected = selected[1:]
        combined.extend(selected)
        segment_max_steps.append(max_step)
    return combined, sum(segment_max_steps), segment_max_steps


def analyze_run(
    runs: Sequence[Path],
    phase: str,
    discard_fraction: float,
    blocks: int,
    require_pressure: bool = True,
) -> Dict[str, Any]:
    if not runs:
        raise ValueError("at least one trajectory segment is required")
    resolved_runs = [run.resolve() for run in runs]
    phase_result = analyze_phase(
        resolved_runs[-1], phase, thermalized_initial=True
    )
    segments = []
    for run in resolved_runs:
        logs = sorted(run.glob("OUT.*/running_md.log"))
        if not logs:
            raise FileNotFoundError(f"no running_md.log below {run}")
        segments.append(parse_md_log(logs[-1]))
    rows, max_step, segment_max_steps = concatenate_segment_rows(segments)
    start = int(len(rows) * discard_fraction)
    production = rows[start:]
    potentials = [row["potential_Ry"] * RY_TO_EV for row in production]
    kinetics = [row["kinetic_Ry"] * RY_TO_EV for row in production]
    totals = [row["total_Ry"] * RY_TO_EV for row in production]
    temperatures = [row["temperature_K"] for row in production]
    pressures = [row["pressure_kbar"] for row in production if "pressure_kbar" in row]
    if require_pressure and len(pressures) != len(production):
        raise RuntimeError(f"stress/pressure output is incomplete in {logs[-1]}")
    midpoint = len(potentials) // 2
    natoms = int(phase_result["trajectory"]["natoms"])
    return {
        "run": str(resolved_runs[-1]),
        "runs": [str(run) for run in resolved_runs],
        "segment_max_steps": segment_max_steps,
        "phase": phase,
        "phase_status": phase_result["status"],
        "max_step": max_step,
        "natoms": natoms,
        "nearest_neighbor_angstrom": phase_result["trajectory"]["nearest_neighbor_A"],
        "minimum_nearest_neighbor_angstrom": phase_result["trajectory"].get(
            "minimum_nearest_neighbor_A",
            phase_result["trajectory"]["nearest_neighbor_A"],
        ),
        "samples": len(rows),
        "production_samples": len(production),
        "temperature_k": series_stats(temperatures),
        "pressure_kbar": series_stats(pressures),
        "trajectory_pressure_complete": len(pressures) == len(production),
        "potential_mean_ev_per_atom": mean(potentials) / natoms,
        "kinetic_mean_ev_per_atom": mean(kinetics) / natoms,
        "total_mean_ev_per_atom": mean(totals) / natoms,
        "potential_block_standard_error_mev_per_atom": 1000.0
        * block_standard_error(potentials, blocks)
        / natoms,
        "total_block_standard_error_mev_per_atom": 1000.0
        * block_standard_error(totals, blocks)
        / natoms,
        "potential_first_half_ev_per_atom": mean(potentials[:midpoint]) / natoms,
        "potential_second_half_ev_per_atom": mean(potentials[midpoint:]) / natoms,
        "total_first_half_ev_per_atom": mean(totals[:midpoint]) / natoms,
        "total_second_half_ev_per_atom": mean(totals[midpoint:]) / natoms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a P=0 fusion-enthalpy series from WT solid/liquid MD"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--temperature-tolerance", type=float, default=20.0)
    parser.add_argument(
        "--phase-temperature-difference-tolerance", type=float, default=20.0
    )
    parser.add_argument("--pressure-tolerance", type=float, default=2.5)
    parser.add_argument("--maximum-half-drift", type=float, default=5.0)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_kedf = str(manifest.get("target_kedf", "wt")).lower()
    if target_kedf == "wt":
        result_schema = "wt-zero-pressure-fusion-enthalpy-series-v2"
    elif target_kedf in {"xwm", "lkt"}:
        result_schema = "kedf-zero-pressure-fusion-enthalpy-series-v1"
    else:
        raise ValueError(f"unsupported target KEDF {target_kedf!r}")
    base = manifest_path.parent
    analyzed_points = []
    for point in sorted(manifest["points"], key=lambda item: item["temperature_k"]):
        target_temperature = float(point["temperature_k"])
        target_pressure = float(point.get("target_pressure_kbar", 0.0))
        solid_runs = point.get("solid_runs", [point["solid_run"]])
        liquid_runs = point.get("liquid_runs", [point["liquid_run"]])
        trajectory_pressure_required = bool(
            point.get("trajectory_pressure_required", True)
        )
        analysis_arguments = (
            args.discard_fraction,
            args.blocks,
        )
        solid_run_paths = [(base / run).resolve() for run in solid_runs]
        liquid_run_paths = [(base / run).resolve() for run in liquid_runs]
        if trajectory_pressure_required:
            solid = analyze_run(
                solid_run_paths, "solid", *analysis_arguments
            )
            liquid = analyze_run(
                liquid_run_paths, "liquid", *analysis_arguments
            )
        else:
            solid = analyze_run(
                solid_run_paths,
                "solid",
                *analysis_arguments,
                require_pressure=False,
            )
            liquid = analyze_run(
                liquid_run_paths,
                "liquid",
                *analysis_arguments,
                require_pressure=False,
            )
        if solid["natoms"] != liquid["natoms"]:
            raise ValueError("solid and liquid atom counts differ")
        solid_volume = float(point["solid_volume_per_atom_A3"])
        liquid_volume = float(point["liquid_volume_per_atom_A3"])
        pv_difference = (
            target_pressure * (liquid_volume - solid_volume) * KBAR_A3_TO_EV
        )
        components = fusion_enthalpy_components(solid, liquid, pv_difference)
        potential_difference = components["potential_difference_ev_per_atom"]
        kinetic_difference = components["kinetic_difference_ev_per_atom"]
        delta_h = components["delta_h_ev_per_atom"]
        first = (
            liquid["total_first_half_ev_per_atom"]
            - solid["total_first_half_ev_per_atom"]
            + pv_difference
        )
        second = (
            liquid["total_second_half_ev_per_atom"]
            - solid["total_second_half_ev_per_atom"]
            + pv_difference
        )
        error = math.sqrt(
            solid["total_block_standard_error_mev_per_atom"] ** 2
            + liquid["total_block_standard_error_mev_per_atom"] ** 2
        )
        half_drift = 1000.0 * abs(second - first)
        phase_temperature_difference = (
            liquid["temperature_k"]["mean"]
            - solid["temperature_k"]["mean"]
        )
        requested_solid_steps = int(
            point.get("solid_steps", point["steps"])
        )
        requested_liquid_steps = int(
            point.get("liquid_steps", point["steps"])
        )
        zero_pressure_evidence_verified = (
            trajectory_pressure_required
            or (
                point.get("zero_pressure_verified") is True
                and bool(point.get("zero_pressure_provenance"))
            )
        )
        pressure_means_within_tolerance = (
            abs(solid["pressure_kbar"]["mean"] - target_pressure)
            <= args.pressure_tolerance
            and abs(liquid["pressure_kbar"]["mean"] - target_pressure)
            <= args.pressure_tolerance
            if trajectory_pressure_required
            else True
        )
        checks = {
            "solid_phase_verified": solid["phase_status"] == "solid_verified",
            "liquid_phase_verified": liquid["phase_status"] == "liquid_verified",
            "requested_steps_reached": (
                solid["max_step"] >= requested_solid_steps
                and liquid["max_step"] >= requested_liquid_steps
            ),
            "temperature_means_within_tolerance": abs(
                solid["temperature_k"]["mean"] - target_temperature
            )
            <= args.temperature_tolerance
            and abs(liquid["temperature_k"]["mean"] - target_temperature)
            <= args.temperature_tolerance,
            "solid_liquid_temperature_means_match": abs(
                phase_temperature_difference
            )
            <= args.phase_temperature_difference_tolerance,
            "pressure_means_within_tolerance": pressure_means_within_tolerance,
            "zero_pressure_evidence_verified": zero_pressure_evidence_verified,
            "nearest_neighbors_gt_2_A": (
                solid["minimum_nearest_neighbor_angstrom"] > 2.0
                and liquid["minimum_nearest_neighbor_angstrom"] > 2.0
            ),
            "positive_fusion_enthalpy": delta_h > 0.0,
            "half_drift_within_tolerance": half_drift <= args.maximum_half_drift,
        }
        analyzed_points.append(
            {
                **point,
                "solid": solid,
                "liquid": liquid,
                "delta_h_ev_per_atom": delta_h,
                "potential_difference_ev_per_atom": potential_difference,
                "kinetic_difference_ev_per_atom": kinetic_difference,
                "liquid_minus_solid_temperature_mean_difference_k": (
                    phase_temperature_difference
                ),
                "block_standard_error_mev_per_atom": error,
                "half_drift_mev_per_atom": half_drift,
                "external_pressure_pv_difference_ev_per_atom": pv_difference,
                "trajectory_pressure_required": trajectory_pressure_required,
                "pressure_gate_mode": (
                    "trajectory_analytic_stress"
                    if trajectory_pressure_required
                    else "verified_posthoc_zero_pressure"
                ),
                "requested_solid_steps": requested_solid_steps,
                "requested_liquid_steps": requested_liquid_steps,
                "checks": checks,
                "status": "verified" if all(checks.values()) else "gate_failed",
            }
        )
    result = {
        **manifest,
        "schema": result_schema,
        "target_kedf": target_kedf,
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "discard_fraction": args.discard_fraction,
        "blocks": args.blocks,
        "temperature_tolerance_k": args.temperature_tolerance,
        "phase_temperature_difference_tolerance_k": (
            args.phase_temperature_difference_tolerance
        ),
        "pressure_tolerance_kbar": args.pressure_tolerance,
        "maximum_half_drift_mev_per_atom": args.maximum_half_drift,
        "points": analyzed_points,
        "status": "verified"
        if all(point["status"] == "verified" for point in analyzed_points)
        else "gate_failed",
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
