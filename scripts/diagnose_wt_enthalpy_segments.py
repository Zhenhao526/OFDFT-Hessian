#!/usr/bin/env python3
"""Diagnose phase-resolved drift and correlation in WT enthalpy segments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.analyze_two_phase_run import RY_TO_EV, parse_md_log


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty series")
    return sum(values) / len(values)


def integrated_autocorrelation_time(
    values: Sequence[float], maximum_lag: int = 400
) -> float:
    """Return the initial-positive-sequence IAT in sample units."""
    if len(values) < 4:
        return 0.5
    average = mean(values)
    centered = [value - average for value in values]
    variance = mean([value * value for value in centered])
    if variance == 0.0:
        return 0.5
    tau = 0.5
    for lag in range(1, min(maximum_lag, len(values) // 2) + 1):
        covariance = mean(
            [
                centered[index] * centered[index + lag]
                for index in range(len(values) - lag)
            ]
        )
        correlation = covariance / variance
        if correlation <= 0.0:
            break
        tau += correlation
    return tau


def split_means(values: Sequence[float], parts: int) -> List[float]:
    if parts < 1 or len(values) < parts:
        raise ValueError("series must contain at least one value per part")
    return [
        mean(values[index * len(values) // parts : (index + 1) * len(values) // parts])
        for index in range(parts)
    ]


def series_diagnostics(
    values: Sequence[float],
    sample_stride_md_steps: float,
    blocks: int,
    maximum_lag: int,
) -> Dict[str, Any]:
    if len(values) < 4:
        raise ValueError("diagnostic series needs at least four samples")
    midpoint = len(values) // 2
    first_half = mean(values[:midpoint])
    second_half = mean(values[midpoint:])
    iat = integrated_autocorrelation_time(values, maximum_lag)
    return {
        "samples": len(values),
        "mean_mev_per_atom": mean(values),
        "standard_deviation_mev_per_atom": statistics.pstdev(values),
        "first_half_mean_mev_per_atom": first_half,
        "second_half_mean_mev_per_atom": second_half,
        "signed_half_change_mev_per_atom": second_half - first_half,
        "absolute_half_drift_mev_per_atom": abs(second_half - first_half),
        "quarter_means_mev_per_atom": split_means(values, 4),
        "block_means_mev_per_atom": split_means(values, min(blocks, len(values))),
        "integrated_autocorrelation_time_samples": iat,
        "integrated_autocorrelation_time_md_steps": iat
        * sample_stride_md_steps,
        "effective_independent_samples": len(values) / (2.0 * iat),
        "sample_stride_md_steps": sample_stride_md_steps,
    }


def only_log(run: Path) -> Path:
    logs = sorted(run.glob("OUT.*/running_md.log"))
    if not logs:
        raise FileNotFoundError(f"no running_md.log below {run}")
    return logs[-1]


def load_rows(run: Path) -> List[Dict[str, Any]]:
    rows, _ = parse_md_log(only_log(run))
    if not rows:
        raise ValueError(f"no MD rows parsed below {run}")
    return rows


def sample_stride(rows: Sequence[Dict[str, Any]]) -> float:
    differences = [
        float(rows[index + 1]["step"]) - float(rows[index]["step"])
        for index in range(len(rows) - 1)
        if float(rows[index + 1]["step"]) > float(rows[index]["step"])
    ]
    return statistics.median(differences) if differences else 1.0


def load_natoms(run: Path) -> int:
    analysis_path = run / "phase_analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"missing phase_analysis.json below {run}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    return int(analysis["trajectory"]["natoms"])


def total_energy_series(
    rows: Sequence[Dict[str, Any]], natoms: int
) -> List[float]:
    return [
        1000.0 * float(row["total_Ry"]) * RY_TO_EV / natoms for row in rows
    ]


def concatenate_segments(
    runs: Sequence[Path],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    combined: List[Dict[str, Any]] = []
    intervals = []
    for index, run in enumerate(runs):
        rows = load_rows(run)
        if index > 0 and rows and int(rows[0].get("step", -1)) == 0:
            rows = rows[1:]
        start = len(combined)
        combined.extend(rows)
        intervals.append(
            {
                "index": index,
                "run": str(run),
                "start_sample": start,
                "stop_sample": len(combined),
                "samples": len(rows),
            }
        )
    return combined, intervals


def paired_fusion_series(
    solid_values: Sequence[float], liquid_values: Sequence[float]
) -> List[float]:
    if len(solid_values) != len(liquid_values):
        raise ValueError("solid and liquid series lengths differ")
    return [
        liquid - solid for solid, liquid in zip(solid_values, liquid_values)
    ]


def selected_interval_counts(
    intervals: Sequence[Dict[str, Any]], start: int, stop: int
) -> List[Dict[str, Any]]:
    selected = []
    for interval in intervals:
        overlap_start = max(start, int(interval["start_sample"]))
        overlap_stop = min(stop, int(interval["stop_sample"]))
        count = max(0, overlap_stop - overlap_start)
        selected.append(
            {
                **interval,
                "selected_samples": count,
                "selected_fraction_of_window": count / (stop - start),
            }
        )
    return selected


def phase_drift_attribution(
    solid: Dict[str, Any], liquid: Dict[str, Any]
) -> Dict[str, Any]:
    solid_change = float(solid["signed_half_change_mev_per_atom"])
    liquid_change = float(liquid["signed_half_change_mev_per_atom"])
    fusion_change = liquid_change - solid_change
    magnitudes = {"solid": abs(solid_change), "liquid": abs(liquid_change)}
    maximum = max(magnitudes.values())
    if maximum == 0.0:
        extend_phases: List[str] = []
    else:
        extend_phases = [
            phase for phase, value in magnitudes.items() if value >= 0.6 * maximum
        ]
    return {
        "solid_signed_half_change_mev_per_atom": solid_change,
        "liquid_signed_half_change_mev_per_atom": liquid_change,
        "fusion_signed_half_change_mev_per_atom": fusion_change,
        "fusion_absolute_half_drift_mev_per_atom": abs(fusion_change),
        "dominant_phase": max(magnitudes, key=magnitudes.get),
        "phases_with_material_drift": extend_phases,
    }


def point_for_temperature(
    manifest: Dict[str, Any], temperature: float
) -> Dict[str, Any]:
    matches = [
        point
        for point in manifest["points"]
        if abs(float(point["temperature_k"]) - temperature) < 1.0e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest point at {temperature:g} K")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--maximum-lag", type=int, default=400)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    point = point_for_temperature(manifest, args.temperature)
    solid_runs = [Path(path).resolve() for path in point["solid_runs"]]
    liquid_runs = [Path(path).resolve() for path in point["liquid_runs"]]
    if len(solid_runs) != len(liquid_runs):
        raise ValueError("solid and liquid segment counts differ")
    natoms = load_natoms(solid_runs[-1])
    if load_natoms(liquid_runs[-1]) != natoms:
        raise ValueError("solid and liquid atom counts differ")

    segment_results = []
    for index, (solid_run, liquid_run) in enumerate(zip(solid_runs, liquid_runs)):
        solid_rows = load_rows(solid_run)
        liquid_rows = load_rows(liquid_run)
        stride = max(sample_stride(solid_rows), sample_stride(liquid_rows))
        solid_values = total_energy_series(solid_rows, natoms)
        liquid_values = total_energy_series(liquid_rows, natoms)
        fusion_values = paired_fusion_series(solid_values, liquid_values)
        solid_result = series_diagnostics(
            solid_values, stride, args.blocks, args.maximum_lag
        )
        liquid_result = series_diagnostics(
            liquid_values, stride, args.blocks, args.maximum_lag
        )
        segment_results.append(
            {
                "index": index,
                "label": "base" if index == 0 else f"round{index}",
                "solid_run": str(solid_run),
                "liquid_run": str(liquid_run),
                "solid": solid_result,
                "liquid": liquid_result,
                "fusion": series_diagnostics(
                    fusion_values, stride, args.blocks, args.maximum_lag
                ),
                "drift_attribution": phase_drift_attribution(
                    solid_result, liquid_result
                ),
            }
        )

    solid_rows, solid_intervals = concatenate_segments(solid_runs)
    liquid_rows, liquid_intervals = concatenate_segments(liquid_runs)
    if len(solid_rows) != len(liquid_rows):
        raise ValueError("concatenated solid and liquid series lengths differ")
    start = int(len(solid_rows) * args.discard_fraction)
    stride = max(sample_stride(solid_rows), sample_stride(liquid_rows))
    solid_values = total_energy_series(solid_rows[start:], natoms)
    liquid_values = total_energy_series(liquid_rows[start:], natoms)
    fusion_values = paired_fusion_series(solid_values, liquid_values)
    solid_window = series_diagnostics(
        solid_values, stride, args.blocks, args.maximum_lag
    )
    liquid_window = series_diagnostics(
        liquid_values, stride, args.blocks, args.maximum_lag
    )
    window = {
        "discard_fraction": args.discard_fraction,
        "start_sample": start,
        "stop_sample": len(solid_rows),
        "solid_intervals": selected_interval_counts(
            solid_intervals, start, len(solid_rows)
        ),
        "liquid_intervals": selected_interval_counts(
            liquid_intervals, start, len(liquid_rows)
        ),
        "solid": solid_window,
        "liquid": liquid_window,
        "fusion": series_diagnostics(
            fusion_values, stride, args.blocks, args.maximum_lag
        ),
        "drift_attribution": phase_drift_attribution(
            solid_window, liquid_window
        ),
    }

    result = {
        "schema": "wt-enthalpy-phase-segment-diagnostics-v1",
        "status": "diagnosis_only",
        "temperature_k": args.temperature,
        "natoms": natoms,
        "energy_definition": "sampled_total_energy_plus_external_pv",
        "manifest": str(manifest_path),
        "segments": segment_results,
        "cumulative_window": window,
        "recommendation": {
            "launch_extension": False,
            "reason": (
                "Report phase-resolved drift and correlation before selecting "
                "any targeted extension."
            ),
            "candidate_phases_for_extension": window["drift_attribution"][
                "phases_with_material_drift"
            ],
        },
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
