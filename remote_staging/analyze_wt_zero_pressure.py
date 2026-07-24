#!/usr/bin/env python3
"""Estimate a common-temperature zero-pressure volume from WT NVT runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path


STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")


def parse_records(run_dir: Path) -> tuple[list[dict[str, float]], int]:
    logs = sorted(run_dir.glob("OUT.*/running_md.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one running_md.log below {run_dir}, found {len(logs)}")
    lines = logs[0].read_text(errors="replace").splitlines()
    records: list[dict[str, float]] = []
    step = -1
    max_step_seen = -1
    for index, line in enumerate(lines):
        match = STEP_RE.search(line)
        if match:
            step = int(match.group(1))
            max_step_seen = max(max_step_seen, step)
        if "Temperature (K)" not in line or index + 1 >= len(lines):
            continue
        fields = lines[index + 1].split()
        if len(fields) < 5:
            continue
        try:
            records.append(
                {"step": float(step), "temperature_K": float(fields[3]), "pressure_kbar": float(fields[4])}
            )
        except ValueError:
            continue
    if not records:
        raise ValueError(f"no MD observables parsed from {logs[0]}")
    return records, max_step_seen


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise ValueError("singular pressure regression")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def fit_pressure(
    datasets: list[tuple[float, list[dict[str, float]]]], target_temperature: float, reference_volume: float
) -> dict[str, float]:
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    observations: list[tuple[list[float], float]] = []
    for volume, records in datasets:
        for record in records:
            row = [1.0, volume - reference_volume, record["temperature_K"] - target_temperature]
            pressure = record["pressure_kbar"]
            observations.append((row, pressure))
            for i in range(3):
                xty[i] += row[i] * pressure
                for j in range(3):
                    xtx[i][j] += row[i] * row[j]
    intercept, volume_slope, temperature_slope = solve_3x3(xtx, xty)
    residuals = [
        pressure - (intercept + volume_slope * row[1] + temperature_slope * row[2])
        for row, pressure in observations
    ]
    mean_pressure = statistics.fmean(pressure for _, pressure in observations)
    total_ss = sum((pressure - mean_pressure) ** 2 for _, pressure in observations)
    residual_ss = sum(value * value for value in residuals)
    zero_volume = reference_volume - intercept / volume_slope
    return {
        "intercept_at_reference_volume_and_target_T_kbar": intercept,
        "dP_dV_kbar_per_A3_per_atom": volume_slope,
        "dP_dT_kbar_per_K": temperature_slope,
        "zero_pressure_volume_per_atom_A3": zero_volume,
        "residual_sd_kbar": math.sqrt(residual_ss / max(len(observations) - 3, 1)),
        "r_squared": 1.0 - residual_ss / total_ss if total_ss > 0.0 else math.nan,
        "observations": float(len(observations)),
    }


def block_resample(records: list[dict[str, float]], block_size: int, rng: random.Random) -> list[dict[str, float]]:
    blocks = [records[index : index + block_size] for index in range(0, len(records), block_size)]
    sampled: list[dict[str, float]] = []
    while len(sampled) < len(records):
        sampled.extend(rng.choice(blocks))
    return sampled[: len(records)]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="VPA=RUN_DIR")
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.discard_fraction < 1.0:
        raise ValueError("discard fraction must be in [0, 1)")
    if args.block_size < 1:
        raise ValueError("block size must be positive")

    datasets: list[tuple[float, list[dict[str, float]]]] = []
    run_summaries = []
    for item in args.run:
        volume_text, path_text = item.split("=", 1)
        volume = float(volume_text)
        run_dir = Path(path_text)
        all_records, max_step_seen = parse_records(run_dir)
        start = int(len(all_records) * args.discard_fraction)
        records = all_records[start:]
        if len(records) < 4:
            raise ValueError(f"too few retained records for {run_dir}")
        datasets.append((volume, records))
        run_summaries.append(
            {
                "run_dir": str(run_dir.resolve()),
                "volume_per_atom_A3": volume,
                "max_step": max_step_seen,
                "total_records": len(all_records),
                "retained_records": len(records),
                "temperature_mean_K": statistics.fmean(record["temperature_K"] for record in records),
                "pressure_mean_kbar": statistics.fmean(record["pressure_kbar"] for record in records),
                "pressure_sd_kbar": statistics.stdev(record["pressure_kbar"] for record in records),
            }
        )
    if len(datasets) < 2:
        raise ValueError("at least two volumes are required")

    reference_volume = statistics.fmean(volume for volume, _ in datasets)
    fit = fit_pressure(datasets, args.target_temperature, reference_volume)
    for summary in run_summaries:
        summary["pressure_adjusted_to_target_T_kbar"] = (
            fit["intercept_at_reference_volume_and_target_T_kbar"]
            + fit["dP_dV_kbar_per_A3_per_atom"]
            * (summary["volume_per_atom_A3"] - reference_volume)
        )

    rng = random.Random(args.seed)
    bootstrap_volumes: list[float] = []
    for _ in range(args.bootstrap_samples):
        resampled = [
            (volume, block_resample(records, args.block_size, rng)) for volume, records in datasets
        ]
        try:
            estimate = fit_pressure(resampled, args.target_temperature, reference_volume)
        except ValueError:
            continue
        value = estimate["zero_pressure_volume_per_atom_A3"]
        if math.isfinite(value):
            bootstrap_volumes.append(value)
    if len(bootstrap_volumes) < max(100, args.bootstrap_samples // 2):
        raise ValueError("too few valid bootstrap fits")

    low = min(volume for volume, _ in datasets)
    high = max(volume for volume, _ in datasets)
    estimate = fit["zero_pressure_volume_per_atom_A3"]
    result = {
        "schema": "wt-zero-pressure-volume-v1",
        "target_temperature_K": args.target_temperature,
        "discard_fraction": args.discard_fraction,
        "block_size_records": args.block_size,
        "fit": fit,
        "bootstrap": {
            "samples": len(bootstrap_volumes),
            "median_volume_per_atom_A3": quantile(bootstrap_volumes, 0.5),
            "ci95_volume_per_atom_A3": [
                quantile(bootstrap_volumes, 0.025),
                quantile(bootstrap_volumes, 0.975),
            ],
        },
        "runs": run_summaries,
        "gate": {
            "negative_dP_dV": fit["dP_dV_kbar_per_A3_per_atom"] < 0.0,
            "zero_pressure_volume_bracketed": low <= estimate <= high,
            "all_runs_reached_400_steps": all(summary["max_step"] >= 400 for summary in run_summaries),
        },
    }
    result["status"] = (
        "zero_pressure_volume_verified"
        if all(result["gate"].values())
        else "zero_pressure_volume_not_verified"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
