#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from mpn_melting.free_energy import exponential_free_energy_difference


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def integrate_simpson(values: List[float], spacing: float) -> float:
    if len(values) < 3 or len(values) % 2 == 0:
        raise ValueError("Simpson integration requires an odd number of points")
    return spacing / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * sum(values[1:-1:2])
        + 2.0 * sum(values[2:-1:2])
    )


def integrate_trapezoid(values: List[float], spacing: float) -> float:
    return spacing * (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1])


def analyze(
    root: Path,
    *,
    discard_fraction: float = 0.25,
    blocks: int = 10,
    du_key: str = "du_pair_minus_suf_ev_per_atom",
    reference_label: str = "sUF",
    minimum_distance: float = 1.5,
    target_minimum_distance: float = 2.0,
    target_lambda_threshold: float = 1.0,
    temperature_tolerance: float = 50.0,
    minimum_liquid_msd: float = 1.0,
    max_block_se: float = 1.0,
    max_half_drift: float = 2.0,
    max_quadrature_difference: float = 1.0,
    minimum_overlap_ess: float = 0.05,
    max_overlap_closure: float = 2.0,
) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    production_du: List[List[float]] = []
    natoms_values = set()
    target_temperatures = set()
    for path in sorted(root.glob("lambda_*/trajectory.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            continue
        summary_path = path.with_name("summary.json")
        if not summary_path.exists():
            raise ValueError(f"missing summary file for {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        natoms_values.add(int(summary["natoms"]))
        target_temperatures.add(float(summary["target_temperature_k"]))

        start = int(len(rows) * discard_fraction)
        production = rows[start:]
        if len(production) < blocks:
            raise ValueError(f"not enough production samples in {path}")
        block_size = len(production) // blocks
        block_means = [
            mean(
                [
                    row[du_key]
                    for row in production[index * block_size : (index + 1) * block_size]
                ]
            )
            for index in range(blocks)
        ]
        du_values = [float(row[du_key]) for row in production]
        temperatures = [float(row["temperature_k"]) for row in production]
        production_du.append(du_values)
        windows.append(
            {
                "lambda": float(rows[0]["lambda"]),
                "path": str(path.resolve()),
                "total_samples": len(rows),
                "production_samples": len(production),
                "du_mean_ev_per_atom": mean(du_values),
                "du_std_ev_per_atom": standard_deviation(du_values),
                "du_block_standard_error_ev_per_atom": standard_deviation(block_means)
                / math.sqrt(len(block_means)),
                "du_first_half_ev_per_atom": mean(du_values[: len(du_values) // 2]),
                "du_second_half_ev_per_atom": mean(du_values[len(du_values) // 2 :]),
                "temperature_mean_k": mean(temperatures),
                "temperature_std_k": standard_deviation(temperatures),
                "minimum_neighbor_angstrom": min(
                    float(summary["minimum_distance_angstrom"]),
                    min(row["nearest_neighbor_angstrom"] for row in rows),
                ),
                "msd_last_angstrom2": float(rows[-1]["msd_angstrom2"]),
                "block_means": block_means,
            }
        )

    if len(natoms_values) != 1:
        raise ValueError(f"inconsistent atom counts: {sorted(natoms_values)}")
    if len(target_temperatures) != 1:
        raise ValueError(f"inconsistent target temperatures: {sorted(target_temperatures)}")
    paired = sorted(zip(windows, production_du), key=lambda item: item[0]["lambda"])
    windows = [item[0] for item in paired]
    production_du = [item[1] for item in paired]
    if len(windows) < 3 or len(windows) % 2 == 0:
        raise ValueError("an odd number of lambda windows is required")
    lambdas = [window["lambda"] for window in windows]
    spacing = lambdas[1] - lambdas[0]
    if any(
        not math.isclose(right - left, spacing, rel_tol=0.0, abs_tol=1.0e-12)
        for left, right in zip(lambdas, lambdas[1:])
    ):
        raise ValueError("lambda windows must be evenly spaced")
    if not math.isclose(lambdas[0], 0.0) or not math.isclose(lambdas[-1], 1.0):
        raise ValueError("lambda grid must include zero and one")

    means = [window["du_mean_ev_per_atom"] for window in windows]
    first = [window["du_first_half_ev_per_atom"] for window in windows]
    second = [window["du_second_half_ev_per_atom"] for window in windows]
    block_integrals = [
        integrate_simpson(
            [window["block_means"][block] for window in windows], spacing
        )
        for block in range(blocks)
    ]
    simpson = integrate_simpson(means, spacing)
    trapezoid = integrate_trapezoid(means, spacing)
    first_integral = integrate_simpson(first, spacing)
    second_integral = integrate_simpson(second, spacing)
    block_se = 1000.0 * standard_deviation(block_integrals) / math.sqrt(len(block_integrals))
    quadrature_difference = 1000.0 * abs(simpson - trapezoid)
    half_drift = 1000.0 * abs(second_integral - first_integral)
    natoms = natoms_values.pop()
    target_temperature = target_temperatures.pop()

    adjacent_overlap = []
    for index, (left, right) in enumerate(zip(windows, windows[1:])):
        delta_lambda = float(right["lambda"]) - float(left["lambda"])
        forward = exponential_free_energy_difference(
            [delta_lambda * value * natoms for value in production_du[index]],
            target_temperature,
        )
        reverse = exponential_free_energy_difference(
            [-delta_lambda * value * natoms for value in production_du[index + 1]],
            target_temperature,
        )
        adjacent_overlap.append(
            {
                "lambda_left": left["lambda"],
                "lambda_right": right["lambda"],
                "forward_delta_f_mev_per_atom": 1000.0
                * forward["delta_f_ev"]
                / natoms,
                "reverse_delta_f_mev_per_atom": 1000.0
                * reverse["delta_f_ev"]
                / natoms,
                "closure_mev_per_atom": 1000.0
                * abs(forward["delta_f_ev"] + reverse["delta_f_ev"])
                / natoms,
                "forward_effective_sample_fraction": forward[
                    "effective_sample_fraction"
                ],
                "reverse_effective_sample_fraction": reverse[
                    "effective_sample_fraction"
                ],
            }
        )

    minimum_adjacent_ess = min(
        min(
            item["forward_effective_sample_fraction"],
            item["reverse_effective_sample_fraction"],
        )
        for item in adjacent_overlap
    )
    maximum_adjacent_closure = max(item["closure_mev_per_atom"] for item in adjacent_overlap)
    numerical_stability = all(
        window["minimum_neighbor_angstrom"] >= minimum_distance for window in windows
    )
    target_windows = [
        window for window in windows if window["lambda"] >= target_lambda_threshold
    ]
    if not target_windows:
        raise ValueError("target lambda threshold selects no windows")
    target_stability = all(
        window["minimum_neighbor_angstrom"] >= target_minimum_distance
        for window in target_windows
    )
    checks = {
        "numerical_stability": numerical_stability,
        "target_windows_stable": target_stability,
        "temperature_control": all(
            abs(window["temperature_mean_k"] - target_temperature)
            <= temperature_tolerance
            for window in windows
        ),
        "liquid_diffusion": all(
            window["msd_last_angstrom2"] >= minimum_liquid_msd for window in windows
        ),
        "block_standard_error": block_se <= max_block_se,
        "half_drift": half_drift <= max_half_drift,
        "quadrature_consistency": quadrature_difference <= max_quadrature_difference,
        "adjacent_overlap_ess": minimum_adjacent_ess >= minimum_overlap_ess,
        "adjacent_overlap_closure": maximum_adjacent_closure <= max_overlap_closure,
    }
    for window in windows:
        del window["block_means"]
    return {
        "schema": "mpn-classical-reference-pair-ti-analysis-v1",
        "status": "verified" if all(checks.values()) else "production_gate_failed",
        "reference_label": reference_label,
        "root": str(root.resolve()),
        "natoms": natoms,
        "target_temperature_k": target_temperature,
        "discard_fraction": discard_fraction,
        "blocks": blocks,
        "windows": windows,
        "adjacent_overlap": adjacent_overlap,
        "delta_f_pair_minus_suf_simpson_ev_per_atom": simpson,
        "delta_f_pair_minus_suf_simpson_mev_per_atom": 1000.0 * simpson,
        "delta_f_pair_minus_suf_trapezoid_ev_per_atom": trapezoid,
        "quadrature_difference_mev_per_atom": quadrature_difference,
        "block_standard_error_mev_per_atom": block_se,
        "first_half_integral_ev_per_atom": first_integral,
        "second_half_integral_ev_per_atom": second_integral,
        "half_drift_mev_per_atom": half_drift,
        "minimum_adjacent_effective_sample_fraction": minimum_adjacent_ess,
        "maximum_adjacent_closure_mev_per_atom": maximum_adjacent_closure,
        "all_windows_stable": numerical_stability,
        "target_windows_stable": target_stability,
        "checks": checks,
        "gates": {
            "minimum_distance_angstrom": minimum_distance,
            "target_minimum_distance_angstrom": target_minimum_distance,
            "target_lambda_threshold": target_lambda_threshold,
            "temperature_tolerance_k": temperature_tolerance,
            "minimum_liquid_msd_angstrom2": minimum_liquid_msd,
            "maximum_block_standard_error_mev_per_atom": max_block_se,
            "maximum_half_drift_mev_per_atom": max_half_drift,
            "maximum_quadrature_difference_mev_per_atom": max_quadrature_difference,
            "minimum_adjacent_effective_sample_fraction": minimum_overlap_ess,
            "maximum_adjacent_closure_mev_per_atom": max_overlap_closure,
        },
        "minimum_distance_gate_angstrom": minimum_distance,
        "delta_f_pair_minus_reference_simpson_ev_per_atom": simpson,
        "delta_f_pair_minus_reference_simpson_mev_per_atom": 1000.0 * simpson,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sUF-to-pair TI windows")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--du-key", default="du_pair_minus_suf_ev_per_atom")
    parser.add_argument("--reference-label", default="sUF")
    parser.add_argument("--minimum-distance", type=float, default=1.5)
    parser.add_argument("--target-minimum-distance", type=float, default=2.0)
    parser.add_argument("--target-lambda-threshold", type=float, default=1.0)
    parser.add_argument("--temperature-tolerance", type=float, default=50.0)
    parser.add_argument("--minimum-liquid-msd", type=float, default=1.0)
    parser.add_argument("--max-block-se", type=float, default=1.0)
    parser.add_argument("--max-half-drift", type=float, default=2.0)
    parser.add_argument("--max-quadrature-difference", type=float, default=1.0)
    parser.add_argument("--minimum-overlap-ess", type=float, default=0.05)
    parser.add_argument("--max-overlap-closure", type=float, default=2.0)
    args = parser.parse_args()

    result = analyze(
        args.root,
        discard_fraction=args.discard_fraction,
        blocks=args.blocks,
        du_key=args.du_key,
        reference_label=args.reference_label,
        minimum_distance=args.minimum_distance,
        target_minimum_distance=args.target_minimum_distance,
        target_lambda_threshold=args.target_lambda_threshold,
        temperature_tolerance=args.temperature_tolerance,
        minimum_liquid_msd=args.minimum_liquid_msd,
        max_block_se=args.max_block_se,
        max_half_drift=args.max_half_drift,
        max_quadrature_difference=args.max_quadrature_difference,
        minimum_overlap_ess=args.minimum_overlap_ess,
        max_overlap_closure=args.max_overlap_closure,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
