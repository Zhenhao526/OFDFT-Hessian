#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from mpn_melting.free_energy import exponential_free_energy_difference
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import RY_TO_EV, parse_md_log
from scripts.prepare_al108_ti_windows import parse_components


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def simpson(values: List[float], spacing: float) -> float:
    if len(values) < 3 or len(values) % 2 == 0:
        raise ValueError("Simpson integration requires an odd number of windows")
    return spacing / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * sum(values[1:-1:2])
        + 2.0 * sum(values[2:-1:2])
    )


def trapezoid(values: List[float], spacing: float) -> float:
    return spacing * (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1])


def overlap_closure_mev_per_atom(
    forward_delta_f_ev_system: float,
    reverse_delta_f_ev_system: float,
    natoms: int,
) -> float:
    if natoms <= 0:
        raise ValueError("natoms must be positive")
    return (
        1000.0
        * abs(forward_delta_f_ev_system + reverse_delta_f_ev_system)
        / natoms
    )


def parse_window_overrides(values: List[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"window override must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or not raw_path:
            raise ValueError(f"window override must be LABEL=PATH, got {value!r}")
        if label in overrides:
            raise ValueError(f"duplicate window override for {label}")
        overrides[label] = Path(raw_path).resolve()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Production analysis of WT-to-pair TI windows")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--minimum-distance", type=float, default=2.0)
    parser.add_argument("--temperature-tolerance", type=float, default=50.0)
    parser.add_argument("--max-block-se", type=float, default=1.0)
    parser.add_argument("--max-half-drift", type=float, default=2.0)
    parser.add_argument("--max-quadrature-difference", type=float, default=2.0)
    parser.add_argument("--minimum-overlap-ess", type=float, default=0.05)
    parser.add_argument("--max-overlap-closure", type=float, default=2.0)
    parser.add_argument(
        "--window-override",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="use an extended trajectory directory for one lambda label",
    )
    args = parser.parse_args()

    root = args.run_root.resolve()
    output_path = args.out.resolve() if args.out else root / "ti_production_analysis.json"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    overrides = parse_window_overrides(args.window_override)
    labels = {str(item["label"]) for item in manifest["windows"]}
    unknown = sorted(set(overrides) - labels)
    if unknown:
        raise ValueError(f"window overrides not present in manifest: {', '.join(unknown)}")
    windows: List[Dict[str, object]] = []
    raw_by_lambda: Dict[float, List[float]] = {}
    for item in manifest["windows"]:
        run = overrides.get(str(item["label"]), root / item["label"])
        logs = sorted(run.glob("OUT.*/running_md.log"))
        if not logs:
            windows.append({**item, "status": "missing_output"})
            continue
        components = parse_components(logs[-1])
        md_rows, max_step = parse_md_log(logs[-1])
        start = int(len(components) * args.discard_fraction)
        production = components[start:]
        if len(production) < args.blocks:
            windows.append({**item, "status": "insufficient_samples", "max_step": max_step})
            continue
        phase = analyze_phase(run, manifest["phase"], thermalized_initial=True)
        (run / "phase_analysis.json").write_text(
            json.dumps(phase, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        values = [row["delta_U_eV"] for row in production]
        target_values = [row["U_target_eV"] for row in production]
        raw_by_lambda[float(item["lambda"])] = values
        block_size = len(values) // args.blocks
        block_means = [
            mean(values[index * block_size : (index + 1) * block_size])
            for index in range(args.blocks)
        ]
        target_block_means = [
            mean(target_values[index * block_size : (index + 1) * block_size])
            for index in range(args.blocks)
        ]
        midpoint = len(values) // 2
        late_md = md_rows[int(len(md_rows) * args.discard_fraction) :]
        if len(late_md) < args.blocks:
            windows.append(
                {
                    **item,
                    "status": "insufficient_md_samples",
                    "max_step": max_step,
                }
            )
            continue
        temperatures = [row["temperature_K"] for row in late_md]
        kinetic_values = [row["kinetic_Ry"] * RY_TO_EV for row in late_md]
        kinetic_block_size = len(kinetic_values) // args.blocks
        kinetic_block_means = [
            mean(
                kinetic_values[
                    index * kinetic_block_size : (index + 1) * kinetic_block_size
                ]
            )
            for index in range(args.blocks)
        ]
        target_total_block_means = [
            potential + kinetic
            for potential, kinetic in zip(target_block_means, kinetic_block_means)
        ]
        windows.append(
            {
                **item,
                "run_path": str(run),
                "status": "complete" if max_step >= manifest["steps"] else "incomplete",
                "max_step": max_step,
                "component_samples": len(components),
                "production_samples": len(values),
                "phase_status": phase["status"],
                "phase_gate_checks": phase["phase_gate"],
                "phase_metrics": {
                    "final_non_affine_msd_A2": phase["trajectory"][
                        "non_affine_MSD_A2"
                    ],
                    "late_msd_slope_A2_per_step": phase["trajectory"][
                        "late_MSD_slope_A2_per_step"
                    ],
                    "lindemann_proxy": phase["trajectory"]["lindemann_proxy"],
                    "final_csp_median_A2": phase["structure"]["median_A2"],
                    "final_ordered_fraction": phase["structure"][
                        "ordered_fraction_CSP_lt_2_5"
                    ],
                },
                "minimum_pair_distance_angstrom": min(
                    row["nearest_neighbor_A"] for row in components
                ),
                "temperature_mean_k": mean(temperatures),
                "temperature_std_k": standard_deviation(temperatures),
                "du_mean_ev_system": mean(values),
                "du_std_ev_system": standard_deviation(values),
                "du_block_standard_error_ev_system": standard_deviation(block_means)
                / math.sqrt(len(block_means)),
                "du_first_half_ev_system": mean(values[:midpoint]),
                "du_second_half_ev_system": mean(values[midpoint:]),
                "target_potential_mean_ev_system": mean(target_values),
                "target_potential_std_ev_system": standard_deviation(target_values),
                "target_potential_block_standard_error_ev_system": standard_deviation(
                    target_block_means
                )
                / math.sqrt(len(target_block_means)),
                "target_kinetic_mean_ev_system": mean(kinetic_values),
                "target_total_energy_mean_ev_system": mean(target_values)
                + mean(kinetic_values),
                "target_total_energy_block_standard_error_ev_system": standard_deviation(
                    target_total_block_means
                )
                / math.sqrt(len(target_total_block_means)),
                "block_means": block_means,
            }
        )

    complete = [window for window in windows if window.get("status") == "complete"]
    complete.sort(key=lambda window: window["lambda"])
    if len(complete) != len(manifest["windows"]):
        result = {
            **manifest,
            "schema": "wt-pair-ti-production-analysis-v1",
            "status": "incomplete",
            "windows": windows,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    lambdas = [float(window["lambda"]) for window in complete]
    spacing = lambdas[1] - lambdas[0]
    if any(
        not math.isclose(right - left, spacing, abs_tol=1.0e-12)
        for left, right in zip(lambdas, lambdas[1:])
    ):
        raise ValueError("lambda windows must be evenly spaced")
    means = [float(window["du_mean_ev_system"]) for window in complete]
    first = [float(window["du_first_half_ev_system"]) for window in complete]
    second = [float(window["du_second_half_ev_system"]) for window in complete]
    block_integrals = [
        simpson([window["block_means"][block] for window in complete], spacing)
        for block in range(args.blocks)
    ]
    integral = simpson(means, spacing)
    trap = trapezoid(means, spacing)
    first_integral = simpson(first, spacing)
    second_integral = simpson(second, spacing)
    natoms = int(manifest["natoms"])
    overlaps = []
    for left, right in zip(complete, complete[1:]):
        delta_lambda = float(right["lambda"]) - float(left["lambda"])
        forward = exponential_free_energy_difference(
            [delta_lambda * value for value in raw_by_lambda[float(left["lambda"])]],
            manifest["target_temperature_K"],
        )
        reverse = exponential_free_energy_difference(
            [-delta_lambda * value for value in raw_by_lambda[float(right["lambda"])]],
            manifest["target_temperature_K"],
        )
        overlaps.append(
            {
                "lambda_left": left["lambda"],
                "lambda_right": right["lambda"],
                "forward_delta_f_ev_system": forward["delta_f_ev"],
                "reverse_delta_f_ev_system": reverse["delta_f_ev"],
                "closure_error_mev_per_atom": overlap_closure_mev_per_atom(
                    forward["delta_f_ev"], reverse["delta_f_ev"], natoms
                ),
                "forward_effective_sample_fraction": forward["effective_sample_fraction"],
                "reverse_effective_sample_fraction": reverse["effective_sample_fraction"],
            }
        )
    for window in complete:
        del window["block_means"]
    block_standard_error = (
        1000.0
        * standard_deviation(block_integrals)
        / math.sqrt(len(block_integrals))
        / natoms
    )
    quadrature_difference = 1000.0 * abs(integral - trap) / natoms
    half_drift = 1000.0 * abs(second_integral - first_integral) / natoms
    minimum_overlap = min(
        min(item["forward_effective_sample_fraction"], item["reverse_effective_sample_fraction"])
        for item in overlaps
    )
    maximum_overlap_closure = max(
        float(item["closure_error_mev_per_atom"]) for item in overlaps
    )
    structural_gate = all(
        window["phase_status"] == f"{manifest['phase']}_verified"
        and window["minimum_pair_distance_angstrom"] >= args.minimum_distance
        for window in complete
    )
    temperature_gate = all(
        abs(window["temperature_mean_k"] - manifest["target_temperature_K"])
        <= args.temperature_tolerance
        for window in complete
    )
    convergence_checks = {
        "structural_gate": structural_gate,
        "temperature_gate": temperature_gate,
        "block_standard_error_gate": block_standard_error <= args.max_block_se,
        "half_drift_gate": half_drift <= args.max_half_drift,
        "quadrature_gate": quadrature_difference <= args.max_quadrature_difference,
        "adjacent_overlap_gate": minimum_overlap >= args.minimum_overlap_ess,
        "adjacent_overlap_closure_gate": maximum_overlap_closure
        <= args.max_overlap_closure,
    }
    lambda_one = next(window for window in complete if math.isclose(window["lambda"], 1.0))
    result = {
        **manifest,
        "schema": "wt-pair-ti-production-analysis-v1",
        "status": "verified" if all(convergence_checks.values()) else "production_gate_failed",
        "discard_fraction": args.discard_fraction,
        "blocks": args.blocks,
        "minimum_distance_gate_angstrom": args.minimum_distance,
        "temperature_tolerance_k": args.temperature_tolerance,
        "maximum_block_standard_error_mev_per_atom": args.max_block_se,
        "maximum_half_drift_mev_per_atom": args.max_half_drift,
        "maximum_quadrature_difference_mev_per_atom": args.max_quadrature_difference,
        "minimum_overlap_effective_sample_fraction_gate": args.minimum_overlap_ess,
        "maximum_overlap_closure_mev_per_atom_gate": args.max_overlap_closure,
        "window_overrides": {label: str(path) for label, path in sorted(overrides.items())},
        "checks": convergence_checks,
        "windows": complete,
        "delta_f_wt_minus_pair_simpson_ev_system": integral,
        "delta_f_wt_minus_pair_simpson_mev_per_atom": 1000.0 * integral / natoms,
        "delta_f_wt_minus_pair_trapezoid_ev_system": trap,
        "quadrature_difference_mev_per_atom": quadrature_difference,
        "block_standard_error_mev_per_atom": block_standard_error,
        "first_half_integral_ev_system": first_integral,
        "second_half_integral_ev_system": second_integral,
        "half_drift_mev_per_atom": half_drift,
        "adjacent_overlap": overlaps,
        "minimum_adjacent_effective_sample_fraction": minimum_overlap,
        "maximum_adjacent_overlap_closure_mev_per_atom": maximum_overlap_closure,
        "lambda_one_target_potential_ev_per_atom": lambda_one[
            "target_potential_mean_ev_system"
        ]
        / natoms,
        "lambda_one_target_potential_block_standard_error_mev_per_atom": 1000.0
        * lambda_one["target_potential_block_standard_error_ev_system"]
        / natoms,
        "lambda_one_target_kinetic_ev_per_atom": lambda_one[
            "target_kinetic_mean_ev_system"
        ]
        / natoms,
        "lambda_one_target_total_energy_ev_per_atom": lambda_one[
            "target_total_energy_mean_ev_system"
        ]
        / natoms,
        "lambda_one_target_total_energy_block_standard_error_mev_per_atom": 1000.0
        * lambda_one["target_total_energy_block_standard_error_ev_system"]
        / natoms,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
