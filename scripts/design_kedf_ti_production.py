#!/usr/bin/env python3
"""Design formal KEDF TI production lengths from verified pilot trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from scripts.diagnose_wt_enthalpy_segments import integrated_autocorrelation_time
from scripts.prepare_al108_ti_windows import parse_components


def simpson_weights(lambdas: Sequence[float]) -> list[float]:
    if len(lambdas) < 3 or len(lambdas) % 2 == 0:
        raise ValueError("Simpson integration requires an odd number of windows")
    spacing = lambdas[1] - lambdas[0]
    if spacing <= 0.0 or any(
        not math.isclose(right - left, spacing, abs_tol=1.0e-12)
        for left, right in zip(lambdas, lambdas[1:])
    ):
        raise ValueError("lambda windows must be uniformly spaced")
    coefficients = [1.0] + [
        4.0 if index % 2 else 2.0 for index in range(1, len(lambdas) - 1)
    ] + [1.0]
    return [spacing * coefficient / 3.0 for coefficient in coefficients]


def projected_window_standard_error(
    standard_deviation: float,
    iat_samples: float,
    steps: int,
    retained_fraction: float,
) -> tuple[float, float]:
    retained_samples = steps * retained_fraction
    effective_samples = retained_samples / (2.0 * iat_samples)
    return standard_deviation / math.sqrt(effective_samples), effective_samples


def projected_integral_standard_error(
    weights: Sequence[float], window_errors: Sequence[float]
) -> float:
    if len(weights) != len(window_errors):
        raise ValueError("weights and window errors differ in length")
    return math.sqrt(
        sum((weight * error) ** 2 for weight, error in zip(weights, window_errors))
    )


def choose_steps(
    candidates: Sequence[int],
    weights: Sequence[float],
    standard_deviations: Sequence[float],
    design_iats: Sequence[float],
    *,
    retained_fraction: float,
    minimum_effective_samples: float,
    maximum_window_standard_error: float,
    maximum_integral_standard_error: float,
) -> dict[str, Any]:
    if not (
        len(weights) == len(standard_deviations) == len(design_iats)
    ):
        raise ValueError("window design arrays differ in length")
    projections = []
    for steps in sorted(set(candidates)):
        errors = []
        effective_samples = []
        for standard_deviation, iat in zip(standard_deviations, design_iats):
            error, effective = projected_window_standard_error(
                standard_deviation, iat, steps, retained_fraction
            )
            errors.append(error)
            effective_samples.append(effective)
        integral_error = projected_integral_standard_error(weights, errors)
        checks = {
            "minimum_effective_samples": min(effective_samples)
            >= minimum_effective_samples,
            "maximum_window_standard_error": max(errors)
            <= maximum_window_standard_error,
            "maximum_integral_standard_error": integral_error
            <= maximum_integral_standard_error,
        }
        projection = {
            "steps": steps,
            "minimum_effective_samples": min(effective_samples),
            "maximum_window_standard_error_mev_per_atom": max(errors),
            "projected_integral_standard_error_mev_per_atom": integral_error,
            "checks": checks,
        }
        projections.append(projection)
        if all(checks.values()):
            return {"selected": projection, "candidates": projections}
    return {"selected": None, "candidates": projections}


def only_log(run: Path) -> Path:
    logs = sorted(run.glob("OUT.*/running_md.log"))
    if not logs:
        raise FileNotFoundError(f"missing running_md.log below {run}")
    return logs[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_diagnostics(
    row: dict[str, Any],
    natoms: int,
    iat_safety_factor: float,
    minimum_design_iat: float,
) -> dict[str, Any]:
    run = Path(row["run"]).resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    components = parse_components(only_log(run))
    production = components[len(components) // 2 :]
    values = [float(item["delta_U_eV"]) * 1000.0 / natoms for item in production]
    observed_iat = integrated_autocorrelation_time(
        values, maximum_lag=min(100, len(values) // 2)
    )
    design_iat = max(minimum_design_iat, observed_iat * iat_safety_factor)
    source_tau = float(metadata["csvr_tau"])
    return {
        "label": row["label"],
        "lambda": float(row["lambda"]),
        "run": str(run),
        "samples_used": len(values),
        "mean_delta_u_mev_per_atom": statistics.mean(values),
        "standard_deviation_mev_per_atom": statistics.pstdev(values),
        "observed_iat_samples": observed_iat,
        "design_iat_samples": design_iat,
        "pilot_effective_samples": len(values) / (2.0 * observed_iat),
        "source_csvr_tau_fs": source_tau,
        "recommended_csvr_tau_fs": min(source_tau, 5.0),
        "source_step": metadata.get("source_step"),
        "source_velocities_discarded": metadata.get(
            "source_velocities_discarded"
        ),
        "target_kedf": metadata["target_kedf"],
        "pair_model": metadata["pair_model"],
    }


def design_method(
    name: str,
    merged_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    merged = json.loads(merged_path.read_text())
    if merged.get("status") != "pilot_grid_verified":
        raise ValueError(f"{name}: pilot grid is not verified")
    method = None
    phase_designs = {}
    pair_models = set()
    recommended_steps = []
    for phase in ("solid", "liquid"):
        phase_result = merged["phase_results"][phase]
        if phase_result.get("status") != "verified":
            raise ValueError(f"{name}: {phase} pilot is not verified")
        rows = sorted(phase_result["window_results"], key=lambda row: row["lambda"])
        lambdas = [float(row["lambda"]) for row in rows]
        weights = simpson_weights(lambdas)
        windows = [
            window_diagnostics(
                row,
                int(phase_result["natoms"]),
                args.iat_safety_factor,
                args.minimum_design_iat,
            )
            for row in rows
        ]
        methods = {str(window["target_kedf"]).lower() for window in windows}
        if len(methods) != 1:
            raise ValueError(f"{name}: inconsistent target KEDF in {phase}")
        phase_method = methods.pop()
        if method is None:
            method = phase_method
        elif method != phase_method:
            raise ValueError(f"{name}: phases use different KEDFs")
        pair_models.update(window["pair_model"] for window in windows)
        selection = choose_steps(
            args.candidate_steps,
            weights,
            [window["standard_deviation_mev_per_atom"] for window in windows],
            [window["design_iat_samples"] for window in windows],
            retained_fraction=args.retained_fraction,
            minimum_effective_samples=args.minimum_effective_samples,
            maximum_window_standard_error=args.maximum_window_standard_error,
            maximum_integral_standard_error=args.maximum_integral_standard_error,
        )
        if selection["selected"] is None:
            raise ValueError(f"{name}: no candidate production length passes for {phase}")
        recommended_steps.append(selection["selected"]["steps"])
        phase_designs[phase] = {
            "pilot_status": phase_result["status"],
            "pilot_minimum_overlap": phase_result[
                "minimum_adjacent_effective_sample_fraction"
            ],
            "pilot_minimum_pair_distance_A": min(
                float(row["minimum_pair_distance_A"]) for row in rows
            ),
            "windows": windows,
            "step_selection": selection,
        }
    if len(pair_models) != 1:
        raise ValueError(f"{name}: phases use different pair models")
    pair_model = Path(pair_models.pop()).resolve()
    steps = max(recommended_steps)
    windows_total = sum(
        len(phase_designs[phase]["windows"]) for phase in ("solid", "liquid")
    )
    waves = math.ceil(windows_total / args.parallel_windows)
    wall_hours = steps * args.seconds_per_step_per_wave * waves / 3600.0
    return {
        "name": name,
        "target_kedf": method,
        "pilot_merged_analysis": str(merged_path),
        "pilot_merged_analysis_sha256": sha256(merged_path),
        "pair_model": str(pair_model),
        "pair_model_sha256": sha256(pair_model),
        "recommended_steps_per_window": steps,
        "recommended_discard_fractions": [0.25, 0.5, 0.75],
        "production_temperature_tolerance_K": 20.0,
        "production_minimum_pair_distance_A": 2.0,
        "phase_designs": phase_designs,
        "resource_plan": {
            "windows_total": windows_total,
            "parallel_windows": args.parallel_windows,
            "mpi_ranks_per_window": 12,
            "waves": waves,
            "estimated_wall_hours": wall_hours,
        },
    }


def parse_method(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--method must use NAME=PATH")
    return name, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate-steps",
        type=int,
        nargs="+",
        default=[2000, 3000, 4000, 6000],
    )
    parser.add_argument("--retained-fraction", type=float, default=0.25)
    parser.add_argument("--iat-safety-factor", type=float, default=2.0)
    parser.add_argument("--minimum-design-iat", type=float, default=5.0)
    parser.add_argument("--minimum-effective-samples", type=float, default=5.0)
    parser.add_argument("--maximum-window-standard-error", type=float, default=1.0)
    parser.add_argument(
        "--maximum-integral-standard-error", type=float, default=0.5
    )
    parser.add_argument("--parallel-windows", type=int, default=6)
    parser.add_argument("--seconds-per-step-per-wave", type=float, default=7.5)
    args = parser.parse_args()

    designs = [
        design_method(name, path, args)
        for name, path in args.method
    ]
    result = {
        "schema": "kedf-ti-production-design-v1",
        "status": "verified",
        "launch_authorized": False,
        "design_basis": {
            "pilot_fraction_used": 0.5,
            "retained_fraction_for_projection": args.retained_fraction,
            "iat_safety_factor": args.iat_safety_factor,
            "minimum_design_iat_samples": args.minimum_design_iat,
            "minimum_effective_samples": args.minimum_effective_samples,
            "maximum_window_standard_error_mev_per_atom": (
                args.maximum_window_standard_error
            ),
            "maximum_phase_integral_standard_error_mev_per_atom": (
                args.maximum_integral_standard_error
            ),
            "formal_discard_convergence_gates": {
                "integral_discard_spread_mev_per_atom": 1.0,
                "window_discard_spread_mev_per_atom": 1.0,
                "window_block_standard_error_mev_per_atom": 1.0,
                "window_half_drift_mev_per_atom": 2.0,
                "quadrature_difference_mev_per_atom": 2.0,
                "minimum_overlap_effective_sample_fraction": 0.05,
                "maximum_overlap_closure_mev_per_atom": 2.0,
            },
            "adaptive_extension_policy": {
                "initial_scope": "all 18 windows for each method",
                "extension_steps": 3000,
                "extension_scope": (
                    "only windows identified by block-SE, half-drift, "
                    "discard-spread, temperature, or phase gates"
                ),
                "lambda_refinement_rule": (
                    "refine lambda only when quadrature, overlap ESS, or "
                    "overlap-closure gates fail"
                ),
            },
        },
        "methods": designs,
        "sequential_node01_estimated_wall_hours": sum(
            method["resource_plan"]["estimated_wall_hours"] for method in designs
        ),
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
