#!/usr/bin/env python3
"""Validate conservative total-OFDFT forces against relaxed scalar energies and loop work."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ofdft.conservative_force import (
    evaluate_total_ofdft_force,
    prepare_differentiable_geometry,
)
from mldft.ofdft.geometry_integrals import FiniteDifferencePySCFIntegralProvider
from mldft.ofdft.stationary_density import (
    refine_sample_density_lbfgs,
    refine_sample_density_newton_pcg,
)

from qm9_hessian_density_relaxed_eval import (
    _load_context,
    _load_geometry,
    _make_mol,
    _optimize_density,
    _parse_run,
    _save_optimization_trace,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_final_coefficients(
    path: Path,
    coefficients: torch.Tensor,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=coefficients.detach().cpu().numpy(),
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        positions_bohr=np.asarray(positions_bohr, dtype=np.float64),
    )
    return str(path)


def _initialization(
    sample: Any,
    base_coeffs: torch.Tensor | None,
    default: str,
    initial_coeffs: torch.Tensor | None = None,
    initialization_mode_override: str | None = None,
):
    if initial_coeffs is not None:
        transformed = transform_tensor_with_sample(
            sample,
            initial_coeffs.to(device=sample.coeffs.device, dtype=sample.coeffs.dtype),
            Representation.VECTOR,
        )
        return transformed, initialization_mode_override or "explicit_density_warm_start"
    if base_coeffs is None:
        return default, initialization_mode_override or default
    transformed = transform_tensor_with_sample(
        sample,
        base_coeffs.to(device=sample.coeffs.device, dtype=sample.coeffs.dtype),
        Representation.VECTOR,
    )
    return transformed, initialization_mode_override or "base_density_warm_start"


def _evaluate_point(
    context: Any,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    charge: int,
    args: argparse.Namespace,
    base_coeffs: torch.Tensor | None,
    need_force: bool,
    initial_coeffs: torch.Tensor | None = None,
    initialization_mode_override: str | None = None,
) -> dict[str, Any]:
    point_started = time.perf_counter()
    sample_started = time.perf_counter()
    mol = _make_mol(atomic_numbers, positions_bohr, charge)
    sample = context.sample_generator.get_sample_from_mol(mol)
    sample_construction_elapsed_s = time.perf_counter() - sample_started
    initialization, initialization_mode = _initialization(
        sample,
        base_coeffs,
        args.initialization,
        initial_coeffs=initial_coeffs,
        initialization_mode_override=initialization_mode_override,
    )
    optimization_started = time.perf_counter()
    tensor_energies = None
    refinement = None
    newton = None
    legacy_energies, final_coeffs, metadata, trace = _optimize_density(
        context, sample, args, initialization, initialization_mode
    )
    if args.lbfgs_refine:
        before_refine_gradient_norm = metadata["final_gradient_norm"]
        tensor_energies, refinement = refine_sample_density_lbfgs(
            sample,
            context.functional_factory,
            n_electron=int(np.sum(atomic_numbers) - charge),
            tolerance=args.lbfgs_tolerance,
            max_iterations=args.lbfgs_max_iterations,
            history_size=args.lbfgs_history_size,
        )
        legacy_energies = tensor_energies.detached()
        final_coeffs = transform_tensor_with_sample(
            sample, sample.coeffs, Representation.VECTOR, invert=True
        ).detach()
        trace.total_energies.extend(refinement.energy_trace)
        trace.gradient_norms.extend(refinement.gradient_norm_trace)
        metadata.update(
            {
                "pre_lbfgs_gradient_norm": before_refine_gradient_norm,
                "lbfgs_refined": True,
                "lbfgs_converged": refinement.converged,
                "lbfgs_closure_evaluations": refinement.closure_evaluations,
                "lbfgs_constraint_residual": refinement.constraint_residual,
                "lbfgs_final_gradient_norm": refinement.final_projected_gradient_norm,
                "final_gradient_norm": refinement.final_projected_gradient_norm,
                "converged": refinement.converged,
                "cycles": metadata["cycles"] + refinement.closure_evaluations,
                "final_total_energy": float(tensor_energies.total_energy.detach().cpu()),
            }
        )
    else:
        metadata.update(
            {
                "lbfgs_refined": False,
                "lbfgs_converged": None,
                "lbfgs_closure_evaluations": 0,
            }
        )
    if args.newton_refine:
        pre_newton_gradient_norm = metadata["final_gradient_norm"]
        tensor_energies, newton = refine_sample_density_newton_pcg(
            sample,
            context.functional_factory,
            n_electron=int(np.sum(atomic_numbers) - charge),
            tolerance=args.newton_tolerance,
            max_newton_iterations=args.newton_max_iterations,
            max_krylov_iterations=args.newton_max_krylov_iterations,
            krylov_tolerance=args.newton_krylov_tolerance,
            diagonal_probes=args.newton_diagonal_probes,
            damping=args.newton_damping,
        )
        legacy_energies = tensor_energies.detached()
        final_coeffs = transform_tensor_with_sample(
            sample, sample.coeffs, Representation.VECTOR, invert=True
        ).detach()
        trace.total_energies.extend(newton.energy_trace)
        trace.gradient_norms.extend(newton.gradient_norm_trace)
        metadata.update(
            {
                "pre_newton_gradient_norm": pre_newton_gradient_norm,
                "newton_refined": True,
                "newton_converged": newton.converged,
                "newton_energy_evaluations": newton.closure_evaluations,
                "newton_constraint_residual": newton.constraint_residual,
                "newton_final_gradient_norm": newton.final_projected_gradient_norm,
                "newton_krylov_iterations": newton.krylov_iterations,
                "newton_krylov_relative_residuals": newton.krylov_relative_residuals,
                "final_gradient_norm": newton.final_projected_gradient_norm,
                "converged": newton.converged,
                "cycles": metadata["cycles"] + newton.closure_evaluations,
                "final_total_energy": float(tensor_energies.total_energy.detach().cpu()),
            }
        )
    else:
        metadata.update(
            {
                "newton_refined": False,
                "newton_converged": None,
                "newton_energy_evaluations": 0,
            }
        )
    if metadata.get("response_predictor_fast_path_used"):
        fast_path_succeeded = bool(metadata.get("converged")) and math.isfinite(
            float(metadata.get("final_gradient_norm", math.inf))
        )
        metadata["response_predictor_fast_path_succeeded"] = fast_path_succeeded
        if not fast_path_succeeded:
            failed_fast_metadata = dict(metadata)
            failed_fast_cycles = int(metadata.get("cycles", 0))
            failed_fast_point_elapsed_s = time.perf_counter() - point_started
            failed_fast_optimization_elapsed_s = (
                time.perf_counter() - optimization_started
            )
            fallback_reason = (
                "fast_refiners_failed_strict_stationarity: "
                f"final_gradient_norm={metadata.get('final_gradient_norm')}"
            )
            fallback_args = argparse.Namespace(**vars(args))
            fallback_args.response_predictor_fast_refine = False

            # The 80-GiB training path cannot retain a failed model graph while
            # constructing the legacy fallback graph. Preserve scalar metadata,
            # then explicitly release all failed-path tensor owners first.
            sample.coeffs = sample.coeffs.detach()
            tensor_energies = None
            refinement = None
            newton = None
            legacy_energies = None
            final_coeffs = None
            trace = None
            sample = None
            metadata = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fallback = _evaluate_point(
                context,
                atomic_numbers,
                positions_bohr,
                charge,
                fallback_args,
                base_coeffs=base_coeffs,
                need_force=need_force,
                initial_coeffs=initial_coeffs,
                initialization_mode_override=initialization_mode_override,
            )
            fallback_metadata = fallback["optimization_metadata"]
            fallback_metadata["cycles"] = (
                int(fallback_metadata.get("cycles", 0)) + failed_fast_cycles
            )
            fallback_metadata.update(
                {
                    "response_predictor_fast_path_attempted": True,
                    "response_predictor_fast_path_used": True,
                    "response_predictor_fast_path_succeeded": False,
                    "response_predictor_fast_path_fallback_reason": fallback_reason,
                    "response_predictor_fast_path_initial_gradient_norm": failed_fast_metadata.get(
                        "response_predictor_fast_path_initial_gradient_norm"
                    ),
                    "response_predictor_fast_path_lbfgs_final_gradient_norm": failed_fast_metadata.get(
                        "lbfgs_final_gradient_norm"
                    ),
                    "response_predictor_fast_path_newton_final_gradient_norm": failed_fast_metadata.get(
                        "newton_final_gradient_norm"
                    ),
                    "response_predictor_fast_path_failed_cycles": failed_fast_cycles,
                    "response_predictor_fast_path_failed_lbfgs_closure_evaluations": failed_fast_metadata.get(
                        "lbfgs_closure_evaluations", 0
                    ),
                    "response_predictor_fast_path_failed_newton_energy_evaluations": failed_fast_metadata.get(
                        "newton_energy_evaluations", 0
                    ),
                    "response_predictor_fast_path_failed_newton_krylov_iterations": failed_fast_metadata.get(
                        "newton_krylov_iterations", []
                    ),
                    "response_predictor_fast_path_failed_point_elapsed_s": (
                        failed_fast_point_elapsed_s
                    ),
                }
            )
            fallback["total_point_elapsed_s"] = float(
                fallback["total_point_elapsed_s"] + failed_fast_point_elapsed_s
            )
            fallback["density_optimization_elapsed_s"] = float(
                fallback["density_optimization_elapsed_s"]
                + failed_fast_optimization_elapsed_s
            )
            return fallback
    density_optimization_elapsed_s = time.perf_counter() - optimization_started
    result: dict[str, Any] = {
        "legacy_total_energy": float(legacy_energies.total_energy),
        "final_coeffs": final_coeffs.detach().clone(),
        "optimization_metadata": metadata,
        "trace": trace,
        "sample_construction_elapsed_s": sample_construction_elapsed_s,
        "density_optimization_elapsed_s": density_optimization_elapsed_s,
        "density_driver_reported_elapsed_s": metadata.get("elapsed_s"),
    }
    if need_force:
        force_started = time.perf_counter()
        provider = FiniteDifferencePySCFIntegralProvider(
            atomic_numbers=atomic_numbers,
            basis=context.sample_generator.basis_info.basis_dict,
            charge=charge,
            derivative_step_bohr=args.integral_derivative_step,
            derivative_workers=args.integral_derivative_workers,
        )
        geometry = prepare_differentiable_geometry(
            context.sample_generator,
            atomic_numbers,
            positions_bohr,
            final_coeffs,
            integral_provider=provider,
            charge=charge,
        )
        total = evaluate_total_ofdft_force(
            context.functional_factory,
            geometry,
            n_electron=int(np.sum(atomic_numbers) - charge),
            model_geometry_fd_step_bohr=(
                args.model_geometry_fd_step
                if args.model_geometry_derivative == "numerical"
                else None
            ),
            model_geometry_fd_richardson=args.model_geometry_fd_richardson,
            sample_generator=context.sample_generator,
            charge=charge,
        )
        tensor_energy = float(total.energies.total_energy.detach().cpu())
        sample.pos = sample.pos.detach().clone().requires_grad_(True)
        sample.coeffs = sample.coeffs.detach().clone()
        with torch.enable_grad():
            _, _, _, incomplete_force = context.model.forward_predictions(
                sample,
                compute_density_gradients=False,
                compute_forces=True,
            )
        result.update(
            {
                "tensor_total_energy": tensor_energy,
                "tensor_legacy_energy_difference": tensor_energy
                - float(legacy_energies.total_energy),
                "total_force": total.force.detach().cpu().numpy(),
                "incomplete_force": incomplete_force.detach().cpu().numpy(),
                "electron_number": float(total.electron_number.detach().cpu()),
                "constraint_residual": float(total.constraint_residual.detach().cpu()),
                "tensor_projected_gradient_norm": float(
                    total.projected_density_gradient_norm.detach().cpu()
                ),
                "lagrange_multiplier": float(total.lagrange_multiplier.detach().cpu()),
                "model_geometry_derivative_mode": total.model_geometry_derivative_mode,
                "force_pipeline_elapsed_s": time.perf_counter() - force_started,
            }
        )
    result["total_point_elapsed_s"] = time.perf_counter() - point_started
    return result


def _loop_points(
    positions: np.ndarray, coordinate_i: int, coordinate_j: int, half_width: float
) -> list[tuple[str, np.ndarray]]:
    points = []
    for name, sign_i, sign_j in (
        ("minus_minus", -1.0, -1.0),
        ("plus_minus", 1.0, -1.0),
        ("plus_plus", 1.0, 1.0),
        ("minus_plus", -1.0, 1.0),
    ):
        flat = positions.reshape(-1).copy()
        flat[coordinate_i] += sign_i * half_width
        flat[coordinate_j] += sign_j * half_width
        points.append((name, flat.reshape(positions.shape)))
    return points


def _loop_work(
    points: list[tuple[str, np.ndarray]], forces: dict[str, np.ndarray]
) -> float:
    work = 0.0
    for index in range(4):
        name_a, position_a = points[index]
        name_b, position_b = points[(index + 1) % 4]
        displacement = position_b.reshape(-1) - position_a.reshape(-1)
        average_force = 0.5 * (
            forces[name_a].reshape(-1) + forces[name_b].reshape(-1)
        )
        work += float(np.dot(average_force, displacement))
    return work


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    molecule_ids = [value for item in args.molecules for value in item.split(",") if value]
    coordinates = [int(value) for value in args.coordinates.split(",") if value]
    loop_coordinates = tuple(int(value) for value in args.loop_coordinates.split(","))
    if len(loop_coordinates) != 2:
        raise ValueError("--loop-coordinates must contain exactly two flattened indices")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.optimization_trace_dir = args.output_dir / "optimization_traces"
    args.coefficients_dir = args.output_dir / "final_coefficients"
    point_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    loop_rows: list[dict[str, Any]] = []

    for context in contexts:
        for molecule_id in molecule_ids:
            atomic_numbers, positions = _load_geometry(
                args.dataset_dir, molecule_id, args.sample_id
            )
            charge = args.charge
            base = _evaluate_point(
                context,
                atomic_numbers,
                positions,
                charge,
                args,
                base_coeffs=None,
                need_force=True,
            )
            if args.save_final_coefficients:
                base["final_coefficients_file"] = _save_final_coefficients(
                    args.coefficients_dir
                    / f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_base.npz",
                    base["final_coeffs"],
                    atomic_numbers,
                    positions,
                )
            base_coeffs = base.pop("final_coeffs")
            base_trace = base.pop("trace")
            base_metadata = base.pop("optimization_metadata")
            base_trace_file = _save_optimization_trace(
                args.optimization_trace_dir,
                context,
                molecule_id,
                args.sample_id,
                None,
                "base_total_force",
                base_trace,
                base_metadata,
            )
            total_force = base.pop("total_force")
            incomplete_force = base.pop("incomplete_force")
            point_rows.append(
                {
                    "run": context.spec.name,
                    "molecule_id": molecule_id,
                    "point": "base",
                    "trace_file": base_trace_file,
                    **base,
                    **base_metadata,
                }
            )

            for coordinate in coordinates if args.run_force_fd else []:
                energies = {}
                coordinate_metadata = []
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    displaced = positions.reshape(-1).copy()
                    displaced[coordinate] += sign * args.force_fd_step
                    point = _evaluate_point(
                        context,
                        atomic_numbers,
                        displaced.reshape(positions.shape),
                        charge,
                        args,
                        base_coeffs=base_coeffs,
                        need_force=False,
                    )
                    if args.save_final_coefficients:
                        point["final_coefficients_file"] = _save_final_coefficients(
                            args.coefficients_dir
                            / (
                                f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_"
                                f"coord_{coordinate:03d}_{side}.npz"
                            ),
                            point["final_coeffs"],
                            atomic_numbers,
                            displaced.reshape(positions.shape),
                        )
                    point.pop("final_coeffs")
                    trace = point.pop("trace")
                    metadata = point.pop("optimization_metadata")
                    trace_file = _save_optimization_trace(
                        args.optimization_trace_dir,
                        context,
                        molecule_id,
                        args.sample_id,
                        coordinate,
                        f"energy_fd_{side}",
                        trace,
                        metadata,
                    )
                    energies[side] = point["legacy_total_energy"]
                    coordinate_metadata.append(metadata)
                    point_rows.append(
                        {
                            "run": context.spec.name,
                            "molecule_id": molecule_id,
                            "point": f"energy_fd_coord_{coordinate}_{side}",
                            "trace_file": trace_file,
                            **point,
                            **metadata,
                        }
                    )
                relaxed_energy_force = -(
                    energies["plus"] - energies["minus"]
                ) / (2.0 * args.force_fd_step)
                force_rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "coordinate": coordinate,
                        "force_fd_step": args.force_fd_step,
                        "total_force": float(total_force.reshape(-1)[coordinate]),
                        "incomplete_force": float(incomplete_force.reshape(-1)[coordinate]),
                        "relaxed_energy_fd_force": relaxed_energy_force,
                        "total_force_abs_error": abs(
                            float(total_force.reshape(-1)[coordinate]) - relaxed_energy_force
                        ),
                        "incomplete_force_abs_error": abs(
                            float(incomplete_force.reshape(-1)[coordinate])
                            - relaxed_energy_force
                        ),
                        "max_displaced_gradient_norm": max(
                            float(item["final_gradient_norm"])
                            for item in coordinate_metadata
                        ),
                    }
                )

            points = (
                _loop_points(
                    positions, loop_coordinates[0], loop_coordinates[1], args.loop_half_width
                )
                if args.run_loop
                else []
            )
            loop_forces: dict[str, dict[str, np.ndarray]] = {
                "total": {},
                "incomplete": {},
            }
            strict_corners = 0
            max_corner_gradient = 0.0
            strict_tolerance = (
                args.newton_tolerance
                if args.newton_refine
                else args.fallback_convergence_tolerance
            )
            for point_name, point_positions in points:
                point = _evaluate_point(
                    context,
                    atomic_numbers,
                    point_positions,
                    charge,
                    args,
                    base_coeffs=base_coeffs,
                    need_force=True,
                )
                if args.save_final_coefficients:
                    point["final_coefficients_file"] = _save_final_coefficients(
                        args.coefficients_dir
                        / (
                            f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_"
                            f"loop_{point_name}.npz"
                        ),
                        point["final_coeffs"],
                        atomic_numbers,
                        point_positions,
                    )
                point.pop("final_coeffs")
                trace = point.pop("trace")
                metadata = point.pop("optimization_metadata")
                loop_forces["total"][point_name] = point.pop("total_force")
                loop_forces["incomplete"][point_name] = point.pop("incomplete_force")
                gradient_norm = float(metadata["final_gradient_norm"])
                strict_corners += int(gradient_norm < strict_tolerance)
                max_corner_gradient = max(max_corner_gradient, gradient_norm)
                trace_file = _save_optimization_trace(
                    args.optimization_trace_dir,
                    context,
                    molecule_id,
                    args.sample_id,
                    None,
                    f"loop_{point_name}",
                    trace,
                    metadata,
                )
                point_rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "point": f"loop_{point_name}",
                        "trace_file": trace_file,
                        **point,
                        **metadata,
                    }
                )
            if points:
                area = (2.0 * args.loop_half_width) ** 2
                for force_definition, forces in loop_forces.items():
                    work = _loop_work(points, forces)
                    loop_rows.append(
                        {
                            "run": context.spec.name,
                            "molecule_id": molecule_id,
                            "force_definition": force_definition,
                            "coordinate_i": loop_coordinates[0],
                            "coordinate_j": loop_coordinates[1],
                            "half_width": args.loop_half_width,
                            "loop_work_hartree": work,
                            "abs_loop_work_hartree": abs(work),
                            "curl_estimate_hartree_per_bohr2": work / area,
                            "strict_corners": strict_corners,
                            "max_corner_gradient_norm": max_corner_gradient,
                            "strict_tolerance": strict_tolerance,
                        }
                    )
            message = [context.spec.name, molecule_id]
            relevant_force_rows = [
                row
                for row in force_rows
                if row["run"] == context.spec.name and row["molecule_id"] == molecule_id
            ]
            if relevant_force_rows:
                message.append(
                    "force_max_abs_error="
                    f"{max(row['total_force_abs_error'] for row in relevant_force_rows):.3e}"
                )
            if points:
                message.append(f"total_loop_work={loop_rows[-2]['loop_work_hartree']:.3e}")
            print(*message, flush=True)

    result = {
        "definition": (
            "Complete total-OFDFT force from one scalar tensor energy containing learned "
            "kin_plus_xc, Hartree, electron-nuclear, nuclear-repulsion, geometry-dependent "
            "overlap/natural basis, electron constraint, and numerical moving-basis/Pulay "
            "integral derivatives. Density is strictly optimized before force evaluation."
        ),
        "runs": args.run,
        "molecules": molecule_ids,
        "sample_id": args.sample_id,
        "force_fd_step": args.force_fd_step,
        "integral_derivative_step": args.integral_derivative_step,
        "model_geometry_derivative": args.model_geometry_derivative,
        "loop_half_width": args.loop_half_width,
        "point_rows": point_rows,
        "force_rows": force_rows,
        "loop_rows": loop_rows,
        "wall_time_s": time.perf_counter() - evaluation_started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "points.csv", point_rows)
    _write_csv(args.output_dir / "force_comparison.csv", force_rows)
    _write_csv(args.output_dir / "loop_comparison.csv", loop_rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--molecules", action="append", required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-final-coefficients", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--coordinates", default="0,1")
    parser.add_argument("--loop-coordinates", default="0,1")
    parser.add_argument("--force-fd-step", type=float, default=1e-3)
    parser.add_argument("--integral-derivative-step", type=float, default=1e-4)
    parser.add_argument("--integral-derivative-workers", type=int, default=1)
    parser.add_argument(
        "--model-geometry-derivative",
        choices=["autograd", "numerical"],
        default="autograd",
        help="Differentiate the learned model preprocessing with autograd or scalar FD.",
    )
    parser.add_argument("--model-geometry-fd-step", type=float, default=1e-6)
    parser.add_argument(
        "--model-geometry-fd-richardson",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lbfgs-refine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lbfgs-tolerance", type=float, default=1e-8)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=200)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument(
        "--newton-refine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--newton-tolerance", type=float, default=1e-8)
    parser.add_argument("--newton-max-iterations", type=int, default=3)
    parser.add_argument("--newton-max-krylov-iterations", type=int, default=100)
    parser.add_argument("--newton-krylov-tolerance", type=float, default=1e-10)
    parser.add_argument("--newton-diagonal-probes", type=int, default=4)
    parser.add_argument("--newton-damping", type=float, default=1e-8)
    parser.add_argument("--loop-half-width", type=float, default=1e-3)
    parser.add_argument(
        "--run-force-fd", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--run-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument(
        "--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--max-cycle", type=int, default=1000)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fallback-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--fallback-max-cycle", type=int, default=10000)
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--fallback-lr", type=float, default=3e-4)
    parser.add_argument(
        "--fallback-always", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
