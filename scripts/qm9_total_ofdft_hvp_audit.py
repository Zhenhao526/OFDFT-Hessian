#!/usr/bin/env python3
"""Compare total-OFDFT implicit density-response HVP with strict relaxed finite differences."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from mldft.ofdft.conservative_force import (
    evaluate_total_ofdft_force,
    prepare_differentiable_geometry,
    prepare_fixed_geometry_scalar_energy,
)
from mldft.ofdft.geometry_integrals import FiniteDifferencePySCFIntegralProvider
from mldft.ofdft.implicit_response import FiniteDifferenceGeometryResponseSystem

from qm9_hessian_density_relaxed_eval import (
    _load_context,
    _load_geometry,
    _parse_run,
    _save_optimization_trace,
)
from qm9_total_ofdft_force_audit import _evaluate_point, _save_final_coefficients


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _vector_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference)
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_frobenius": float(
            np.linalg.norm(difference) / max(reference_norm, np.finfo(float).tiny)
        ),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _directional_curvatures(
    plus_energy: float,
    base_energy: float,
    minus_energy: float,
    plus_force: np.ndarray,
    minus_force: np.ndarray,
    direction: np.ndarray,
    step_bohr: float,
) -> dict[str, float]:
    energy_curvature = (
        plus_energy - 2.0 * base_energy + minus_energy
    ) / step_bohr**2
    force_hvp = -(plus_force - minus_force) / (2.0 * step_bohr)
    force_curvature = float(np.sum(direction * force_hvp))
    scale = max(abs(energy_curvature), abs(force_curvature), np.finfo(float).tiny)
    return {
        "energy_curvature": float(energy_curvature),
        "force_curvature": force_curvature,
        "absolute_difference": abs(float(energy_curvature) - force_curvature),
        "relative_difference": abs(float(energy_curvature) - force_curvature) / scale,
    }


def _load_label_coefficients(
    dataset_dir: Path, molecule_id: str, sample_id: int, device: torch.device
) -> torch.Tensor:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    root = zarr.open(label_path, mode="r")
    coefficients = np.asarray(root["of_labels/spatial/coeffs"][-1], dtype=np.float64)
    return torch.as_tensor(coefficients, dtype=torch.float64, device=device)


def _select_direction(
    payload: Any, direction_index: int, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, str, int]:
    directions = np.asarray(payload["direction"], dtype=np.float64)
    if directions.ndim == 2:
        directions = directions[None, ...]
    if directions.ndim != 3 or tuple(directions.shape[1:]) != expected_shape:
        raise ValueError(
            f"Direction array shape {directions.shape} is incompatible with {expected_shape}"
        )
    if direction_index < 0 or direction_index >= directions.shape[0]:
        raise IndexError(
            f"direction_index={direction_index} outside [0, {directions.shape[0]})"
        )
    kinds = np.asarray(payload.get("direction_kind", "sidecar"))
    if kinds.ndim == 0:
        kind = str(kinds.item())
    else:
        kind = str(kinds.reshape(-1)[direction_index])
    return directions[direction_index], kind, int(directions.shape[0])


def _force_at_fixed_coefficients(
    context: Any,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    coefficients: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | str]]:
    started = time.perf_counter()
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers,
        context.sample_generator.basis_info.basis_dict,
        charge=args.charge,
        derivative_step_bohr=args.integral_derivative_step,
        derivative_workers=args.integral_derivative_workers,
    )
    geometry = prepare_differentiable_geometry(
        context.sample_generator,
        atomic_numbers,
        positions_bohr,
        coefficients,
        integral_provider=provider,
        charge=args.charge,
    )
    result = evaluate_total_ofdft_force(
        context.functional_factory,
        geometry,
        n_electron=int(np.sum(atomic_numbers) - args.charge),
        model_geometry_fd_step_bohr=(
            args.model_geometry_fd_step
            if args.model_geometry_derivative == "numerical"
            else None
        ),
        model_geometry_fd_richardson=args.model_geometry_fd_richardson,
        sample_generator=context.sample_generator,
        charge=args.charge,
    )
    metadata: dict[str, float | str] = {
        "elapsed_s": time.perf_counter() - started,
        "projected_density_gradient_norm": float(
            result.projected_density_gradient_norm.detach().cpu()
        ),
        "constraint_residual": float(result.constraint_residual.detach().cpu()),
        "total_energy": float(result.energies.total_energy.detach().cpu()),
        "model_geometry_derivative_mode": result.model_geometry_derivative_mode,
    }
    return result.force.detach().cpu().numpy(), metadata


def _extract_optimized_point(
    point: dict[str, Any],
    output_dir: Path,
    context: Any,
    molecule_id: str,
    sample_id: int,
    point_name: str,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray | None, float, dict[str, Any]]:
    coefficients = point.pop("final_coeffs")
    trace = point.pop("trace")
    metadata = point.pop("optimization_metadata")
    force = point.pop("total_force", None)
    point.pop("incomplete_force", None)
    trace_file = _save_optimization_trace(
        output_dir / "optimization_traces",
        context,
        molecule_id,
        sample_id,
        None,
        point_name,
        trace,
        metadata,
    )
    coefficients_file = _save_final_coefficients(
        output_dir
        / "final_coefficients"
        / f"{context.spec.name}_{molecule_id}_{sample_id:07d}_{point_name}.npz",
        coefficients,
        atomic_numbers,
        positions_bohr,
    )
    row = {
        "run": context.spec.name,
        "molecule_id": molecule_id,
        "point": point_name,
        "trace_file": trace_file,
        "final_coefficients_file": coefficients_file,
        **point,
        **metadata,
    }
    return coefficients, force, float(point["legacy_total_energy"]), row


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.optimization_trace_dir = args.output_dir / "optimization_traces"
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    molecule_ids = [value for item in args.molecules for value in item.split(",") if value]
    point_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curvature_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for context in contexts:
        for molecule_id in molecule_ids:
            molecule_started = time.perf_counter()
            atomic_numbers, positions = _load_geometry(
                args.dataset_dir, molecule_id, args.sample_id
            )
            direction_kind = "cartesian_coordinate"
            if args.direction_sidecar_dir is not None:
                direction_file = (
                    args.direction_sidecar_dir
                    / f"{molecule_id}.{args.sample_id:07d}.npz"
                )
                if not direction_file.exists():
                    raise FileNotFoundError(direction_file)
                direction_payload = np.load(direction_file)
                direction, direction_kind, direction_count = _select_direction(
                    direction_payload, args.direction_index, tuple(positions.shape)
                )
            else:
                direction = np.zeros_like(positions)
                direction.reshape(-1)[args.direction_coordinate] = 1.0
                direction_count = 1
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 0:
                raise ValueError(f"Zero HVP direction for {molecule_id}")
            direction = direction / direction_norm
            displaced_positions = {
                "plus": positions + args.hvp_step * direction,
                "minus": positions - args.hvp_step * direction,
            }

            base_initial_coefficients = None
            if args.base_initialization == "label_reference":
                base_initial_coefficients = _load_label_coefficients(
                    args.dataset_dir, molecule_id, args.sample_id, device
                )

            if args.base_coefficients_file is None:
                base_point = _evaluate_point(
                    context,
                    atomic_numbers,
                    positions,
                    args.charge,
                    args,
                    base_coeffs=None,
                    need_force=False,
                    initial_coeffs=base_initial_coefficients,
                    initialization_mode_override=args.base_initialization,
                )
                base_coefficients, _, base_energy, base_row = _extract_optimized_point(
                    base_point,
                    args.output_dir,
                    context,
                    molecule_id,
                    args.sample_id,
                    "base",
                    atomic_numbers,
                    positions,
                )
                point_rows.append(base_row)
            else:
                cached = np.load(args.base_coefficients_file)
                cached_atomic_numbers = np.asarray(cached["atomic_numbers"], dtype=np.int64)
                cached_positions = np.asarray(cached["positions_bohr"], dtype=np.float64)
                if not np.array_equal(cached_atomic_numbers, atomic_numbers) or not np.allclose(
                    cached_positions, positions, atol=0.0, rtol=0.0
                ):
                    raise ValueError(
                        "Cached base coefficients do not match the requested molecule geometry"
                    )
                base_coefficients = torch.as_tensor(
                    cached["coefficients"], dtype=torch.float64, device=device
                )
                cached_base_energy = prepare_fixed_geometry_scalar_energy(
                    context.sample_generator,
                    context.functional_factory,
                    atomic_numbers,
                    positions,
                    charge=args.charge,
                )
                base_energy = float(
                    cached_base_energy(base_coefficients).detach().cpu()
                )
                point_rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "point": "base_reused",
                        "legacy_total_energy": base_energy,
                        "final_coefficients_file": str(args.base_coefficients_file),
                    }
                )

            relaxed: dict[str, dict[str, Any]] = {}
            for side in ("plus", "minus"):
                displaced_base_coefficients = (
                    base_coefficients
                    if args.displaced_initialization == "base_continuation"
                    else None
                )
                point = _evaluate_point(
                    context,
                    atomic_numbers,
                    displaced_positions[side],
                    args.charge,
                    args,
                    base_coeffs=displaced_base_coefficients,
                    need_force=True,
                    initialization_mode_override=args.displaced_initialization,
                )
                coefficients, force, energy, row = _extract_optimized_point(
                    point,
                    args.output_dir,
                    context,
                    molecule_id,
                    args.sample_id,
                    f"relaxed_{side}",
                    atomic_numbers,
                    displaced_positions[side],
                )
                point_rows.append(row)
                relaxed[side] = {
                    "coefficients": coefficients,
                    "force": force,
                    "energy": energy,
                    "metadata": row,
                }

            curvature_scan: list[dict[str, Any]] = []
            relaxed_hvp_step_scan: list[np.ndarray] = []
            curvature_steps = list(dict.fromkeys([args.hvp_step, *args.curvature_step]))
            for curvature_step in curvature_steps:
                if curvature_step <= 0.0:
                    raise ValueError("Curvature steps must be positive")
                if curvature_step == args.hvp_step:
                    curvature_points = relaxed
                else:
                    curvature_points: dict[str, dict[str, Any]] = {}
                    step_tag = (
                        f"{curvature_step:.1e}".replace(".", "p").replace("-", "m")
                    )
                    for side, sign in (("plus", 1.0), ("minus", -1.0)):
                        geometry = positions + sign * curvature_step * direction
                        point = _evaluate_point(
                            context,
                            atomic_numbers,
                            geometry,
                            args.charge,
                            args,
                            base_coeffs=(
                                base_coefficients
                                if args.displaced_initialization == "base_continuation"
                                else None
                            ),
                            need_force=True,
                            initialization_mode_override=args.displaced_initialization,
                        )
                        coefficients, force, energy, row = _extract_optimized_point(
                            point,
                            args.output_dir,
                            context,
                            molecule_id,
                            args.sample_id,
                            f"curvature_h{step_tag}_{side}",
                            atomic_numbers,
                            geometry,
                        )
                        row["curvature_step_bohr"] = curvature_step
                        point_rows.append(row)
                        curvature_points[side] = {
                            "coefficients": coefficients,
                            "force": force,
                            "energy": energy,
                            "metadata": row,
                        }
                curvature = _directional_curvatures(
                    curvature_points["plus"]["energy"],
                    base_energy,
                    curvature_points["minus"]["energy"],
                    curvature_points["plus"]["force"],
                    curvature_points["minus"]["force"],
                    direction,
                    curvature_step,
                )
                relaxed_hvp_step_scan.append(
                    -(
                        curvature_points["plus"]["force"]
                        - curvature_points["minus"]["force"]
                    )
                    / (2.0 * curvature_step)
                )
                curvature_row = {
                    "run": context.spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": args.sample_id,
                    "natoms": int(atomic_numbers.size),
                    "direction_coordinate": args.direction_coordinate,
                    "direction_kind": direction_kind,
                    "step_bohr": curvature_step,
                    **curvature,
                    "max_projected_gradient_norm": max(
                        float(curvature_points[side]["metadata"]["final_gradient_norm"])
                        for side in ("plus", "minus")
                    ),
                    "max_cycles": max(
                        int(curvature_points[side]["metadata"]["cycles"])
                        for side in ("plus", "minus")
                    ),
                }
                curvature_scan.append(curvature_row)
                curvature_rows.append(curvature_row)

            prepared_geometries: dict[bytes, Any] = {}

            def energy_function(coefficients: torch.Tensor, geometry: np.ndarray) -> torch.Tensor:
                geometry = np.asarray(geometry, dtype=np.float64)
                key = geometry.tobytes()
                if key not in prepared_geometries:
                    prepared_geometries[key] = prepare_fixed_geometry_scalar_energy(
                        context.sample_generator,
                        context.functional_factory,
                        atomic_numbers,
                        geometry,
                        charge=args.charge,
                    )
                return prepared_geometries[key](coefficients)

            base_prepared = prepare_fixed_geometry_scalar_energy(
                context.sample_generator,
                context.functional_factory,
                atomic_numbers,
                positions,
                charge=args.charge,
            )
            base_coulomb_matrix = FiniteDifferencePySCFIntegralProvider(
                atomic_numbers,
                context.sample_generator.basis_info.basis_dict,
                charge=args.charge,
                derivative_step_bohr=args.integral_derivative_step,
                derivative_workers=args.integral_derivative_workers,
            ).evaluate(positions).coulomb
            response_coefficients = base_coefficients.to(
                device=base_prepared.normalization_untransformed.device,
                dtype=base_prepared.normalization_untransformed.dtype,
            )
            response_system = FiniteDifferenceGeometryResponseSystem(
                energy_function=energy_function,
                coeffs=response_coefficients,
                positions_bohr=positions,
                normalization=base_prepared.normalization_untransformed,
                n_electron=int(np.sum(atomic_numbers) - args.charge),
            )
            response_started = time.perf_counter()
            pcg_response = None
            tangent_minres_response = None
            kkt_minres_response = None
            response_attempts: list[dict[str, Any]] = []

            def record_attempt(attempt: Any) -> None:
                response_attempts.append(
                    {**vars(attempt.krylov), "solution": "omitted"}
                )

            if args.response_solver == "dense":
                response = response_system.solve_tangent_dense_reference(
                    direction, geometry_step_bohr=args.mixed_derivative_step
                )
            elif args.response_solver == "deflated":
                low_eigenvalues, low_eigenvectors = (
                    response_system.estimate_low_tangent_modes(
                        args.deflation_modes,
                        tolerance=args.eigensolver_tolerance,
                        max_iterations=args.eigensolver_max_iterations,
                    )
                )
                response_system.low_tangent_eigenvalues = low_eigenvalues.detach()
                inverse_diagonal = response_system.estimate_tangent_inverse_diagonal(
                    probes=args.preconditioner_probes,
                    damping=args.preconditioner_damping,
                    seed=args.seed,
                )
                response = response_system.solve_tangent_deflated_pcg(
                    direction,
                    geometry_step_bohr=args.mixed_derivative_step,
                    deflation_vectors=low_eigenvectors,
                    tolerance=args.krylov_tolerance,
                    max_iterations=args.max_krylov_iterations,
                    tangent_inverse_diagonal=inverse_diagonal,
                )
            elif args.response_solver == "minres":
                response = response_system.solve_kkt_minres(
                    direction,
                    geometry_step_bohr=args.mixed_derivative_step,
                    tolerance=args.krylov_tolerance,
                    max_iterations=args.max_krylov_iterations,
                )
            elif args.response_solver == "tangent-minres":
                inverse_diagonal = response_system.estimate_tangent_inverse_diagonal(
                    probes=args.preconditioner_probes,
                    damping=args.preconditioner_damping,
                    seed=args.seed,
                )
                response = response_system.solve_tangent_minres(
                    direction,
                    geometry_step_bohr=args.mixed_derivative_step,
                    tolerance=args.krylov_tolerance,
                    max_iterations=args.max_krylov_iterations,
                    tangent_inverse_diagonal=inverse_diagonal,
                )
            else:
                inverse_diagonal = response_system.estimate_tangent_inverse_diagonal(
                    probes=args.preconditioner_probes,
                    damping=args.preconditioner_damping,
                    seed=args.seed,
                )
                pcg_response = response_system.solve_tangent_pcg(
                    direction,
                    geometry_step_bohr=args.mixed_derivative_step,
                    tolerance=args.krylov_tolerance,
                    max_iterations=args.max_krylov_iterations,
                    tangent_inverse_diagonal=inverse_diagonal,
                )
                response = pcg_response
                record_attempt(response)
                if not response.krylov.converged and args.minres_fallback:
                    tangent_minres_response = response_system.solve_tangent_minres(
                        direction,
                        geometry_step_bohr=args.mixed_derivative_step,
                        tolerance=args.krylov_tolerance,
                        max_iterations=args.max_krylov_iterations,
                        tangent_inverse_diagonal=inverse_diagonal,
                        initial_guess=pcg_response.krylov.solution,
                    )
                    response = tangent_minres_response
                    record_attempt(response)
                if not response.krylov.converged and args.minres_fallback:
                    kkt_minres_response = response_system.solve_kkt_minres(
                        direction,
                        geometry_step_bohr=args.mixed_derivative_step,
                        tolerance=args.krylov_tolerance,
                        max_iterations=args.max_krylov_iterations,
                    )
                    response = kkt_minres_response
                    record_attempt(response)
                if (
                    not response.krylov.converged
                    and args.dense_fallback_max_coefficients > 0
                    and response_coefficients.numel()
                    <= args.dense_fallback_max_coefficients
                ):
                    response = response_system.solve_tangent_dense_reference(
                        direction,
                        geometry_step_bohr=args.mixed_derivative_step,
                        compute_spectrum=False,
                    )
                    record_attempt(response)
            if not response_attempts:
                record_attempt(response)
            response_elapsed_s = time.perf_counter() - response_started
            if not response.krylov.converged and args.require_response_convergence:
                pcg_payload = (
                    {**vars(pcg_response.krylov), "solution": "omitted"}
                    if pcg_response is not None
                    else None
                )
                final_payload = {**vars(response.krylov), "solution": "omitted"}
                failure = {
                    "run": context.spec.name,
                    "molecule_id": molecule_id,
                    "pcg": pcg_payload,
                    "attempts": response_attempts,
                    "final_solver": final_payload,
                    "stationarity_direction_residual": response.stationarity_direction_residual,
                    "constraint_direction_residual": response.constraint_direction_residual,
                }
                (args.output_dir / "response_failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n"
                )
                raise RuntimeError(
                    "Density response solver did not converge; see response_failure.json"
                )

            fixed_forces: dict[str, np.ndarray] = {}
            fixed_force_metadata: dict[str, dict[str, Any]] = {}
            implicit_forces: dict[str, np.ndarray] = {}
            implicit_force_metadata: dict[str, dict[str, Any]] = {}
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                fixed_forces[side], fixed_force_metadata[side] = (
                    _force_at_fixed_coefficients(
                        context,
                        atomic_numbers,
                        displaced_positions[side],
                        response_coefficients,
                        args,
                    )
                )
                predicted_coefficients = (
                    response_coefficients
                    + sign * args.hvp_step * response.density_response
                )
                implicit_forces[side], implicit_force_metadata[side] = (
                    _force_at_fixed_coefficients(
                        context,
                        atomic_numbers,
                        displaced_positions[side],
                        predicted_coefficients,
                        args,
                    )
                )

            denominator = 2.0 * args.hvp_step
            relaxed_hvp = -(
                relaxed["plus"]["force"] - relaxed["minus"]["force"]
            ) / denominator
            fixed_hvp = -(fixed_forces["plus"] - fixed_forces["minus"]) / denominator
            implicit_hvp = -(
                implicit_forces["plus"] - implicit_forces["minus"]
            ) / denominator
            relaxed_density_response = (
                relaxed["plus"]["coefficients"].to(response_coefficients)
                - relaxed["minus"]["coefficients"].to(response_coefficients)
            ) / denominator
            local_curvature = curvature_scan[0]
            energy_curvature = local_curvature["energy_curvature"]
            force_curvature = local_curvature["force_curvature"]

            implicit_metrics = _vector_metrics(implicit_hvp, relaxed_hvp)
            fixed_metrics = _vector_metrics(fixed_hvp, relaxed_hvp)
            pbe_reference_file = None
            pbe_hvp = None
            relaxed_vs_pbe = None
            implicit_vs_pbe = None
            fixed_vs_pbe = None
            if args.reference_dir is not None:
                pbe_reference_file = (
                    args.reference_dir
                    / f"pbe_hessian_{molecule_id}_{args.sample_id:07d}.npz"
                )
                if not pbe_reference_file.exists():
                    raise FileNotFoundError(pbe_reference_file)
                pbe_hessian = np.asarray(
                    np.load(pbe_reference_file)["pbe_hessian"], dtype=np.float64
                )
                if pbe_hessian.shape != (positions.size, positions.size):
                    raise ValueError(
                        f"PBE Hessian shape {pbe_hessian.shape} does not match "
                        f"{positions.size} coordinates for {molecule_id}"
                    )
                pbe_hvp = (pbe_hessian @ direction.reshape(-1)).reshape(
                    positions.shape
                )
                relaxed_vs_pbe = _vector_metrics(relaxed_hvp, pbe_hvp)
                implicit_vs_pbe = _vector_metrics(implicit_hvp, pbe_hvp)
                fixed_vs_pbe = _vector_metrics(fixed_hvp, pbe_hvp)
            density_response_metrics = _vector_metrics(
                response.density_response.detach().cpu().numpy(),
                relaxed_density_response.detach().cpu().numpy(),
            )
            dense_eigenvalues = getattr(
                response_system, "last_dense_tangent_eigenvalues", None
            )
            low_eigenvalues = getattr(
                response_system, "low_tangent_eigenvalues", None
            )
            summary = {
                "run": context.spec.name,
                "molecule_id": molecule_id,
                "sample_id": args.sample_id,
                "natoms": int(atomic_numbers.size),
                "direction_coordinate": args.direction_coordinate,
                "direction_kind": direction_kind,
                "direction_index": args.direction_index,
                "direction_count": direction_count,
                "base_initialization": args.base_initialization,
                "displaced_initialization": args.displaced_initialization,
                "hvp_step_bohr": args.hvp_step,
                "mixed_derivative_step_bohr": args.mixed_derivative_step,
                "base_projected_gradient_norm": float(
                    torch.linalg.vector_norm(response_system.projected_gradient).detach().cpu()
                ),
                "base_constraint_residual": float(
                    torch.abs(response_system.constraint).detach().cpu()
                ),
                "response_solver": response.krylov.method,
                "response_attempts": response_attempts,
                "response_converged": response.krylov.converged,
                "response_iterations": response.krylov.iterations,
                "response_relative_residual": response.krylov.relative_residual,
                "stationarity_direction_residual": response.stationarity_direction_residual,
                "constraint_direction_residual": response.constraint_direction_residual,
                "response_elapsed_s": response_elapsed_s,
                "dense_tangent_eigenvalue_min": (
                    float(dense_eigenvalues.min().cpu())
                    if dense_eigenvalues is not None
                    else None
                ),
                "dense_tangent_eigenvalue_max": (
                    float(dense_eigenvalues.max().cpu())
                    if dense_eigenvalues is not None
                    else None
                ),
                "dense_tangent_condition_number": (
                    float(
                        torch.max(torch.abs(dense_eigenvalues))
                        / torch.min(torch.abs(dense_eigenvalues))
                    )
                    if dense_eigenvalues is not None
                    else None
                ),
                "deflation_modes": (
                    int(low_eigenvalues.numel()) if low_eigenvalues is not None else 0
                ),
                "deflation_eigenvalue_min": (
                    float(low_eigenvalues.min().cpu())
                    if low_eigenvalues is not None
                    else None
                ),
                "deflation_eigenvalue_max": (
                    float(low_eigenvalues.max().cpu())
                    if low_eigenvalues is not None
                    else None
                ),
                "pcg_converged": (
                    pcg_response.krylov.converged if pcg_response is not None else None
                ),
                "pcg_iterations": (
                    pcg_response.krylov.iterations if pcg_response is not None else None
                ),
                "pcg_relative_residual": (
                    pcg_response.krylov.relative_residual
                    if pcg_response is not None
                    else None
                ),
                "pcg_breakdown": (
                    pcg_response.krylov.breakdown if pcg_response is not None else None
                ),
                "relaxed_energy_curvature": energy_curvature,
                "relaxed_force_hvp_directional_curvature": force_curvature,
                "curvature_abs_difference": abs(energy_curvature - force_curvature),
                "curvature_step_scan": curvature_scan,
                "implicit_vs_relaxed": implicit_metrics,
                "fixed_vs_relaxed": fixed_metrics,
                "pbe_reference_file": (
                    str(pbe_reference_file) if pbe_reference_file is not None else None
                ),
                "strict_relaxed_vs_pbe": relaxed_vs_pbe,
                "implicit_vs_pbe": implicit_vs_pbe,
                "fixed_vs_pbe": fixed_vs_pbe,
                "density_response_vs_relaxed": density_response_metrics,
                "fixed_force_metadata": fixed_force_metadata,
                "implicit_force_metadata": implicit_force_metadata,
                "strict_relaxed_max_gradient_norm": max(
                    float(relaxed[side]["metadata"]["final_gradient_norm"])
                    for side in ("plus", "minus")
                ),
                "wall_time_s": time.perf_counter() - molecule_started,
            }
            summaries.append(summary)
            metric_rows.append(
                {
                    "run": context.spec.name,
                    "molecule_id": molecule_id,
                    "natoms": int(atomic_numbers.size),
                    "direction_kind": direction_kind,
                    "response_solver": response.krylov.method,
                    "response_iterations": response.krylov.iterations,
                    "response_relative_residual": response.krylov.relative_residual,
                    "implicit_hvp_mae": implicit_metrics["mae"],
                    "implicit_hvp_rmse": implicit_metrics["rmse"],
                    "implicit_hvp_relative_frobenius": implicit_metrics[
                        "relative_frobenius"
                    ],
                    "fixed_hvp_mae": fixed_metrics["mae"],
                    "fixed_hvp_rmse": fixed_metrics["rmse"],
                    "fixed_hvp_relative_frobenius": fixed_metrics[
                        "relative_frobenius"
                    ],
                    "strict_relaxed_vs_pbe_mae": (
                        relaxed_vs_pbe["mae"] if relaxed_vs_pbe is not None else None
                    ),
                    "strict_relaxed_vs_pbe_rmse": (
                        relaxed_vs_pbe["rmse"] if relaxed_vs_pbe is not None else None
                    ),
                    "strict_relaxed_vs_pbe_relative_frobenius": (
                        relaxed_vs_pbe["relative_frobenius"]
                        if relaxed_vs_pbe is not None
                        else None
                    ),
                    "implicit_vs_pbe_mae": (
                        implicit_vs_pbe["mae"] if implicit_vs_pbe is not None else None
                    ),
                    "implicit_vs_pbe_rmse": (
                        implicit_vs_pbe["rmse"] if implicit_vs_pbe is not None else None
                    ),
                    "implicit_vs_pbe_relative_frobenius": (
                        implicit_vs_pbe["relative_frobenius"]
                        if implicit_vs_pbe is not None
                        else None
                    ),
                    "fixed_vs_pbe_mae": (
                        fixed_vs_pbe["mae"] if fixed_vs_pbe is not None else None
                    ),
                    "fixed_vs_pbe_rmse": (
                        fixed_vs_pbe["rmse"] if fixed_vs_pbe is not None else None
                    ),
                    "fixed_vs_pbe_relative_frobenius": (
                        fixed_vs_pbe["relative_frobenius"]
                        if fixed_vs_pbe is not None
                        else None
                    ),
                    "density_response_relative_frobenius": density_response_metrics[
                        "relative_frobenius"
                    ],
                    "energy_curvature": energy_curvature,
                    "force_curvature": force_curvature,
                    "curvature_abs_difference": abs(energy_curvature - force_curvature),
                    "strict_relaxed_max_gradient_norm": summary[
                        "strict_relaxed_max_gradient_norm"
                    ],
                    "wall_time_s": summary["wall_time_s"],
                }
            )
            np.savez_compressed(
                args.output_dir
                / f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_hvp_arrays.npz",
                atomic_numbers=atomic_numbers,
                positions_bohr=positions,
                direction=direction,
                direction_index=np.asarray(args.direction_index, dtype=np.int64),
                direction_kind=np.asarray(direction_kind),
                curvature_steps_bohr=np.asarray(curvature_steps, dtype=np.float64),
                relaxed_hvp_step_scan=np.stack(relaxed_hvp_step_scan),
                relaxed_hvp=relaxed_hvp,
                fixed_hvp=fixed_hvp,
                implicit_hvp=implicit_hvp,
                pbe_hvp=(
                    np.asarray([], dtype=np.float64) if pbe_hvp is None else pbe_hvp
                ),
                implicit_density_response=response.density_response.detach().cpu().numpy(),
                relaxed_density_response=relaxed_density_response.detach().cpu().numpy(),
                base_coefficients=base_coefficients.detach().cpu().numpy(),
                relaxed_plus_coefficients=relaxed["plus"]["coefficients"].detach().cpu().numpy(),
                relaxed_minus_coefficients=relaxed["minus"]["coefficients"].detach().cpu().numpy(),
                relaxed_plus_force=relaxed["plus"]["force"],
                relaxed_minus_force=relaxed["minus"]["force"],
                base_coulomb_matrix=base_coulomb_matrix,
            )
            print(
                context.spec.name,
                molecule_id,
                f"solver={response.krylov.method}:{response.krylov.iterations}",
                f"implicit_rel_fro={implicit_metrics['relative_frobenius']:.3e}",
                f"curvature_error={abs(energy_curvature - force_curvature):.3e}",
                flush=True,
            )

    result = {
        "definition": (
            "Total-OFDFT HVP from an autograd coefficient Hessian, rebuilt-geometry numerical "
            "E_cR mixed derivative, constrained KKT density response, and finite difference of "
            "complete scalar-derived total forces at linearly responded densities. The learned "
            f"model geometry derivative uses {args.model_geometry_derivative}."
        ),
        "model_geometry_derivative": args.model_geometry_derivative,
        "direction_index": args.direction_index,
        "base_initialization": args.base_initialization,
        "displaced_initialization": args.displaced_initialization,
        "runs": args.run,
        "molecules": molecule_ids,
        "reference_dir": (
            str(args.reference_dir.resolve()) if args.reference_dir is not None else None
        ),
        "summaries": summaries,
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
    _write_csv(args.output_dir / "metrics.csv", metric_rows)
    _write_csv(args.output_dir / "curvature_scan.csv", curvature_rows)
    _write_csv(args.output_dir / "points.csv", point_rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--molecules", action="append", required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional directory containing pbe_hessian_<molecule>_<sample>.npz files.",
    )
    parser.add_argument(
        "--direction-sidecar-dir",
        type=Path,
        default=None,
        help="Optional directory containing normalized per-molecule HVP direction sidecars.",
    )
    parser.add_argument("--base-coefficients-file", type=Path)
    parser.add_argument("--direction-coordinate", type=int, default=0)
    parser.add_argument("--direction-index", type=int, default=0)
    parser.add_argument(
        "--base-initialization",
        choices=["configured", "label_reference"],
        default="configured",
    )
    parser.add_argument(
        "--displaced-initialization",
        choices=["base_continuation", "configured"],
        default="base_continuation",
    )
    parser.add_argument("--hvp-step", type=float, default=1e-3)
    parser.add_argument(
        "--curvature-step",
        action="append",
        type=float,
        default=[],
        help=(
            "Additional strictly relaxed R+/-h directional points for scalar-energy versus "
            "total-force curvature stability. May be repeated."
        ),
    )
    parser.add_argument("--mixed-derivative-step", type=float, default=1e-4)
    parser.add_argument("--integral-derivative-step", type=float, default=1e-4)
    parser.add_argument("--integral-derivative-workers", type=int, default=1)
    parser.add_argument(
        "--model-geometry-derivative",
        choices=["autograd", "numerical"],
        default="autograd",
        help="Differentiate learned-model preprocessing by autograd or scalar FD in force calls.",
    )
    parser.add_argument("--model-geometry-fd-step", type=float, default=1e-6)
    parser.add_argument(
        "--model-geometry-fd-richardson",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--preconditioner-probes", type=int, default=8)
    parser.add_argument("--preconditioner-damping", type=float, default=1e-8)
    parser.add_argument("--krylov-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-krylov-iterations", type=int, default=300)
    parser.add_argument(
        "--response-solver",
        choices=["auto", "deflated", "minres", "tangent-minres", "dense"],
        default="auto",
    )
    parser.add_argument(
        "--dense-fallback-max-coefficients",
        type=int,
        default=1024,
        help="For auto mode, assemble a dense tangent solve after Krylov failures up to this size.",
    )
    parser.add_argument("--deflation-modes", type=int, default=24)
    parser.add_argument("--eigensolver-tolerance", type=float, default=1e-6)
    parser.add_argument("--eigensolver-max-iterations", type=int, default=3000)
    parser.add_argument(
        "--minres-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-response-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1729)
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
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
