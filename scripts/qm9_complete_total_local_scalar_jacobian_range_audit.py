#!/usr/bin/env python3
"""Matrix-free stable5 audit of the parameter-to-Hessian Jacobian range."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

try:
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        ScalarModel,
        _build_model_and_optimizer,
        _correction,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )
except ModuleNotFoundError:
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        ScalarModel,
        _build_model_and_optimizer,
        _correction,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )


def is_stable5_protocol(protocol: dict[str, Any]) -> bool:
    """Accept both legacy scope-tagged and current stage-tagged stable5 protocols."""
    scope = str(protocol.get("definitions", {}).get("scope", ""))
    stage = str(protocol.get("stage", ""))
    return scope.split(";", 1)[0].strip().startswith("stable5") or stage.startswith(
        "stable5_fit_only_"
    )


def select_audit_parents(
    parents: list[CapacityParent], parent_id: str | None
) -> list[CapacityParent]:
    if parent_id is None:
        return parents
    selected = [parent for parent in parents if parent.molecule_id == parent_id]
    if len(selected) != 1:
        raise ValueError("requested Jacobian-audit parent is absent from stable5")
    return selected


def symmetric_frobenius_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Vectorize a symmetric matrix while preserving its Frobenius norm."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected a square matrix")
    indices = torch.triu_indices(
        matrix.shape[0], matrix.shape[1], device=matrix.device
    )
    values = 0.5 * (matrix + matrix.T)[indices[0], indices[1]]
    weights = torch.where(
        indices[0] == indices[1],
        torch.ones_like(values),
        torch.full_like(values, math.sqrt(2.0)),
    )
    return weights * values


def cgls(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    parameter_count: int,
    *,
    iterations: int,
    relative_tolerance: float,
    callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Solve min ||A*x-b|| by matrix-free conjugate gradients on normal equations."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    solution = right_hand_side.new_zeros(parameter_count)
    residual = right_hand_side.clone()
    gradient = rmatvec(residual)
    direction = gradient.clone()
    gradient_squared = torch.dot(gradient, gradient)
    initial_gradient_squared = gradient_squared.detach().clone()
    initial_residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())
    rows: list[dict[str, float]] = []
    if float(initial_gradient_squared) <= torch.finfo(gradient_squared.dtype).tiny:
        return solution, residual, rows
    for iteration in range(1, iterations + 1):
        action = matvec(direction)
        action_squared = torch.dot(action, action)
        if not bool(torch.isfinite(action_squared)) or float(action_squared) <= 0.0:
            raise RuntimeError("CGLS encountered a non-finite or null Jacobian action")
        step_size = gradient_squared / action_squared
        solution = solution + step_size * direction
        residual = residual - step_size * action
        new_gradient = rmatvec(residual)
        new_gradient_squared = torch.dot(new_gradient, new_gradient)
        residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())
        row = {
            "iteration": float(iteration),
            "residual_norm": residual_norm,
            "relative_residual_norm": residual_norm
            / max(initial_residual_norm, np.finfo(float).tiny),
            "normal_gradient_norm": float(
                torch.sqrt(new_gradient_squared).detach().cpu()
            ),
            "solution_norm": float(torch.linalg.vector_norm(solution).detach().cpu()),
            "step_size": float(step_size.detach().cpu()),
        }
        rows.append(row)
        if callback is not None:
            callback(row)
        if row["relative_residual_norm"] <= relative_tolerance:
            break
        if not bool(torch.isfinite(new_gradient_squared)):
            raise RuntimeError("CGLS normal gradient became non-finite")
        if float(new_gradient_squared) <= float(initial_gradient_squared) * max(
            relative_tolerance**2, torch.finfo(new_gradient_squared.dtype).eps
        ):
            break
        if float(gradient_squared) <= torch.finfo(gradient_squared.dtype).tiny:
            break
        beta = new_gradient_squared / gradient_squared
        direction = new_gradient + beta * direction
        gradient = new_gradient
        gradient_squared = new_gradient_squared
    return solution, residual, rows


def estimate_right_jacobi_preconditioner(
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    reference_output: torch.Tensor,
    parameter_count: int,
    *,
    probes: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Estimate diag(A.T A)^-1/2 using deterministic Rademacher probes."""
    if probes <= 0:
        raise ValueError("preconditioner probes must be positive")
    generator = torch.Generator(device=reference_output.device).manual_seed(seed)
    diagonal = reference_output.new_zeros(parameter_count)
    for _ in range(probes):
        direction = 2.0 * torch.randint(
            0,
            2,
            reference_output.shape,
            device=reference_output.device,
            generator=generator,
            dtype=torch.int64,
        ).to(reference_output.dtype) - 1.0
        gradient = rmatvec(direction)
        diagonal = diagonal + gradient * gradient
    diagonal = diagonal / float(probes)
    active = torch.isfinite(diagonal) & (diagonal > 0.0)
    if not bool(torch.any(active)):
        raise RuntimeError("Hutchinson preconditioner found no active parameters")
    scale = torch.zeros_like(diagonal)
    scale[active] = torch.rsqrt(diagonal[active])
    median_scale = torch.median(scale[active])
    scale[active] = scale[active] / median_scale
    active_values = scale[active]
    return scale, {
        "probes": probes,
        "active_parameter_count": int(torch.count_nonzero(active).cpu()),
        "inactive_parameter_count": int(torch.count_nonzero(~active).cpu()),
        "scale_min": float(torch.min(active_values).cpu()),
        "scale_median": float(torch.median(active_values).cpu()),
        "scale_max": float(torch.max(active_values).cpu()),
    }


def _jet_output_vector(
    model: ScalarModel,
    parents: list[CapacityParent],
    *,
    create_parameter_graph: bool,
    include_energy_force: bool = False,
    energy_scale: float = 0.1,
    force_scale: float = 0.05,
) -> torch.Tensor:
    rows = []
    for parent in parents:
        energy, force, hessian = _correction(
            model,
            parent,
            create_parameter_graph=create_parameter_graph,
            anchor_energy_force=False,
        )
        if include_energy_force:
            force_block_scale = force_scale * math.sqrt(float(force.numel()))
            rows.extend(
                (
                    energy.reshape(1) / energy_scale,
                    force.reshape(-1) / force_block_scale,
                )
            )
        reference_norm = max(
            float(np.linalg.norm(parent.pbe_hessian)), np.finfo(float).tiny
        )
        rows.append(symmetric_frobenius_vector(hessian) / reference_norm)
    return torch.cat(rows)


def _target_vector(
    parents: list[CapacityParent],
    device: torch.device,
    *,
    include_energy_force: bool = False,
    energy_scale: float = 0.1,
    force_scale: float = 0.05,
) -> torch.Tensor:
    rows = []
    for parent in parents:
        if include_energy_force:
            force_count = 3 * parent.natoms
            force_block_scale = force_scale * math.sqrt(float(force_count))
            rows.extend(
                (
                    torch.as_tensor(
                        [parent.pbe_energy - parent.source_energy],
                        dtype=torch.float64,
                        device=device,
                    )
                    / energy_scale,
                    torch.as_tensor(
                        parent.pbe_force - parent.source_force,
                        dtype=torch.float64,
                        device=device,
                    ).reshape(-1)
                    / force_block_scale,
                )
            )
        target = torch.as_tensor(
            parent.pbe_hessian - parent.source_hessian_symmetric,
            dtype=torch.float64,
            device=device,
        )
        reference_norm = max(
            float(np.linalg.norm(parent.pbe_hessian)), np.finfo(float).tiny
        )
        rows.append(symmetric_frobenius_vector(target) / reference_norm)
    return torch.cat(rows)


def _split_parent_residuals(
    vector: torch.Tensor,
    parents: list[CapacityParent],
    *,
    include_energy_force: bool = False,
    energy_scale: float = 0.1,
    force_scale: float = 0.05,
) -> list[dict[str, float | int | str]]:
    rows = []
    offset = 0
    for parent in parents:
        width = 3 * parent.natoms
        row: dict[str, float | int | str] = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
        }
        if include_energy_force:
            row["energy_abs_error_hartree"] = abs(float(vector[offset].cpu())) * energy_scale
            offset += 1
            force = (
                vector[offset : offset + width]
                * force_scale
                * math.sqrt(float(width))
            )
            row["force_mae_hartree_per_bohr"] = float(torch.mean(torch.abs(force)).cpu())
            row["force_rmse_hartree_per_bohr"] = float(torch.sqrt(torch.mean(force * force)).cpu())
            offset += width
        count = width * (width + 1) // 2
        value = float(torch.linalg.vector_norm(vector[offset : offset + count]).cpu())
        row["relative_frobenius"] = value
        rows.append(row)
        offset += count
    if offset != vector.numel():
        raise ValueError("parent vector partition mismatch")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "formal")
    if not is_stable5_protocol(protocol):
        raise ValueError("Jacobian audit must use a stable5-only protocol")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    checkpoint_parent_ids = [parent.molecule_id for parent in parents]
    model, _, architecture = _build_model_and_optimizer(arm, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not certify frozen Test100")
    if int(checkpoint.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("checkpoint Test100 count is nonzero")
    checkpoint_protocol = args.checkpoint_protocol or args.protocol
    if checkpoint.get("protocol_sha256") != _sha256(checkpoint_protocol):
        raise ValueError("checkpoint/protocol hash mismatch")
    if checkpoint.get("arm_id") != args.arm_id:
        raise ValueError("checkpoint arm mismatch")
    if checkpoint.get("selected_parent_ids") != checkpoint_parent_ids:
        raise ValueError("checkpoint parent list mismatch")
    parents = select_audit_parents(parents, args.parent_id)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_parameters = parameters_to_vector(trainable).detach().clone()
    base_parameter_norm = float(torch.linalg.vector_norm(base_parameters).cpu())
    if args.energy_scale <= 0.0 or args.force_scale <= 0.0:
        raise ValueError("energy and force scales must be positive")
    vector_options = {
        "include_energy_force": args.include_energy_force_targets,
        "energy_scale": args.energy_scale,
        "force_scale": args.force_scale,
    }
    target = _target_vector(parents, device, **vector_options)
    base_output = _jet_output_vector(
        model, parents, create_parameter_graph=False, **vector_options
    ).detach()
    right_hand_side = target - base_output
    initial_rows = _split_parent_residuals(
        right_hand_side, parents, **vector_options
    )
    initial_norm = float(torch.linalg.vector_norm(right_hand_side).cpu())
    matvec_calls = 0
    rmatvec_calls = 0
    fd_stability: dict[str, float | str] = {}
    zero_output: torch.Tensor | None = None
    if args.exact_linear_parameter_map:
        if architecture["architecture"] != "mace_frozen_invariant_random_readout_scalar":
            raise ValueError("exact linear parameter map is restricted to frozen MACE readout")
        with torch.no_grad():
            vector_to_parameters(torch.zeros_like(base_parameters), trainable)
        zero_output = _jet_output_vector(
            model, parents, create_parameter_graph=False, **vector_options
        ).detach()
        with torch.no_grad():
            vector_to_parameters(base_parameters, trainable)
        if float(torch.linalg.vector_norm(zero_output).cpu()) > 1e-10:
            raise ValueError("exact linear parameter map has a nonzero origin")

    def matvec(direction: torch.Tensor) -> torch.Tensor:
        nonlocal matvec_calls
        matvec_calls += 1
        direction_norm = torch.linalg.vector_norm(direction)
        if not bool(torch.isfinite(direction_norm)) or float(direction_norm) == 0.0:
            raise RuntimeError("invalid parameter direction")
        if args.exact_linear_parameter_map:
            if zero_output is None:
                raise RuntimeError("exact linear origin was not initialized")
            with torch.no_grad():
                vector_to_parameters(direction, trainable)
            action = _jet_output_vector(
                model, parents, create_parameter_graph=False, **vector_options
            ).detach()
            with torch.no_grad():
                vector_to_parameters(base_parameters, trainable)
            return action - zero_output
        step = (
            float(args.fd_relative_step)
            * max(base_parameter_norm, 1.0)
            / float(direction_norm.detach().cpu())
        )
        with torch.no_grad():
            vector_to_parameters(base_parameters + step * direction, trainable)
        plus = _jet_output_vector(
            model, parents, create_parameter_graph=False, **vector_options
        ).detach()
        with torch.no_grad():
            vector_to_parameters(base_parameters - step * direction, trainable)
        minus = _jet_output_vector(
            model, parents, create_parameter_graph=False, **vector_options
        ).detach()
        with torch.no_grad():
            vector_to_parameters(base_parameters, trainable)
        return (plus - minus) / (2.0 * step)

    def rmatvec(output_direction: torch.Tensor) -> torch.Tensor:
        nonlocal rmatvec_calls
        rmatvec_calls += 1
        with torch.no_grad():
            vector_to_parameters(base_parameters, trainable)
        output = _jet_output_vector(
            model, parents, create_parameter_graph=True, **vector_options
        )
        scalar = torch.dot(output, output_direction)
        gradients = torch.autograd.grad(scalar, trainable, allow_unused=True)
        return torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(trainable, gradients, strict=True)
            ]
        ).detach()

    parameter_preconditioner = torch.ones_like(base_parameters)
    preconditioner_metadata: dict[str, float | int | str] = {
        "method": "none",
        "probes": 0,
    }
    if args.hutchinson_preconditioner_probes > 0:
        parameter_preconditioner, estimated = estimate_right_jacobi_preconditioner(
            rmatvec,
            right_hand_side,
            base_parameters.numel(),
            probes=args.hutchinson_preconditioner_probes,
            seed=args.preconditioner_seed,
        )
        preconditioner_metadata = {
            "method": "hutchinson_right_jacobi",
            **estimated,
        }

    def solver_matvec(value: torch.Tensor) -> torch.Tensor:
        return matvec(parameter_preconditioner * value)

    def solver_rmatvec(value: torch.Tensor) -> torch.Tensor:
        return parameter_preconditioner * rmatvec(value)

    initial_parameter_step = torch.zeros_like(base_parameters)
    initial_parameter_step_metadata: dict[str, Any] = {"used": False}
    if args.initial_parameter_step is not None:
        payload = torch.load(
            args.initial_parameter_step, map_location=device, weights_only=False
        )
        candidate = payload.get("linearized_parameter_step")
        if candidate is None:
            raise ValueError("restart artifact has no linearized_parameter_step")
        candidate = candidate.to(device=device, dtype=base_parameters.dtype).reshape(-1)
        if candidate.shape != base_parameters.shape or not bool(torch.isfinite(candidate).all()):
            raise ValueError("restart parameter step is incompatible or non-finite")
        if payload.get("source_checkpoint_sha256") != _sha256(args.checkpoint):
            raise ValueError("restart/source checkpoint hash mismatch")
        initial_parameter_step = candidate
        initial_parameter_step_metadata = {
            "used": True,
            "path": args.initial_parameter_step.resolve().as_posix(),
            "sha256": _sha256(args.initial_parameter_step),
            "norm": float(torch.linalg.vector_norm(candidate).cpu()),
        }
    restart_right_hand_side = right_hand_side - matvec(initial_parameter_step) if bool(
        torch.any(initial_parameter_step != 0.0)
    ) else right_hand_side
    restart_residual_norm = float(
        torch.linalg.vector_norm(restart_right_hand_side).cpu()
    )

    metrics_path = args.output_dir / "cgls_metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as handle:
        solver_solution, linear_residual, cgls_rows = cgls(
            solver_matvec,
            solver_rmatvec,
            restart_right_hand_side,
            base_parameters.numel(),
            iterations=args.iterations,
            relative_tolerance=args.relative_tolerance,
            callback=lambda row: (
                handle.write(json.dumps(row, sort_keys=True) + "\n"),
                handle.flush(),
                print(json.dumps(row, sort_keys=True), flush=True),
            ),
        )
    solution = initial_parameter_step + parameter_preconditioner * solver_solution

    if solution.numel() and float(torch.linalg.vector_norm(solution)) > 0.0:
        probe = solution / torch.linalg.vector_norm(solution)
        primary = matvec(probe)
        original_step = args.fd_relative_step
        args.fd_relative_step = original_step / 3.0
        refined = matvec(probe)
        args.fd_relative_step = original_step
        fd_stability = {
            "method": (
                "exact_linear_parameter_map"
                if args.exact_linear_parameter_map
                else "centered_parameter_finite_difference"
            ),
            "relative_difference": float(
                torch.linalg.vector_norm(primary - refined)
                / torch.clamp(torch.linalg.vector_norm(refined), min=torch.finfo(torch.float64).tiny)
            ),
            "cosine": float(
                torch.dot(primary, refined)
                / torch.clamp(
                    torch.linalg.vector_norm(primary) * torch.linalg.vector_norm(refined),
                    min=torch.finfo(torch.float64).tiny,
                )
            ),
        }

    linear_rows = _split_parent_residuals(
        linear_residual, parents, **vector_options
    )
    nonlinear_rows: list[dict[str, Any]] = []
    nonlinear_summaries: list[dict[str, Any]] = []
    solution_norm = float(torch.linalg.vector_norm(solution).cpu())
    for alpha in args.nonlinear_alphas:
        with torch.no_grad():
            vector_to_parameters(base_parameters + alpha * solution, trainable)
        evaluation, rows, _ = _evaluate(
            model,
            parents,
            protocol["capacity_gate"],
            anchor_energy_force=args.anchor_energy_force,
        )
        nonlinear_summaries.append(
            {
                "alpha": alpha,
                "parameter_step_over_base_norm": (
                    alpha * solution_norm / base_parameter_norm
                    if base_parameter_norm > np.finfo(float).tiny
                    else None
                ),
                "hessian_relative_frobenius": evaluation[
                    "hessian_relative_frobenius"
                ],
                "energy_median_ratio_to_source": evaluation[
                    "energy_median_ratio_to_source"
                ],
                "force_median_ratio_to_source": evaluation[
                    "force_median_ratio_to_source"
                ],
                "max_antisymmetric_over_symmetric_frobenius": evaluation[
                    "max_antisymmetric_over_symmetric_frobenius"
                ],
                "capacity_gate_passed": evaluation["gate"]["passed"],
            }
        )
        nonlinear_rows.extend({"alpha": alpha, **row} for row in rows)
    with torch.no_grad():
        vector_to_parameters(base_parameters, trainable)
    torch.save(
        {
            "linearized_parameter_step": solution.cpu(),
            "right_parameter_preconditioner": parameter_preconditioner.cpu(),
            "source_checkpoint": args.checkpoint.resolve().as_posix(),
            "source_checkpoint_sha256": _sha256(args.checkpoint),
            "protocol": args.protocol.resolve().as_posix(),
            "protocol_sha256": _sha256(args.protocol),
            "arm_id": args.arm_id,
            "test100_accessed": False,
            "test100_evaluations_used": 0,
        },
        args.output_dir / "linearized_parameter_step.pt",
    )
    _write_csv(args.output_dir / "initial_per_parent.csv", initial_rows)
    _write_csv(args.output_dir / "linearized_per_parent.csv", linear_rows)
    _write_csv(args.output_dir / "nonlinear_probe_per_parent.csv", nonlinear_rows)
    elapsed = time.perf_counter() - started
    result = {
        "definition": (
            "Matrix-free CGLS projection of the stable5 target energy/force/Hessian "
            "correction onto one frozen scalar-readout parameter-to-jet range."
            if args.include_energy_force_targets
            else "Matrix-free CGLS projection of the stable5 target Hessian correction "
            "onto the local parameter-to-Hessian Jacobian range at one frozen checkpoint."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "source_checkpoint": args.checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "source_checkpoint_protocol": checkpoint_protocol.resolve().as_posix(),
        "source_checkpoint_protocol_sha256": _sha256(checkpoint_protocol),
        "source_step": int(checkpoint["step"]),
        "architecture": architecture,
        "parameter_count": int(base_parameters.numel()),
        "output_dimension": int(target.numel()),
        "output_blocks": (
            {
                "energy": True,
                "force": True,
                "hessian": True,
                "energy_scale_hartree": args.energy_scale,
                "force_scale_hartree_per_bohr": args.force_scale,
                "force_objective": "mean_squared_component_error",
                "hessian_scale": "per_parent_pbe_frobenius_norm",
            }
            if args.include_energy_force_targets
            else {"energy": False, "force": False, "hessian": True}
        ),
        "iterations_requested": args.iterations,
        "iterations_completed": len(cgls_rows),
        "matvec_calls": matvec_calls,
        "rmatvec_calls": rmatvec_calls,
        "fd_relative_step": args.fd_relative_step,
        "parameter_matvec_mode": (
            "exact_linear" if args.exact_linear_parameter_map else "centered_fd"
        ),
        "nonlinear_evaluation_anchor_energy_force": args.anchor_energy_force,
        "right_preconditioner": preconditioner_metadata,
        "initial_parameter_step": initial_parameter_step_metadata,
        "restart_residual_norm": restart_residual_norm,
        "fd_directional_stability": fd_stability,
        "initial_global_residual_norm": initial_norm,
        "final_global_residual_norm": float(torch.linalg.vector_norm(linear_residual).cpu()),
        "final_relative_to_initial_residual": float(
            torch.linalg.vector_norm(linear_residual).cpu()
        )
        / max(initial_norm, np.finfo(float).tiny),
        "linearized_hessian_relative_frobenius": _distribution(
            linear_rows, "relative_frobenius"
        ),
        "initial_hessian_relative_frobenius": _distribution(
            initial_rows, "relative_frobenius"
        ),
        "linearized_parameter_step_norm": solution_norm,
        "base_parameter_norm": base_parameter_norm,
        "linearized_parameter_step_over_base_norm": (
            solution_norm / base_parameter_norm
            if base_parameter_norm > np.finfo(float).tiny
            else None
        ),
        "nonlinear_probes": nonlinear_summaries,
        "wall_time_s": elapsed,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "selected_parent_ids": [parent.molecule_id for parent in parents],
        "checkpoint_parent_ids": checkpoint_parent_ids,
        "opened_label_paths": provenance["opened_label_paths"],
        "opened_capacity_arrays": provenance["opened_capacity_arrays"],
        "unselected_parent_artifacts_opened": provenance[
            "unselected_parent_artifacts_opened"
        ],
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    if args.include_energy_force_targets:
        result["initial_energy_abs_error_hartree"] = _distribution(
            initial_rows, "energy_abs_error_hartree"
        )
        result["initial_force_mae_hartree_per_bohr"] = _distribution(
            initial_rows, "force_mae_hartree_per_bohr"
        )
        result["linearized_energy_abs_error_hartree"] = _distribution(
            linear_rows, "energy_abs_error_hartree"
        )
        result["linearized_force_mae_hartree_per_bohr"] = _distribution(
            linear_rows, "force_mae_hartree_per_bohr"
        )
    artifacts = {}
    for path in sorted(args.output_dir.iterdir()):
        # The wrapper closes resource.time only after this process exits.
        if path.name in {"summary.json", "resource.time"}:
            continue
        artifacts[path.name] = {
            "path": path.resolve().as_posix(),
            "sha256": _sha256(path),
        }
    result["artifacts"] = artifacts
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-protocol",
        type=Path,
        help="Hash-bound source protocol when reusing a compatible frozen checkpoint.",
    )
    parser.add_argument("--parent-id")
    parser.add_argument("--initial-parameter-step", type=Path)
    parser.add_argument("--anchor-energy-force", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--fd-relative-step", type=float, default=3e-6)
    parser.add_argument("--exact-linear-parameter-map", action="store_true")
    parser.add_argument("--include-energy-force-targets", action="store_true")
    parser.add_argument("--energy-scale", type=float, default=0.1)
    parser.add_argument("--force-scale", type=float, default=0.05)
    parser.add_argument("--hutchinson-preconditioner-probes", type=int, default=0)
    parser.add_argument("--preconditioner-seed", type=int, default=20260806)
    parser.add_argument("--nonlinear-alphas", default="0.01,0.03,0.1,0.3,1.0")
    args = parser.parse_args()
    args.nonlinear_alphas = [
        float(value) for value in args.nonlinear_alphas.split(",") if value
    ]
    if not args.nonlinear_alphas or any(value <= 0.0 for value in args.nonlinear_alphas):
        parser.error("nonlinear alphas must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
