#!/usr/bin/env python3
"""Fail-closed single-parent audit of analytic complete-total relaxed HVPs."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mldft.ofdft.complete_total_training import (
    consume_implicit_response_diagnostics,
)
from mldft.ofdft.implicit_response import (
    deflated_preconditioned_conjugate_gradient,
)
from qm9_complete_total_capacity_train import (
    IntegralBundleCache,
    _analytic_center_response,
    _analytic_full_hessian_metrics,
    _analytic_relaxed_direction_prediction,
    _density_namespace,
    _load_molecule,
    _parse_run,
    _refresh_base_densities,
    _sha256,
)
from qm9_graphformer_complete_total_relaxed_hvp_verify import (
    _fresh_hvp,
    _relative_l2,
    _require_finite,
    _require_hash,
)
from qm9_hessian_density_relaxed_eval import _load_context


def _common_args(
    output_dir: Path, protocol: dict[str, Any], device: str
) -> argparse.Namespace:
    density = protocol["numerics"]["density"]
    implicit = protocol["numerics"]["implicit_parameter_response"]
    analytic = protocol["numerics"]["analytic_geometry_response"]
    return argparse.Namespace(
        output_dir=output_dir,
        device=device,
        transform_device="cpu",
        charge=0,
        negative_integrated_density_penalty_weight=0.0,
        base_initialization=density["initialization"],
        density_lr=float(density["stage1_adam_lr"]),
        density_max_cycles=int(density["stage1_max_cycles"]),
        density_first_threshold=float(density["stage1_threshold"]),
        density_fallback_lr=float(density["fallback_adam_lr"]),
        density_fallback_max_cycles=int(density["fallback_max_cycles"]),
        density_fallback_threshold=float(density["fallback_threshold"]),
        density_strict_threshold=float(
            density["strict_projected_gradient_threshold"]
        ),
        lbfgs_refine=bool(density.get("lbfgs_refine", True)),
        lbfgs_max_iterations=int(density.get("lbfgs_max_iterations", 500)),
        newton_refine=bool(density.get("newton_refine", True)),
        newton_max_iterations=int(density.get("newton_max_iterations", 6)),
        displacement=float(protocol["numerics"]["displacement_bohr"]),
        integral_derivative_step=float(
            protocol["numerics"]["integral_derivative_step_bohr"]
        ),
        integral_directional_second_step=float(
            protocol["numerics"][
                "integral_directional_second_step_bohr"
            ]
        ),
        integral_derivative_workers=8,
        integral_cache_entries=48,
        analytic_response_damping=float(analytic["damping"]),
        analytic_response_residual_tolerance=float(
            analytic["residual_tolerance"]
        ),
        implicit_density_parameter_response=True,
        implicit_response_tolerance=float(implicit["tolerance"]),
        implicit_response_max_iterations=int(implicit["max_iterations"]),
        implicit_response_damping=float(implicit["damping"]),
        implicit_response_diagonal_probes=int(implicit["diagonal_probes"]),
        # The registered protocol requires a training-grade implicit adjoint
        # and its solver benchmark names dense direct as the only implemented
        # parameter-adjoint-capable method.
        implicit_response_solver=str(implicit.get("solver", "direct")),
        implicit_response_warm_start=bool(implicit["warm_start"]),
        density_response_unroll_steps=0,
        density_response_unroll_lr=1.0e-3,
        connect_lagrange_multiplier_response=True,
    )


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _solver_benchmark(
    system: Any,
    position_direction: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    rows = []

    _synchronize(position_direction)
    started = time.perf_counter()
    direct = system.solve_tangent_direct_implicit(
        position_direction,
        damping=0.0,
        create_graph=False,
    )
    _synchronize(direct.density_response)
    rows.append(
        {
            "method": "dense_direct_implicit_adjoint",
            "parameter_adjoint_capable": True,
            "converged": direct.krylov.converged,
            "iterations": direct.krylov.iterations,
            "relative_residual": direct.krylov.relative_residual,
            "density_response_relative_l2_vs_direct": 0.0,
            "wall_time_s": time.perf_counter() - started,
            "breakdown": direct.krylov.breakdown,
        }
    )

    direct_norm = torch.linalg.vector_norm(direct.density_response).clamp_min(
        torch.finfo(direct.density_response.dtype).tiny
    )

    def response_error(response: torch.Tensor) -> float:
        return float(
            (
                torch.linalg.vector_norm(response - direct.density_response)
                / direct_norm
            )
            .detach()
            .cpu()
        )

    _synchronize(position_direction)
    started = time.perf_counter()
    pcg = system.solve_tangent_pcg(
        position_direction,
        tolerance=tolerance,
        max_iterations=max(500, 2 * system.coeffs.numel()),
    )
    _synchronize(pcg.density_response)
    rows.append(
        {
            "method": "matrix_free_tangent_pcg",
            "parameter_adjoint_capable": False,
            "converged": pcg.krylov.converged,
            "iterations": pcg.krylov.iterations,
            "relative_residual": pcg.krylov.relative_residual,
            "density_response_relative_l2_vs_direct": response_error(
                pcg.density_response
            ),
            "wall_time_s": time.perf_counter() - started,
            "breakdown": pcg.krylov.breakdown,
        }
    )

    _synchronize(position_direction)
    started = time.perf_counter()
    minres = system.solve_kkt_minres(
        position_direction,
        tolerance=tolerance,
        max_iterations=max(500, 2 * system.coeffs.numel()),
    )
    _synchronize(minres.density_response)
    rows.append(
        {
            "method": "matrix_free_kkt_minres",
            "parameter_adjoint_capable": False,
            "converged": minres.krylov.converged,
            "iterations": minres.krylov.iterations,
            "relative_residual": minres.krylov.relative_residual,
            "density_response_relative_l2_vs_direct": response_error(
                minres.density_response
            ),
            "wall_time_s": time.perf_counter() - started,
            "breakdown": minres.krylov.breakdown,
        }
    )

    _synchronize(position_direction)
    matrix_started = time.perf_counter()
    matrix = system.tangent_matrix(create_graph=False).detach()
    tangent = system.tangent.detach()
    q = system.constraint_gradient.detach()
    q_norm_squared = torch.dot(q, q)

    def tangent_rhs(direction: torch.Tensor) -> torch.Tensor:
        mixed = system.mixed_coefficient_direction(direction).detach()
        constraint_rhs = -system.constraint_position_direction(direction).detach()
        particular = q * (constraint_rhs / q_norm_squared)
        return tangent.T @ (
            -mixed - system.coefficient_hvp(particular).detach()
        )

    rhs = tangent_rhs(position_direction)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    _synchronize(eigenvalues)
    deflation_prepare_seconds = time.perf_counter() - matrix_started
    constraint_rhs = -system.constraint_position_direction(
        position_direction
    ).detach()
    particular = q * (constraint_rhs / q_norm_squared)
    deflation_count = min(8, matrix.shape[0])
    started = time.perf_counter()
    deflated = deflated_preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        eigenvectors[:, :deflation_count],
        tolerance=tolerance,
        max_iterations=max(500, 2 * matrix.shape[0]),
    )
    _synchronize(deflated.solution)
    rows.append(
        {
            "method": f"dense_operator_deflated_pcg_{deflation_count}",
            "parameter_adjoint_capable": False,
            "converged": deflated.converged,
            "iterations": deflated.iterations,
            "relative_residual": deflated.relative_residual,
            "density_response_relative_l2_vs_direct": response_error(
                particular + tangent @ deflated.solution
            ),
            "wall_time_s": (
                time.perf_counter() - started + deflation_prepare_seconds
            ),
            "preparation_wall_time_s": deflation_prepare_seconds,
            "breakdown": deflated.breakdown,
        }
    )

    strict_rows = [
        row
        for row in rows
        if bool(row["converged"])
        and float(row["relative_residual"]) <= tolerance
        and float(row["density_response_relative_l2_vs_direct"]) <= 1.0e-7
    ]
    selected = (
        min(strict_rows, key=lambda row: float(row["wall_time_s"]))["method"]
        if strict_rows
        else None
    )
    training_rows = [
        row for row in strict_rows if bool(row["parameter_adjoint_capable"])
    ]
    selected_training = (
        min(training_rows, key=lambda row: float(row["wall_time_s"]))["method"]
        if training_rows
        else None
    )
    return {
        "tangent_dimension": int(matrix.shape[0]),
        "minimum_tangent_eigenvalue": float(eigenvalues[0].cpu()),
        "maximum_tangent_eigenvalue": float(eigenvalues[-1].cpu()),
        "condition_number": float(
            torch.abs(eigenvalues[-1] / eigenvalues[0]).cpu()
        ),
        "methods": rows,
        "strict_relative_residual_tolerance": tolerance,
        "strict_response_relative_l2_tolerance": 1.0e-7,
        "selected_fastest_strict_method": selected,
        "selected_fastest_strict_parameter_adjoint_method": selected_training,
        "fastest_strict_method_is_training_eligible": (
            selected is not None and selected == selected_training
        ),
    }


def _parameter_gradient_audit(
    context: Any,
    cache: IntegralBundleCache,
    molecule: Any,
    direction: Any,
    density_args: argparse.Namespace,
    parameter_steps: list[float],
) -> dict[str, Any]:
    consume_implicit_response_diagnostics()
    parameter_device = next(context.model.parameters()).device
    if parameter_device.type == "cuda":
        torch.cuda.synchronize(parameter_device)
    started = time.perf_counter()
    graph_hvp, density_norms, graph_timings = (
        _analytic_relaxed_direction_prediction(
            context,
            cache,
            molecule,
            direction,
            create_graph=True,
        )
    )
    _synchronize(graph_hvp)
    forward_seconds = time.perf_counter() - started
    _require_finite(graph_hvp, "analytic relaxed HVP")
    generator = torch.Generator(device=graph_hvp.device)
    generator.manual_seed(20260724)
    probe = torch.randn(
        graph_hvp.shape,
        generator=generator,
        dtype=graph_hvp.dtype,
        device=graph_hvp.device,
    )
    probe = probe / torch.linalg.vector_norm(probe)
    scalar = torch.sum(graph_hvp * probe)
    named_parameters = [
        (name, parameter)
        for name, parameter in context.model.net.named_parameters()
        if parameter.requires_grad
    ]
    backward_started = time.perf_counter()
    gradients = torch.autograd.grad(
        scalar,
        [parameter for _, parameter in named_parameters],
        allow_unused=True,
    )
    if graph_hvp.device.type == "cuda":
        torch.cuda.synchronize(graph_hvp.device)
    backward_seconds = time.perf_counter() - backward_started
    implicit_diagnostics = consume_implicit_response_diagnostics()
    if not implicit_diagnostics or any(
        not bool(row["converged"]) for row in implicit_diagnostics
    ):
        raise RuntimeError(
            "stationary center density parameter VJP was absent or failed"
        )

    selected = None
    total_norm_squared = 0.0
    for (name, parameter), gradient in zip(
        named_parameters, gradients, strict=True
    ):
        if gradient is None:
            continue
        _require_finite(gradient, f"parameter gradient {name}")
        total_norm_squared += float(torch.sum(gradient.detach().square()).cpu())
        flat = gradient.detach().reshape(-1)
        index = int(torch.argmax(torch.abs(flat)).cpu())
        magnitude = float(torch.abs(flat[index]).cpu())
        if selected is None or magnitude > selected[-1]:
            selected = (name, parameter, gradient, index, magnitude)
    if selected is None or selected[-1] == 0.0:
        raise RuntimeError("analytic relaxed HVP parameter gradient is zero")
    parameter_name, parameter, gradient, flat_index, _ = selected
    analytic_gradient = float(
        gradient.detach().reshape(-1)[flat_index].cpu()
    )
    original_value = float(parameter.detach().reshape(-1)[flat_index].cpu())
    original_base = molecule.base
    fd_rows = []
    try:
        for step in parameter_steps:
            values = {}
            point_rows = {}
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                with torch.no_grad():
                    parameter.reshape(-1)[flat_index] = (
                        original_value + sign * step
                    )
                refreshed = _refresh_base_densities(
                    context,
                    [molecule],
                    density_args,
                    refresh_index=0,
                )
                value, _, _ = _analytic_relaxed_direction_prediction(
                    context,
                    cache,
                    molecule,
                    direction,
                    create_graph=False,
                )
                values[side] = float(
                    torch.sum(value.detach() * probe.detach()).cpu()
                )
                point_rows[side] = refreshed[0]
            finite_difference = (values["plus"] - values["minus"]) / (
                2.0 * step
            )
            error = abs(analytic_gradient - finite_difference)
            fd_rows.append(
                {
                    "parameter_step": step,
                    "analytic_gradient": analytic_gradient,
                    "finite_difference_gradient": finite_difference,
                    "relative_error": error
                    / max(
                        abs(analytic_gradient),
                        abs(finite_difference),
                        1.0e-12,
                    ),
                    "strict_center_points": point_rows,
                }
            )
    finally:
        with torch.no_grad():
            parameter.reshape(-1)[flat_index] = original_value
        molecule.base = original_base
    return {
        "graph_hvp": graph_hvp.detach().cpu(),
        "probe": probe.detach().cpu(),
        "forward_seconds": forward_seconds,
        "higher_order_backward_seconds": backward_seconds,
        "graph_timings": graph_timings,
        "density_gradient_norms": [
            float(value.detach().cpu()) for value in density_norms
        ],
        "stationary_density_parameter_vjp": implicit_diagnostics,
        "parameter_name": parameter_name,
        "parameter_flat_index": flat_index,
        "total_parameter_gradient_norm": math.sqrt(total_norm_squared),
        "parameter_fd": fd_rows,
        "best_parameter_fd_relative_error": min(
            row["relative_error"] for row in fd_rows
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ["MLDFT_SYMMETRIC_MATRIX_POWER_MODE"] = (
        args.symmetric_matrix_power_mode
    )
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20260724)
    np.random.seed(20260724)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text())
    if (
        protocol["validation_access_allowed"]
        or protocol["test100_access_allowed"]
        or protocol.get("method_name")
        != "解析密度/KKT响应的hybrid relaxed-HVP"
        or protocol.get("fully_analytic_libcint_second_integrals") is not False
        or protocol["identity"].get("old_original_a_identity_allowed") is not False
        or protocol["identity"].get(
            "old_checkpoint_or_manifest_hash_reuse_allowed"
        )
        is not False
    ):
        raise ValueError("new hybrid rebuild protocol identity/access boundary failed")

    registration = json.loads(args.checkpoint_registration.read_text())
    protocol_hash = _sha256(args.protocol)
    if (
        int(registration["global_step"]) != int(protocol["baseline"]["max_steps"])
        or registration["selection"]
        != "fixed_final_step_without_validation"
        or registration.get("baseline_name")
        != protocol["identity"]["baseline_name"]
        or registration.get("protocol_id") != protocol["protocol_id"]
        or registration.get("protocol_sha256") != protocol_hash
        or registration.get("old_original_a_identity_used") is not False
        or registration.get("validation_accessed") is not False
        or registration.get("test100_accessed") is not False
    ):
        raise ValueError("new baseline checkpoint registration failed")
    checkpoint = Path(registration["checkpoint"])
    checkpoint_hash = str(registration["checkpoint_sha256"])
    _require_hash(checkpoint, checkpoint_hash, "new baseline checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("complete_total_capacity") is not None:
        raise ValueError("baseline checkpoint already contains capacity-training state")
    parent_path = args.parent_manifest
    direction_path = args.direction_manifest
    parent_manifest = json.loads(parent_path.read_text())
    direction_manifest = json.loads(direction_path.read_text())
    if (
        parent_manifest.get("protocol_id") != protocol["protocol_id"]
        or direction_manifest.get("protocol_id") != protocol["protocol_id"]
        or parent_manifest.get("protocol_sha256") != protocol_hash
        or direction_manifest.get("protocol_sha256") != protocol_hash
        or parent_manifest.get("old_original_a_identity_used") is not False
        or direction_manifest.get("old_original_a_identity_used") is not False
        or parent_manifest.get("test100_accessed") is not False
        or direction_manifest.get("test100_accessed") is not False
        or direction_manifest.get("validation_accessed") is not False
    ):
        raise ValueError("frozen parent/direction provenance failed")
    if direction_manifest.get("parent_manifest_sha256") != _sha256(parent_path):
        raise ValueError("direction manifest is not bound to the supplied parent manifest")
    parent_entries = {
        str(row["molecule_id"]): row
        for row in parent_manifest["parents"]
    }
    direction_entries = {
        str(row["molecule_id"]): row
        for row in direction_manifest["parents"]
    }
    molecule = _load_molecule(
        parent_entries[args.molecule],
        low_mode_count=0,
        direction_entry=direction_entries[args.molecule],
        direction_role="all",
    )
    expected_count = int(
        protocol["verification"]["full_basis_direction_count"]
    )
    if len(molecule.directions) != expected_count:
        raise ValueError(
            f"full internal basis has {len(molecule.directions)} != {expected_count}"
        )

    common = _common_args(args.output_dir, protocol, args.device)
    if args.strict_fd_displacement is not None:
        common.displacement = float(args.strict_fd_displacement)
    density_args = _density_namespace(common)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    run = (
        f"{protocol['identity']['baseline_name']}="
        f"{registration['run_dir']}={checkpoint}"
    )
    context = _load_context(_parse_run(run), density_args, device)
    context.model.to(torch.float64)
    if context.model.net.__class__.__name__ != "Graphformer":
        raise TypeError("source model is not Graphformer")
    cache = IntegralBundleCache(context, common)

    density_started = time.perf_counter()
    density_rows = _refresh_base_densities(
        context, [molecule], density_args, refresh_index=0
    )
    density_seconds = time.perf_counter() - density_started
    direction = molecule.directions[args.direction_index]
    parameter_audit = _parameter_gradient_audit(
        context,
        cache,
        molecule,
        direction,
        density_args,
        [
            float(value)
            for value in protocol["verification"]["parameter_fd_steps"]
        ],
    )

    (
        _,
        position_direction,
        response_system,
        response,
        response_timings,
    ) = _analytic_center_response(
        context,
        cache,
        molecule,
        direction,
        create_graph=False,
    )
    response_cpu = response.density_response.detach().cpu()
    displacement = float(common.displacement)
    plus_start = molecule.base.coefficients + displacement * response_cpu
    minus_start = molecule.base.coefficients - displacement * response_cpu
    fd_started = time.perf_counter()
    fresh_hvp, fresh_points = _fresh_hvp(
        context,
        cache,
        molecule,
        direction,
        density_args,
        plus_start,
        minus_start,
    )
    fd_seconds = time.perf_counter() - fd_started
    analytic_hvp = parameter_audit["graph_hvp"]
    analytic_vs_fd = _relative_l2(analytic_hvp, fresh_hvp)

    solver_benchmark = _solver_benchmark(
        response_system,
        position_direction,
        tolerance=float(
            protocol["numerics"]["analytic_geometry_response"][
                "residual_tolerance"
            ]
        ),
    )
    full_started = time.perf_counter()
    full_metrics = _analytic_full_hessian_metrics(
        context, cache, [molecule], step=0
    )[0]
    full_seconds = time.perf_counter() - full_started

    legacy_started = time.perf_counter()
    legacy_rows = []
    for legacy_direction in molecule.directions[:4]:
        legacy_hvp, legacy_points = _fresh_hvp(
            context,
            cache,
            molecule,
            legacy_direction,
            density_args,
            molecule.base.coefficients,
            molecule.base.coefficients,
        )
        _require_finite(legacy_hvp, "legacy strict finite-difference HVP")
        legacy_rows.append(
            {
                "direction_index": legacy_direction.index,
                "direction_kind": legacy_direction.kind,
                "points": legacy_points,
            }
        )
    legacy_four_direction_seconds = time.perf_counter() - legacy_started

    analytic_gate = float(
        protocol["verification"][
            "analytic_vs_fresh_strict_fd_relative_l2_max"
        ]
    )
    parameter_gate = float(
        protocol["verification"][
            "parameter_gradient_best_relative_error_max"
        ]
    )
    symmetry_gate = float(
        protocol["verification"]["full_basis_asym_over_sym_max"]
    )
    estimated_analytic_step_seconds = (
        density_seconds
        + float(parameter_audit["forward_seconds"])
        + float(parameter_audit["higher_order_backward_seconds"])
    )
    measured_speedup = legacy_four_direction_seconds / max(
        estimated_analytic_step_seconds, 1.0e-12
    )
    maximum_step_seconds = float(
        protocol["verification"]["maximum_training_step_seconds"]
    )
    minimum_speedup = float(
        protocol["verification"]["target_training_step_speedup_min"]
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "molecule_id": args.molecule,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "checkpoint_registration": str(
            args.checkpoint_registration.resolve()
        ),
        "checkpoint_registration_sha256": _sha256(
            args.checkpoint_registration
        ),
        "method_name": protocol["method_name"],
        "fully_analytic_libcint_second_integrals": False,
        "direction_index": direction.index,
        "direction_kind": direction.kind,
        "symmetric_matrix_power_mode": args.symmetric_matrix_power_mode,
        "strict_fd_displacement_bohr": displacement,
        "strict_density_threshold": common.density_strict_threshold,
        "density_optimization_seconds": density_seconds,
        "density_points": density_rows,
        "analytic_hvp": analytic_hvp.numpy().tolist(),
        "strict_reoptimized_fd_hvp": fresh_hvp.numpy().tolist(),
        "analytic_vs_strict_fd_relative_l2": analytic_vs_fd,
        "strict_fd_seconds": fd_seconds,
        "strict_fd_points": fresh_points,
        "response_prediction_norm": float(
            torch.linalg.vector_norm(response_cpu)
        ),
        "response_timings": response_timings,
        "parameter_audit": {
            key: value
            for key, value in parameter_audit.items()
            if key not in {"graph_hvp", "probe"}
        },
        "solver_benchmark": solver_benchmark,
        "full_basis_metrics": full_metrics,
        "full_basis_seconds": full_seconds,
        "estimated_analytic_training_step_seconds": (
            estimated_analytic_step_seconds
        ),
        "legacy_four_direction_fd_seconds": legacy_four_direction_seconds,
        "legacy_four_direction_fd_points": legacy_rows,
        "measured_training_step_speedup": measured_speedup,
        "performance_gate_passed": (
            estimated_analytic_step_seconds <= maximum_step_seconds
            and measured_speedup >= minimum_speedup
        ),
        "analytic_vs_fd_gate_passed": analytic_vs_fd <= analytic_gate,
        "parameter_gradient_gate_passed": (
            parameter_audit["best_parameter_fd_relative_error"]
            <= parameter_gate
        ),
        "symmetry_gate_passed": (
            full_metrics["antisymmetric_over_symmetric_frobenius"]
            <= symmetry_gate
        ),
        "full39_nonregression_gate_passed": bool(
            full_metrics["full_hessian_complete"]
            and int(full_metrics["direction_count"]) == expected_count
            and all(
                math.isfinite(float(full_metrics[key]))
                for key in (
                    "mae",
                    "rmse",
                    "relative_frobenius",
                    "antisymmetric_over_symmetric_frobenius",
                    "total_energy_abs_error_hartree",
                    "complete_total_force_mae_hartree_per_bohr",
                )
            )
        ),
        "proxy_fallback_used": False,
        "displaced_density_used_by_analytic_hvp": False,
        "independent_force_or_hessian_head_used": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    density_norms = [
        float(row["final_gradient_norm"]) for row in density_rows
    ]
    density_norms.extend(
        float(row["final_projected_density_gradient_norm"])
        for row in fresh_points
    )
    density_norms.extend(
        float(point["final_gradient_norm"])
        for fd_row in parameter_audit["parameter_fd"]
        for point in fd_row["strict_center_points"].values()
    )
    density_norms.extend(
        float(point["final_projected_density_gradient_norm"])
        for legacy_row in legacy_rows
        for point in legacy_row["points"]
    )
    result["maximum_audited_projected_density_gradient_norm"] = max(
        density_norms
    )
    result["density_residual_gate_passed"] = bool(
        result["maximum_audited_projected_density_gradient_norm"]
        <= float(
            protocol["verification"][
                "strict_density_projected_gradient_norm_max"
            ]
        )
    )
    result["solver_selection_gate_passed"] = bool(
        solver_benchmark["selected_fastest_strict_method"] is not None
        and solver_benchmark[
            "selected_fastest_strict_parameter_adjoint_method"
        ]
        is not None
        and solver_benchmark[
            "fastest_strict_method_is_training_eligible"
        ]
    )
    result["passed"] = bool(
        result["analytic_vs_fd_gate_passed"]
        and result["parameter_gradient_gate_passed"]
        and result["symmetry_gate_passed"]
        and result["full39_nonregression_gate_passed"]
        and result["performance_gate_passed"]
        and result["density_residual_gate_passed"]
        and result["solver_selection_gate_passed"]
    )
    result["single_parent_capacity_training_allowed"] = result["passed"]
    result["stable5_train20_allowed"] = False
    result["curvature_corrector_evaluation_required"] = not result["passed"]
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "audit_arrays.npz",
        analytic_hvp=analytic_hvp.numpy(),
        strict_reoptimized_fd_hvp=fresh_hvp.numpy(),
        probe=parameter_audit["probe"].numpy(),
        direction=direction.vector,
        pbe_hvp=direction.target_hvp,
    )
    if not result["passed"]:
        raise RuntimeError(
            f"fail-closed analytic HVP audit failed; see {args.output_dir / 'summary.json'}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/audit/"
            "qm9_graphformer_hybrid_relaxed_hvp_rebuild_v1.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint-registration", type=Path, required=True
    )
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecule", default="0028399")
    parser.add_argument("--direction-index", type=int, default=0)
    parser.add_argument("--strict-fd-displacement", type=float)
    parser.add_argument(
        "--symmetric-matrix-power-mode",
        choices=("stable_first_order", "eigh_second_order_audit"),
        default="eigh_second_order_audit",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = audit(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "passed": False,
                    "proxy_fallback_used": False,
                    "validation_accessed": False,
                    "test100_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
