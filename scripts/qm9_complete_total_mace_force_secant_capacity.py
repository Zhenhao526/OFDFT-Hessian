#!/usr/bin/env python3
"""Fit a scalar MACE correction with force secants, then gate on full Hessians."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mldft.ml.models.components.local_mace_scalar_residual import LocalMACEScalarResidual

try:
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _build_model_and_optimizer,
        _distribution,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _periodic_evaluation_metrics,
        _sha256,
        _write_rows,
    )
    from scripts.qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )
except ModuleNotFoundError:
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _build_model_and_optimizer,
        _distribution,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _periodic_evaluation_metrics,
        _sha256,
        _write_rows,
    )
    from qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )


def central_force_secant_hvp(
    plus_force: torch.Tensor, minus_force: torch.Tensor, displacement: float
) -> torch.Tensor:
    """Return H*v from scalar-derived forces at R +/- h*v."""
    if displacement <= 0.0:
        raise ValueError("displacement must be positive")
    if plus_force.shape != minus_force.shape:
        raise ValueError("force secant endpoints must have matching shapes")
    return (minus_force - plus_force) / (2.0 * displacement)


def _correction_energy_force(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    positions: torch.Tensor,
    *,
    create_parameter_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = positions
    if not geometry.requires_grad:
        geometry = geometry.detach().requires_grad_(True)
    energy = model.forward_energy(
        geometry, parent.atomic_numbers, parent.topology
    )
    gradient = torch.autograd.grad(
        energy, geometry, create_graph=create_parameter_graph
    )[0]
    return energy, -gradient


def correction_force_secant_hvp(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    direction: torch.Tensor,
    *,
    displacement: float,
    create_parameter_graph: bool,
) -> torch.Tensor:
    if direction.shape != parent.positions.shape:
        raise ValueError("direction shape must match molecular positions")
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("direction must be finite and nonzero")
    unit = direction / norm
    _, plus_force = _correction_energy_force(
        model,
        parent,
        parent.positions + displacement * unit,
        create_parameter_graph=create_parameter_graph,
    )
    _, minus_force = _correction_energy_force(
        model,
        parent,
        parent.positions - displacement * unit,
        create_parameter_graph=create_parameter_graph,
    )
    return central_force_secant_hvp(plus_force, minus_force, displacement)


def _cartesian_direction(parent: CapacityParent, index: int) -> torch.Tensor:
    size = parent.positions.numel()
    if not 0 <= index < size:
        raise ValueError("Cartesian direction index is out of range")
    direction = torch.zeros_like(parent.positions).reshape(-1)
    direction[index] = 1.0
    return direction.reshape_as(parent.positions)


def _parent_loss(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    direction_index: int,
    training: dict[str, Any],
    *,
    create_parameter_graph: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _parent_loss_multi_direction(
        model,
        parent,
        [direction_index],
        training,
        create_parameter_graph=create_parameter_graph,
    )


def _parent_loss_multi_direction(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    direction_indices: list[int],
    training: dict[str, Any],
    *,
    create_parameter_graph: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not direction_indices or len(set(direction_indices)) != len(direction_indices):
        raise ValueError("direction indices must be nonempty and unique")
    if min(direction_indices) < 0 or max(direction_indices) >= parent.positions.numel():
        raise ValueError("direction index exceeds parent coordinate count")
    base_terms = _base_loss_terms(
        model,
        parent,
        training,
        create_parameter_graph=create_parameter_graph,
    )
    directional_hvp_losses = [
        _directional_hvp_loss(
            model,
            parent,
            direction_index,
            training,
            create_parameter_graph=create_parameter_graph,
        )
        for direction_index in direction_indices
    ]
    hvp_loss = torch.stack(directional_hvp_losses).mean()
    total = (
        float(training["lambda_energy"]) * base_terms["energy_loss"]
        + float(training["lambda_force"]) * base_terms["force_loss"]
        + float(training["lambda_hvp"]) * hvp_loss
        + float(training["lambda_parameter"]) * base_terms["parameter_loss"]
    )
    return total, {
        **base_terms,
        "hvp_loss": hvp_loss,
    }


def _base_loss_terms(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    training: dict[str, Any],
    *,
    create_parameter_graph: bool,
) -> dict[str, torch.Tensor]:
    correction_energy, correction_force = _correction_energy_force(
        model,
        parent,
        parent.positions,
        create_parameter_graph=create_parameter_graph,
    )
    device = parent.positions.device
    target_energy = torch.as_tensor(
        parent.pbe_energy - parent.source_energy, dtype=torch.float64, device=device
    )
    target_force = torch.as_tensor(
        parent.pbe_force - parent.source_force, dtype=torch.float64, device=device
    )
    return {
        "energy_loss": (
            (correction_energy - target_energy) / float(training["energy_scale"])
        )
        ** 2,
        "force_loss": torch.mean(
            ((correction_force - target_force) / float(training["force_scale"])) ** 2
        ),
        "parameter_loss": torch.mean(
            torch.cat(
                [parameter.reshape(-1) for parameter in model.trainable_parameters()]
            )
            ** 2
        ),
    }


def _directional_hvp_loss(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    direction_index: int,
    training: dict[str, Any],
    *,
    create_parameter_graph: bool,
) -> torch.Tensor:
    direction = _cartesian_direction(parent, direction_index)
    predicted_hvp = correction_force_secant_hvp(
        model,
        parent,
        direction,
        displacement=float(training["force_secant_displacement_bohr"]),
        create_parameter_graph=create_parameter_graph,
    )
    target_hessian = torch.as_tensor(
        parent.pbe_hessian - parent.source_hessian_symmetric,
        dtype=torch.float64,
        device=parent.positions.device,
    )
    target_hvp = (target_hessian @ direction.reshape(-1)).reshape_as(direction)
    hvp_error = predicted_hvp - target_hvp
    absolute_hvp = torch.mean(
        (hvp_error / float(training["hvp_absolute_scale"])) ** 2
    )
    relative_hvp = torch.sum(hvp_error * hvp_error) / torch.clamp(
        torch.sum(target_hvp * target_hvp),
        min=float(training["hvp_reference_floor"]) ** 2,
    )
    relative_fraction = float(training["hvp_relative_loss_fraction"])
    return (1.0 - relative_fraction) * absolute_hvp + relative_fraction * relative_hvp


def _flatten_parameter_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def project_conflicting_gradients(
    task_gradients: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply deterministic PCGrad projections in insertion order."""
    projected = {name: gradient.clone() for name, gradient in task_gradients.items()}
    for name, gradient in projected.items():
        for other_name, other in task_gradients.items():
            if name == other_name:
                continue
            dot = torch.dot(gradient, other)
            squared_norm = torch.dot(other, other)
            if bool(dot < 0.0) and float(squared_norm) > torch.finfo(other.dtype).tiny:
                gradient = gradient - dot / squared_norm * other
        projected[name] = gradient
    return projected


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= torch.finfo(first.dtype).tiny:
        return 0.0
    return float((torch.dot(first, second) / denominator).detach().cpu())


def assign_flat_parameter_gradient(
    parameters: list[torch.nn.Parameter], flat_gradient: torch.Tensor
) -> None:
    expected = sum(parameter.numel() for parameter in parameters)
    if flat_gradient.numel() != expected:
        raise ValueError("flat gradient size does not match parameters")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat_gradient[offset : offset + count].reshape_as(parameter).clone()
        offset += count


def apply_fixed_scale_pcgrad(
    terms: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    training: dict[str, Any],
) -> dict[str, float]:
    reference_norms = training["task_gradient_reference_norms"]
    weighted_losses = {
        "energy": float(training["lambda_energy"]) * terms["energy_loss"],
        "force": float(training["lambda_force"]) * terms["force_loss"],
        "hvp": float(training["lambda_hvp"]) * terms["hvp_loss"],
    }
    raw_gradients = {
        name: _flatten_parameter_gradients(loss, parameters, retain_graph=True)
        for name, loss in weighted_losses.items()
    }
    return apply_fixed_scale_pcgrad_from_gradients(
        raw_gradients, parameters, training
    )


def apply_fixed_scale_pcgrad_from_gradients(
    raw_gradients: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    training: dict[str, Any],
) -> dict[str, float]:
    reference_norms = training["task_gradient_reference_norms"]
    if set(raw_gradients) != {"energy", "force", "hvp"}:
        raise ValueError("fixed-scale PCGrad requires energy, force, and HVP gradients")
    scaled_gradients = {}
    diagnostics: dict[str, float] = {}
    for name, gradient in raw_gradients.items():
        reference = float(reference_norms[name])
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError(f"invalid task gradient reference norm for {name}")
        diagnostics[f"raw_{name}_gradient_norm"] = float(
            torch.linalg.vector_norm(gradient).detach().cpu()
        )
        scaled_gradients[name] = gradient / reference
    for first, second in (("energy", "force"), ("energy", "hvp"), ("force", "hvp")):
        diagnostics[f"raw_cosine_{first}_{second}"] = _gradient_cosine(
            raw_gradients[first], raw_gradients[second]
        )
    projected = project_conflicting_gradients(scaled_gradients)
    combined = torch.stack(list(projected.values())).mean(dim=0)
    parameter_weight = float(training.get("lambda_parameter", 0.0))
    if parameter_weight:
        parameter_count = sum(parameter.numel() for parameter in parameters)
        parameter_gradient = torch.cat(
            [
                (2.0 * parameter_weight / parameter_count) * parameter.detach().reshape(-1)
                for parameter in parameters
            ]
        )
        combined = combined + parameter_gradient
    diagnostics["combined_gradient_norm_before_clip"] = float(
        torch.linalg.vector_norm(combined).detach().cpu()
    )
    for first, second in (("energy", "force"), ("energy", "hvp"), ("force", "hvp")):
        diagnostics[f"projected_cosine_{first}_{second}"] = _gradient_cosine(
            projected[first], projected[second]
        )
    assign_flat_parameter_gradient(parameters, combined)
    return diagnostics


def accumulated_full_basis_task_gradients(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    parameters: list[torch.nn.Parameter],
    training: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_terms = _base_loss_terms(
        model, parent, training, create_parameter_graph=True
    )
    raw_gradients = {
        "energy": _flatten_parameter_gradients(
            float(training["lambda_energy"]) * base_terms["energy_loss"],
            parameters,
            retain_graph=True,
        ),
        "force": _flatten_parameter_gradients(
            float(training["lambda_force"]) * base_terms["force_loss"],
            parameters,
            retain_graph=False,
        ),
    }
    hvp_gradient = torch.zeros_like(raw_gradients["energy"])
    hvp_loss_sum = torch.zeros((), dtype=parent.positions.dtype, device=parent.positions.device)
    direction_count = parent.positions.numel()
    for direction_index in range(direction_count):
        directional_loss = _directional_hvp_loss(
            model,
            parent,
            direction_index,
            training,
            create_parameter_graph=True,
        )
        hvp_loss_sum = hvp_loss_sum + directional_loss.detach()
        hvp_gradient = hvp_gradient + _flatten_parameter_gradients(
            float(training["lambda_hvp"]) * directional_loss,
            parameters,
            retain_graph=False,
        )
    raw_gradients["hvp"] = hvp_gradient / direction_count
    terms = {
        key: value.detach() for key, value in base_terms.items()
    }
    terms["hvp_loss"] = hvp_loss_sum / direction_count
    return raw_gradients, terms


def _weighted_task_values(
    terms: dict[str, torch.Tensor], training: dict[str, Any]
) -> dict[str, float]:
    return {
        "energy": float(training["lambda_energy"])
        * float(terms["energy_loss"].detach().cpu()),
        "force": float(training["lambda_force"])
        * float(terms["force_loss"].detach().cpu()),
        "hvp": float(training["lambda_hvp"])
        * float(terms["hvp_loss"].detach().cpu()),
    }


def backtracking_tasks_are_acceptable(
    baseline: dict[str, float],
    trial: dict[str, float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if baseline.keys() != trial.keys():
        raise ValueError("line-search task keys do not match")
    if not all(math.isfinite(value) for value in trial.values()):
        return False
    each_nonincreasing = all(
        trial[name]
        <= baseline[name] * (1.0 + relative_tolerance) + absolute_tolerance
        for name in baseline
    )
    return each_nonincreasing and sum(trial.values()) < sum(baseline.values())


def backtracking_tasks_are_budget_acceptable(
    baseline: dict[str, float],
    trial: dict[str, float],
    budgets: dict[str, float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if baseline.keys() != trial.keys() or set(budgets) != {"energy", "force"}:
        raise ValueError("budget line-search task keys do not match")
    if not all(math.isfinite(value) for value in trial.values()):
        return False
    for name in ("energy", "force"):
        limit = max(
            baseline[name] * (1.0 + relative_tolerance) + absolute_tolerance,
            float(budgets[name]),
        )
        if trial[name] > limit:
            return False
    return trial["hvp"] < baseline["hvp"]


def apply_backtracking_pcgrad_step(
    model: LocalMACEScalarResidual,
    parent: CapacityParent,
    direction_indices: list[int],
    terms: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    training: dict[str, Any],
    *,
    initial_step_size: float,
) -> dict[str, float | int | bool]:
    baseline = _weighted_task_values(terms, training)
    original = [parameter.detach().clone() for parameter in parameters]
    gradients = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    factor = float(training["line_search_backtrack_factor"])
    max_backtracks = int(training["line_search_max_backtracks"])
    relative_tolerance = float(training["line_search_relative_task_tolerance"])
    absolute_tolerance = float(training["line_search_absolute_task_tolerance"])
    if not 0.0 < factor < 1.0:
        raise ValueError("line-search backtrack factor must be between zero and one")
    if max_backtracks < 0 or initial_step_size <= 0.0:
        raise ValueError("invalid line-search step configuration")
    for backtracks in range(max_backtracks + 1):
        step_size = initial_step_size * factor**backtracks
        with torch.no_grad():
            for parameter, value, gradient in zip(
                parameters, original, gradients, strict=True
            ):
                parameter.copy_(value - step_size * gradient)
        _, trial_terms = _parent_loss_multi_direction(
            model,
            parent,
            direction_indices,
            training,
            create_parameter_graph=False,
        )
        trial = _weighted_task_values(trial_terms, training)
        acceptance_policy = str(
            training.get("line_search_acceptance_policy", "monotone_all")
        )
        if acceptance_policy == "monotone_all":
            accepted = backtracking_tasks_are_acceptable(
                baseline,
                trial,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        elif acceptance_policy == "physical_energy_force_budgets":
            accepted = backtracking_tasks_are_budget_acceptable(
                baseline,
                trial,
                {
                    "energy": float(training["line_search_energy_loss_budget"]),
                    "force": float(training["line_search_force_loss_budget"]),
                },
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            raise ValueError(
                f"unsupported line-search acceptance policy: {acceptance_policy}"
            )
        del trial_terms
        if accepted:
            return {
                "line_search_accepted": True,
                "line_search_backtracks": backtracks,
                "accepted_step_size": step_size,
                "line_search_baseline_total": sum(baseline.values()),
                "line_search_trial_total": sum(trial.values()),
            }
    with torch.no_grad():
        for parameter, value in zip(parameters, original, strict=True):
            parameter.copy_(value)
    return {
        "line_search_accepted": False,
        "line_search_backtracks": max_backtracks + 1,
        "accepted_step_size": 0.0,
        "line_search_baseline_total": sum(baseline.values()),
        "line_search_trial_total": math.nan,
    }


def _strict_capacity_gate(
    evaluation: dict[str, Any], rows: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, bool]:
    return {
        "hessian_median": evaluation["hessian_relative_frobenius"]["median"]
        <= float(gate["training_median_relative_frobenius_max"]),
        "hessian_all_parent": evaluation["hessian_relative_frobenius"]["max"]
        <= float(gate["training_all_parent_relative_frobenius_max"]),
        "energy_median": evaluation["energy_abs_error_hartree"]["median"]
        <= float(gate["energy_abs_error_median_max_hartree"]),
        "energy_all_parent": evaluation["energy_abs_error_hartree"]["max"]
        <= float(gate["energy_abs_error_all_parent_max_hartree"]),
        "force_median": evaluation["force_mae_hartree_per_bohr"]["median"]
        <= float(gate["force_mae_median_max_hartree_per_bohr"]),
        "force_all_parent": evaluation["force_mae_hartree_per_bohr"]["max"]
        <= float(gate["force_mae_all_parent_max_hartree_per_bohr"]),
        "symmetry": max(
            float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
        )
        <= float(gate["antisymmetric_over_symmetric_fro_max"]),
    }


def _validate_initial_checkpoint_metadata(
    checkpoint: dict[str, Any],
    inputs: dict[str, Any],
    selected_parent_ids: list[str],
) -> None:
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("initial checkpoint does not certify frozen Test100")
    if int(checkpoint.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("initial checkpoint Test100 count is not zero")
    if checkpoint.get("protocol_sha256") != inputs["initial_protocol_sha256"]:
        raise ValueError("initial checkpoint protocol hash drift")
    if checkpoint.get("arm_id") != inputs["initial_arm_id"]:
        raise ValueError("initial checkpoint arm drift")
    if [str(value) for value in checkpoint.get("selected_parent_ids", [])] != selected_parent_ids:
        raise ValueError("initial checkpoint parent list drift")


def _initialize_model_from_protocol(
    model: LocalMACEScalarResidual,
    protocol: dict[str, Any],
    selected_parent_ids: list[str],
    device: torch.device,
) -> dict[str, Any] | None:
    inputs = protocol["inputs"]
    checkpoint_value = inputs.get("initial_checkpoint")
    if checkpoint_value is None:
        return None
    checkpoint_path = Path(checkpoint_value)
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != inputs["initial_checkpoint_sha256"]:
        raise ValueError("initial checkpoint hash drift")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_initial_checkpoint_metadata(checkpoint, inputs, selected_parent_ids)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "path": checkpoint_path.resolve().as_posix(),
        "sha256": checkpoint_hash,
        "source_step": int(checkpoint["step"]),
        "source_arm_id": checkpoint["arm_id"],
        "source_protocol_sha256": checkpoint["protocol_sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol, arm = _load_protocol(args.protocol, args.arm_id, args.run_mode)
    training = protocol["training"]
    steps = int(training[f"{args.run_mode}_steps"])
    log_interval = int(training[f"log_interval_{args.run_mode}"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(arm["seed"]))
    parents, provenance = _load_selected_parents(protocol, device)
    parents = select_audit_parents(parents, args.parent_id)
    model, optimizer, architecture = _build_model_and_optimizer(arm, device)
    if not isinstance(model, LocalMACEScalarResidual):
        raise ValueError("force-secant capacity path is restricted to scalar MACE")
    selected_parent_ids = [parent.molecule_id for parent in parents]
    initialization = _initialize_model_from_protocol(
        model, protocol, selected_parent_ids, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(int(arm["seed"]) + 8191)
    best_score = math.inf
    best_step = -1
    metrics_path = args.output_dir / "training_metrics.jsonl"
    with metrics_path.open("w") as handle:
        for step in range(steps + 1):
            sampled = {key: math.nan for key in ("energy_loss", "force_loss", "hvp_loss", "parameter_loss")}
            gradient_norm = math.nan
            gradient_diagnostics: dict[str, float] = {}
            step_diagnostics: dict[str, float | int | bool] = {}
            parent_id = ""
            direction_index = -1
            direction_indices: list[int] = []
            if step > 0:
                parent_index = int(torch.randint(len(parents), (1,), generator=generator))
                parent = parents[parent_index]
                gradient_strategy = training.get("gradient_strategy", "summed_loss")
                if gradient_strategy == "accumulated_full_basis_pcgrad":
                    direction_indices = list(range(parent.positions.numel()))
                else:
                    directions_per_step = int(training.get("directions_per_step", 1))
                    if not 1 <= directions_per_step <= parent.positions.numel():
                        raise ValueError("directions_per_step exceeds coordinate count")
                    direction_indices = torch.randperm(
                        parent.positions.numel(), generator=generator
                    )[:directions_per_step].tolist()
                direction_index = direction_indices[0]
                parent_id = parent.molecule_id
                optimizer.zero_grad(set_to_none=True)
                if gradient_strategy == "accumulated_full_basis_pcgrad":
                    raw_gradients, terms = accumulated_full_basis_task_gradients(
                        model, parent, model.trainable_parameters(), training
                    )
                    gradient_diagnostics = apply_fixed_scale_pcgrad_from_gradients(
                        raw_gradients,
                        model.trainable_parameters(),
                        training,
                    )
                    del raw_gradients
                else:
                    loss, terms = _parent_loss_multi_direction(
                        model, parent, direction_indices, training
                    )
                if gradient_strategy == "summed_loss":
                    loss.backward()
                elif gradient_strategy == "fixed_scale_pcgrad":
                    gradient_diagnostics = apply_fixed_scale_pcgrad(
                        terms, model.trainable_parameters(), training
                    )
                elif gradient_strategy != "accumulated_full_basis_pcgrad":
                    raise ValueError(f"unsupported gradient strategy: {gradient_strategy}")
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_parameters(), float(training["gradient_clip"])
                    ).detach().cpu()
                )
                parameter_step_strategy = training.get(
                    "parameter_step_strategy", "optimizer"
                )
                if parameter_step_strategy == "optimizer":
                    optimizer.step()
                elif parameter_step_strategy == "backtracking_sgd":
                    if not isinstance(optimizer, torch.optim.SGD):
                        raise ValueError("backtracking PCGrad requires SGD")
                    step_diagnostics = apply_backtracking_pcgrad_step(
                        model,
                        parent,
                        direction_indices,
                        terms,
                        model.trainable_parameters(),
                        training,
                        initial_step_size=float(optimizer.param_groups[0]["lr"]),
                    )
                else:
                    raise ValueError(
                        f"unsupported parameter step strategy: {parameter_step_strategy}"
                    )
                sampled = {key: float(value.detach().cpu()) for key, value in terms.items()}
            if step % log_interval == 0 or step == steps:
                evaluation, rows, arrays = _evaluate(
                    model, parents, protocol["capacity_gate"], anchor_energy_force=False
                )
                strict_checks = _strict_capacity_gate(
                    evaluation, rows, protocol["strict_capacity_gate"]
                )
                score = (
                    evaluation["hessian_relative_frobenius"]["median"]
                    + evaluation["hessian_relative_frobenius"]["max"]
                    + evaluation["energy_abs_error_hartree"]["median"]
                    / float(training["energy_scale"])
                    + evaluation["force_mae_hartree_per_bohr"]["median"]
                    / float(training["force_scale"])
                )
                log_row = {
                    "step": step,
                    "wall_time_s": time.perf_counter() - started,
                    "sampled_parent_id": parent_id,
                    "sampled_direction_index": direction_index,
                    "sampled_direction_indices": direction_indices,
                    "sampled_direction_count": len(direction_indices),
                    "gradient_norm": gradient_norm,
                    "gradient_strategy": training.get(
                        "gradient_strategy", "summed_loss"
                    ),
                    **gradient_diagnostics,
                    **step_diagnostics,
                    **{f"sampled_{key}": value for key, value in sampled.items()},
                    **_periodic_evaluation_metrics(evaluation),
                    "strict_capacity_gate_passed": all(strict_checks.values()),
                    "strict_capacity_checks": strict_checks,
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0
                    ),
                }
                print(json.dumps(log_row, sort_keys=True), flush=True)
                handle.write(json.dumps(log_row, sort_keys=True) + "\n")
                handle.flush()
                checkpoint = {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "protocol": args.protocol.resolve().as_posix(),
                    "protocol_sha256": _sha256(args.protocol),
                    "arm_id": args.arm_id,
                    "selected_parent_ids": selected_parent_ids,
                    "initialization": initialization,
                    "test100_accessed": False,
                    "test100_evaluations_used": 0,
                }
                torch.save(checkpoint, args.output_dir / "last.ckpt")
                if score < best_score:
                    best_score = score
                    best_step = step
                    torch.save(checkpoint, args.output_dir / "best.ckpt")
                    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
                    for molecule_id, payload in arrays.items():
                        np.savez_compressed(
                            args.output_dir / f"{molecule_id}_result.npz", **payload
                        )
    best = torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    evaluation, rows, arrays = _evaluate(
        model, parents, protocol["capacity_gate"], anchor_energy_force=False
    )
    strict_checks = _strict_capacity_gate(
        evaluation, rows, protocol["strict_capacity_gate"]
    )
    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
    for molecule_id, payload in arrays.items():
        np.savez_compressed(args.output_dir / f"{molecule_id}_result.npz", **payload)
    result = {
        "definition": protocol["definitions"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "architecture": architecture,
        "initialization": initialization,
        "run_mode": args.run_mode,
        "steps": steps,
        "best_step": best_step,
        "final": evaluation,
        "strict_capacity_checks": strict_checks,
        "strict_capacity_gate_passed": all(strict_checks.values()),
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "selected_parent_ids": selected_parent_ids,
        "opened_label_paths": provenance["opened_label_paths"],
        "opened_capacity_arrays": provenance["opened_capacity_arrays"],
        "unselected_parent_artifacts_opened": 0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--parent-id")
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
