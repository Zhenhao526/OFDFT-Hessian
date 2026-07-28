#!/usr/bin/env python3
"""Audit the stable5 linear ceiling of angular plus bonded-torsion scalars."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)

try:
    from scripts.qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from scripts.qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _design_blocks,
        _distribution,
    )
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )
except ModuleNotFoundError:
    from qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _design_blocks,
        _distribution,
    )
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )


FeatureFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ParentFeaturePlan:
    four_body_function: FeatureFunction
    four_body_keys: tuple[tuple[int, ...], ...]
    global_columns: torch.Tensor
    angular_feature_mode: str
    central_element_indices: tuple[int, ...]
    chain_group_count: int
    bond_count: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _central_resolved_angular_features(
    atomic_features: torch.Tensor,
    element_index: torch.Tensor,
    *,
    element_count: int,
    central_element_indices: tuple[int, ...],
) -> torch.Tensor:
    central_counts = atomic_features[:, :element_count].sum(dim=0)
    environment = atomic_features[:, element_count:]
    blocks = [
        environment[element_index == central_index].sum(dim=0)
        for central_index in central_element_indices
    ]
    return torch.cat((central_counts, *blocks))


def _central_resolved_angular_columns(
    *,
    atomic_feature_count: int,
    element_count: int,
    central_element_indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    environment_count = atomic_feature_count - element_count
    blocks = [torch.arange(element_count, dtype=torch.long, device=device)]
    for central_index in central_element_indices:
        start = element_count + central_index * environment_count
        blocks.append(
            torch.arange(
                start,
                start + environment_count,
                dtype=torch.long,
                device=device,
            )
        )
    return torch.cat(blocks)


def _build_feature_plans(
    parents: list[CapacityParent],
    model: LocalAngularScalarResidual,
    settings: dict[str, Any],
) -> tuple[dict[str, ParentFeaturePlan], int, tuple[tuple[int, ...], ...]]:
    centers = np.linspace(
        float(settings["center_min_bohr"]),
        float(settings["center_max_bohr"]),
        int(settings["center_count"]),
    )
    local: dict[
        str,
        tuple[FeatureFunction, tuple[tuple[int, ...], ...], int, int],
    ] = {}
    all_keys: set[tuple[int, ...]] = set()
    for parent in parents:
        atomic_numbers = parent.atomic_numbers.detach().cpu().numpy()
        positions = parent.positions.detach().cpu().numpy()
        groups, bonds = build_bonded_chain_groups(
            atomic_numbers,
            positions,
            float(settings["bond_scale"]),
        )
        if not groups:
            raise ValueError(f"no bonded torsion groups for {parent.molecule_id}")
        function, raw_keys = make_four_body_feature_function(
            groups,
            centers,
            float(settings["sigma_bohr"]),
            int(settings["torsion_order"]),
        )
        keys = tuple(tuple(int(value) for value in key) for key in raw_keys)
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate torsion keys for {parent.molecule_id}")
        all_keys.update(keys)
        local[parent.molecule_id] = (function, keys, len(groups), len(bonds))

    first = parents[0]
    with torch.no_grad():
        raw_angular_feature_count = int(
            model.network.atomic_features(
                first.positions,
                model._element_index(first.atomic_numbers),
                first.topology,
            ).shape[1]
        )
    angular_feature_mode = str(
        settings.get("angular_feature_mode", "raw_atomic_sum")
    )
    if angular_feature_mode == "raw_atomic_sum":
        angular_feature_count = raw_angular_feature_count
    elif angular_feature_mode == "central_element_resolved":
        angular_feature_count = model.network.element_count + (
            model.network.element_count
            * (raw_angular_feature_count - model.network.element_count)
        )
    else:
        raise ValueError(f"unsupported angular feature mode: {angular_feature_mode}")
    global_keys = tuple(sorted(all_keys))
    global_lookup = {key: index for index, key in enumerate(global_keys)}
    plans = {}
    for parent in parents:
        function, keys, chain_count, bond_count = local[parent.molecule_id]
        parent_element_index = model._element_index(parent.atomic_numbers)
        central_element_indices = tuple(
            int(value)
            for value in torch.unique(parent_element_index, sorted=True).tolist()
        )
        if angular_feature_mode == "raw_atomic_sum":
            angular_columns = torch.arange(
                angular_feature_count,
                dtype=torch.long,
                device=parent.positions.device,
            )
        else:
            angular_columns = _central_resolved_angular_columns(
                atomic_feature_count=raw_angular_feature_count,
                element_count=model.network.element_count,
                central_element_indices=central_element_indices,
                device=parent.positions.device,
            )
        torsion_columns = torch.as_tensor(
            [angular_feature_count + global_lookup[key] for key in keys],
            dtype=torch.long,
            device=parent.positions.device,
        )
        plans[parent.molecule_id] = ParentFeaturePlan(
            four_body_function=function,
            four_body_keys=keys,
            global_columns=torch.cat((angular_columns, torsion_columns)),
            angular_feature_mode=angular_feature_mode,
            central_element_indices=central_element_indices,
            chain_group_count=chain_count,
            bond_count=bond_count,
        )
    return plans, angular_feature_count, global_keys


def _parent_feature_jet(
    model: LocalAngularScalarResidual,
    parent: CapacityParent,
    plan: ParentFeaturePlan,
    *,
    feature_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    element_index = model._element_index(parent.atomic_numbers)

    def feature_function(flat_positions: torch.Tensor) -> torch.Tensor:
        positions = flat_positions.reshape(parent.positions.shape)
        atomic_features = model.network.atomic_features(
            positions,
            element_index,
            parent.topology,
        )
        if plan.angular_feature_mode == "raw_atomic_sum":
            angular = atomic_features.sum(dim=0)
        else:
            angular = _central_resolved_angular_features(
                atomic_features,
                element_index,
                element_count=model.network.element_count,
                central_element_indices=plan.central_element_indices,
            )
        four_body = plan.four_body_function(positions)
        return torch.cat((angular, four_body))

    return _chunked_vector_jet(
        feature_function,
        parent.positions.detach().reshape(-1),
        feature_chunk_size=feature_chunk_size,
    )


def _cgls_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    column_floor: float,
    max_iterations: int,
    tolerance: float,
    log_interval: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if ridge < 0.0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid CGLS settings")
    column_norm = torch.linalg.vector_norm(design, dim=0)
    active = column_norm > column_floor
    if not bool(torch.any(active)):
        raise ValueError("combined design has no active columns")
    scale = column_norm[active]
    matrix = design[:, active] / scale[None, :]
    solution = torch.zeros(matrix.shape[1], dtype=design.dtype, device=design.device)
    residual = target.clone()
    normal_residual = matrix.T @ residual
    direction = normal_residual.clone()
    gamma = torch.dot(normal_residual, normal_residual)
    initial_normal = torch.sqrt(gamma).clamp_min(torch.finfo(design.dtype).tiny)
    history = []
    converged = False
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        projected = matrix @ direction
        denominator = torch.dot(projected, projected) + ridge * torch.dot(
            direction, direction
        )
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise FloatingPointError("non-positive/non-finite CGLS denominator")
        step = gamma / denominator
        solution = solution + step * direction
        residual = residual - step * projected
        next_normal = matrix.T @ residual - ridge * solution
        next_gamma = torch.dot(next_normal, next_normal)
        relative_normal = float(torch.sqrt(next_gamma) / initial_normal)
        completed_iterations = iteration
        if iteration == 1 or iteration % log_interval == 0 or relative_normal <= tolerance:
            row = {
                "iteration": iteration,
                "relative_normal_residual": relative_normal,
                "relative_data_residual": float(
                    torch.linalg.vector_norm(residual)
                    / torch.clamp(torch.linalg.vector_norm(target), min=1e-30)
                ),
                "normalized_solution_norm": float(torch.linalg.vector_norm(solution)),
            }
            history.append(row)
            print(json.dumps({"event": "cgls", **row}, sort_keys=True), flush=True)
        if relative_normal <= tolerance:
            converged = True
            gamma = next_gamma
            break
        beta = next_gamma / gamma
        direction = next_normal + beta * direction
        normal_residual = next_normal
        gamma = next_gamma

    coefficients = torch.zeros(
        design.shape[1], dtype=design.dtype, device=design.device
    )
    coefficients[active] = solution / scale
    data_residual = design @ coefficients - target
    return coefficients, {
        "solver": "column_normalized_cgls",
        "active_feature_count": int(torch.sum(active)),
        "design_row_count": int(design.shape[0]),
        "design_column_count": int(design.shape[1]),
        "ridge": ridge,
        "max_iterations": max_iterations,
        "completed_iterations": completed_iterations,
        "tolerance": tolerance,
        "converged": converged,
        "final_relative_normal_residual": float(torch.sqrt(gamma) / initial_normal),
        "design_residual_relative": float(
            torch.linalg.vector_norm(data_residual)
            / torch.clamp(torch.linalg.vector_norm(target), min=1e-30)
        ),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "column_norm_min_active": float(torch.min(scale)),
        "column_norm_max": float(torch.max(column_norm)),
        "history": history,
    }


def _normal_equation_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    column_floor: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve a column-normalized ridge problem by float64 Gram/Cholesky."""
    if ridge <= 0.0:
        raise ValueError("normal-equation Cholesky requires a positive ridge")
    column_norm = torch.linalg.vector_norm(design, dim=0)
    active = column_norm > column_floor
    if not bool(torch.any(active)):
        raise ValueError("combined design has no active columns")
    scale = column_norm[active]
    matrix = design[:, active] / scale[None, :]
    right_hand_side = matrix.T @ target
    gram = matrix.T @ matrix
    gram.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(gram)
    if bool(torch.any(info != 0)):
        raise FloatingPointError(
            f"ridge Gram matrix is not positive definite: info={int(torch.max(info))}"
        )
    normalized_solution = torch.cholesky_solve(
        right_hand_side[:, None], factor
    ).squeeze(1)
    coefficients = torch.zeros(
        design.shape[1], dtype=design.dtype, device=design.device
    )
    coefficients[active] = normalized_solution / scale
    residual = design @ coefficients - target
    normalized_residual = matrix @ normalized_solution - target
    normal_residual = matrix.T @ normalized_residual + ridge * normalized_solution
    initial_normal = torch.linalg.vector_norm(right_hand_side).clamp_min(
        torch.finfo(design.dtype).tiny
    )
    return coefficients, {
        "solver": "column_normalized_ridge_gram_cholesky_float64",
        "active_feature_count": int(torch.sum(active)),
        "design_row_count": int(design.shape[0]),
        "design_column_count": int(design.shape[1]),
        "ridge": ridge,
        "converged": True,
        "final_relative_normal_residual": float(
            torch.linalg.vector_norm(normal_residual) / initial_normal
        ),
        "design_residual_relative": float(
            torch.linalg.vector_norm(residual)
            / torch.clamp(torch.linalg.vector_norm(target), min=1e-30)
        ),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "normalized_solution_norm": float(
            torch.linalg.vector_norm(normalized_solution)
        ),
        "column_norm_min_active": float(torch.min(scale)),
        "column_norm_max": float(torch.max(column_norm)),
        "cholesky_info_max": int(torch.max(info)),
        "history": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "smoke")
    if str(arm["architecture"]) != "local_angular_symmetry_network":
        raise ValueError("combined ceiling requires the local angular architecture")
    prerequisite_path = Path(protocol["inputs"]["angular_linear_ceiling_summary"])
    if _sha256(prerequisite_path) != str(
        protocol["inputs"]["angular_linear_ceiling_summary_sha256"]
    ):
        raise ValueError("angular linear-ceiling prerequisite hash drift")
    prerequisite = json.loads(prerequisite_path.read_text())
    if prerequisite.get("test100_accessed") is not False:
        raise ValueError("angular prerequisite does not certify frozen Test100")
    if prerequisite["gate"]["passed"] is not False:
        raise ValueError("four-body contingency requires failed angular-only ceiling")
    continuation_path_value = protocol["inputs"].get("four_body_initial_summary")
    if continuation_path_value is not None:
        continuation_path = Path(continuation_path_value)
        if _sha256(continuation_path) != str(
            protocol["inputs"]["four_body_initial_summary_sha256"]
        ):
            raise ValueError("four-body initial-solve summary hash drift")
        continuation = json.loads(continuation_path.read_text())
        if continuation.get("test100_accessed") is not False:
            raise ValueError("four-body initial solve does not freeze Test100")
        if continuation["gate"]["passed"] is not False:
            raise ValueError("solver continuation is unnecessary after a passed gate")
        if continuation["solve"]["converged"] is not False:
            raise ValueError("solver continuation requires a truncated initial CGLS solve")
    exact_path_value = protocol["inputs"].get("four_body_exact_summary")
    if exact_path_value is not None:
        exact_path = Path(exact_path_value)
        if _sha256(exact_path) != str(
            protocol["inputs"]["four_body_exact_summary_sha256"]
        ):
            raise ValueError("four-body exact-ridge summary hash drift")
        exact = json.loads(exact_path.read_text())
        if exact.get("test100_accessed") is not False:
            raise ValueError("four-body exact solve does not freeze Test100")
        if exact["solve"]["converged"] is not True:
            raise ValueError("central typing requires a converged prior exact solve")
        if exact["gate"]["passed"] is not False:
            raise ValueError("central typing is unnecessary after a passed exact gate")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    model = LocalAngularScalarResidual(
        hidden_size=int(arm["hidden_size"]),
        radial_size=int(arm["radial_size"]),
        angular_order=int(arm["angular_order"]),
        cutoff_bohr=float(arm["cutoff_bohr"]),
        seed=int(arm["seed"]),
        activation=str(arm["activation"]),
        radial_feature_scale=float(arm["radial_feature_scale"]),
        angular_feature_scale=float(arm["angular_feature_scale"]),
    ).to(device)
    settings = protocol["four_body_ceiling"]
    offload_parent_tensors = bool(
        settings.get("offload_parent_tensors_to_cpu", False)
    )
    cache_parent_jets = bool(settings.get("cache_parent_feature_jets", False))
    if cache_parent_jets and not offload_parent_tensors:
        raise ValueError("parent feature-jet caching requires CPU offload")
    feature_jet_cache_dir = args.output_dir / "feature_jets"
    if cache_parent_jets:
        feature_jet_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_read_dir_value = settings.get("parent_feature_jet_cache_read_dir")
    feature_jet_cache_read_dir = (
        Path(cache_read_dir_value)
        if cache_read_dir_value is not None
        else feature_jet_cache_dir
    )
    protocol_sha256 = _sha256(args.protocol)
    compatible_cache_protocols = {
        str(value)
        for value in settings.get("compatible_feature_jet_protocol_sha256s", [])
    }
    accepted_cache_protocols = {protocol_sha256, *compatible_cache_protocols}
    plans, angular_feature_count, global_four_body_keys = _build_feature_plans(
        parents, model, settings
    )
    global_feature_count = angular_feature_count + len(global_four_body_keys)
    training = protocol["training"]
    designs = []
    targets = []
    jets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    jet_seconds = {}
    progress_path = args.output_dir / "feature_jet_progress.jsonl"
    with progress_path.open("w") as progress_handle:
        for parent_index, parent in enumerate(parents, start=1):
            plan = plans[parent.molecule_id]
            jet_started = time.perf_counter()
            cache_path = feature_jet_cache_dir / f"{parent.molecule_id}.pt"
            read_cache_path = (
                cache_path
                if cache_path.is_file()
                else feature_jet_cache_read_dir / f"{parent.molecule_id}.pt"
            )
            cache_hit = False
            if cache_parent_jets and read_cache_path.is_file():
                cached = torch.load(
                    read_cache_path, map_location="cpu", weights_only=False
                )
                expected = {
                    "arm_id": args.arm_id,
                    "molecule_id": parent.molecule_id,
                    "angular_feature_mode": plan.angular_feature_mode,
                    "global_feature_count": global_feature_count,
                }
                observed = {key: cached.get(key) for key in expected}
                if observed != expected:
                    raise ValueError(
                        f"feature-jet cache provenance mismatch for {parent.molecule_id}: "
                        f"expected={expected}, observed={observed}"
                    )
                cached_protocol = str(cached.get("protocol_sha256"))
                if cached_protocol not in accepted_cache_protocols:
                    raise ValueError(
                        f"feature-jet cache protocol mismatch for {parent.molecule_id}: "
                        f"observed={cached_protocol}, accepted={sorted(accepted_cache_protocols)}"
                    )
                jet = tuple(cached[key] for key in ("features", "jacobian", "hessian"))
                cache_hit = True
                if read_cache_path != cache_path:
                    migrated = {**cached, "protocol_sha256": protocol_sha256}
                    temporary_path = cache_path.with_suffix(".tmp")
                    torch.save(migrated, temporary_path)
                    temporary_path.replace(cache_path)
            else:
                jet = _parent_feature_jet(
                    model,
                    parent,
                    plan,
                    feature_chunk_size=int(settings["feature_chunk_size"]),
                )
                if offload_parent_tensors:
                    jet = tuple(value.detach().cpu() for value in jet)
                if cache_parent_jets:
                    cache_payload = {
                        "protocol_sha256": protocol_sha256,
                        "arm_id": args.arm_id,
                        "molecule_id": parent.molecule_id,
                        "angular_feature_mode": plan.angular_feature_mode,
                        "global_feature_count": global_feature_count,
                        "features": jet[0],
                        "jacobian": jet[1],
                        "hessian": jet[2],
                    }
                    temporary_path = cache_path.with_suffix(".tmp")
                    torch.save(cache_payload, temporary_path)
                    temporary_path.replace(cache_path)
            jets[parent.molecule_id] = jet
            jet_seconds[parent.molecule_id] = time.perf_counter() - jet_started
            local_designs, parent_targets = _design_blocks(
                parent,
                *jet,
                energy_scale=float(training["energy_scale"]),
                force_scale=float(training["force_scale"]),
                hessian_scale=float(training["absolute_hessian_scale"]),
                relative_fraction=float(training["relative_loss_fraction"]),
                hessian_reference_floor=float(training["hessian_reference_floor"]),
            )
            for local_design in local_designs:
                global_columns = plan.global_columns.to(local_design.device)
                designs.append(
                    torch.zeros(
                        (local_design.shape[0], global_feature_count),
                        dtype=local_design.dtype,
                        device=local_design.device,
                    ).index_copy(1, global_columns, local_design)
                )
            targets.extend(parent_targets)
            progress = {
                "event": "combined_feature_jet_complete",
                "parent_index": parent_index,
                "parent_count": len(parents),
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "local_feature_count": jet[0].numel(),
                "global_feature_count": global_feature_count,
                "angular_feature_count": angular_feature_count,
                "angular_feature_mode": plan.angular_feature_mode,
                "central_element_indices": plan.central_element_indices,
                "local_four_body_feature_count": len(plan.four_body_keys),
                "global_four_body_feature_count": len(global_four_body_keys),
                "chain_group_count": plan.chain_group_count,
                "feature_jet_cache_hit": cache_hit,
                "feature_jet_cache_source": (
                    read_cache_path.resolve().as_posix() if cache_hit else None
                ),
                "parent_tensors_offloaded_to_cpu": offload_parent_tensors,
                "feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
                "elapsed_wall_time_s": time.perf_counter() - started,
                "gpu_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            progress_handle.write(json.dumps(progress, sort_keys=True) + "\n")
            progress_handle.flush()
            print(json.dumps(progress, sort_keys=True), flush=True)
            if offload_parent_tensors and device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    design = torch.cat(designs, dim=0)
    target = torch.cat(targets, dim=0)
    designs.clear()
    targets.clear()
    gc.collect()
    if offload_parent_tensors and device.type == "cuda":
        design = design.to(device)
        target = target.to(device)
    print(
        json.dumps(
            {
                "event": "combined_linear_solve_start",
                "design_shape": list(design.shape),
                "elapsed_wall_time_s": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    solver = str(settings.get("solver", "cgls"))
    if solver == "cgls":
        coefficients, solve = _cgls_solve(
            design,
            target,
            ridge=float(settings["ridge"]),
            column_floor=float(settings["column_floor"]),
            max_iterations=int(settings["cgls_max_iterations"]),
            tolerance=float(settings["cgls_tolerance"]),
            log_interval=int(settings["cgls_log_interval"]),
        )
    elif solver == "normal_equation_cholesky":
        coefficients, solve = _normal_equation_solve(
            design,
            target,
            ridge=float(settings["ridge"]),
            column_floor=float(settings["column_floor"]),
        )
    else:
        raise ValueError(f"unsupported linear solver: {solver}")
    if offload_parent_tensors:
        coefficients = coefficients.detach().cpu()
        del design, target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for parent in parents:
        plan = plans[parent.molecule_id]
        features, jacobian, feature_hessian = jets[parent.molecule_id]
        local_coefficients = coefficients[
            plan.global_columns.to(coefficients.device)
        ]
        correction_energy = torch.dot(features, local_coefficients)
        correction_force = -(jacobian.T @ local_coefficients).reshape(
            parent.positions.shape
        )
        correction_hessian = torch.einsum(
            "fij,f->ij", feature_hessian, local_coefficients
        )
        predicted_energy = parent.source_energy + float(correction_energy)
        predicted_force = parent.source_force + correction_force.detach().cpu().numpy()
        predicted_hessian = (
            parent.source_hessian_symmetric
            + correction_hessian.detach().cpu().numpy()
        )
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
            "local_four_body_feature_count": len(plan.four_body_keys),
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "source_energy_abs_error_hartree": abs(
                parent.source_energy - parent.pbe_energy
            ),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(predicted_force - parent.pbe_force))
            ),
            "source_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.source_force - parent.pbe_force))
            ),
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_force=predicted_force,
            predicted_hessian=predicted_hessian,
            pbe_hessian=parent.pbe_hessian,
            source_hessian_symmetric=parent.source_hessian_symmetric,
        )
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hessian_distribution = _distribution(rows, "relative_frobenius")
    energy_distribution = _distribution(rows, "energy_abs_error_hartree")
    source_energy_distribution = _distribution(rows, "source_energy_abs_error_hartree")
    force_distribution = _distribution(rows, "force_mae_hartree_per_bohr")
    source_force_distribution = _distribution(rows, "source_force_mae_hartree_per_bohr")
    energy_ratio = energy_distribution["median"] / max(
        source_energy_distribution["median"], np.finfo(float).tiny
    )
    force_ratio = force_distribution["median"] / max(
        source_force_distribution["median"], np.finfo(float).tiny
    )
    gate = protocol["capacity_gate"]
    checks = {
        "training_median_relative_frobenius": hessian_distribution["median"]
        <= float(gate["training_median_relative_frobenius_max"]),
        "training_all_parent_relative_frobenius": hessian_distribution["max"]
        <= float(gate["training_all_parent_relative_frobenius_max"]),
        "energy_median_ratio_to_source": energy_ratio
        <= float(gate["energy_median_ratio_to_source_max"]),
        "force_median_ratio_to_source": force_ratio
        <= float(gate["force_median_ratio_to_source_max"]),
        "antisymmetric_over_symmetric_frobenius": max(
            float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
        )
        <= float(gate["antisymmetric_over_symmetric_frobenius_max"]),
    }
    torch.save(
        {
            "coefficients": coefficients.detach().cpu(),
            "angular_feature_count": angular_feature_count,
            "global_four_body_keys": global_four_body_keys,
            "protocol": args.protocol.resolve().as_posix(),
            "protocol_sha256": _sha256(args.protocol),
            "arm_id": args.arm_id,
            "test100_accessed": False,
            **provenance,
        },
        args.output_dir / "linear_ceiling.pt",
    )
    result = {
        "definition": (
            "Column-normalized linear E/F/full-Hessian ceiling of a local angular scalar "
            "plus parity-even bonded four-body torsion RBF scalar."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "angular_feature_count": angular_feature_count,
        "global_four_body_feature_count": len(global_four_body_keys),
        "global_feature_count": global_feature_count,
        "four_body_settings": settings,
        "offload_parent_tensors_to_cpu": offload_parent_tensors,
        "cache_parent_feature_jets": cache_parent_jets,
        "parent_feature_jet_cache_read_dir": feature_jet_cache_read_dir.resolve().as_posix(),
        "compatible_feature_jet_protocol_sha256s": sorted(compatible_cache_protocols),
        "solve": solve,
        "hessian_relative_frobenius": hessian_distribution,
        "energy_abs_error_hartree": energy_distribution,
        "force_mae_hartree_per_bohr": force_distribution,
        "energy_median_ratio_to_source": energy_ratio,
        "force_median_ratio_to_source": force_ratio,
        "gate": {**checks, "passed": all(checks.values())},
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "parent_cv_design_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", default="M1_local_angular_tanh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
