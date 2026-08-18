#!/usr/bin/env python3
"""Single-parent full39 Graphformer Hessian capacity-only optimizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mldft.ofdft.complete_total_training import (
    consume_implicit_response_diagnostics,
    hessian_error_metrics,
)
from qm9_complete_total_capacity_train import (
    IntegralBundleCache,
    MoleculeState,
    _analytic_relaxed_direction_prediction,
    _density_namespace,
    _evaluate_point_graph,
    _load_molecule,
    _parse_run,
    _refresh_base_densities,
    _save_checkpoint,
    _sha256,
)
from qm9_graphformer_analytic_relaxed_hvp_audit import _common_args
from qm9_hessian_density_relaxed_eval import _load_context


PROTOCOL_IDS = {
    "qm9_graphformer_0028399_full39_capacity_only_v1",
    "qm9_graphformer_0028399_full39_capacity_only_v2",
    "qm9_graphformer_0028399_full_network_capacity_smoke_v3",
}
MOLECULE_ID = "0028399"
SCOPES = (
    "energy_readout",
    "readout_last1",
    "readout_last2",
    "full_graphformer",
)
OPTIMIZERS = ("adamw", "lbfgs", "levenberg_marquardt")


@dataclass(frozen=True)
class Full39Result:
    objective: float
    residual_vector: torch.Tensor
    predicted_columns: torch.Tensor
    predicted_hessian: torch.Tensor
    reference_hessian: torch.Tensor
    metrics: dict[str, float]
    density_residual: float
    maximum_response_residual: float
    wall_time_s: float
    kkt_solve_s: float
    center_graph_s: float


def capacity_passed(relative_frobenius: float, threshold: float = 0.05) -> bool:
    return math.isfinite(relative_frobenius) and relative_frobenius <= threshold


def configure_capacity_numerics(protocol: dict[str, Any]) -> None:
    """Apply the registered process-wide numeric mode before sample construction."""
    if protocol["numerics"]["dtype"] != "float64":
        raise ValueError("capacity-only protocol requires float64")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20260727)
    np.random.seed(20260727)


def _parameter_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _reset_leaf_modules(module: torch.nn.Module) -> int:
    reset_count = 0
    for child in module.modules():
        if any(True for _ in child.children()):
            continue
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


def initialize_capacity_model(
    net: torch.nn.Module,
    *,
    arm: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply a registered pretrained or deterministic scratch initialization."""

    if arm not in settings:
        raise ValueError(f"unregistered initialization arm: {arm}")
    definition = settings[arm]
    kind = definition["type"]
    before_sha256 = _parameter_state_sha256(net)
    reset_count = 0
    if kind == "source_checkpoint":
        seed = None
    elif kind == "deterministic_reset":
        seed = int(definition["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        roots = tuple(definition["randomized_roots"])
        if not roots:
            raise ValueError("scratch initialization has no randomized roots")
        for root_name in roots:
            module = getattr(net, root_name, None)
            if module is None:
                raise ValueError(
                    f"scratch initialization root is absent: {root_name}"
                )
            reset_count += _reset_leaf_modules(module)
        named_parameters = dict(net.named_parameters())
        with torch.no_grad():
            for name, value in definition.get(
                "constant_parameter_overrides", {}
            ).items():
                if name not in named_parameters:
                    raise ValueError(
                        f"scratch constant override is absent: {name}"
                    )
                named_parameters[name].fill_(float(value))
            for name, limits in definition.get(
                "uniform_parameter_overrides", {}
            ).items():
                if name not in named_parameters:
                    raise ValueError(
                        f"scratch uniform override is absent: {name}"
                    )
                if len(limits) != 2:
                    raise ValueError(
                        f"scratch uniform override is invalid: {name}"
                    )
                torch.nn.init.uniform_(
                    named_parameters[name],
                    float(limits[0]),
                    float(limits[1]),
                )
        if reset_count == 0:
            raise RuntimeError("scratch initialization reset no leaf modules")
    else:
        raise ValueError(f"unsupported initialization type: {kind}")
    after_sha256 = _parameter_state_sha256(net)
    changed = after_sha256 != before_sha256
    if kind == "source_checkpoint" and changed:
        raise RuntimeError("pretrained initialization changed model parameters")
    if kind == "deterministic_reset" and not changed:
        raise RuntimeError("scratch initialization did not change model parameters")
    return {
        "arm": arm,
        "type": kind,
        "seed": seed,
        "randomized_roots": list(definition.get("randomized_roots", [])),
        "preserved_roots": list(definition.get("preserved_roots", [])),
        "constant_parameter_overrides": dict(
            definition.get("constant_parameter_overrides", {})
        ),
        "uniform_parameter_overrides": dict(
            definition.get("uniform_parameter_overrides", {})
        ),
        "reset_leaf_module_count": reset_count,
        "before_parameter_state_sha256": before_sha256,
        "after_parameter_state_sha256": after_sha256,
        "parameter_state_changed": changed,
    }


def implementation_provenance() -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/qm9_graphformer_full39_capacity_only.py",
        "scripts/qm9_complete_total_capacity_train.py",
        "scripts/qm9_graphformer_analytic_relaxed_hvp_audit.py",
        "scripts/qm9_total_ofdft_force_audit.py",
        "mldft/ofdft/complete_total_training.py",
        "mldft/ofdft/conservative_force.py",
        "mldft/ofdft/geometry_integrals.py",
        "mldft/ofdft/implicit_response.py",
        "mldft/ofdft/stationary_density.py",
    )
    provenance = {}
    for relative_path in relative_paths:
        path = root / relative_path
        provenance[relative_path] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
        }
    return provenance


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_full39_progress(
    cache: IntegralBundleCache,
    *,
    phase: str,
    completed_directions: int,
    started: float,
    density_residual: float,
    maximum_response_residual: float,
    kkt_solve_s: float,
    center_graph_s: float,
) -> None:
    output = Path(cache.args.output_dir) / "full39_progress.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "phase": phase,
                "completed_directions": completed_directions,
                "total_directions": 39,
                "wall_time_s": time.perf_counter() - started,
                "density_residual": density_residual,
                "maximum_response_residual": maximum_response_residual,
                "kkt_solve_s": kkt_solve_s,
                "center_graph_s": center_graph_s,
                "validation_accessed": False,
                "test100_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(output)


def _parameter_scope_names(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    scope: str,
) -> list[str]:
    names = [name for name, _ in named_parameters]
    if scope not in SCOPES:
        raise ValueError(f"unsupported parameter scope: {scope}")
    layer_indices = sorted(
        {
            int(match.group(1))
            for name in names
            if (
                match := re.match(
                    r"gnn_module\.g3d_layers\.(\d+)\.", name
                )
            )
        }
    )
    if not layer_indices or layer_indices != list(range(max(layer_indices) + 1)):
        raise ValueError("Graphformer block indices are incomplete")
    last = layer_indices[-1]
    selected = []
    for name in names:
        readout = name.startswith("energy_mlp.")
        block_match = re.match(r"gnn_module\.g3d_layers\.(\d+)\.", name)
        block = int(block_match.group(1)) if block_match else None
        include = {
            "energy_readout": readout,
            "readout_last1": readout or block == last,
            "readout_last2": readout or (
                block is not None and block >= max(0, last - 1)
            ),
            "full_graphformer": True,
        }[scope]
        if include:
            selected.append(name)
    if not selected or not any(name.startswith("energy_mlp.") for name in selected):
        raise ValueError(f"parameter scope {scope} omitted the energy readout")
    return selected


def configure_parameter_scope(
    net: torch.nn.Module,
    scope: str,
) -> tuple[list[str], list[torch.Tensor]]:
    named = list(net.named_parameters())
    selected_names = _parameter_scope_names(named, scope)
    selected_set = set(selected_names)
    for name, parameter in named:
        parameter.requires_grad_(name in selected_set)
        parameter.grad = None
    parameters = [parameter for name, parameter in named if name in selected_set]
    return selected_names, parameters


def _parameter_norm(parameters: list[torch.Tensor]) -> float:
    return float(
        torch.linalg.vector_norm(parameters_to_vector(parameters).detach()).cpu()
    )


def _set_parameter_vector(
    parameters: list[torch.Tensor], vector: torch.Tensor
) -> None:
    with torch.no_grad():
        vector_to_parameters(vector, parameters)


def _load_and_validate_assets(
    protocol_path: Path,
    asset_registration_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("protocol_id") not in PROTOCOL_IDS:
        raise ValueError("unexpected capacity-only protocol")
    boundary = protocol["access_boundary"]
    if (
        boundary["molecule_ids"] != [MOLECULE_ID]
        or boundary["sample_ids"] != [0]
        or any(
            boundary[key] is not False
            for key in (
                "stable5_access_allowed",
                "train20_access_allowed",
                "held_direction_access_allowed",
                "validation_access_allowed",
                "test100_access_allowed",
            )
        )
    ):
        raise ValueError("capacity-only access boundary drift")
    initialization_arms = protocol.get("initialization_arms")
    if protocol["protocol_id"].endswith("_v3"):
        if (
            set(initialization_arms or {}) != {"pretrained", "scratch"}
            or initialization_arms["pretrained"]["type"]
            != "source_checkpoint"
            or initialization_arms["scratch"]["type"]
            != "deterministic_reset"
        ):
            raise ValueError("v3 initialization arms are not frozen")
    if protocol["definitions"]["symmetric_matrix_power_mode"] != (
        "eigh_second_order_audit"
    ):
        raise ValueError("native-eigh second-order mode is not frozen")
    if protocol["definitions"]["classical_energy_basis"] != "physical":
        raise ValueError("physical-basis classical energy is not frozen")
    if protocol["definitions"]["forbidden_fallbacks"] != [
        "fixed_density",
        "stale_density",
        "detached_hvp",
        "force_head",
        "hessian_head",
    ]:
        raise ValueError("forbidden fallback list drift")
    loss = protocol["loss"]
    if (
        float(loss["lambda_energy"]) != 0.0
        or float(loss["lambda_force"]) != 0.0
        or float(loss["lambda_density"]) != 0.0
        or float(loss["lambda_hessian"]) != 1.0
    ):
        raise ValueError("capacity run must be Hessian-only")

    registration = json.loads(asset_registration_path.read_text())
    if (
        registration["protocol_sha256"] != _sha256(protocol_path)
        or registration["molecule_ids"] != [MOLECULE_ID]
        or int(registration["direction_count"]) != 39
        or any(
            registration[key] is not False
            for key in (
                "stable5_accessed",
                "train20_accessed",
                "validation_accessed",
                "test100_accessed",
            )
        )
    ):
        raise ValueError("capacity asset registration boundary drift")
    expected_checkpoint = Path(protocol["source"]["checkpoint"])
    if (
        Path(registration["source_checkpoint"]) != expected_checkpoint
        or registration["source_checkpoint_sha256"]
        != protocol["source"]["checkpoint_sha256"]
        or _sha256(expected_checkpoint)
        != protocol["source"]["checkpoint_sha256"]
    ):
        raise ValueError("fresh checkpoint identity drift")
    parent_path = Path(registration["parent_manifest"])
    direction_path = Path(registration["direction_manifest"])
    if (
        _sha256(parent_path) != registration["parent_manifest_sha256"]
        or _sha256(direction_path) != registration["direction_manifest_sha256"]
    ):
        raise ValueError("capacity manifest hash drift")
    parent_manifest = json.loads(parent_path.read_text())
    direction_manifest = json.loads(direction_path.read_text())
    if (
        parent_manifest["parent_count"] != 1
        or parent_manifest["molecule_ids"] != [MOLECULE_ID]
        or direction_manifest["parent_count"] != 1
        or direction_manifest["total_direction_count"] != 39
        or direction_manifest["roles"] != ["capacity_train"]
        or direction_manifest["held_directions_present"] is not False
    ):
        raise ValueError("capacity manifests are not one-parent full39")
    return protocol, registration, parent_manifest, direction_manifest


def _load_density_rescue(
    path: Path | None,
    protocol: dict[str, Any],
    protocol_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    rescue = yaml.safe_load(path.read_text())
    decision = rescue.get("decision", {})
    if (
        rescue.get("rescue_id")
        != "qm9_graphformer_0028399_density_solver_rescue_v1"
        or rescue.get("applies_to_protocol_id") != protocol["protocol_id"]
        or rescue.get("applies_to_protocol_sha256") != _sha256(protocol_path)
        or rescue.get("scope", {}).get("molecule_ids") != [MOLECULE_ID]
        or rescue.get("scope", {}).get("sample_ids") != [0]
        or rescue.get("scope", {}).get("density_solver_only") is not True
        or rescue.get("scope", {}).get("model_or_loss_change") is not False
        or decision.get("lbfgs_refine") is not False
        or decision.get("newton_refine") is not True
        or int(decision.get("newton_max_iterations", -1)) != 20
        or float(
            decision.get("strict_projected_gradient_threshold", math.nan)
        )
        != 1.0e-8
        or rescue.get("validation_accessed") is not False
        or rescue.get("test100_accessed") is not False
    ):
        raise ValueError("density-solver rescue registration drift")
    return rescue, _sha256(path)


def _direction_tensors(
    molecule: MoleculeState,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    ordered = sorted(molecule.directions, key=lambda item: item.index)
    if (
        len(ordered) != 39
        or molecule.expected_internal_dimension != 39
        or molecule.loaded_direction_role != "all"
        or any(direction.role != "capacity_train" for direction in ordered)
    ):
        raise ValueError("formal capacity objective requires all 39 capacity directions")
    basis = torch.as_tensor(
        np.stack([direction.vector.reshape(-1) for direction in ordered]),
        dtype=torch.float64,
        device=device,
    )
    target = torch.as_tensor(
        np.stack([direction.target_hvp.reshape(-1) for direction in ordered]),
        dtype=torch.float64,
        device=device,
    ).T
    projector = basis.T @ basis
    projected_target = projector @ target
    target_norm = float(
        torch.linalg.vector_norm(projected_target).detach().cpu()
    )
    if target_norm <= torch.finfo(torch.float64).tiny:
        raise ValueError("PBE internal Hessian norm is zero")
    return basis, projector, projected_target, target_norm


def _align_direction_tensors(
    basis: torch.Tensor,
    projector: torch.Tensor,
    projected_target: torch.Tensor,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {"device": like.device, "dtype": like.dtype}
    return (
        basis.to(**kwargs),
        projector.to(**kwargs),
        projected_target.to(**kwargs),
    )


def _assemble_result(
    molecule: MoleculeState,
    basis: torch.Tensor,
    projector: torch.Tensor,
    projected_target: torch.Tensor,
    predicted_columns: torch.Tensor,
    *,
    density_residual: float,
    maximum_response_residual: float,
    wall_time_s: float,
    kkt_solve_s: float,
    center_graph_s: float,
) -> Full39Result:
    basis, projector, projected_target = _align_direction_tensors(
        basis, projector, projected_target, predicted_columns
    )
    prediction = projector @ predicted_columns
    residual = prediction - projected_target
    target_norm = torch.linalg.vector_norm(projected_target).clamp_min(
        torch.finfo(prediction.dtype).tiny
    )
    objective = torch.sum(residual.square()) / target_norm.square()
    predicted_hessian = projector @ (prediction @ basis) @ projector
    reference_hessian = projector @ (
        torch.as_tensor(
            molecule.pbe_hessian,
            dtype=prediction.dtype,
            device=prediction.device,
        )
    ) @ projector
    metrics = {
        key: float(value.detach().cpu())
        for key, value in hessian_error_metrics(
            predicted_hessian, reference_hessian
        ).items()
    }
    return Full39Result(
        objective=float(objective.detach().cpu()),
        residual_vector=(residual / target_norm).T.reshape(-1).detach(),
        predicted_columns=prediction.detach(),
        predicted_hessian=predicted_hessian.detach(),
        reference_hessian=reference_hessian.detach(),
        metrics=metrics,
        density_residual=float(density_residual),
        maximum_response_residual=float(maximum_response_residual),
        wall_time_s=float(wall_time_s),
        kkt_solve_s=float(kkt_solve_s),
        center_graph_s=float(center_graph_s),
    )


def evaluate_full39(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
) -> Full39Result:
    if molecule.base is None:
        raise RuntimeError("strict center density is missing")
    started = time.perf_counter()
    device = next(context.model.parameters()).device
    basis, projector, projected_target, _ = _direction_tensors(
        molecule, device
    )
    columns = []
    maximum_response_residual = 0.0
    density_residual = float(molecule.base.final_gradient_norm)
    kkt_solve_s = 0.0
    center_graph_s = 0.0
    with torch.enable_grad():
        for completed, direction in enumerate(
            sorted(molecule.directions, key=lambda item: item.index), start=1
        ):
            hvp, norms, diagnostics = _analytic_relaxed_direction_prediction(
                context,
                cache,
                molecule,
                direction,
                create_graph=False,
            )
            basis, projector, projected_target = _align_direction_tensors(
                basis, projector, projected_target, hvp
            )
            columns.append(hvp.reshape(-1).detach())
            maximum_response_residual = max(
                maximum_response_residual,
                float(diagnostics["response_stationarity_residual"]),
                float(diagnostics["response_constraint_residual"]),
            )
            kkt_solve_s += float(diagnostics["kkt_solve_seconds"])
            center_graph_s += float(diagnostics["center_graph_seconds"])
            density_residual = max(
                density_residual,
                *(float(value.detach().cpu()) for value in norms),
            )
            _write_full39_progress(
                cache,
                phase="evaluate",
                completed_directions=completed,
                started=started,
                density_residual=density_residual,
                maximum_response_residual=maximum_response_residual,
                kkt_solve_s=kkt_solve_s,
                center_graph_s=center_graph_s,
            )
    if density_residual >= cache.args.density_strict_threshold:
        raise RuntimeError(
            "full39 evaluation used a non-strict center density: "
            f"{density_residual:.3e} >= "
            f"{cache.args.density_strict_threshold:.3e}"
        )
    return _assemble_result(
        molecule,
        basis,
        projector,
        projected_target,
        torch.stack(columns, dim=1),
        density_residual=density_residual,
        maximum_response_residual=maximum_response_residual,
        wall_time_s=time.perf_counter() - started,
        kkt_solve_s=kkt_solve_s,
        center_graph_s=center_graph_s,
    )


def full39_objective_and_gradient(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    *,
    residual_weights: torch.Tensor | None = None,
) -> tuple[Full39Result, list[torch.Tensor], dict[str, Any]]:
    """Accumulate 39 exact VJPs without changing parameters."""
    if molecule.base is None:
        raise RuntimeError("strict center density is missing")
    started = time.perf_counter()
    device = parameters[0].device
    basis, projector, projected_target, target_norm = _direction_tensors(
        molecule, device
    )
    if residual_weights is not None:
        residual_weights = residual_weights.to(
            device=device, dtype=torch.float64
        ).reshape(39, -1).T
        if residual_weights.shape != projected_target.shape:
            raise ValueError("linearized residual weight shape drift")
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    columns = []
    maximum_response_residual = 0.0
    maximum_density_residual = float(molecule.base.final_gradient_norm)
    kkt_solve_s = 0.0
    center_graph_s = 0.0
    implicit_rows: list[dict[str, Any]] = []
    for offset, direction in enumerate(
        sorted(molecule.directions, key=lambda item: item.index)
    ):
        hvp, norms, diagnostics = _analytic_relaxed_direction_prediction(
            context,
            cache,
            molecule,
            direction,
            create_graph=True,
        )
        basis, projector, projected_target = _align_direction_tensors(
            basis, projector, projected_target, hvp
        )
        flat = hvp.reshape(-1)
        projected = projector @ flat
        target = projected_target[:, offset]
        residual = projected - target
        if residual_weights is None:
            scalar = torch.sum(residual.square()) / (target_norm * target_norm)
        else:
            scalar = torch.dot(
                residual / target_norm, residual_weights[:, offset]
            )
        gradients = torch.autograd.grad(
            scalar,
            parameters,
            allow_unused=True,
        )
        for accumulator, gradient in zip(
            accumulated, gradients, strict=True
        ):
            if gradient is not None:
                accumulator.add_(gradient.detach())
        columns.append(flat.detach())
        maximum_density_residual = max(
            maximum_density_residual,
            *(float(value.detach().cpu()) for value in norms),
        )
        maximum_response_residual = max(
            maximum_response_residual,
            float(diagnostics["response_stationarity_residual"]),
            float(diagnostics["response_constraint_residual"]),
        )
        kkt_solve_s += float(diagnostics["kkt_solve_seconds"])
        center_graph_s += float(diagnostics["center_graph_seconds"])
        implicit_rows.extend(consume_implicit_response_diagnostics())
        _write_full39_progress(
            cache,
            phase="gradient",
            completed_directions=offset + 1,
            started=started,
            density_residual=maximum_density_residual,
            maximum_response_residual=maximum_response_residual,
            kkt_solve_s=kkt_solve_s,
            center_graph_s=center_graph_s,
        )
        del hvp, scalar, gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if maximum_density_residual >= cache.args.density_strict_threshold:
        raise RuntimeError(
            "full39 gradient used a non-strict center density: "
            f"{maximum_density_residual:.3e} >= "
            f"{cache.args.density_strict_threshold:.3e}"
        )
    result = _assemble_result(
        molecule,
        basis,
        projector,
        projected_target,
        torch.stack(columns, dim=1),
        density_residual=maximum_density_residual,
        maximum_response_residual=maximum_response_residual,
        wall_time_s=time.perf_counter() - started,
        kkt_solve_s=kkt_solve_s,
        center_graph_s=center_graph_s,
    )
    return result, accumulated, {
        "implicit_adjoint_solve_count": len(implicit_rows),
        "implicit_adjoint_rows": implicit_rows,
    }


def assign_gradients(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
) -> float:
    if len(parameters) != len(gradients):
        raise ValueError("parameter/gradient count mismatch")
    squared = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite full39 parameter gradient")
        parameter.grad = gradient
        squared = squared + torch.sum(gradient.to(torch.float64).square())
    return float(torch.sqrt(squared).detach().cpu())


def one_direction_no_update_gradient_smoke(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
) -> dict[str, Any]:
    """Exercise the scoped implicit adjoint without performing an update."""
    if molecule.base is None:
        raise RuntimeError("strict center density is missing")
    direction = sorted(molecule.directions, key=lambda item: item.index)[0]
    before = parameters_to_vector(parameters).detach().clone()
    hvp, norms, diagnostics = _analytic_relaxed_direction_prediction(
        context,
        cache,
        molecule,
        direction,
        create_graph=True,
    )
    basis, projector, projected_target, target_norm = _direction_tensors(
        molecule, parameters[0].device
    )
    basis, projector, projected_target = _align_direction_tensors(
        basis, projector, projected_target, hvp
    )
    residual = (
        projector @ hvp.reshape(-1) - projected_target[:, 0]
    )
    loss = torch.sum(residual.square()) / (target_norm * target_norm)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    gradient_norm_squared = torch.zeros(
        (), dtype=torch.float64, device=parameters[0].device
    )
    used = 0
    for gradient in gradients:
        if gradient is not None:
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("one-direction smoke gradient is non-finite")
            gradient_norm_squared = (
                gradient_norm_squared
                + torch.sum(gradient.to(torch.float64).square())
            )
            used += 1
    after = parameters_to_vector(parameters).detach()
    parameter_change = float(torch.linalg.vector_norm(after - before).cpu())
    if parameter_change != 0.0:
        raise RuntimeError("gradient smoke changed parameters before an optimizer step")
    density_residual = max(
        float(molecule.base.final_gradient_norm),
        *(float(value.detach().cpu()) for value in norms),
    )
    if density_residual >= cache.args.density_strict_threshold:
        raise RuntimeError("gradient smoke used a non-strict center density")
    return {
        "formal_capacity_evaluation": False,
        "formal_projection_path_checked": True,
        "hvp_device": str(hvp.device),
        "update_performed": False,
        "direction_index": int(direction.index),
        "direction_kind": direction.kind,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(torch.sqrt(gradient_norm_squared).cpu()),
        "gradient_parameter_tensor_count": used,
        "parameter_change_norm": parameter_change,
        "density_residual": density_residual,
        "response_stationarity_residual": float(
            diagnostics["response_stationarity_residual"]
        ),
        "response_constraint_residual": float(
            diagnostics["response_constraint_residual"]
        ),
        "implicit_adjoint": consume_implicit_response_diagnostics(),
    }


def _refresh_center(
    context: Any,
    molecule: MoleculeState,
    density_args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    row = _refresh_base_densities(
        context,
        [molecule],
        density_args,
        index,
        parameter_step=index,
    )[0]
    if float(row["final_gradient_norm"]) >= density_args.density_strict_threshold:
        raise RuntimeError("center density failed the strict capacity threshold")
    return row


def _result_row(
    result: Full39Result,
    *,
    update: int,
    scope: str,
    optimizer: str,
    gradient_norm: float | None,
    update_norm: float | None,
    parameter_norm: float,
    closure_calls: int = 0,
) -> dict[str, Any]:
    return {
        "update": update,
        "scope": scope,
        "optimizer": optimizer,
        "objective": result.objective,
        "full39_relative_frobenius": result.metrics["relative_frobenius"],
        "full39_mae": result.metrics["mae"],
        "full39_rmse": result.metrics["rmse"],
        "hessian_symmetry_max_abs": result.metrics["symmetry_max_abs"],
        "antisymmetric_over_symmetric_frobenius": result.metrics[
            "antisymmetric_over_symmetric_frobenius"
        ],
        "density_residual": result.density_residual,
        "maximum_response_residual": result.maximum_response_residual,
        "full39_wall_time_s": result.wall_time_s,
        "kkt_solve_s": result.kkt_solve_s,
        "center_graph_s": result.center_graph_s,
        "gradient_norm": gradient_norm,
        "update_norm": update_norm,
        "parameter_norm": parameter_norm,
        "closure_calls": closure_calls,
    }


def _save_result_arrays(
    output_dir: Path, tag: str, result: Full39Result
) -> None:
    array_dir = output_dir / "full39_arrays"
    array_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        array_dir / f"{tag}.npz",
        predicted_hessian=result.predicted_hessian.cpu().numpy(),
        reference_hessian=result.reference_hessian.cpu().numpy(),
        predicted_columns=result.predicted_columns.cpu().numpy(),
        normalized_residual_vector=result.residual_vector.cpu().numpy(),
    )


def _persist_optimizer_progress(
    args: argparse.Namespace,
    context: Any,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    update: int,
    definition: str,
) -> None:
    checkpoint = args.output_dir / "last.ckpt"
    _save_checkpoint(
        Path(args.source_checkpoint),
        context,
        optimizer,
        None,
        checkpoint,
        update,
        definition=definition,
        provenance_update=args.checkpoint_provenance,
    )
    _write_csv(args.output_dir / "training_metrics.partial.csv", rows)
    state = {
        "status": "running",
        "last_fully_evaluated_update": update,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "full39_relative_frobenius": rows[-1][
            "full39_relative_frobenius"
        ],
        "parameter_scope": args.scope,
        "optimizer": args.optimizer,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    temporary = args.output_dir / "run_state.json.tmp"
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(args.output_dir / "run_state.json")


def _plateaued(
    values: list[float], patience: int, relative_tolerance: float
) -> bool:
    if len(values) <= patience:
        return False
    previous = values[-patience - 1]
    best_recent = min(values[-patience:])
    scale = max(abs(previous), torch.finfo(torch.float64).tiny)
    return (previous - best_recent) / scale <= relative_tolerance


def run_adamw(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    density_args: argparse.Namespace,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rows = []
    values: list[float] = []
    status = "max_updates"
    for update in range(int(settings["max_updates"]) + 1):
        _refresh_center(context, molecule, density_args, update)
        if update == 0:
            result = evaluate_full39(context, cache, molecule)
            gradient_norm = None
            update_norm = None
        else:
            before = parameters_to_vector(parameters).detach().clone()
            optimizer.zero_grad(set_to_none=True)
            result_before, gradients, _ = full39_objective_and_gradient(
                context, cache, molecule, parameters
            )
            gradient_norm = assign_gradients(parameters, gradients)
            torch.nn.utils.clip_grad_norm_(
                parameters, float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            after = parameters_to_vector(parameters).detach()
            update_norm = float(torch.linalg.vector_norm(after - before).cpu())
            _refresh_center(context, molecule, density_args, update)
            result = evaluate_full39(context, cache, molecule)
            del result_before, before, after
        row = _result_row(
            result,
            update=update,
            scope=args.scope,
            optimizer=args.optimizer,
            gradient_norm=gradient_norm,
            update_norm=update_norm,
            parameter_norm=_parameter_norm(parameters),
        )
        rows.append(row)
        values.append(result.metrics["relative_frobenius"])
        _save_result_arrays(args.output_dir, f"update_{update:04d}", result)
        print(json.dumps(row, sort_keys=True), flush=True)
        _persist_optimizer_progress(
            args,
            context,
            optimizer,
            rows,
            update=update,
            definition=(
                "single-parent exact-full39 Hessian-only AdamW capacity audit"
            ),
        )
        if capacity_passed(
            result.metrics["relative_frobenius"],
            float(args.capacity_threshold),
        ):
            status = "capacity_passed"
            break
        if _plateaued(
            values,
            int(settings["convergence_patience"]),
            float(settings["relative_change_tolerance"]),
        ):
            status = "plateau"
            break
    _save_checkpoint(
        Path(args.source_checkpoint),
        context,
        optimizer,
        None,
        args.output_dir / "last.ckpt",
        len(rows) - 1,
        definition="single-parent exact-full39 Hessian-only AdamW capacity audit",
        provenance_update=args.checkpoint_provenance,
    )
    return rows, status


def run_lbfgs(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    density_args: argparse.Namespace,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=float(settings["learning_rate"]),
        max_iter=1,
        max_eval=int(settings["max_evaluations_per_update"]),
        tolerance_grad=float(settings["tolerance_grad"]),
        tolerance_change=float(settings["tolerance_change"]),
        history_size=int(settings["history_size"]),
        line_search_fn="strong_wolfe",
    )
    rows = []
    values: list[float] = []
    closure_calls = 0
    latest_gradient_norm: float | None = None
    status = "max_updates"
    for update in range(int(settings["max_updates"]) + 1):
        _refresh_center(context, molecule, density_args, closure_calls)
        result = evaluate_full39(context, cache, molecule)
        row = _result_row(
            result,
            update=update,
            scope=args.scope,
            optimizer=args.optimizer,
            gradient_norm=latest_gradient_norm,
            update_norm=None,
            parameter_norm=_parameter_norm(parameters),
            closure_calls=closure_calls,
        )
        rows.append(row)
        values.append(result.metrics["relative_frobenius"])
        _save_result_arrays(args.output_dir, f"update_{update:04d}", result)
        print(json.dumps(row, sort_keys=True), flush=True)
        _persist_optimizer_progress(
            args,
            context,
            optimizer,
            rows,
            update=update,
            definition=(
                "single-parent exact-full39 Hessian-only L-BFGS capacity audit"
            ),
        )
        if capacity_passed(
            result.metrics["relative_frobenius"],
            float(args.capacity_threshold),
        ):
            status = "capacity_passed"
            break
        if update == int(settings["max_updates"]):
            break
        before = parameters_to_vector(parameters).detach().clone()

        def closure() -> torch.Tensor:
            nonlocal closure_calls, latest_gradient_norm
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            _refresh_center(context, molecule, density_args, closure_calls)
            objective, gradients, _ = full39_objective_and_gradient(
                context, cache, molecule, parameters
            )
            latest_gradient_norm = assign_gradients(parameters, gradients)
            return torch.as_tensor(
                objective.objective,
                dtype=parameters[0].dtype,
                device=parameters[0].device,
            )

        optimizer.step(closure)
        after = parameters_to_vector(parameters).detach()
        rows[-1]["next_update_norm"] = float(
            torch.linalg.vector_norm(after - before).cpu()
        )
        if _plateaued(values, int(settings["convergence_patience"]), 1.0e-5):
            status = "plateau"
            break
    _save_checkpoint(
        Path(args.source_checkpoint),
        context,
        optimizer,
        None,
        args.output_dir / "last.ckpt",
        len(rows) - 1,
        definition="single-parent exact-full39 Hessian-only L-BFGS capacity audit",
        provenance_update=args.checkpoint_provenance,
    )
    return rows, status


def damped_cgls(
    residual: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    *,
    parameter_count: int,
    damping: float,
    max_iterations: int,
    relative_tolerance: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve min ||residual + J x||^2 + damping ||x||^2."""
    if damping < 0.0:
        raise ValueError("CGLS damping must be nonnegative")
    data_rhs = -residual.detach()
    x = torch.zeros(
        parameter_count, dtype=residual.dtype, device=residual.device
    )
    data_remainder = data_rhs.clone()
    normal = rmatvec(data_remainder)
    direction = normal.clone()
    initial_normal_norm = float(torch.linalg.vector_norm(normal).cpu())
    if initial_normal_norm == 0.0:
        return x, {
            "iterations": 0,
            "converged": True,
            "normal_relative_residual": 0.0,
            "linearized_residual_upper_bound": float(
                torch.linalg.vector_norm(residual).cpu()
            ),
        }
    normal_squared = torch.dot(normal, normal)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        action = matvec(direction)
        denominator = torch.dot(action, action) + damping * torch.dot(
            direction, direction
        )
        if not torch.isfinite(denominator) or float(denominator.cpu()) <= 0.0:
            break
        alpha = normal_squared / denominator
        x = x + alpha * direction
        data_remainder = data_remainder - alpha * action
        next_normal = rmatvec(data_remainder) - damping * x
        next_normal_norm = float(torch.linalg.vector_norm(next_normal).cpu())
        if next_normal_norm <= relative_tolerance * initial_normal_norm:
            normal = next_normal
            converged = True
            break
        next_squared = torch.dot(next_normal, next_normal)
        beta = next_squared / normal_squared
        direction = next_normal + beta * direction
        normal = next_normal
        normal_squared = next_squared
    linearized_remainder = residual + matvec(x)
    return x, {
        "iterations": iterations,
        "converged": converged,
        "normal_relative_residual": (
            float(torch.linalg.vector_norm(normal).cpu()) / initial_normal_norm
        ),
        "linearized_residual_upper_bound": float(
            torch.linalg.vector_norm(linearized_remainder).cpu()
        ),
        "linearized_minimum_claimed": converged,
    }


def _linearized_operators(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    density_args: argparse.Namespace,
    *,
    fd_relative_step: float,
) -> tuple[
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    dict[str, int],
]:
    base_parameters = parameters_to_vector(parameters).detach().clone()
    _refresh_center(context, molecule, density_args, 0)
    base_result = evaluate_full39(context, cache, molecule)
    counters = {"matvec_calls": 0, "rmatvec_calls": 0}

    def restore() -> None:
        _set_parameter_vector(parameters, base_parameters)

    def residual_at(vector: torch.Tensor, refresh_index: int) -> torch.Tensor:
        _set_parameter_vector(parameters, vector)
        _refresh_center(context, molecule, density_args, refresh_index)
        return evaluate_full39(context, cache, molecule).residual_vector

    def matvec(direction: torch.Tensor) -> torch.Tensor:
        counters["matvec_calls"] += 1
        direction_norm = float(torch.linalg.vector_norm(direction).cpu())
        if direction_norm == 0.0:
            return torch.zeros_like(base_result.residual_vector)
        step = (
            fd_relative_step
            * max(float(torch.linalg.vector_norm(base_parameters).cpu()), 1.0)
            / direction_norm
        )
        plus = residual_at(
            base_parameters + step * direction,
            2 * counters["matvec_calls"],
        )
        minus = residual_at(
            base_parameters - step * direction,
            2 * counters["matvec_calls"] + 1,
        )
        restore()
        return (plus - minus) / (2.0 * step)

    def rmatvec(output_direction: torch.Tensor) -> torch.Tensor:
        counters["rmatvec_calls"] += 1
        restore()
        _refresh_center(
            context, molecule, density_args, 100000 + counters["rmatvec_calls"]
        )
        _, gradients, _ = full39_objective_and_gradient(
            context,
            cache,
            molecule,
            parameters,
            residual_weights=output_direction,
        )
        return parameters_to_vector(gradients).detach()

    restore()
    return base_result.residual_vector, matvec, rmatvec, counters


def run_linearization(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    density_args: argparse.Namespace,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    residual, matvec, rmatvec, counters = _linearized_operators(
        context,
        cache,
        molecule,
        parameters,
        density_args,
        fd_relative_step=float(settings["parameter_fd_relative_step"]),
    )
    step, diagnostics = damped_cgls(
        residual,
        matvec,
        rmatvec,
        parameter_count=int(parameters_to_vector(parameters).numel()),
        damping=float(settings["initial_damping"]),
        max_iterations=int(settings["cgls_max_iterations"]),
        relative_tolerance=float(settings["cgls_relative_tolerance"]),
    )
    result = {
        "scope": args.scope,
        "fresh_checkpoint": args.source_checkpoint.as_posix(),
        "fresh_checkpoint_sha256": _sha256(args.source_checkpoint),
        "initial_relative_frobenius": float(
            torch.linalg.vector_norm(residual).cpu()
        ),
        "linearized_step_norm": float(torch.linalg.vector_norm(step).cpu()),
        **diagnostics,
        **counters,
        "minimum_interpretation": (
            "converged damped linearized minimum"
            if diagnostics["converged"]
            else "achieved linearized residual upper bound only"
        ),
        "validation_accessed": False,
        "test100_accessed": False,
    }
    (args.output_dir / "linearized_capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def run_lm(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    parameters: list[torch.Tensor],
    density_args: argparse.Namespace,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    rows = []
    linearized_rows = []
    damping = float(settings["initial_damping"])
    patience = int(settings["convergence_patience"])
    no_improvement = 0
    status = "max_updates"
    dummy_optimizer = torch.optim.SGD(parameters, lr=1.0)
    for update in range(int(settings["max_updates"]) + 1):
        _refresh_center(context, molecule, density_args, update)
        result = evaluate_full39(context, cache, molecule)
        row = _result_row(
            result,
            update=update,
            scope=args.scope,
            optimizer=args.optimizer,
            gradient_norm=None,
            update_norm=None,
            parameter_norm=_parameter_norm(parameters),
        )
        row["lm_damping"] = damping
        rows.append(row)
        _save_result_arrays(args.output_dir, f"update_{update:04d}", result)
        print(json.dumps(row, sort_keys=True), flush=True)
        _persist_optimizer_progress(
            args,
            context,
            dummy_optimizer,
            rows,
            update=update,
            definition=(
                "single-parent exact-full39 damped Gauss-Newton/LM capacity audit"
            ),
        )
        if capacity_passed(
            result.metrics["relative_frobenius"],
            float(args.capacity_threshold),
        ):
            status = "capacity_passed"
            break
        if update == int(settings["max_updates"]):
            break
        base_parameters = parameters_to_vector(parameters).detach().clone()
        residual, matvec, rmatvec, counters = _linearized_operators(
            context,
            cache,
            molecule,
            parameters,
            density_args,
            fd_relative_step=float(settings["parameter_fd_relative_step"]),
        )
        step, diagnostics = damped_cgls(
            residual,
            matvec,
            rmatvec,
            parameter_count=int(base_parameters.numel()),
            damping=damping,
            max_iterations=int(settings["cgls_max_iterations"]),
            relative_tolerance=float(settings["cgls_relative_tolerance"]),
        )
        linearized_rows.append(
            {"update": update, "damping": damping, **diagnostics, **counters}
        )
        accepted = False
        for trial_index, scale in enumerate(settings["trial_scales"]):
            trial = base_parameters + float(scale) * step
            _set_parameter_vector(parameters, trial)
            _refresh_center(
                context,
                molecule,
                density_args,
                200000 + update * 100 + trial_index,
            )
            trial_result = evaluate_full39(context, cache, molecule)
            if trial_result.objective < result.objective:
                rows[-1]["next_update_norm"] = float(
                    torch.linalg.vector_norm(trial - base_parameters).cpu()
                )
                rows[-1]["accepted_trial_scale"] = float(scale)
                accepted = True
                no_improvement = 0
                damping = max(
                    float(settings["minimum_damping"]), damping / 3.0
                )
                break
        if not accepted:
            _set_parameter_vector(parameters, base_parameters)
            no_improvement += 1
            damping = min(float(settings["maximum_damping"]), damping * 10.0)
        if no_improvement >= patience:
            status = "plateau"
            break
    _save_checkpoint(
        Path(args.source_checkpoint),
        context,
        dummy_optimizer,
        None,
        args.output_dir / "last.ckpt",
        len(rows) - 1,
        definition="single-parent exact-full39 damped Gauss-Newton/LM capacity audit",
        provenance_update=args.checkpoint_provenance,
    )
    return rows, status, linearized_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (
        protocol,
        registration,
        parent_manifest,
        direction_manifest,
    ) = _load_and_validate_assets(args.protocol, args.asset_registration)
    configure_capacity_numerics(protocol)
    density_rescue, density_rescue_sha256 = _load_density_rescue(
        args.density_rescue, protocol, args.protocol
    )
    if args.scope not in protocol["parameter_scopes"]["order"]:
        raise ValueError("scope is not in frozen execution order")
    if args.mode == "optimize" and args.optimizer not in protocol["optimizers"]["order"]:
        raise ValueError("optimizer is not in frozen execution order")

    os.environ["MLDFT_SYMMETRIC_MATRIX_POWER_MODE"] = (
        "eigh_second_order_audit"
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    common = _common_args(args.output_dir, protocol, args.device)
    common.integral_derivative_workers = int(
        protocol["numerics"]["integral_derivative_workers"]
    )
    common.integral_cache_entries = int(
        protocol["numerics"]["integral_cache_entries"]
    )
    density_settings = protocol["numerics"]["density"]
    common.optimizer = "adam"
    common.lr = float(density_settings["stage1_adam_lr"])
    common.max_cycle = int(density_settings["stage1_max_cycles"])
    common.convergence_tolerance = float(
        density_settings["stage1_threshold"]
    )
    common.momentum = 0.9
    if density_rescue is not None:
        decision = density_rescue["decision"]
        common.lbfgs_refine = bool(decision["lbfgs_refine"])
        common.newton_refine = bool(decision["newton_refine"])
        common.newton_max_iterations = int(
            decision["newton_max_iterations"]
        )
    run_spec = _parse_run(
        f"fresh_capacity={protocol['source']['run_dir']}="
        f"{protocol['source']['checkpoint']}"
    )
    density_args = _density_namespace(common)
    context = _load_context(run_spec, density_args, device)
    initialization = initialize_capacity_model(
        context.model.net,
        arm=args.initialization,
        settings=protocol.get(
            "initialization_arms",
            {"pretrained": {"type": "source_checkpoint"}},
        ),
    )
    names, parameters = configure_parameter_scope(context.model.net, args.scope)
    if not parameters:
        raise ValueError("selected parameter scope is empty")
    molecule = _load_molecule(
        parent_manifest["parents"][0],
        low_mode_count=0,
        direction_entry=direction_manifest["parents"][0],
        direction_role="all",
    )
    if molecule.molecule_id != MOLECULE_ID:
        raise ValueError("unexpected capacity molecule")
    cache = IntegralBundleCache(context, common)
    args.source_checkpoint = Path(protocol["source"]["checkpoint"])
    args.capacity_threshold = float(
        protocol["capacity_gate"]["full39_relative_frobenius_max"]
    )
    args.checkpoint_provenance = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(args.protocol),
        "asset_registration": args.asset_registration.resolve().as_posix(),
        "asset_registration_sha256": _sha256(args.asset_registration),
        "parent_manifest_sha256": registration["parent_manifest_sha256"],
        "direction_manifest_sha256": registration["direction_manifest_sha256"],
        "source_checkpoint_sha256": registration[
            "source_checkpoint_sha256"
        ],
        "parameter_scope": args.scope,
        "parameter_names": names,
        "optimizer_name": args.optimizer,
        "initialization": initialization,
        "density_rescue": (
            args.density_rescue.resolve().as_posix()
            if args.density_rescue is not None
            else None
        ),
        "density_rescue_sha256": density_rescue_sha256,
        "lambda_energy": 0.0,
        "lambda_force": 0.0,
        "lambda_density": 0.0,
        "lambda_hessian": 1.0,
        "complete_direction_count_per_update": 39,
        "proxy_hvp_fallback_allowed": False,
        "process_default_dtype": str(torch.get_default_dtype()),
        "command_argv": sys.argv,
        "implementation": implementation_provenance(),
        "validation_accessed": False,
        "test100_accessed": False,
    }

    linearized_rows: list[dict[str, Any]] = []
    gradient_smoke = None
    if args.mode == "density_preflight":
        density_row = _refresh_center(context, molecule, density_args, 0)
        rows = [
            {
                "update": 0,
                "scope": args.scope,
                "optimizer": "none",
                "objective": math.nan,
                "full39_relative_frobenius": math.nan,
                "density_residual": float(
                    density_row["final_gradient_norm"]
                ),
                "density_cycles": int(density_row["cycles"]),
                "total_energy_hartree": float(density_row["total_energy"]),
            }
        ]
        linearized = None
        status = "strict_center_density_preflight_passed"
    elif args.mode == "gradient_smoke":
        density_row = _refresh_center(context, molecule, density_args, 0)
        gradient_smoke = one_direction_no_update_gradient_smoke(
            context, cache, molecule, parameters
        )
        rows = [
            {
                "update": 0,
                "scope": args.scope,
                "optimizer": "none",
                "objective": math.nan,
                "full39_relative_frobenius": math.nan,
                "density_residual": float(
                    density_row["final_gradient_norm"]
                ),
                **gradient_smoke,
            }
        ]
        linearized = None
        status = "one_direction_no_update_gradient_smoke_passed"
    elif args.mode == "evaluate":
        _refresh_center(context, molecule, density_args, 0)
        result = evaluate_full39(context, cache, molecule)
        rows = [
            _result_row(
                result,
                update=0,
                scope=args.scope,
                optimizer="none",
                gradient_norm=None,
                update_norm=None,
                parameter_norm=_parameter_norm(parameters),
            )
        ]
        _save_result_arrays(args.output_dir, "fresh_checkpoint", result)
        linearized = None
        status = "fresh_checkpoint_evaluated"
    elif args.mode == "linearize":
        linearized = run_linearization(
            context,
            cache,
            molecule,
            parameters,
            density_args,
            protocol["optimizers"]["levenberg_marquardt"],
            args,
        )
        rows: list[dict[str, Any]] = []
        status = "linearization_complete"
    elif args.optimizer == "adamw":
        rows, status = run_adamw(
            context,
            cache,
            molecule,
            parameters,
            density_args,
            protocol["optimizers"]["adamw"],
            args,
        )
        linearized = None
    elif args.optimizer == "lbfgs":
        rows, status = run_lbfgs(
            context,
            cache,
            molecule,
            parameters,
            density_args,
            protocol["optimizers"]["lbfgs"],
            args,
        )
        linearized = None
    else:
        rows, status, linearized_rows = run_lm(
            context,
            cache,
            molecule,
            parameters,
            density_args,
            protocol["optimizers"]["levenberg_marquardt"],
            args,
        )
        linearized = None

    _write_csv(args.output_dir / "training_metrics.csv", rows)
    _write_csv(args.output_dir / "linearized_iterations.csv", linearized_rows)
    final_relative = (
        float(rows[-1]["full39_relative_frobenius"]) if rows else math.nan
    )
    passed = bool(rows) and capacity_passed(
        final_relative, args.capacity_threshold
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(args.protocol),
        "asset_registration": args.asset_registration.resolve().as_posix(),
        "asset_registration_sha256": _sha256(args.asset_registration),
        "source_checkpoint": args.source_checkpoint.as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "initialization": initialization,
        "registered_hashes": {
            "protocol_sha256": _sha256(args.protocol),
            "asset_registration_sha256": _sha256(
                args.asset_registration
            ),
            "parent_manifest_sha256": registration[
                "parent_manifest_sha256"
            ],
            "direction_manifest_sha256": registration[
                "direction_manifest_sha256"
            ],
            "direction_artifact_sha256": registration[
                "direction_artifact_sha256"
            ],
            "source_checkpoint_sha256": registration[
                "source_checkpoint_sha256"
            ],
            "density_rescue_sha256": density_rescue_sha256,
        },
        "implementation": implementation_provenance(),
        "process_default_dtype": str(torch.get_default_dtype()),
        "command_argv": sys.argv,
        "working_directory": Path.cwd().as_posix(),
        "molecule_ids": [MOLECULE_ID],
        "sample_ids": [0],
        "direction_count_per_update": 39,
        "parameter_scope": args.scope,
        "parameter_tensor_count": len(parameters),
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_names": names,
        "mode": args.mode,
        "optimizer": args.optimizer,
        "density_rescue": (
            args.density_rescue.resolve().as_posix()
            if args.density_rescue is not None
            else None
        ),
        "density_rescue_sha256": density_rescue_sha256,
        "status": status,
        "updates_completed": max(0, len(rows) - 1),
        "final": rows[-1] if rows else None,
        "capacity_threshold": args.capacity_threshold,
        "capacity_passed": passed,
        "capacity_conclusion": (
            "Graphformer具有单分子Hessian表达能力" if passed else None
        ),
        "linearized_capacity": linearized,
        "one_direction_no_update_gradient_smoke": gradient_smoke,
        "linearized_iterations": linearized_rows,
        "insufficient_capacity_conclusion_allowed": False,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "integral_cache_build_count": cache.build_count,
        "integral_cache_hit_count": cache.hit_count,
        "stable5_accessed": False,
        "train20_accessed": False,
        "held_directions_accessed": False,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if (args.output_dir / "last.ckpt").exists():
        summary["checkpoint"] = (args.output_dir / "last.ckpt").as_posix()
        summary["checkpoint_sha256"] = _sha256(args.output_dir / "last.ckpt")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_state_path = args.output_dir / "run_state.json"
    if run_state_path.exists():
        run_state = json.loads(run_state_path.read_text())
        run_state.update(
            {
                "status": status,
                "capacity_passed": passed,
                "summary": summary_path.as_posix(),
                "summary_sha256": _sha256(summary_path),
            }
        )
        run_state_path.write_text(
            json.dumps(run_state, indent=2, sort_keys=True) + "\n"
        )
    if passed:
        frozen = {
            "conclusion": "Graphformer具有单分子Hessian表达能力",
            "summary": summary_path.resolve().as_posix(),
            "summary_sha256": _sha256(summary_path),
            "checkpoint": summary.get("checkpoint"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "remaining_arms_authorized": False,
            "validation_accessed": False,
            "test100_accessed": False,
        }
        (args.output_dir / "CAPACITY_PASSED_FROZEN.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-registration", type=Path, required=True)
    parser.add_argument("--density-rescue", type=Path)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "density_preflight",
            "gradient_smoke",
            "evaluate",
            "optimize",
            "linearize",
        ),
        default="optimize",
    )
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument(
        "--initialization",
        choices=("pretrained", "scratch"),
        default="pretrained",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "proxy_fallback_used": False,
                    "stable5_accessed": False,
                    "train20_accessed": False,
                    "validation_accessed": False,
                    "test100_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
