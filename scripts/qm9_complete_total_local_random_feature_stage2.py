#!/usr/bin/env python3
"""Run frozen train20 direction generalization for the nonlinear local scalar."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr

from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)
from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)

try:
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from scripts.qm9_complete_total_local_angular_four_body_linear_ceiling import (
        _build_feature_plans,
        _central_resolved_angular_features,
    )
    from scripts.qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _distribution,
    )
    from scripts.qm9_complete_total_local_random_feature_kernel_ceiling import (
        _central_random_features,
        _random_feature_columns,
    )
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
    )
    from scripts.qm9_complete_total_structured_scalar_subspace import (
        build_structured_scalar_subspace_mapping,
        expand_structured_subspace_coefficients,
        project_design_to_structured_subspace,
    )
    from scripts.prepare_qm9_complete_total_direction_generalization import (
        _external_basis,
    )
except ModuleNotFoundError:
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from qm9_complete_total_local_angular_four_body_linear_ceiling import (
        _build_feature_plans,
        _central_resolved_angular_features,
    )
    from qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _distribution,
    )
    from qm9_complete_total_local_random_feature_kernel_ceiling import (
        _central_random_features,
        _random_feature_columns,
    )
    from qm9_complete_total_local_scalar_full_hessian_capacity import CapacityParent
    from qm9_complete_total_structured_scalar_subspace import (
        build_structured_scalar_subspace_mapping,
        expand_structured_subspace_coefficients,
        project_design_to_structured_subspace,
    )
    from prepare_qm9_complete_total_direction_generalization import _external_basis


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    if protocol.get("stage") != "train20_unseen_direction_nonlinear_local_scalar":
        raise ValueError("unexpected Stage-2 protocol stage")
    inputs = protocol["inputs"]
    for path_key, hash_key in (
        ("selected_manifest", "selected_manifest_sha256"),
        ("v9_summary", "v9_summary_sha256"),
        ("v9_vibrational_summary", "v9_vibrational_summary_sha256"),
        ("v9_checkpoint", "v9_checkpoint_sha256"),
    ):
        source = Path(inputs[path_key])
        if _sha256(source) != str(inputs[hash_key]):
            raise ValueError(f"input hash drift: {path_key}")
    v9 = json.loads(Path(inputs["v9_summary"]).read_text())
    if v9.get("gate", {}).get("passed") is not True:
        raise ValueError("v9 stable5 matrix gate did not pass")
    vibration = json.loads(Path(inputs["v9_vibrational_summary"]).read_text())
    if vibration.get("test100_accessed") is not False:
        raise ValueError("v9 vibration prerequisite does not freeze Test100")
    return protocol


def _load_parent(row: dict[str, Any], device: torch.device) -> CapacityParent:
    label = zarr.open(str(row["label_path"]), mode="r")
    atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
    positions = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
    pbe_force = np.asarray(
        label["metadata/pbe_derivatives/forces"], dtype=np.float64
    )
    energy_trace = np.asarray(label["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        label["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    with np.load(row["capacity_array"]) as payload:
        source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
        pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        source_force = np.asarray(payload["predicted_base_force"], dtype=np.float64)
    atomic_tensor = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    positions_tensor = torch.as_tensor(positions, dtype=torch.float64, device=device)
    return CapacityParent(
        molecule_id=str(row["molecule_id"]),
        natoms=int(atomic_numbers.size),
        atomic_numbers=atomic_tensor,
        positions=positions_tensor,
        topology=build_body_order_topology(
            torch.as_tensor(atomic_numbers, dtype=torch.long),
            torch.as_tensor(positions, dtype=torch.float64),
            bond_scale=1.25,
        ),
        pbe_energy=float(energy_trace[has_energy][-1]),
        pbe_force=pbe_force,
        pbe_hessian=pbe_hessian,
        source_energy=float(row["baseline_total_energy_hartree"]),
        source_force=source_force,
        source_hessian_raw=source_hessian,
        source_hessian_symmetric=0.5 * (source_hessian + source_hessian.T),
    )


def _move_parent(parent: CapacityParent, device: torch.device) -> CapacityParent:
    return replace(
        parent,
        atomic_numbers=parent.atomic_numbers.to(device),
        positions=parent.positions.to(device),
    )


def _load_train20(
    protocol: dict[str, Any], device: torch.device
) -> tuple[list[CapacityParent], list[dict[str, Any]], dict[str, Any]]:
    inputs = protocol["inputs"]
    manifest_path = Path(inputs["selected_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("selected manifest does not freeze Test100")
    if manifest.get("source_split_sha256") != inputs["source_split_sha256"]:
        raise ValueError("selected manifest source split drift")
    rows = list(manifest["parents"])
    if len(rows) != int(protocol["data"]["parent_count"]):
        raise ValueError("train20 parent count mismatch")
    ids = [str(row["molecule_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate train20 parent")
    for row in rows:
        for path_key, hash_key in (
            ("label_path", "label_sha256"),
            ("capacity_array", "capacity_array_sha256"),
            ("direction_path", "direction_sha256"),
        ):
            if _sha256(Path(row[path_key])) != str(row[hash_key]):
                raise ValueError(f"parent artifact hash drift: {row['molecule_id']} {path_key}")
        with np.load(row["direction_path"]) as directions:
            roles = np.asarray(directions["roles"]).astype(str)
            if int(np.sum(roles == "train")) != int(protocol["directions"]["train_count"]):
                raise ValueError("train direction count drift")
            if int(np.sum(roles == "heldout")) != int(
                protocol["directions"]["heldout_count"]
            ):
                raise ValueError("held direction count drift")
    parents = [_load_parent(row, device) for row in rows]
    provenance = {
        "selected_manifest": manifest_path.resolve().as_posix(),
        "selected_manifest_sha256": _sha256(manifest_path),
        "selected_parent_ids": ids,
        "opened_parent_count": len(ids),
        "unselected_parent_artifacts_opened": 0,
        "source_split_sha256": manifest["source_split_sha256"],
    }
    return parents, rows, provenance


def _build_model(protocol: dict[str, Any], device: torch.device) -> LocalAngularScalarResidual:
    arm = protocol["representation"]["angular_model"]
    return LocalAngularScalarResidual(
        hidden_size=int(arm["hidden_size"]),
        radial_size=int(arm["radial_size"]),
        angular_order=int(arm["angular_order"]),
        cutoff_bohr=float(arm["cutoff_bohr"]),
        seed=int(arm["seed"]),
        activation=str(arm["activation"]),
        radial_feature_scale=float(arm["radial_feature_scale"]),
        angular_feature_scale=float(arm["angular_feature_scale"]),
    ).to(device)


def _load_kernel_checkpoint(
    protocol: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(
        protocol["inputs"]["v9_checkpoint"], map_location="cpu", weights_only=False
    )
    expected = str(protocol["inputs"]["v9_protocol_sha256"])
    if str(payload.get("protocol_sha256")) != expected:
        raise ValueError("v9 checkpoint protocol drift")
    return tuple(
        payload[key].to(device=device, dtype=torch.float64)
        for key in (
            "environment_inverse_rms",
            "projection",
            "bias",
            "activation_scale",
        )
    )


def _expand_v9_coefficient_prior(
    protocol: dict[str, Any],
    parents: list[CapacityParent],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Align the fitted stable5 v9 scalar coefficients to the train20 union schema."""
    checkpoint_path = Path(protocol["inputs"]["v9_checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stable_ids = tuple(str(value) for value in payload.get("selected_parent_ids", ()))
    if not stable_ids:
        raise ValueError("v9 checkpoint does not identify its fitted parents")
    parent_by_id = {parent.molecule_id: parent for parent in parents}
    if any(molecule_id not in parent_by_id for molecule_id in stable_ids):
        raise ValueError("v9 fitted-parent set is not contained in train20")
    stable_parents = [parent_by_id[molecule_id] for molecule_id in stable_ids]
    model = _build_model(protocol, torch.device("cpu"))
    settings = protocol["representation"]["base_feature_definition"]
    _, stable_angular_count, stable_four_body_keys = _build_feature_plans(
        stable_parents, model, settings
    )
    _, train20_angular_count, train20_four_body_keys = _build_feature_plans(
        parents, model, settings
    )
    if stable_angular_count != train20_angular_count:
        raise ValueError("v9/train20 angular schema mismatch")
    source_coefficients = payload["coefficients"].to(dtype=torch.float64, device="cpu")
    random_width = int(payload["projection"].shape[1])
    random_count = int(model.network.element_count) * random_width
    source_base_count = stable_angular_count + len(stable_four_body_keys)
    target_base_count = train20_angular_count + len(train20_four_body_keys)
    if source_coefficients.numel() != source_base_count + random_count:
        raise ValueError("v9 coefficient/schema size mismatch")
    target = torch.zeros(target_base_count + random_count, dtype=torch.float64)
    target[:stable_angular_count] = source_coefficients[:stable_angular_count]
    target_lookup = {key: index for index, key in enumerate(train20_four_body_keys)}
    missing = [key for key in stable_four_body_keys if key not in target_lookup]
    if missing:
        raise ValueError("stable5 four-body key is absent from train20 union")
    for source_index, key in enumerate(stable_four_body_keys):
        target_index = target_lookup[key]
        target[train20_angular_count + target_index] = source_coefficients[
            stable_angular_count + source_index
        ]
    target[target_base_count:] = source_coefficients[source_base_count:]
    digest = hashlib.sha256(target.numpy().tobytes(order="C")).hexdigest()
    metadata = {
        "definition": (
            "stable5 full-Hessian-fitted v9 scalar coefficients aligned by explicit "
            "angular/four-body/random feature schema"
        ),
        "source_checkpoint": checkpoint_path.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_coefficient_count": int(source_coefficients.numel()),
        "target_coefficient_count": int(target.numel()),
        "stable_four_body_key_count": len(stable_four_body_keys),
        "train20_four_body_key_count": len(train20_four_body_keys),
        "new_train20_four_body_key_count": len(train20_four_body_keys)
        - len(stable_four_body_keys),
        "fitted_parent_ids": list(stable_ids),
        "unseen_train20_parent_ids": [
            parent.molecule_id
            for parent in parents
            if parent.molecule_id not in set(stable_ids)
        ],
        "aligned_coefficient_sha256": digest,
    }
    return target, metadata


def _combined_feature_function(
    model: LocalAngularScalarResidual,
    parent: CapacityParent,
    plan: Any,
    kernel_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Any:
    inverse_rms, projection, bias, activation_scale = kernel_tensors
    element_index = model._element_index(parent.atomic_numbers)

    def feature_function(flat_positions: torch.Tensor) -> torch.Tensor:
        positions = flat_positions.reshape(parent.positions.shape)
        atomic = model.network.atomic_features(
            positions, element_index, parent.topology
        )
        angular = _central_resolved_angular_features(
            atomic,
            element_index,
            element_count=model.network.element_count,
            central_element_indices=plan.central_element_indices,
        )
        four_body = plan.four_body_function(positions)
        random = _central_random_features(
            atomic[:, model.network.element_count :],
            element_index,
            inverse_rms=inverse_rms,
            projection=projection,
            bias=bias,
            activation_scale=activation_scale,
            central_element_indices=plan.central_element_indices,
        )
        return torch.cat((angular, four_body, random))

    return feature_function


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol(args.protocol)
    parents, rows, provenance = _load_train20(protocol, torch.device("cpu"))
    model = _build_model(protocol, torch.device("cpu"))
    plans, angular_count, four_body_keys = _build_feature_plans(
        parents, model, protocol["representation"]["base_feature_definition"]
    )
    checkpoint = torch.load(
        protocol["inputs"]["v9_checkpoint"], map_location="cpu", weights_only=False
    )
    random_width = int(checkpoint["projection"].shape[1])
    base_count = angular_count + len(four_body_keys)
    global_count = base_count + model.network.element_count * random_width
    parent_rows = []
    for parent, row in zip(parents, rows, strict=True):
        plan = plans[parent.molecule_id]
        random_columns = _random_feature_columns(
            base_feature_count=base_count,
            width=random_width,
            central_element_indices=plan.central_element_indices,
            device=torch.device("cpu"),
        )
        columns = torch.cat((plan.global_columns, random_columns))
        parent_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "direction_path": row["direction_path"],
                "direction_sha256": row["direction_sha256"],
                "local_feature_count": int(columns.numel()),
                "global_column_min": int(torch.min(columns)),
                "global_column_max": int(torch.max(columns)),
            }
        )
    output = {
        "definition": "train20 frozen representation/schema preflight; no model fit",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "angular_feature_count": angular_count,
        "global_four_body_feature_count": len(four_body_keys),
        "random_width_per_element": random_width,
        "global_feature_count": global_count,
        "parents": parent_rows,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def feature_jet(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(args.protocol)
    cpu_parents, rows, provenance = _load_train20(protocol, torch.device("cpu"))
    if not 0 <= args.parent_index < len(cpu_parents):
        raise ValueError("parent index out of range")
    cpu_model = _build_model(protocol, torch.device("cpu"))
    plans, angular_count, four_body_keys = _build_feature_plans(
        cpu_parents,
        cpu_model,
        protocol["representation"]["base_feature_definition"],
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = cpu_model.to(device)
    parent = _move_parent(cpu_parents[args.parent_index], device)
    plan = plans[parent.molecule_id]
    kernel_tensors = _load_kernel_checkpoint(protocol, device)
    random_width = int(kernel_tensors[1].shape[1])
    base_count = angular_count + len(four_body_keys)
    global_count = base_count + model.network.element_count * random_width
    feature_function = _combined_feature_function(model, parent, plan, kernel_tensors)
    chunk_size = int(protocol["feature_jet"]["feature_chunk_size"])
    if parent.natoms >= int(protocol["feature_jet"]["large_parent_min_natoms"]):
        chunk_size = int(protocol["feature_jet"]["large_parent_feature_chunk_size"])
    jet = tuple(
        value.detach().cpu()
        for value in _chunked_vector_jet(
            feature_function,
            parent.positions.detach().reshape(-1),
            feature_chunk_size=chunk_size,
        )
    )
    random_columns = _random_feature_columns(
        base_feature_count=base_count,
        width=random_width,
        central_element_indices=plan.central_element_indices,
        device=torch.device("cpu"),
    )
    global_columns = torch.cat((plan.global_columns.detach().cpu(), random_columns))
    if int(jet[0].numel()) != int(global_columns.numel()):
        raise AssertionError("local feature/global-column mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / f"{args.parent_index:02d}_{parent.molecule_id}.pt"
    temporary = artifact.with_suffix(".tmp")
    torch.save(
        {
            "protocol_sha256": _sha256(args.protocol),
            "parent_index": args.parent_index,
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "global_feature_count": global_count,
            "global_columns": global_columns,
            "features": jet[0],
            "jacobian": jet[1],
            "hessian": jet[2],
            "validation_accessed": False,
            "test100_accessed": False,
        },
        temporary,
    )
    temporary.replace(artifact)
    result = {
        "parent_index": args.parent_index,
        "molecule_id": parent.molecule_id,
        "natoms": parent.natoms,
        "feature_chunk_size": chunk_size,
        "local_feature_count": int(jet[0].numel()),
        "global_feature_count": global_count,
        "artifact": artifact.resolve().as_posix(),
        "artifact_sha256": _sha256(artifact),
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    summary = args.output_dir / f"{args.parent_index:02d}_{parent.molecule_id}.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _stage2_design_blocks(
    parent: CapacityParent,
    jet: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    directions: np.ndarray,
    roles: np.ndarray,
    training: dict[str, Any],
    *,
    fit_full_hessian: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features, jacobian, feature_hessian = jet
    coordinate_count = parent.positions.numel()
    target_energy = features.new_tensor(parent.pbe_energy - parent.source_energy)
    target_force = features.new_tensor(parent.pbe_force - parent.source_force).reshape(-1)
    delta_hessian = features.new_tensor(
        parent.pbe_hessian - parent.source_hessian_symmetric
    )
    pbe_hessian = features.new_tensor(parent.pbe_hessian)
    if fit_full_hessian:
        curvature_design = feature_hessian.permute(1, 2, 0).reshape(
            coordinate_count * coordinate_count, features.numel()
        )
        target_curvature = delta_hessian.reshape(-1)
        reference_curvature = pbe_hessian.reshape(-1)
        curvature_count = coordinate_count * coordinate_count
    else:
        train_directions = features.new_tensor(directions[roles == "train"])
        curvature_design = torch.einsum(
            "fij,dj->dif", feature_hessian, train_directions
        ).reshape(-1, features.numel())
        target_curvature = torch.einsum(
            "ij,dj->di", delta_hessian, train_directions
        ).reshape(-1)
        reference_curvature = torch.einsum(
            "ij,dj->di", pbe_hessian, train_directions
        ).reshape(-1)
        curvature_count = int(target_curvature.numel())
    absolute_weight = math.sqrt(1.0 - float(training["relative_loss_fraction"])) / (
        float(training["absolute_hvp_scale"]) * math.sqrt(curvature_count)
    )
    relative_weight = math.sqrt(float(training["relative_loss_fraction"])) / max(
        float(torch.linalg.vector_norm(reference_curvature)),
        float(training["hvp_reference_floor"]),
    )
    return (
        [
            features[None, :] / float(training["energy_scale"]),
            -jacobian.T / (
                float(training["force_scale"]) * math.sqrt(coordinate_count)
            ),
            absolute_weight * curvature_design,
            relative_weight * curvature_design,
        ],
        [
            target_energy.reshape(1) / float(training["energy_scale"]),
            target_force
            / (float(training["force_scale"]) * math.sqrt(coordinate_count)),
            absolute_weight * target_curvature,
            relative_weight * target_curvature,
        ],
    )


def _extend_training_direction_bank(
    positions: np.ndarray,
    directions: np.ndarray,
    roles: np.ndarray,
    kinds: np.ndarray,
    *,
    target_train_count: int | None,
    maximal_train_complement: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if (target_train_count is None) == (not maximal_train_complement):
        raise ValueError("select exactly one direction-extension target")
    external = _external_basis(positions)
    internal_dimension = int(positions.size - external.shape[1])
    heldout_count = int(np.sum(roles == "heldout"))
    original_train_count = int(np.sum(roles == "train"))
    maximum_train_count = internal_dimension - heldout_count
    target = maximum_train_count if maximal_train_complement else int(target_train_count)
    if not original_train_count <= target <= maximum_train_count:
        raise ValueError(
            f"invalid train-direction target {target}; allowed "
            f"[{original_train_count}, {maximum_train_count}]"
        )
    accepted = [np.asarray(vector, dtype=np.float64).copy() for vector in directions]
    new_directions: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    attempts = 0
    while original_train_count + len(new_directions) < target:
        projected = rng.normal(size=positions.size)
        for _ in range(2):
            projected -= external @ (external.T @ projected)
            for previous in accepted:
                projected -= np.dot(previous, projected) * previous
        norm = np.linalg.norm(projected)
        if norm <= 1e-9:
            attempts += 1
            if attempts > 20 * positions.size:
                raise RuntimeError("could not extend frozen internal direction bank")
            continue
        vector = projected / norm
        accepted.append(vector)
        new_directions.append(vector)
    if new_directions:
        extended_directions = np.concatenate(
            (directions, np.stack(new_directions)), axis=0
        )
        extended_roles = np.concatenate(
            (roles, np.full(len(new_directions), "train", dtype=roles.dtype))
        )
        extended_kinds = np.concatenate(
            (
                kinds,
                np.full(
                    len(new_directions),
                    "extension_random_internal",
                    dtype="U32",
                ),
            )
        )
    else:
        extended_directions, extended_roles, extended_kinds = directions, roles, kinds
    gram_error = float(
        np.max(
            np.abs(
                extended_directions @ extended_directions.T
                - np.eye(extended_directions.shape[0])
            )
        )
    )
    external_overlap = float(np.max(np.abs(extended_directions @ external)))
    diagnostics = {
        "internal_dimension": internal_dimension,
        "original_train_count": original_train_count,
        "heldout_count": heldout_count,
        "added_train_count": len(new_directions),
        "final_train_count": int(np.sum(extended_roles == "train")),
        "maximum_train_count": maximum_train_count,
        "orthonormality_max_abs": gram_error,
        "external_overlap_max_abs": external_overlap,
    }
    if gram_error > 1e-10 or external_overlap > 1e-10:
        raise FloatingPointError("extended direction bank lost internal orthonormality")
    return extended_directions, extended_roles, extended_kinds, diagnostics


def _streamed_normal_equation_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    column_floor: float,
    row_chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if design.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("streamed solver expects CPU design and target")
    if ridge <= 0.0 or row_chunk_size <= 0:
        raise ValueError("invalid streamed ridge settings")
    column_norm = torch.linalg.vector_norm(design, dim=0)
    active = column_norm > column_floor
    scale = column_norm[active]
    active_count = int(torch.sum(active))
    if active_count == 0:
        raise ValueError("no active Stage-2 design columns")
    scale_device = scale.to(device)
    gram = torch.zeros((active_count, active_count), dtype=torch.float64, device=device)
    right_hand_side = torch.zeros(active_count, dtype=torch.float64, device=device)
    target_norm_squared = float(torch.dot(target, target))
    for start in range(0, design.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, design.shape[0])
        matrix_chunk = design[start:stop, active].to(device) / scale_device[None, :]
        target_chunk = target[start:stop].to(device)
        gram.addmm_(matrix_chunk.T, matrix_chunk)
        right_hand_side.addmv_(matrix_chunk.T, target_chunk)
        del matrix_chunk, target_chunk
    gram.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(gram)
    if bool(torch.any(info != 0)):
        raise FloatingPointError("streamed ridge Gram matrix is not positive definite")
    normalized_solution = torch.cholesky_solve(
        right_hand_side[:, None], factor
    ).squeeze(1)
    normal_residual = gram @ normalized_solution - right_hand_side
    data_quadratic = (
        torch.dot(normalized_solution, gram @ normalized_solution)
        - ridge * torch.dot(normalized_solution, normalized_solution)
        - 2.0 * torch.dot(normalized_solution, right_hand_side)
        + target_norm_squared
    )
    coefficients = torch.zeros(design.shape[1], dtype=torch.float64)
    coefficients[active] = (normalized_solution / scale_device).cpu()
    diagnostics = {
        "solver": "streamed_column_normalized_ridge_gram_cholesky_float64",
        "active_feature_count": active_count,
        "design_row_count": int(design.shape[0]),
        "design_column_count": int(design.shape[1]),
        "ridge": ridge,
        "row_chunk_size": row_chunk_size,
        "converged": True,
        "cholesky_info_max": int(torch.max(info)),
        "final_relative_normal_residual": float(
            torch.linalg.vector_norm(normal_residual)
            / torch.linalg.vector_norm(right_hand_side).clamp_min(1e-30)
        ),
        "design_residual_relative": float(
            torch.sqrt(torch.clamp(data_quadratic, min=0.0))
            / math.sqrt(max(target_norm_squared, 1e-30))
        ),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "normalized_solution_norm": float(torch.linalg.vector_norm(normalized_solution)),
        "column_norm_min_active": float(torch.min(scale)),
        "column_norm_max": float(torch.max(column_norm)),
    }
    return coefficients, diagnostics


def _resolve_effective_ridge(
    protocol_ridge: float,
    ridge_override: float | None,
    diagnostic_override: bool,
) -> float:
    if ridge_override is not None and not diagnostic_override:
        raise ValueError(
            "--ridge may only override the frozen protocol with "
            "--diagnostic-ridge-override"
        )
    effective = protocol_ridge if ridge_override is None else ridge_override
    if not math.isfinite(effective) or effective <= 0.0:
        raise ValueError("effective ridge must be finite and positive")
    return effective


def _random_feature_subset_mask(
    *,
    global_feature_count: int,
    element_count: int,
    activation_scale: torch.Tensor,
    width_per_scale: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if activation_scale.ndim != 1 or activation_scale.numel() == 0:
        raise ValueError("invalid random-feature activation scales")
    _, counts = torch.unique_consecutive(activation_scale, return_counts=True)
    if not bool(torch.all(counts == counts[0])):
        raise ValueError("random-feature scales do not have equal frozen widths")
    full_width_per_scale = int(counts[0])
    if not 0 <= width_per_scale <= full_width_per_scale:
        raise ValueError("random-feature subset width is outside the frozen kernel")
    scale_count = int(counts.numel())
    random_width = int(activation_scale.numel())
    base_feature_count = global_feature_count - element_count * random_width
    if base_feature_count <= 0:
        raise ValueError("global feature schema is incompatible with the frozen kernel")
    mask = torch.zeros(global_feature_count, dtype=torch.bool)
    mask[:base_feature_count] = True
    for element in range(element_count):
        element_start = base_feature_count + element * random_width
        for scale in range(scale_count):
            start = element_start + scale * full_width_per_scale
            mask[start : start + width_per_scale] = True
    diagnostics = {
        "base_feature_count": base_feature_count,
        "element_count": element_count,
        "scale_count": scale_count,
        "full_width_per_scale": full_width_per_scale,
        "selected_width_per_scale": width_per_scale,
        "selected_global_feature_count": int(torch.sum(mask)),
        "full_global_feature_count": global_feature_count,
    }
    return mask, diagnostics


def _hashed_subspace_mapping(
    *,
    feature_count: int,
    subspace_dimension: int,
    repetitions: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if feature_count <= 0 or not 0 < subspace_dimension <= feature_count:
        raise ValueError("invalid hashed scalar subspace dimension")
    if repetitions <= 0:
        raise ValueError("hashed scalar subspace repetitions must be positive")
    rng = np.random.default_rng(seed)
    bucket_rows = []
    sign_rows = []
    for _ in range(repetitions):
        bucket_rows.append(
            rng.integers(0, subspace_dimension, size=feature_count, dtype=np.int64)
        )
        sign_rows.append(
            np.where(
                rng.integers(0, 2, size=feature_count, dtype=np.int8) == 0,
                -1.0,
                1.0,
            )
        )
    buckets_array = np.stack(bucket_rows)
    signs_array = np.stack(sign_rows).astype(np.float64, copy=False)
    digest = hashlib.sha256()
    digest.update(buckets_array.tobytes(order="C"))
    digest.update(signs_array.tobytes(order="C"))
    counts = np.bincount(
        buckets_array.reshape(-1), minlength=subspace_dimension
    )
    diagnostics = {
        "definition": "label-independent multi-hash CountSketch scalar coefficient subspace",
        "feature_count": feature_count,
        "subspace_dimension": subspace_dimension,
        "repetitions": repetitions,
        "seed": seed,
        "normalization": 1.0 / math.sqrt(repetitions),
        "mapping_sha256": digest.hexdigest(),
        "bucket_occupancy_min": int(np.min(counts)),
        "bucket_occupancy_median": float(np.median(counts)),
        "bucket_occupancy_max": int(np.max(counts)),
        "empty_bucket_count": int(np.sum(counts == 0)),
    }
    return (
        torch.from_numpy(buckets_array),
        torch.from_numpy(signs_array),
        diagnostics,
    )


def _project_design_to_hashed_subspace(
    design: torch.Tensor,
    buckets: torch.Tensor,
    signs: torch.Tensor,
    *,
    subspace_dimension: int,
    column_chunk_size: int,
) -> torch.Tensor:
    if design.device.type != "cpu" or design.dtype != torch.float64:
        raise ValueError("hashed scalar projection expects a float64 CPU design")
    if buckets.shape != signs.shape or buckets.shape[1] != design.shape[1]:
        raise ValueError("hashed scalar mapping shape mismatch")
    if column_chunk_size <= 0:
        raise ValueError("hashed scalar projection chunk must be positive")
    projected = torch.zeros(
        (design.shape[0], subspace_dimension), dtype=torch.float64
    )
    normalization = 1.0 / math.sqrt(buckets.shape[0])
    for repetition in range(buckets.shape[0]):
        for start in range(0, design.shape[1], column_chunk_size):
            stop = min(start + column_chunk_size, design.shape[1])
            source = design[:, start:stop] * (
                normalization * signs[repetition, start:stop]
            )[None, :]
            projected.index_add_(
                1, buckets[repetition, start:stop], source
            )
            del source
    return projected


def _expand_hashed_subspace_coefficients(
    subspace_coefficients: torch.Tensor,
    buckets: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    if buckets.shape != signs.shape:
        raise ValueError("hashed scalar mapping shape mismatch")
    normalization = 1.0 / math.sqrt(buckets.shape[0])
    coefficients = torch.zeros(buckets.shape[1], dtype=torch.float64)
    for repetition in range(buckets.shape[0]):
        coefficients.add_(
            normalization
            * signs[repetition]
            * subspace_coefficients[buckets[repetition]]
        )
    return coefficients


def fit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(args.protocol)
    parents, rows, provenance = _load_train20(protocol, torch.device("cpu"))
    protocol_sha256 = _sha256(args.protocol)
    direction_extension_requested = (
        args.target_train_direction_count is not None
        or args.maximal_train_complement
    )
    if direction_extension_requested != args.diagnostic_direction_extension:
        raise ValueError(
            "direction extension requires --diagnostic-direction-extension, and vice versa"
        )
    if direction_extension_requested != (args.direction_extension_seed is not None):
        raise ValueError("direction extension requires an explicit deterministic seed")
    if args.diagnostic_full_hessian_fit and direction_extension_requested:
        raise ValueError("full-Hessian capacity fit cannot use direction extension")
    hashed_subspace_requested = args.hashed_subspace_dimension is not None
    if hashed_subspace_requested != args.diagnostic_hashed_subspace:
        raise ValueError(
            "hashed subspace requires --diagnostic-hashed-subspace, and vice versa"
        )
    if hashed_subspace_requested != (
        args.hashed_subspace_seed is not None
        and args.hashed_subspace_repetitions is not None
    ):
        raise ValueError("hashed subspace requires an explicit seed and repetition count")
    if hashed_subspace_requested and (
        args.diagnostic_ridge_override
        or args.diagnostic_feature_subset
        or args.diagnostic_direction_extension
        or args.diagnostic_full_hessian_fit
    ):
        raise ValueError("hashed subspace diagnostic must be isolated from other arms")
    structured_subspace_requested = args.structured_subspace_config is not None
    if structured_subspace_requested != args.diagnostic_structured_subspace:
        raise ValueError(
            "structured subspace requires --diagnostic-structured-subspace and a config"
        )
    if structured_subspace_requested and (
        args.diagnostic_ridge_override
        or args.diagnostic_feature_subset
        or args.diagnostic_direction_extension
        or args.diagnostic_full_hessian_fit
        or args.diagnostic_hashed_subspace
    ):
        raise ValueError("structured subspace diagnostic must be isolated from other arms")
    if args.v9_prior_only and args.v9_prior_residual:
        raise ValueError("select at most one v9 coefficient-prior mode")
    v9_prior_requested = args.v9_prior_only or args.v9_prior_residual
    if v9_prior_requested and (
        args.diagnostic_ridge_override
        or args.diagnostic_feature_subset
        or args.diagnostic_direction_extension
        or args.diagnostic_full_hessian_fit
        or args.diagnostic_hashed_subspace
    ):
        raise ValueError("v9 coefficient-prior diagnostic must be isolated")
    if args.v9_prior_only and structured_subspace_requested:
        raise ValueError("prior-only evaluation cannot also fit a structured residual")
    designs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    jet_paths: dict[str, Path] = {}
    direction_banks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    direction_extension_metrics: list[dict[str, Any]] = []
    global_count = None
    for index, (parent, row) in enumerate(zip(parents, rows, strict=True)):
        matches = list(args.jet_dir.glob(f"{index:02d}_{parent.molecule_id}.pt"))
        if len(matches) != 1:
            raise ValueError(f"missing Stage-2 jet for {parent.molecule_id}")
        jet_path = matches[0]
        payload = torch.load(jet_path, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"jet protocol drift for {parent.molecule_id}")
        if payload.get("molecule_id") != parent.molecule_id:
            raise ValueError("jet parent drift")
        jet = tuple(payload[key] for key in ("features", "jacobian", "hessian"))
        if not all(bool(torch.isfinite(value).all()) for value in jet):
            raise FloatingPointError(f"non-finite Stage-2 jet: {parent.molecule_id}")
        columns = payload["global_columns"]
        count = int(payload["global_feature_count"])
        global_count = count if global_count is None else global_count
        if count != global_count:
            raise ValueError("global Stage-2 feature count drift")
        with np.load(row["direction_path"]) as direction_payload:
            directions = np.asarray(direction_payload["directions"], dtype=np.float64)
            roles = np.asarray(direction_payload["roles"]).astype(str)
            kinds = np.asarray(direction_payload["kinds"]).astype(str)
        if direction_extension_requested:
            directions, roles, kinds, extension = _extend_training_direction_bank(
                parent.positions.detach().cpu().numpy(),
                directions,
                roles,
                kinds,
                target_train_count=args.target_train_direction_count,
                maximal_train_complement=args.maximal_train_complement,
                seed=int(args.direction_extension_seed) + index,
            )
            direction_extension_metrics.append(
                {"molecule_id": parent.molecule_id, **extension}
            )
        direction_banks[parent.molecule_id] = (directions, roles, kinds)
        local_designs, local_targets = _stage2_design_blocks(
            parent,
            jet,
            directions,
            roles,
            protocol["training"],
            fit_full_hessian=args.diagnostic_full_hessian_fit,
        )
        for local_design in local_designs:
            designs.append(
                torch.zeros(
                    (local_design.shape[0], global_count), dtype=torch.float64
                ).index_copy(1, columns, local_design)
            )
        targets.extend(local_targets)
        jet_paths[parent.molecule_id] = jet_path
        del payload, jet, local_designs, local_targets
        gc.collect()
    design = torch.cat(designs, dim=0)
    target = torch.cat(targets, dim=0)
    designs.clear()
    targets.clear()
    gc.collect()
    v9_prior = None
    v9_prior_metadata = None
    v9_prior_residual_relative = None
    if v9_prior_requested:
        v9_prior, v9_prior_metadata = _expand_v9_coefficient_prior(
            protocol, parents
        )
        if v9_prior.numel() != design.shape[1]:
            raise ValueError("aligned v9 prior does not match the Stage-2 design")
        prior_residual = target - design @ v9_prior
        v9_prior_residual_relative = float(
            torch.linalg.vector_norm(prior_residual)
            / torch.linalg.vector_norm(target).clamp_min(1e-30)
        )
        if args.v9_prior_residual:
            target = prior_residual
        del prior_residual
        gc.collect()
    if (args.random_width_per_scale is None) != (
        not args.diagnostic_feature_subset
    ):
        raise ValueError(
            "--random-width-per-scale and --diagnostic-feature-subset must be used together"
        )
    feature_subset = None
    solve_columns = None
    if args.random_width_per_scale is not None:
        checkpoint = torch.load(
            protocol["inputs"]["v9_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        element_count = int(_build_model(protocol, torch.device("cpu")).network.element_count)
        solve_columns, feature_subset = _random_feature_subset_mask(
            global_feature_count=int(global_count),
            element_count=element_count,
            activation_scale=checkpoint["activation_scale"].to(dtype=torch.float64),
            width_per_scale=int(args.random_width_per_scale),
        )
        design = design[:, solve_columns].contiguous()
        del checkpoint
        gc.collect()
    hashed_subspace = None
    hashed_buckets = None
    hashed_signs = None
    if hashed_subspace_requested:
        hashed_buckets, hashed_signs, hashed_subspace = _hashed_subspace_mapping(
            feature_count=int(global_count),
            subspace_dimension=int(args.hashed_subspace_dimension),
            repetitions=int(args.hashed_subspace_repetitions),
            seed=int(args.hashed_subspace_seed),
        )
        projected_design = _project_design_to_hashed_subspace(
            design,
            hashed_buckets,
            hashed_signs,
            subspace_dimension=int(args.hashed_subspace_dimension),
            column_chunk_size=int(args.hashed_subspace_column_chunk_size),
        )
        del design
        design = projected_design
        gc.collect()
    structured_subspace = None
    structured_mapping = None
    structured_config = None
    structured_config_sha256 = None
    if structured_subspace_requested:
        structured_config_sha256 = _sha256(args.structured_subspace_config)
        structured_config = yaml.safe_load(args.structured_subspace_config.read_text())
        if structured_config.get("test100_access_allowed") is not False:
            raise ValueError("structured subspace config does not freeze Test100")
        if int(structured_config.get("test100_evaluations_used", -1)) != 0:
            raise ValueError("structured subspace config Test100 count is not zero")
        source = structured_config.get("source", {})
        if source.get("stage2_protocol_sha256") != protocol_sha256:
            raise ValueError("structured subspace source protocol hash drift")
        arm = structured_config["structured_subspace"]
        model = _build_model(protocol, torch.device("cpu"))
        _, angular_count, four_body_keys = _build_feature_plans(
            parents,
            model,
            protocol["representation"]["base_feature_definition"],
        )
        inverse_rms, projection, bias, activation_scale = _load_kernel_checkpoint(
            protocol, torch.device("cpu")
        )
        structured_mapping, structured_subspace = (
            build_structured_scalar_subspace_mapping(
                element_count=int(model.network.element_count),
                radial_size=int(model.network.radial_size),
                angular_order=int(model.network.angular_order),
                angular_feature_count=int(angular_count),
                four_body_keys=four_body_keys,
                four_body_center_count=int(
                    protocol["representation"]["base_feature_definition"][
                        "center_count"
                    ]
                ),
                angular_radial_modes=int(arm["angular_radial_modes"]),
                four_body_radial_modes=int(arm["four_body_radial_modes"]),
                random_environment_radial_modes=int(
                    arm["random_environment_radial_modes"]
                ),
                random_bias_order=int(arm["random_bias_order"]),
                inverse_rms=inverse_rms,
                projection=projection,
                bias=bias,
                activation_scale=activation_scale,
            )
        )
        if int(structured_mapping.shape[0]) != int(global_count):
            raise ValueError("structured mapping does not match the frozen global schema")
        structured_subspace.update(
            {
                "config": args.structured_subspace_config.resolve().as_posix(),
                "config_sha256": structured_config_sha256,
                "arm_id": structured_config.get("arm_id"),
            }
        )
        projected_design = project_design_to_structured_subspace(
            design,
            structured_mapping,
            row_chunk_size=int(arm["projection_row_chunk_size"]),
            device=torch.device(args.device),
        )
        del design
        design = projected_design
        gc.collect()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    protocol_ridge = float(protocol["solver"]["ridge"])
    effective_ridge = _resolve_effective_ridge(
        protocol_ridge,
        args.ridge,
        args.diagnostic_ridge_override,
    )
    if args.v9_prior_only:
        coefficients = v9_prior
        solver = {
            "solver": "no_fit_stable5_v9_coefficient_prior_only",
            "active_feature_count": 0,
            "design_row_count": int(design.shape[0]),
            "design_column_count": int(design.shape[1]),
            "ridge": None,
            "converged": True,
            "design_residual_relative": v9_prior_residual_relative,
            "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
            "normalized_solution_norm": 0.0,
        }
        solved_coefficients = None
    else:
        solved_coefficients, solver = _streamed_normal_equation_solve(
            design,
            target,
            ridge=effective_ridge,
            column_floor=float(protocol["solver"]["column_floor"]),
            row_chunk_size=int(protocol["solver"]["row_chunk_size"]),
            device=device,
        )
    if args.v9_prior_only:
        pass
    elif structured_subspace_requested:
        coefficients = expand_structured_subspace_coefficients(
            solved_coefficients, structured_mapping
        )
        solver["full_design_column_count"] = int(global_count)
        solver["structured_subspace"] = structured_subspace
        solver["expanded_coefficient_norm"] = float(
            torch.linalg.vector_norm(coefficients)
        )
    elif hashed_subspace_requested:
        coefficients = _expand_hashed_subspace_coefficients(
            solved_coefficients,
            hashed_buckets,
            hashed_signs,
        )
        solver["full_design_column_count"] = int(global_count)
        solver["hashed_subspace"] = hashed_subspace
        solver["expanded_coefficient_norm"] = float(
            torch.linalg.vector_norm(coefficients)
        )
    elif solve_columns is None:
        coefficients = solved_coefficients
    else:
        coefficients = torch.zeros(int(global_count), dtype=torch.float64)
        coefficients[solve_columns] = solved_coefficients
        solver["full_design_column_count"] = int(global_count)
        solver["feature_subset"] = feature_subset
    if args.v9_prior_residual:
        solver["residual_coefficient_norm"] = float(
            torch.linalg.vector_norm(coefficients)
        )
        coefficients = coefficients + v9_prior
        solver["total_coefficient_norm"] = float(
            torch.linalg.vector_norm(coefficients)
        )
        solver["v9_prior_design_residual_relative"] = v9_prior_residual_relative
    del solved_coefficients
    del design, target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = []
    direction_metrics = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for parent, row in zip(parents, rows, strict=True):
        payload = torch.load(
            jet_paths[parent.molecule_id], map_location="cpu", weights_only=False
        )
        features, jacobian, feature_hessian = (
            payload[key] for key in ("features", "jacobian", "hessian")
        )
        local_coefficients = coefficients[payload["global_columns"]]
        correction_energy = torch.dot(features, local_coefficients)
        correction_force = -(jacobian.T @ local_coefficients).reshape(
            parent.positions.shape
        )
        correction_hessian = torch.einsum(
            "fij,f->ij", feature_hessian, local_coefficients
        )
        predicted_energy = parent.source_energy + float(correction_energy)
        predicted_force = parent.source_force + correction_force.numpy()
        predicted_hessian = parent.source_hessian_symmetric + correction_hessian.numpy()
        difference = predicted_hessian - parent.pbe_hessian
        directions, roles, kinds = direction_banks[parent.molecule_id]
        local_direction = {}
        for role in ("train", "heldout"):
            selected = directions[roles == role]
            error_hvp = np.einsum("ij,dj->di", difference, selected)
            reference_hvp = np.einsum("ij,dj->di", parent.pbe_hessian, selected)
            local_direction[role] = float(
                np.linalg.norm(error_hvp)
                / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
            )
        for direction_index, (direction, role, kind) in enumerate(
            zip(directions, roles, kinds, strict=True)
        ):
            error = difference @ direction
            reference = parent.pbe_hessian @ direction
            direction_metrics.append(
                {
                    "molecule_id": parent.molecule_id,
                    "direction_index": direction_index,
                    "role": role,
                    "kind": kind,
                    "hvp_mae": float(np.mean(np.abs(error))),
                    "hvp_rmse": float(np.sqrt(np.mean(error * error))),
                    "hvp_relative_l2": float(
                        np.linalg.norm(error)
                        / max(np.linalg.norm(reference), np.finfo(float).tiny)
                    ),
                }
            )
        metric = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "v9_prior_fitted_parent": bool(
                v9_prior_metadata is not None
                and parent.molecule_id
                in set(v9_prior_metadata["fitted_parent_ids"])
            ),
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
            "train_hvp_relative_frobenius": local_direction["train"],
            "heldout_hvp_relative_frobenius": local_direction["heldout"],
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        metrics.append(metric)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_force=predicted_force,
            predicted_hessian=predicted_hessian,
            pbe_hessian=parent.pbe_hessian,
            source_hessian_symmetric=parent.source_hessian_symmetric,
        )
        del payload, features, jacobian, feature_hessian
        gc.collect()

    hessian_distribution = _distribution(metrics, "relative_frobenius")
    train_distribution = _distribution(metrics, "train_hvp_relative_frobenius")
    heldout_distribution = _distribution(metrics, "heldout_hvp_relative_frobenius")
    energy_distribution = _distribution(metrics, "energy_abs_error_hartree")
    source_energy_distribution = _distribution(metrics, "source_energy_abs_error_hartree")
    force_distribution = _distribution(metrics, "force_mae_hartree_per_bohr")
    source_force_distribution = _distribution(metrics, "source_force_mae_hartree_per_bohr")
    v9_prior_stratified = None
    if v9_prior_requested:
        v9_prior_stratified = {}
        for group_name, group_value in (
            ("prior_fitted5", True),
            ("prior_unseen15", False),
        ):
            selected_metrics = [
                row
                for row in metrics
                if bool(row["v9_prior_fitted_parent"]) is group_value
            ]
            if not selected_metrics:
                raise ValueError(f"empty v9 prior stratum: {group_name}")
            v9_prior_stratified[group_name] = {
                "parent_count": len(selected_metrics),
                "parent_ids": [row["molecule_id"] for row in selected_metrics],
                "train_hvp_relative_frobenius": _distribution(
                    selected_metrics, "train_hvp_relative_frobenius"
                ),
                "heldout_hvp_relative_frobenius": _distribution(
                    selected_metrics, "heldout_hvp_relative_frobenius"
                ),
                "hessian_relative_frobenius": _distribution(
                    selected_metrics, "relative_frobenius"
                ),
                "energy_abs_error_hartree": _distribution(
                    selected_metrics, "energy_abs_error_hartree"
                ),
                "force_mae_hartree_per_bohr": _distribution(
                    selected_metrics, "force_mae_hartree_per_bohr"
                ),
            }
    energy_ratio = energy_distribution["median"] / max(
        source_energy_distribution["median"], np.finfo(float).tiny
    )
    force_ratio = force_distribution["median"] / max(
        source_force_distribution["median"], np.finfo(float).tiny
    )
    full_values = np.asarray([row["relative_frobenius"] for row in metrics])
    gate = protocol["stage2_gates"]
    checks = {
        "train_hvp_all_parent": train_distribution["max"]
        <= float(gate["train_hvp_all_parent_max"]),
        "heldout_hvp_median": heldout_distribution["median"]
        <= float(gate["heldout_hvp_median_max"]),
        "heldout_hvp_p90": heldout_distribution["p90"]
        <= float(gate["heldout_hvp_p90_max"]),
        "full_hessian_median": hessian_distribution["median"]
        <= float(gate["full_hessian_median_max"]),
        "full_hessian_p90": hessian_distribution["p90"]
        <= float(gate["full_hessian_p90_max"]),
        "full_hessian_fraction_below_0p15": float(np.mean(full_values <= 0.15))
        >= float(gate["full_hessian_fraction_below_0p15_min"]),
        "energy_median_ratio": energy_ratio <= float(gate["energy_median_ratio_max"]),
        "force_median_ratio": force_ratio <= float(gate["force_median_ratio_max"]),
        "asymmetry": max(
            float(row["antisymmetric_over_symmetric_frobenius"]) for row in metrics
        )
        <= float(gate["antisymmetric_over_symmetric_frobenius_max"]),
    }
    torch.save(
        {
            "coefficients": coefficients,
            "protocol": args.protocol.resolve().as_posix(),
            "protocol_sha256": protocol_sha256,
            "global_feature_count": global_count,
            "protocol_ridge": protocol_ridge,
            "effective_ridge": effective_ridge,
            "diagnostic_ridge_override": bool(args.diagnostic_ridge_override),
            "diagnostic_feature_subset": bool(args.diagnostic_feature_subset),
            "feature_subset": feature_subset,
            "diagnostic_direction_extension": bool(
                args.diagnostic_direction_extension
            ),
            "direction_extension": direction_extension_metrics,
            "diagnostic_full_hessian_fit": bool(args.diagnostic_full_hessian_fit),
            "diagnostic_hashed_subspace": bool(args.diagnostic_hashed_subspace),
            "hashed_subspace": hashed_subspace,
            "hashed_subspace_buckets": hashed_buckets,
            "hashed_subspace_signs": hashed_signs,
            "diagnostic_structured_subspace": bool(
                args.diagnostic_structured_subspace
            ),
            "structured_subspace": structured_subspace,
            "structured_subspace_mapping": structured_mapping,
            "structured_subspace_config": (
                args.structured_subspace_config.resolve().as_posix()
                if args.structured_subspace_config is not None
                else None
            ),
            "structured_subspace_config_sha256": structured_config_sha256,
            "v9_prior_only": bool(args.v9_prior_only),
            "v9_prior_residual": bool(args.v9_prior_residual),
            "v9_coefficient_prior": v9_prior_metadata,
            "v9_prior_design_residual_relative": v9_prior_residual_relative,
            "promotion_allowed": not (
                bool(args.diagnostic_ridge_override)
                or bool(args.diagnostic_feature_subset)
                or bool(args.diagnostic_direction_extension)
                or bool(args.diagnostic_full_hessian_fit)
                or bool(args.diagnostic_hashed_subspace)
                or bool(args.diagnostic_structured_subspace)
                or bool(v9_prior_requested)
            ),
            "test100_accessed": False,
            **provenance,
        },
        args.output_dir / "stage2_direction_fit.pt",
    )
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output_dir / "per_direction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_metrics[0]))
        writer.writeheader()
        writer.writerows(direction_metrics)
    result = {
        "definition": (
            "Diagnostic full-Hessian fit-only capacity of frozen v9 local-scalar features "
            "on all 20 train parents; no direction or matrix is held out."
            if args.diagnostic_full_hessian_fit
            else (
                "Diagnostic stable5 full-Hessian v9 scalar-coefficient prior on train20; "
                "the five fitted parents are explicitly marked and only the other fifteen "
                "are unseen-parent evidence."
                if args.v9_prior_only
                else (
                "Diagnostic train20 residual fit around the stable5 full-Hessian v9 scalar "
                "coefficient prior; prior-fitted and unseen parents remain stratified."
                if args.v9_prior_residual
                else (
                "Diagnostic chemistry-aware structured scalar-subspace fit to train20 E/F "
                "and registered train HVP directions; held directions and full Hessians "
                "remain evaluation-only."
                if args.diagnostic_structured_subspace
                else (
                "Diagnostic label-independent hashed scalar-subspace fit to train20 E/F and "
                "registered train HVP directions; held directions and full Hessians remain "
                "evaluation-only."
                if args.diagnostic_hashed_subspace
                else (
                    "Exact linear-output fit of frozen v9 local-scalar features to train20 "
                    "E/F and registered training HVP directions; seven held directions and "
                    "full Hessians are evaluation-only."
                )
                )
                )
                )
            )
        ),
        "fit_label_scope": (
            "energy_force_full_hessian"
            if args.diagnostic_full_hessian_fit
            else "energy_force_train_hvp_only"
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "direction_manifest": provenance["selected_manifest"],
        "direction_manifest_sha256": provenance["selected_manifest_sha256"],
        "global_feature_count": global_count,
        "protocol_ridge": protocol_ridge,
        "effective_ridge": effective_ridge,
        "diagnostic_ridge_override": bool(args.diagnostic_ridge_override),
        "diagnostic_feature_subset": bool(args.diagnostic_feature_subset),
        "feature_subset": feature_subset,
        "diagnostic_direction_extension": bool(args.diagnostic_direction_extension),
        "direction_extension": direction_extension_metrics,
        "diagnostic_full_hessian_fit": bool(args.diagnostic_full_hessian_fit),
        "diagnostic_hashed_subspace": bool(args.diagnostic_hashed_subspace),
        "hashed_subspace": hashed_subspace,
        "diagnostic_structured_subspace": bool(
            args.diagnostic_structured_subspace
        ),
        "structured_subspace": structured_subspace,
        "v9_prior_only": bool(args.v9_prior_only),
        "v9_prior_residual": bool(args.v9_prior_residual),
        "v9_coefficient_prior": v9_prior_metadata,
        "v9_prior_design_residual_relative": v9_prior_residual_relative,
        "v9_prior_stratified": v9_prior_stratified,
        "promotion_allowed": not (
            bool(args.diagnostic_ridge_override)
            or bool(args.diagnostic_feature_subset)
            or bool(args.diagnostic_direction_extension)
            or bool(args.diagnostic_full_hessian_fit)
            or bool(args.diagnostic_hashed_subspace)
            or bool(args.diagnostic_structured_subspace)
            or bool(v9_prior_requested)
        ),
        "solver": solver,
        "train_hvp_relative_frobenius": train_distribution,
        "heldout_hvp_relative_frobenius": heldout_distribution,
        "hessian_relative_frobenius": hessian_distribution,
        "fraction_full_hessian_below_0p10": float(np.mean(full_values <= 0.10)),
        "fraction_full_hessian_below_0p15": float(np.mean(full_values <= 0.15)),
        "fraction_full_hessian_below_0p20": float(np.mean(full_values <= 0.20)),
        "energy_abs_error_hartree": energy_distribution,
        "force_mae_hartree_per_bohr": force_distribution,
        "energy_median_ratio_to_source": energy_ratio,
        "force_median_ratio_to_source": force_ratio,
        "stage2_direction_generalization_gate": {**checks, "passed": all(checks.values())},
        "stage2_direction_generalization_gate_passed": all(checks.values()),
        "per_parent": metrics,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "feature-jet", "fit"):
        child = subparsers.add_parser(command)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--device", default="cuda:0")
        if command == "feature-jet":
            child.add_argument("--parent-index", type=int, required=True)
        if command == "fit":
            child.add_argument("--jet-dir", type=Path, required=True)
            child.add_argument("--ridge", type=float)
            child.add_argument("--diagnostic-ridge-override", action="store_true")
            child.add_argument("--random-width-per-scale", type=int)
            child.add_argument("--diagnostic-feature-subset", action="store_true")
            direction_target = child.add_mutually_exclusive_group()
            direction_target.add_argument("--target-train-direction-count", type=int)
            direction_target.add_argument(
                "--maximal-train-complement", action="store_true"
            )
            child.add_argument("--direction-extension-seed", type=int)
            child.add_argument("--diagnostic-direction-extension", action="store_true")
            child.add_argument("--diagnostic-full-hessian-fit", action="store_true")
            child.add_argument("--hashed-subspace-dimension", type=int)
            child.add_argument("--hashed-subspace-repetitions", type=int)
            child.add_argument("--hashed-subspace-seed", type=int)
            child.add_argument(
                "--hashed-subspace-column-chunk-size", type=int, default=128
            )
            child.add_argument("--diagnostic-hashed-subspace", action="store_true")
            child.add_argument("--structured-subspace-config", type=Path)
            child.add_argument(
                "--diagnostic-structured-subspace", action="store_true"
            )
            prior = child.add_mutually_exclusive_group()
            prior.add_argument("--v9-prior-only", action="store_true")
            prior.add_argument("--v9-prior-residual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "feature-jet": feature_jet, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
