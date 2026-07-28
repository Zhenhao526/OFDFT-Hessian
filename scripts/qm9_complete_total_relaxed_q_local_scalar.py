#!/usr/bin/env python3
"""Fit a scalar local model to frozen relaxed-q train/held directions."""

from __future__ import annotations

import argparse
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

from mldft.ml.models.components.local_body_order_residual import build_body_order_topology
from scripts.qm9_complete_total_local_angular_four_body_linear_ceiling import (
    _build_feature_plans,
)
from scripts.qm9_complete_total_local_angular_linear_ceiling import _chunked_vector_jet
from scripts.qm9_complete_total_local_random_feature_kernel_ceiling import (
    _random_feature_columns,
)
from scripts.qm9_complete_total_local_random_feature_stage2 import (
    CapacityParent,
    _build_model,
    _combined_feature_function,
    _load_kernel_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("stage") != "train59_relaxed_q_unseen_direction_local_scalar_diagnostic":
        raise ValueError("unexpected relaxed-q local-scalar stage")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    for key in ("direction_split_manifest", "v9_checkpoint"):
        source = Path(protocol["inputs"][key])
        if _sha256(source) != str(protocol["inputs"][f"{key}_sha256"]):
            raise ValueError(f"input hash drift: {key}")
    return protocol


def _load_manifest(protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(protocol["inputs"]["direction_split_manifest"])
    manifest = json.loads(path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("direction split does not freeze Test100")
    if manifest.get("validation_accessed") is not False:
        raise ValueError("direction split opened validation parents")
    if manifest.get("formal_stage3_authorized") is not False:
        raise ValueError("diagnostic split unexpectedly authorizes Stage 3")
    if manifest.get("protocol_sha256") != protocol["inputs"]["direction_split_protocol_sha256"]:
        raise ValueError("direction split protocol drift")
    checks = {
        "parent_count": int(protocol["data"]["parent_count"]),
        "train_direction_count": int(protocol["data"]["train_direction_count"]),
        "heldout_direction_count": int(protocol["data"]["heldout_direction_count"]),
    }
    for key, expected in checks.items():
        if int(manifest[key]) != expected:
            raise ValueError(f"direction split {key} drift")
    rows = list(manifest["parents"])
    if len(rows) != checks["parent_count"]:
        raise ValueError("direction split parent rows drift")
    for row in rows:
        for key in ("sidecar", "baseline_array", "pbe_hessian"):
            source = Path(row[key])
            if _sha256(source) != str(row[f"{key}_sha256"]):
                raise ValueError(f"parent artifact hash drift: {row['molecule_id']} {key}")
    return manifest, rows


def _load_parent(row: dict[str, Any], device: torch.device) -> CapacityParent:
    with np.load(row["sidecar"]) as sidecar, np.load(row["baseline_array"]) as base, np.load(
        row["pbe_hessian"]
    ) as hessian:
        atomic_numbers = np.asarray(sidecar["atomic_numbers"], dtype=np.int64)
        positions = np.asarray(sidecar["positions_bohr"], dtype=np.float64)
        if not np.array_equal(atomic_numbers, np.asarray(base["atomic_numbers"])):
            raise ValueError(f"atomic-number drift: {row['molecule_id']}")
        if not np.allclose(positions, np.asarray(base["positions_bohr"]), atol=1e-12, rtol=0.0):
            raise ValueError(f"position drift: {row['molecule_id']}")
        pbe_hessian = np.asarray(hessian["pbe_hessian"], dtype=np.float64)
        pbe_energy = float(base["pbe_total_energy"])
        pbe_force = np.asarray(base["pbe_force"], dtype=np.float64)
        source_energy = float(base["baseline_total_energy"])
        source_force = np.asarray(base["baseline_total_force"], dtype=np.float64)
    atomic = torch.as_tensor(atomic_numbers, dtype=torch.long, device=device)
    positions_tensor = torch.as_tensor(positions, dtype=torch.float64, device=device)
    zero_hessian = np.zeros_like(pbe_hessian)
    return CapacityParent(
        molecule_id=str(row["molecule_id"]),
        natoms=int(atomic_numbers.size),
        atomic_numbers=atomic,
        positions=positions_tensor,
        topology=build_body_order_topology(
            torch.as_tensor(atomic_numbers, dtype=torch.long),
            torch.as_tensor(positions, dtype=torch.float64),
            bond_scale=1.25,
        ),
        pbe_energy=pbe_energy,
        pbe_force=pbe_force,
        pbe_hessian=pbe_hessian,
        source_energy=source_energy,
        source_force=source_force,
        source_hessian_raw=zero_hessian,
        source_hessian_symmetric=zero_hessian,
    )


def _load_q(row: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(row["sidecar"]) as payload:
        indices = np.asarray(payload["direction_index"], dtype=np.int64)
        lookup = {int(value): index for index, value in enumerate(indices)}
        role_lookup = {
            int(item["direction_index"]): str(item["role"])
            for item in row["directions"]
        }
        selected = np.asarray(sorted(role_lookup), dtype=np.int64)
        local = np.asarray([lookup[int(value)] for value in selected], dtype=np.int64)
        return {
            "direction_index": selected,
            "direction": np.asarray(payload["direction"], dtype=np.float64)[local],
            "kind": np.asarray(payload["direction_kind"]).astype(str)[local],
            "baseline_q": np.asarray(payload["baseline_q_hartree_per_bohr2"], dtype=np.float64)[local],
            "pbe_q": np.asarray(payload["pbe_q_hartree_per_bohr2"], dtype=np.float64)[local],
            "role": np.asarray([role_lookup[int(value)] for value in selected]),
        }


def _parents_and_schema(protocol: dict[str, Any]) -> tuple[list[CapacityParent], list[dict[str, Any]], Any, dict[str, Any], int, int]:
    manifest, rows = _load_manifest(protocol)
    parents = [_load_parent(row, torch.device("cpu")) for row in rows]
    model = _build_model(protocol, torch.device("cpu"))
    plans, angular_count, four_body_keys = _build_feature_plans(
        parents, model, protocol["representation"]["base_feature_definition"]
    )
    checkpoint = torch.load(
        protocol["inputs"]["v9_checkpoint"], map_location="cpu", weights_only=False
    )
    random_width = int(checkpoint["projection"].shape[1])
    global_count = angular_count + len(four_body_keys) + model.network.element_count * random_width
    return parents, rows, model, plans, angular_count + len(four_body_keys), global_count


def preflight(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    parents, rows, model, plans, base_count, global_count = _parents_and_schema(protocol)
    checkpoint = torch.load(protocol["inputs"]["v9_checkpoint"], map_location="cpu", weights_only=False)
    random_width = int(checkpoint["projection"].shape[1])
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
        q = _load_q(row)
        parent_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "local_feature_count": int(columns.numel()),
                "train_direction_count": int(np.sum(q["role"] == "train")),
                "heldout_direction_count": int(np.sum(q["role"] == "heldout")),
            }
        )
    result = {
        "definition": protocol["definitions"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "global_feature_count": global_count,
        "base_feature_count": base_count,
        "random_width_per_element": random_width,
        "parents": parent_rows,
        "train800_full_replay_included": False,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "preflight.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _resolve_feature_chunk_size(
    protocol: dict[str, Any], natoms: int, override: int | None
) -> tuple[int, bool]:
    chunk_size = int(protocol["feature_jet"]["feature_chunk_size"])
    if natoms >= int(protocol["feature_jet"]["large_parent_min_natoms"]):
        chunk_size = int(protocol["feature_jet"]["large_parent_feature_chunk_size"])
    if override is None:
        return chunk_size, False
    if override <= 0 or override > chunk_size:
        raise ValueError("feature chunk override must be positive and no larger than default")
    return int(override), True


def feature_jet(
    protocol_path: Path,
    output_dir: Path,
    parent_index: int,
    device_name: str,
    feature_chunk_size: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(protocol_path)
    parents, rows, cpu_model, plans, base_count, global_count = _parents_and_schema(protocol)
    if not 0 <= parent_index < len(parents):
        raise ValueError("parent index out of range")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parent = replace(
        parents[parent_index],
        atomic_numbers=parents[parent_index].atomic_numbers.to(device),
        positions=parents[parent_index].positions.to(device),
    )
    model = cpu_model.to(device)
    plan = plans[parent.molecule_id]
    kernel = _load_kernel_checkpoint(protocol, device)
    random_width = int(kernel[1].shape[1])
    function = _combined_feature_function(model, parent, plan, kernel)
    chunk_size, chunk_override = _resolve_feature_chunk_size(
        protocol, parent.natoms, feature_chunk_size
    )
    jet = tuple(
        value.detach().cpu()
        for value in _chunked_vector_jet(
            function,
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
    columns = torch.cat((plan.global_columns.detach().cpu(), random_columns))
    if jet[0].numel() != columns.numel():
        raise AssertionError("feature/global-column mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{parent_index:02d}_{parent.molecule_id}.pt"
    temporary = artifact.with_suffix(".tmp")
    torch.save(
        {
            "protocol_sha256": _sha256(protocol_path),
            "parent_index": parent_index,
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "global_feature_count": global_count,
            "feature_chunk_size": chunk_size,
            "feature_chunk_size_override": chunk_override,
            "global_columns": columns,
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
        "parent_index": parent_index,
        "molecule_id": parent.molecule_id,
        "natoms": parent.natoms,
        "feature_chunk_size": chunk_size,
        "feature_chunk_size_override": chunk_override,
        "local_feature_count": int(jet[0].numel()),
        "global_feature_count": global_count,
        "artifact": artifact.resolve().as_posix(),
        "artifact_sha256": _sha256(artifact),
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    summary = output_dir / f"{parent_index:02d}_{parent.molecule_id}.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _q_relative(error: np.ndarray, reference: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(error) / np.maximum(np.abs(reference), floor)


def _dual_normalized_ridge_grid(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridges: list[float],
    column_floor: float,
    device: torch.device,
) -> dict[float, tuple[torch.Tensor, dict[str, Any]]]:
    if design.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("dual ridge expects CPU source tensors")
    if design.shape[0] > design.shape[1]:
        raise ValueError("dual ridge is reserved for row-limited designs")
    column_norm = torch.linalg.vector_norm(design, dim=0)
    active = column_norm > column_floor
    scale = column_norm[active]
    if scale.numel() == 0:
        raise ValueError("no active relaxed-q design columns")
    normalized = design[:, active].to(device) / scale.to(device)[None, :]
    target_device = target.to(device)
    gram = normalized @ normalized.T
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=device)
    solutions: dict[float, tuple[torch.Tensor, dict[str, Any]]] = {}
    for raw_ridge in ridges:
        ridge = float(raw_ridge)
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        factor, info = torch.linalg.cholesky_ex(gram + ridge * identity)
        if bool(torch.any(info != 0)):
            raise FloatingPointError(f"dual ridge Cholesky failed: {ridge}")
        dual = torch.cholesky_solve(target_device[:, None], factor).squeeze(1)
        normalized_solution = normalized.T @ dual
        coefficients = torch.zeros(design.shape[1], dtype=torch.float64)
        coefficients[active] = (normalized_solution / scale.to(device)).cpu()
        residual = design @ coefficients - target
        solutions[ridge] = (
            coefficients,
            {
                "solver": "column_normalized_float64_dual_gram_cholesky",
                "ridge": ridge,
                "design_row_count": int(design.shape[0]),
                "design_column_count": int(design.shape[1]),
                "active_feature_count": int(scale.numel()),
                "dual_gram_dimension": int(gram.shape[0]),
                "cholesky_info_max": int(torch.max(info)),
                "design_residual_relative": float(
                    torch.linalg.vector_norm(residual)
                    / torch.linalg.vector_norm(target).clamp_min(1e-30)
                ),
                "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
            },
        )
    return solutions


def _design_blocks(parent: CapacityParent, q: dict[str, np.ndarray], jet: tuple[torch.Tensor, ...], training: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features, jacobian, feature_hessian = jet
    coordinate_count = int(parent.positions.numel())
    train = q["role"] == "train"
    directions = features.new_tensor(q["direction"][train].reshape(np.sum(train), -1))
    q_design = torch.einsum("fij,di,dj->df", feature_hessian, directions, directions)
    q_target = features.new_tensor(q["pbe_q"][train] - q["baseline_q"][train])
    q_reference = features.new_tensor(q["pbe_q"][train])
    count = int(q_target.numel())
    relative_fraction = float(training["q_relative_loss_fraction"])
    absolute_weight = math.sqrt(1.0 - relative_fraction) / (
        float(training["q_absolute_scale_hartree_per_bohr2"]) * math.sqrt(count)
    )
    relative_weight = math.sqrt(relative_fraction) / math.sqrt(count) / torch.maximum(
        torch.abs(q_reference),
        q_reference.new_full(q_reference.shape, float(training["q_reference_floor_hartree_per_bohr2"])),
    )
    energy_target = features.new_tensor(parent.pbe_energy - parent.source_energy)
    force_target = features.new_tensor(parent.pbe_force - parent.source_force).reshape(-1)
    return (
        [
            features[None, :] / float(training["energy_scale_hartree"]),
            -jacobian.T / (float(training["force_scale_hartree_per_bohr"]) * math.sqrt(coordinate_count)),
            absolute_weight * q_design,
            relative_weight[:, None] * q_design,
        ],
        [
            energy_target.reshape(1) / float(training["energy_scale_hartree"]),
            force_target / (float(training["force_scale_hartree_per_bohr"]) * math.sqrt(coordinate_count)),
            absolute_weight * q_target,
            relative_weight * q_target,
        ],
    )


def _evaluate(coefficients: torch.Tensor, parents: list[CapacityParent], rows: list[dict[str, Any]], jets: dict[str, tuple[torch.Tensor, ...]], columns: dict[str, torch.Tensor], floor: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    parent_metrics = []
    direction_metrics = []
    for parent, row in zip(parents, rows, strict=True):
        features, jacobian, feature_hessian = jets[parent.molecule_id]
        local = coefficients[columns[parent.molecule_id]]
        correction_energy = float(torch.dot(features, local))
        correction_force = -(jacobian.T @ local).numpy().reshape(parent.pbe_force.shape)
        correction_hessian = torch.einsum("fij,f->ij", feature_hessian, local).numpy()
        predicted_energy = parent.source_energy + correction_energy
        predicted_force = parent.source_force + correction_force
        q = _load_q(row)
        flat_directions = q["direction"].reshape(q["direction"].shape[0], -1)
        correction_q = np.einsum("di,ij,dj->d", flat_directions, correction_hessian, flat_directions)
        predicted_q = q["baseline_q"] + correction_q
        errors = predicted_q - q["pbe_q"]
        relative = _q_relative(errors, q["pbe_q"], floor)
        for index in range(q["direction_index"].size):
            direction_metrics.append(
                {
                    "molecule_id": parent.molecule_id,
                    "direction_index": int(q["direction_index"][index]),
                    "direction_kind": str(q["kind"][index]),
                    "role": str(q["role"][index]),
                    "baseline_q": float(q["baseline_q"][index]),
                    "pbe_q": float(q["pbe_q"][index]),
                    "predicted_q": float(predicted_q[index]),
                    "absolute_error": float(abs(errors[index])),
                    "relative_error_with_floor": float(relative[index]),
                }
            )
        symmetric = 0.5 * (correction_hessian + correction_hessian.T)
        antisymmetric = 0.5 * (correction_hessian - correction_hessian.T)
        parent_metrics.append(
            {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
                "source_energy_abs_error_hartree": abs(parent.source_energy - parent.pbe_energy),
                "force_mae_hartree_per_bohr": float(np.mean(np.abs(predicted_force - parent.pbe_force))),
                "source_force_mae_hartree_per_bohr": float(np.mean(np.abs(parent.source_force - parent.pbe_force))),
                "train_q_median_relative_error": float(np.median(relative[q["role"] == "train"])),
                "heldout_q_relative_error": float(relative[q["role"] == "heldout"][0]),
                "correction_hessian_asym_over_sym": float(np.linalg.norm(antisymmetric) / max(np.linalg.norm(symmetric), np.finfo(float).tiny)),
            }
        )
    train = [row["relative_error_with_floor"] for row in direction_metrics if row["role"] == "train"]
    heldout = [row["relative_error_with_floor"] for row in direction_metrics if row["role"] == "heldout"]
    energy = _distribution([row["energy_abs_error_hartree"] for row in parent_metrics])
    source_energy = _distribution([row["source_energy_abs_error_hartree"] for row in parent_metrics])
    force = _distribution([row["force_mae_hartree_per_bohr"] for row in parent_metrics])
    source_force = _distribution([row["source_force_mae_hartree_per_bohr"] for row in parent_metrics])
    aggregate = {
        "train_q_relative_error": _distribution(train),
        "heldout_q_relative_error": _distribution(heldout),
        "heldout_q_fraction_at_or_below_0_15": float(np.mean(np.asarray(heldout) <= 0.15)),
        "energy_abs_error_hartree": energy,
        "source_energy_abs_error_hartree": source_energy,
        "energy_median_ratio": energy["median"] / max(source_energy["median"], np.finfo(float).tiny),
        "force_mae_hartree_per_bohr": force,
        "source_force_mae_hartree_per_bohr": source_force,
        "force_median_ratio": force["median"] / max(source_force["median"], np.finfo(float).tiny),
        "correction_hessian_asym_over_sym_max": max(row["correction_hessian_asym_over_sym"] for row in parent_metrics),
    }
    return aggregate, parent_metrics, direction_metrics


def _load_jet_audit(
    path: Path, protocol_hash: str, parents: list[CapacityParent]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("feature-jet audit protocol drift")
    if manifest.get("test100_accessed") is not False:
        raise ValueError("feature-jet audit opened Test100")
    if int(manifest.get("parent_count", -1)) != len(parents):
        raise ValueError("feature-jet audit parent count drift")
    entries = {str(row["molecule_id"]): row for row in manifest["entries"]}
    if set(entries) != {parent.molecule_id for parent in parents}:
        raise ValueError("feature-jet audit parent identity drift")
    return manifest, entries


def fit(
    protocol_path: Path,
    jet_dir: Path,
    jet_manifest_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(protocol_path)
    parents, rows, _, _, _, global_count = _parents_and_schema(protocol)
    protocol_hash = _sha256(protocol_path)
    jet_manifest, audited_jets = _load_jet_audit(
        jet_manifest_path, protocol_hash, parents
    )
    designs = []
    targets = []
    jets = {}
    columns = {}
    jet_hashes = {}
    for index, (parent, row) in enumerate(zip(parents, rows, strict=True)):
        path = jet_dir / f"{index:02d}_{parent.molecule_id}.pt"
        audit_row = audited_jets[parent.molecule_id]
        if Path(audit_row["artifact"]).resolve() != path.resolve():
            raise ValueError(f"feature jet audit path drift: {parent.molecule_id}")
        if _sha256(path) != str(audit_row["artifact_sha256"]):
            raise ValueError(f"feature jet audit hash drift: {parent.molecule_id}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != protocol_hash or payload.get("molecule_id") != parent.molecule_id:
            raise ValueError(f"feature jet provenance drift: {parent.molecule_id}")
        jet = tuple(payload[key].to(dtype=torch.float64) for key in ("features", "jacobian", "hessian"))
        if not all(bool(torch.isfinite(value).all()) for value in jet):
            raise FloatingPointError(f"non-finite feature jet: {parent.molecule_id}")
        local_designs, local_targets = _design_blocks(parent, _load_q(row), jet, protocol["training"])
        local_columns = payload["global_columns"].to(dtype=torch.long)
        for design in local_designs:
            designs.append(torch.zeros((design.shape[0], global_count), dtype=torch.float64).index_copy(1, local_columns, design))
        targets.extend(local_targets)
        jets[parent.molecule_id] = jet
        columns[parent.molecule_id] = local_columns
        jet_hashes[parent.molecule_id] = _sha256(path)
        del payload, local_designs, local_targets
        gc.collect()
    design = torch.cat(designs, dim=0)
    target = torch.cat(targets, dim=0)
    designs.clear()
    targets.clear()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    arms = []
    output_dir.mkdir(parents=True, exist_ok=True)
    floor = float(protocol["training"]["q_reference_floor_hartree_per_bohr2"])
    ridge_solutions = _dual_normalized_ridge_grid(
        design,
        target,
        ridges=[float(value) for value in protocol["solver"]["ridge_grid"]],
        column_floor=float(protocol["solver"]["column_floor"]),
        device=device,
    )
    for ridge in protocol["solver"]["ridge_grid"]:
        coefficients, solver = ridge_solutions[float(ridge)]
        aggregate, parent_metrics, direction_metrics = _evaluate(
            coefficients, parents, rows, jets, columns, floor
        )
        arm_id = f"ridge_{float(ridge):.0e}".replace("-", "m")
        arm_dir = output_dir / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = arm_dir / "model.pt"
        torch.save(
            {
                "coefficients": coefficients,
                "protocol": protocol_path.resolve().as_posix(),
                "protocol_sha256": protocol_hash,
                "ridge": float(ridge),
                "global_feature_count": global_count,
                "jet_hashes": jet_hashes,
                "formal_stage3_authorized": False,
                "test100_accessed": False,
            },
            checkpoint,
        )
        for name, table in (("per_parent.json", parent_metrics), ("per_direction.json", direction_metrics)):
            (arm_dir / name).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        arm_summary = {
            "arm_id": arm_id,
            "ridge": float(ridge),
            "solver": solver,
            "metrics": aggregate,
            "checkpoint": checkpoint.resolve().as_posix(),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        (arm_dir / "summary.json").write_text(json.dumps(arm_summary, indent=2, sort_keys=True) + "\n")
        arms.append(arm_summary)
        del coefficients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    energy_limit = float(protocol["diagnostic_gates"]["energy_median_ratio_max"])
    force_limit = float(protocol["diagnostic_gates"]["force_median_ratio_max"])
    def score(arm: dict[str, Any]) -> tuple[float, ...]:
        metrics = arm["metrics"]
        rejected = float(metrics["energy_median_ratio"] > energy_limit or metrics["force_median_ratio"] > force_limit)
        return (
            rejected,
            metrics["heldout_q_relative_error"]["median"],
            metrics["heldout_q_relative_error"]["p90"],
            metrics["train_q_relative_error"]["median"],
        )
    selected = min(arms, key=score)
    metrics = selected["metrics"]
    gates = protocol["diagnostic_gates"]
    checks = {
        "train_q_median": metrics["train_q_relative_error"]["median"] <= float(gates["train_q_median_relative_error_max"]),
        "train_q_p90": metrics["train_q_relative_error"]["p90"] <= float(gates["train_q_p90_relative_error_max"]),
        "heldout_q_median": metrics["heldout_q_relative_error"]["median"] <= float(gates["heldout_q_median_relative_error_max"]),
        "heldout_q_p90": metrics["heldout_q_relative_error"]["p90"] <= float(gates["heldout_q_p90_relative_error_max"]),
        "heldout_q_coverage": metrics["heldout_q_fraction_at_or_below_0_15"] >= float(gates["heldout_q_fraction_at_or_below_0_15_min"]),
        "energy": metrics["energy_median_ratio"] <= energy_limit,
        "force": metrics["force_median_ratio"] <= force_limit,
        "symmetry": metrics["correction_hessian_asym_over_sym_max"] <= float(gates["correction_hessian_asym_over_sym_max"]),
    }
    result = {
        "definition": protocol["definitions"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": protocol_hash,
        "feature_jet_manifest": jet_manifest_path.resolve().as_posix(),
        "feature_jet_manifest_sha256": _sha256(jet_manifest_path),
        "feature_jet_chunk_override_count": int(
            jet_manifest["chunk_override_count"]
        ),
        "design_shape": list(design.shape),
        "arms": arms,
        "selected_arm_id": selected["arm_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
        "train800_full_replay_included": False,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
    }
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "feature-jet", "fit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jet-dir", type=Path)
    parser.add_argument("--jet-manifest", type=Path)
    parser.add_argument("--parent-index", type=int)
    parser.add_argument("--feature-chunk-size", type=int)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    if args.mode == "preflight":
        preflight(args.protocol, args.output_dir)
    elif args.mode == "feature-jet":
        if args.parent_index is None:
            raise ValueError("feature-jet requires --parent-index")
        feature_jet(
            args.protocol,
            args.output_dir,
            args.parent_index,
            args.device,
            args.feature_chunk_size,
        )
    else:
        if args.jet_dir is None or args.jet_manifest is None:
            raise ValueError("fit requires --jet-dir and --jet-manifest")
        fit(
            args.protocol,
            args.jet_dir,
            args.jet_manifest,
            args.output_dir,
            args.device,
        )
