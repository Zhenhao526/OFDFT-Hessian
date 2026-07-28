#!/usr/bin/env python3
"""Fit a smooth shared nonlinear scalar residual on frozen Stage-1 Hessians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from mldft.ofdft.complete_total_training import (
    assign_two_task_pcgrad,
    parameter_gradient_diagnostics,
)

try:
    from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
    from scripts.qm9_complete_total_geometry_shared_resolve import (
        _column_squared_norms,
        _stage_target,
    )
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qm9_complete_total_geometry_shared_capacity import _load_parents
    from qm9_complete_total_geometry_shared_resolve import (
        _column_squared_norms,
        _stage_target,
    )
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


class SmoothDescriptorResidual(torch.nn.Module):
    """One-hidden-layer Softplus scalar with analytic descriptor derivatives."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        initial_linear: torch.Tensor,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.linear = torch.nn.Parameter(initial_linear.clone())
        self.weight = torch.nn.Parameter(
            torch.randn(
                hidden_size, feature_count, dtype=torch.float64, generator=generator
            )
            / math.sqrt(feature_count)
        )
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float64))
        self.output = torch.nn.Parameter(torch.zeros(hidden_size, dtype=torch.float64))

    def energy_force(
        self,
        descriptor: torch.Tensor,
        descriptor_jacobian: torch.Tensor,
        extensivity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate only the scalar and its negative geometry gradient."""
        if extensivity_scale <= 0.0:
            raise ValueError("extensivity_scale must be positive")
        descriptor = descriptor / extensivity_scale
        descriptor_jacobian = descriptor_jacobian / extensivity_scale
        activation = self.weight @ descriptor + self.bias
        sigmoid = torch.sigmoid(activation)
        energy = torch.dot(self.linear, descriptor) + torch.dot(
            self.output, torch.nn.functional.softplus(activation)
        )
        descriptor_gradient = self.linear + self.weight.T @ (
            self.output * sigmoid
        )
        force = -(descriptor_jacobian @ descriptor_gradient)
        return extensivity_scale * energy, extensivity_scale * force

    def derivatives(
        self,
        descriptor: torch.Tensor,
        descriptor_jacobian: torch.Tensor,
        weighted_descriptor_hessian: torch.Tensor,
        hessian_weight: float,
        extensivity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extensivity_scale <= 0.0:
            raise ValueError("extensivity_scale must be positive")
        descriptor = descriptor / extensivity_scale
        descriptor_jacobian = descriptor_jacobian / extensivity_scale
        weighted_descriptor_hessian = (
            weighted_descriptor_hessian / extensivity_scale
        )
        activation = self.weight @ descriptor + self.bias
        sigmoid = torch.sigmoid(activation)
        energy = torch.dot(self.linear, descriptor) + torch.dot(
            self.output, torch.nn.functional.softplus(activation)
        )
        descriptor_gradient = self.linear + self.weight.T @ (
            self.output * sigmoid
        )
        force = -(descriptor_jacobian @ descriptor_gradient)
        coordinate_projection = descriptor_jacobian @ self.weight.T
        nonlinear_curvature = self.output * sigmoid * (1.0 - sigmoid)
        coordinate_count = descriptor_jacobian.shape[0]
        weighted_hessian = (
            weighted_descriptor_hessian @ descriptor_gradient
        ).reshape(coordinate_count, coordinate_count)
        weighted_hessian = weighted_hessian + hessian_weight * (
            coordinate_projection * nonlinear_curvature[None, :]
        ) @ coordinate_projection.T
        return (
            extensivity_scale * energy,
            extensivity_scale * force,
            extensivity_scale * weighted_hessian,
        )


class DeepSmoothDescriptorResidual(SmoothDescriptorResidual):
    """One-hidden scalar plus a smooth depth-two residual branch."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        deep_hidden_size: int,
        initial_linear: torch.Tensor,
        seed: int,
    ) -> None:
        if deep_hidden_size <= 0:
            raise ValueError("deep_hidden_size must be positive")
        super().__init__(feature_count, hidden_size, initial_linear, seed)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.deep_weight = torch.nn.Parameter(
            torch.randn(
                deep_hidden_size,
                hidden_size,
                dtype=torch.float64,
                generator=generator,
            )
            / math.sqrt(hidden_size)
        )
        self.deep_bias = torch.nn.Parameter(
            torch.zeros(deep_hidden_size, dtype=torch.float64)
        )
        self.deep_output = torch.nn.Parameter(
            torch.zeros(deep_hidden_size, dtype=torch.float64)
        )

    def _descriptor_derivatives(
        self, descriptor: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        first_activation = self.weight @ descriptor + self.bias
        first_value = torch.nn.functional.softplus(first_activation)
        first_sigmoid = torch.sigmoid(first_activation)
        deep_activation = self.deep_weight @ first_value + self.deep_bias
        deep_sigmoid = torch.sigmoid(deep_activation)
        deep_backprop = self.deep_weight.T @ (self.deep_output * deep_sigmoid)
        first_backprop = self.output + deep_backprop
        descriptor_gradient = self.linear + self.weight.T @ (
            first_backprop * first_sigmoid
        )
        energy = (
            torch.dot(self.linear, descriptor)
            + torch.dot(self.output, first_value)
            + torch.dot(
                self.deep_output,
                torch.nn.functional.softplus(deep_activation),
            )
        )
        return (
            energy,
            descriptor_gradient,
            first_sigmoid,
            deep_sigmoid,
            first_backprop,
        )

    def energy_force(
        self,
        descriptor: torch.Tensor,
        descriptor_jacobian: torch.Tensor,
        extensivity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extensivity_scale <= 0.0:
            raise ValueError("extensivity_scale must be positive")
        descriptor = descriptor / extensivity_scale
        descriptor_jacobian = descriptor_jacobian / extensivity_scale
        energy, descriptor_gradient, _, _, _ = self._descriptor_derivatives(
            descriptor
        )
        force = -(descriptor_jacobian @ descriptor_gradient)
        return extensivity_scale * energy, extensivity_scale * force

    def derivatives(
        self,
        descriptor: torch.Tensor,
        descriptor_jacobian: torch.Tensor,
        weighted_descriptor_hessian: torch.Tensor,
        hessian_weight: float,
        extensivity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extensivity_scale <= 0.0:
            raise ValueError("extensivity_scale must be positive")
        descriptor = descriptor / extensivity_scale
        descriptor_jacobian = descriptor_jacobian / extensivity_scale
        weighted_descriptor_hessian = weighted_descriptor_hessian / extensivity_scale
        (
            energy,
            descriptor_gradient,
            first_sigmoid,
            deep_sigmoid,
            first_backprop,
        ) = self._descriptor_derivatives(descriptor)
        force = -(descriptor_jacobian @ descriptor_gradient)
        coordinate_count = descriptor_jacobian.shape[0]
        weighted_hessian = (
            weighted_descriptor_hessian @ descriptor_gradient
        ).reshape(coordinate_count, coordinate_count)

        first_projection = descriptor_jacobian @ self.weight.T
        first_curvature = (
            first_sigmoid * (1.0 - first_sigmoid) * first_backprop
        )
        deep_projection = (
            first_projection * first_sigmoid[None, :]
        ) @ self.deep_weight.T
        deep_curvature = (
            self.deep_output * deep_sigmoid * (1.0 - deep_sigmoid)
        )
        nonlinear_hessian = (
            (first_projection * first_curvature[None, :])
            @ first_projection.T
            + (deep_projection * deep_curvature[None, :])
            @ deep_projection.T
        )
        weighted_hessian = weighted_hessian + hessian_weight * nonlinear_hessian
        return (
            extensivity_scale * energy,
            extensivity_scale * force,
            extensivity_scale * weighted_hessian,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _design_provenance(design_root: Path) -> dict[str, object]:
    stages = {}
    for stage in ("three_body", "four_body"):
        root = design_root / stage
        summary = root / "summary.json"
        feature_keys = root / "feature_keys.npy"
        feature_key_container = feature_keys
        feature_key_dataset = None
        if not feature_keys.is_file():
            feature_key_container = root / "shared_coefficients.npz"
            feature_key_dataset = "feature_keys"
        design = root / "weighted_hessian_design.npy"
        constraints = root / "anchor_constraints.npy"
        stages[stage] = {
            "summary": summary.resolve().as_posix(),
            "summary_sha256": _sha256(summary),
            "feature_keys": feature_key_container.resolve().as_posix(),
            "feature_keys_dataset": feature_key_dataset,
            "feature_keys_sha256": _sha256(feature_key_container),
            "weighted_hessian_design": design.resolve().as_posix(),
            "weighted_hessian_design_size_bytes": design.stat().st_size,
            "anchor_constraints": constraints.resolve().as_posix(),
            "anchor_constraints_size_bytes": constraints.stat().st_size,
        }
    root_summary = design_root / "summary.json"
    return {
        "root": design_root.resolve().as_posix(),
        "summary": root_summary.resolve().as_posix(),
        "summary_sha256": _sha256(root_summary),
        "stages": stages,
    }


@dataclass
class CachedParent:
    molecule_id: str
    natoms: int
    descriptor: torch.Tensor
    jacobian: torch.Tensor
    weighted_feature_hessian: torch.Tensor
    hessian_weight: float
    energy_target: torch.Tensor
    force_target: torch.Tensor
    weighted_hessian_target: torch.Tensor
    baseline_energy: float
    baseline_force: np.ndarray
    baseline_hessian: np.ndarray
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    pbe_hessian_tensor: torch.Tensor
    baseline_hessian_tensor: torch.Tensor
    train_directions: torch.Tensor | None = None
    heldout_directions: torch.Tensor | None = None
    train_low_mode_directions: torch.Tensor | None = None


@dataclass(frozen=True)
class ReplayRecord:
    molecule_id: str
    sample_id: int
    cache_path: Path
    cache_sha256: str
    natoms: int


class FrozenReplayCache:
    """Hash-bound lazy CPU cache for train800 scalar E/F replay tensors."""

    def __init__(
        self,
        manifest_path: Path,
        feature_count: int,
        *,
        cpu_cache_size: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = _sha256(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("test100_accessed") is not False:
            raise ValueError("replay cache manifest does not freeze Test100")
        if self.manifest.get("complete") is not True:
            raise ValueError("replay cache manifest is not complete")
        if int(self.manifest["active_feature_count"]) != feature_count:
            raise ValueError(
                "replay active feature count does not match the scalar residual"
            )
        artifact = self.manifest["artifacts"]["success_csv"]
        success_csv = Path(artifact["path"])
        if _sha256(success_csv) != artifact["sha256"]:
            raise ValueError("replay cache success CSV hash mismatch")
        with success_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("replay cache success CSV is empty")
        self.records = [
            ReplayRecord(
                molecule_id=str(row["molecule_id"]),
                sample_id=int(row["sample_id"]),
                cache_path=Path(row["descriptor_cache"]),
                cache_sha256=str(row["descriptor_cache_sha256"]),
                natoms=int(row["natoms"]),
            )
            for row in rows
        ]
        if len({(row.molecule_id, row.sample_id) for row in self.records}) != len(
            self.records
        ):
            raise ValueError("replay cache has duplicate molecule/sample records")
        if cpu_cache_size < 0:
            raise ValueError("replay_cpu_cache_size must be non-negative")
        self.cpu_cache_size = cpu_cache_size
        self._cpu_cache: OrderedDict[int, tuple[torch.Tensor, ...]] = OrderedDict()
        self._verified: set[int] = set()
        self.disk_load_count = 0
        self.cache_hit_count = 0

    def __len__(self) -> int:
        return len(self.records)

    def _load_cpu(self, index: int) -> tuple[torch.Tensor, ...]:
        cached = self._cpu_cache.get(index)
        if cached is not None:
            self._cpu_cache.move_to_end(index)
            self.cache_hit_count += 1
            return cached
        record = self.records[index]
        if index not in self._verified:
            if _sha256(record.cache_path) != record.cache_sha256:
                raise ValueError(f"replay descriptor hash mismatch: {record.cache_path}")
            self._verified.add(index)
        with np.load(record.cache_path) as payload:
            descriptor = np.array(payload["descriptor"], dtype=np.float64, copy=True)
            jacobian = np.array(
                payload["descriptor_jacobian"], dtype=np.float64, copy=True
            )
            energy_target = np.array(
                payload["energy_target"], dtype=np.float64, copy=True
            ).reshape(())
            force_target = np.array(
                payload["force_target"], dtype=np.float64, copy=True
            ).reshape(-1)
        if descriptor.shape != (int(self.manifest["active_feature_count"]),):
            raise ValueError(f"invalid replay descriptor shape: {record.cache_path}")
        if jacobian.shape != (force_target.size, descriptor.size):
            raise ValueError(f"invalid replay Jacobian shape: {record.cache_path}")
        if not all(
            np.all(np.isfinite(array))
            for array in (descriptor, jacobian, energy_target, force_target)
        ):
            raise FloatingPointError(f"non-finite replay cache: {record.cache_path}")
        tensors = tuple(
            torch.from_numpy(array)
            for array in (descriptor, jacobian, energy_target, force_target)
        )
        self.disk_load_count += 1
        if self.cpu_cache_size > 0:
            self._cpu_cache[index] = tensors
            self._cpu_cache.move_to_end(index)
            while len(self._cpu_cache) > self.cpu_cache_size:
                self._cpu_cache.popitem(last=False)
        return tensors

    def batch_losses(
        self,
        model: SmoothDescriptorResidual,
        indices: torch.Tensor | list[int],
        device: torch.device,
        *,
        energy_scale: float,
        force_scale: float,
        atom_count_extensive: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy_terms = []
        force_terms = []
        for raw_index in indices:
            index = int(raw_index)
            descriptor, jacobian, energy_target, force_target = self._load_cpu(
                index
            )
            descriptor = descriptor.to(device=device, non_blocking=True)
            jacobian = jacobian.to(device=device, non_blocking=True)
            energy_target = energy_target.to(device=device, non_blocking=True)
            force_target = force_target.to(device=device, non_blocking=True)
            energy, force = model.energy_force(
                descriptor,
                jacobian,
                extensivity_scale=(
                    float(self.records[index].natoms)
                    if atom_count_extensive
                    else 1.0
                ),
            )
            energy_terms.append(((energy - energy_target) / energy_scale).square())
            force_terms.append(
                torch.mean(((force - force_target) / force_scale).square())
            )
        return torch.mean(torch.stack(energy_terms)), torch.mean(
            torch.stack(force_terms)
        )


def _sample_replay_indices(
    count: int, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("replay_batch_size must be positive")
    return torch.randint(count, (batch_size,), generator=generator)


def _active_stage(
    design_root: Path,
    stage: str,
    initial_root: Path | None,
    cutoff: float,
    chunk_rows: int,
    zero_linear_initialization: bool,
    feature_scale_mode: str = "column_norm",
    feature_scale_floor: float = 0.0,
    inventory_stage: dict[str, np.ndarray] | None = None,
):
    source = design_root / stage
    design = np.load(source / "weighted_hessian_design.npy", mmap_mode="r")
    constraints = np.load(source / "anchor_constraints.npy", mmap_mode="r")
    squared = _column_squared_norms(design, chunk_rows)
    norms = np.sqrt(squared)
    threshold = cutoff * max(float(np.max(norms)), np.finfo(float).tiny)
    active = norms >= threshold
    if not np.any(active):
        raise ValueError(f"no active {stage} columns at cutoff {cutoff}")
    if feature_scale_mode not in {"column_norm", "floored_column_norm", "unit"}:
        raise ValueError(f"unsupported feature_scale_mode: {feature_scale_mode}")
    if feature_scale_mode == "floored_column_norm" and feature_scale_floor <= 0.0:
        raise ValueError("feature_scale_floor must be positive for floored_column_norm")
    key_path = source / "feature_keys.npy"
    if key_path.is_file():
        source_keys = np.asarray(np.load(key_path), dtype=np.int64)
    else:
        with np.load(source / "shared_coefficients.npz") as payload:
            source_keys = np.asarray(payload["feature_keys"], dtype=np.int64)
    if source_keys.shape[0] != active.size:
        raise ValueError(f"{stage} feature-key/design column mismatch")
    active_norms = norms[active]
    if feature_scale_mode == "column_norm":
        feature_scales = active_norms
    elif feature_scale_mode == "floored_column_norm":
        feature_scales = np.maximum(active_norms, feature_scale_floor)
    else:
        feature_scales = np.ones_like(active_norms)
    if zero_linear_initialization:
        initial_raw = np.zeros(len(source_keys), dtype=np.float64)
    else:
        if initial_root is None:
            raise ValueError("--initial-root is required without zero initialization")
        with np.load(initial_root / stage / "shared_coefficients.npz") as payload:
            initial_raw = np.asarray(payload["coefficients"], dtype=np.float64)
            initial_keys = np.asarray(payload["feature_keys"], dtype=np.int64)
        if not np.array_equal(initial_keys, source_keys):
            raise ValueError(f"{stage} initial/source feature keys differ")

    if inventory_stage is not None:
        if cutoff != 0.0:
            raise ValueError("feature inventory requires zero column-norm cutoff")
        if not zero_linear_initialization:
            raise ValueError("feature inventory requires zero linear initialization")
        inventory_keys = np.asarray(inventory_stage["feature_keys"], dtype=np.int64)
        inventory_scales = np.asarray(inventory_stage["column_norms"], dtype=np.float64)
        if inventory_keys.shape[0] != inventory_scales.size:
            raise ValueError(f"{stage} inventory key/scale mismatch")
        inventory_lookup = {
            tuple(int(value) for value in key): index
            for index, key in enumerate(inventory_keys)
        }
        global_indices = np.asarray(
            [
                inventory_lookup.get(tuple(int(value) for value in key), -1)
                for key in source_keys
            ],
            dtype=np.int64,
        )
        if np.any(global_indices < 0):
            raise ValueError(f"{stage} inventory omits Stage2 design feature keys")
        active = np.ones_like(active, dtype=np.bool_)
        feature_scales = inventory_scales[global_indices]
        active_norms = norms
        keys = inventory_keys
        initial_linear = np.zeros(len(inventory_keys), dtype=np.float64)
        active_count = len(inventory_keys)
        total_count = len(inventory_keys)
        threshold = 0.0
    else:
        keys = source_keys[active]
        global_indices = np.arange(int(np.sum(active)), dtype=np.int64)
        initial_linear = initial_raw[active] * feature_scales
        active_count = int(np.sum(active))
        total_count = int(active.size)

    return {
        "stage": stage,
        "design": design,
        "constraints": constraints,
        "active": active,
        "global_indices": global_indices,
        "global_count": active_count,
        "norms": feature_scales,
        "source_column_norms": active_norms,
        "feature_scale_mode": feature_scale_mode,
        "feature_scale_floor": float(feature_scale_floor),
        "initial_linear": initial_linear,
        "keys": keys,
        "total_count": total_count,
        "active_count": active_count,
        "source_design_feature_count": int(active.size),
        "threshold": threshold,
    }


def _load_feature_inventory(path: Path | None):
    if path is None:
        return None, {}
    manifest = json.loads(path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("feature inventory does not freeze Test100")
    stages = {}
    for row in manifest["stages"]:
        schema = Path(row["schema"])
        if _sha256(schema) != row["schema_sha256"]:
            raise ValueError(f"feature inventory schema hash mismatch: {schema}")
        with np.load(schema) as payload:
            stages[str(row["stage"])] = {
                "feature_keys": np.asarray(payload["feature_keys"], dtype=np.int64),
                "column_norms": np.asarray(payload["column_norms"], dtype=np.float64),
            }
    if set(stages) != {"three_body", "four_body"}:
        raise ValueError("feature inventory requires three_body and four_body stages")
    return manifest, stages


def _prepare_parents(args: argparse.Namespace, device: torch.device):
    parents, manifest = _load_parents(args.baseline_manifest)
    direction_manifest = None
    direction_lookup = {}
    if args.direction_manifest is not None:
        direction_manifest = json.loads(args.direction_manifest.read_text())
        if direction_manifest.get("test100_accessed") is not False:
            raise ValueError("direction manifest does not certify frozen Test100")
        direction_lookup = {
            str(row["molecule_id"]): row for row in direction_manifest["directions"]
        }
    feature_inventory, inventory_stages = _load_feature_inventory(
        args.feature_inventory_manifest
    )
    stages = [
        _active_stage(
            args.design_root,
            stage,
            args.initial_root,
            args.column_norm_relative_cutoff,
            args.norm_chunk_rows,
            args.zero_linear_initialization,
            getattr(args, "feature_scale_mode", "column_norm"),
            getattr(args, "feature_scale_floor", 0.0),
            inventory_stages.get(stage),
        )
        for stage in ("three_body", "four_body")
    ]
    _, parent_weights, _ = _stage_target(
        parents, args.relative_loss_fraction, args.absolute_hessian_scale
    )
    cached = []
    hessian_offset = 0
    constraint_offset = 0
    for parent, hessian_weight in zip(parents, parent_weights, strict=True):
        hessian_rows = parent.hessian.size
        constraint_rows = 1 + parent.force.size
        descriptors = []
        jacobians = []
        hessians = []
        for stage in stages:
            active = stage["active"]
            global_indices = stage["global_indices"]
            global_count = stage["global_count"]
            norms = stage["norms"]
            local_constraints = np.asarray(
                stage["constraints"][
                    constraint_offset : constraint_offset + constraint_rows, active
                ]
            )
            local_descriptor = local_constraints[0] / norms
            local_jacobian = local_constraints[1:] / norms[None, :]
            descriptor_stage = np.zeros(global_count, dtype=np.float64)
            jacobian_stage = np.zeros(
                (constraint_rows - 1, global_count), dtype=np.float64
            )
            descriptor_stage[global_indices] = local_descriptor
            jacobian_stage[:, global_indices] = local_jacobian
            descriptors.append(descriptor_stage)
            jacobians.append(jacobian_stage)
            local_hessian = np.asarray(
                stage["design"][
                    hessian_offset : hessian_offset + hessian_rows, active
                ]
            ) / norms[None, :]
            coordinate_count = parent.force.size
            local_hessian = local_hessian.reshape(
                coordinate_count, coordinate_count, -1
            )
            # A scalar descriptor has commuting mixed partials. Average only the
            # finite-difference noise in its cached derivative, not the final Hessian.
            local_hessian = 0.5 * (
                local_hessian + local_hessian.transpose(1, 0, 2)
            )
            hessian_stage = np.zeros((hessian_rows, global_count), dtype=np.float64)
            hessian_stage[:, global_indices] = local_hessian.reshape(hessian_rows, -1)
            hessians.append(hessian_stage)
        descriptor = np.concatenate(descriptors)
        jacobian = np.concatenate(jacobians, axis=1)
        weighted_feature_hessian = np.concatenate(hessians, axis=1)
        train_directions = None
        heldout_directions = None
        train_low_mode_directions = None
        if direction_manifest is not None:
            direction_row = direction_lookup.get(parent.molecule_id)
            if direction_row is None:
                raise ValueError(f"no frozen directions for {parent.molecule_id}")
            with np.load(direction_row["direction_path"]) as payload:
                directions = np.asarray(payload["directions"], dtype=np.float64)
                roles = np.asarray(payload["roles"]).astype(str)
                kinds = np.asarray(payload["kinds"]).astype(str)
            if directions.shape[1] != parent.force.size:
                raise ValueError(f"direction shape mismatch for {parent.molecule_id}")
            train_directions = torch.as_tensor(
                directions[roles == "train"], dtype=torch.float64, device=device
            )
            heldout_directions = torch.as_tensor(
                directions[roles == "heldout"], dtype=torch.float64, device=device
            )
            train_low_mode_directions = torch.as_tensor(
                directions[(roles == "train") & (kinds == "low_mode")],
                dtype=torch.float64,
                device=device,
            )
            if train_directions.numel() == 0 or heldout_directions.numel() == 0:
                raise ValueError(f"empty train/heldout directions for {parent.molecule_id}")
        cached.append(
            CachedParent(
                molecule_id=parent.molecule_id,
                natoms=int(parent.atomic_numbers.size),
                descriptor=torch.as_tensor(descriptor, dtype=torch.float64, device=device),
                jacobian=torch.as_tensor(jacobian, dtype=torch.float64, device=device),
                weighted_feature_hessian=torch.as_tensor(
                    weighted_feature_hessian, dtype=torch.float64, device=device
                ),
                hessian_weight=hessian_weight,
                energy_target=torch.tensor(
                    parent.pbe_energy - parent.energy, dtype=torch.float64, device=device
                ),
                force_target=torch.as_tensor(
                    parent.pbe_force - parent.force, dtype=torch.float64, device=device
                ).reshape(-1),
                weighted_hessian_target=torch.as_tensor(
                    (parent.pbe_hessian - parent.hessian) * hessian_weight,
                    dtype=torch.float64,
                    device=device,
                ),
                baseline_energy=parent.energy,
                baseline_force=parent.force.copy(),
                baseline_hessian=parent.hessian.copy(),
                pbe_energy=parent.pbe_energy,
                pbe_force=parent.pbe_force.copy(),
                pbe_hessian=parent.pbe_hessian.copy(),
                pbe_hessian_tensor=torch.as_tensor(
                    parent.pbe_hessian, dtype=torch.float64, device=device
                ),
                baseline_hessian_tensor=torch.as_tensor(
                    parent.hessian, dtype=torch.float64, device=device
                ),
                train_directions=train_directions,
                heldout_directions=heldout_directions,
                train_low_mode_directions=train_low_mode_directions,
            )
        )
        hessian_offset += hessian_rows
        constraint_offset += constraint_rows
    initial_linear = torch.as_tensor(
        np.concatenate([stage["initial_linear"] for stage in stages]),
        dtype=torch.float64,
        device=device,
    )
    metadata = {
        "source_split_sha256": manifest["source_split_sha256"],
        "feature_hessian_mixed_partials_symmetrized": True,
        "stages": [
            {
                "stage": stage["stage"],
                "total_feature_count": stage["total_count"],
                "active_feature_count": stage["active_count"],
                "column_norm_threshold": stage["threshold"],
                "feature_scale_mode": stage["feature_scale_mode"],
                "feature_scale_floor": stage["feature_scale_floor"],
                "source_design_feature_count": stage["source_design_feature_count"],
            }
            for stage in stages
        ],
        "direction_manifest": (
            str(args.direction_manifest.resolve())
            if args.direction_manifest is not None
            else None
        ),
        "direction_manifest_sha256": (
            hashlib.sha256(args.direction_manifest.read_bytes()).hexdigest()
            if args.direction_manifest is not None
            else None
        ),
        "zero_linear_initialization": args.zero_linear_initialization,
        "feature_inventory_manifest": (
            str(args.feature_inventory_manifest.resolve())
            if args.feature_inventory_manifest is not None
            else None
        ),
        "feature_inventory_manifest_sha256": (
            _sha256(args.feature_inventory_manifest)
            if args.feature_inventory_manifest is not None
            else None
        ),
    }
    return cached, initial_linear, metadata


def _parent_cv_partition(
    parents: list[CachedParent], args: argparse.Namespace
) -> tuple[list[CachedParent], list[CachedParent], dict[str, object] | None]:
    protocol_path = getattr(args, "parent_cv_protocol", None)
    variant_id = getattr(args, "parent_cv_variant_id", None)
    fold_index = getattr(args, "parent_cv_fold_index", None)
    if protocol_path is None:
        if variant_id is not None or fold_index is not None:
            raise ValueError("parent-CV variant/fold requires --parent-cv-protocol")
        return parents, parents, None
    if variant_id is None or fold_index is None:
        raise ValueError("parent-CV protocol requires variant ID and fold index")

    protocol = yaml.safe_load(protocol_path.read_text())
    scope = protocol["scope"]
    if scope.get("validation_parent_access_allowed") is not False:
        raise ValueError("parent-CV protocol must prohibit validation-parent access")
    if scope.get("test100_accessed") is not False:
        raise ValueError("parent-CV protocol must freeze Test100")
    if int(scope.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("parent-CV protocol records Test100 access")

    frozen = protocol["frozen_inputs"]
    artifacts = (
        (args.baseline_manifest, "baseline_manifest_sha256"),
        (args.design_root / "summary.json", "design_summary_sha256"),
        (args.feature_inventory_manifest, "feature_inventory_manifest_sha256"),
    )
    for path, hash_key in artifacts:
        if path is None or _sha256(path) != str(frozen[hash_key]):
            raise ValueError(f"parent-CV frozen artifact differs: {hash_key}")
    if args.checkpoint is not None or args.initial_root is not None:
        raise ValueError("parent-CV must start from a zero residual, not a fitted checkpoint")
    if not args.zero_linear_initialization:
        raise ValueError("parent-CV requires zero linear initialization")
    if args.direction_manifest is not None:
        raise ValueError("this parent-CV protocol uses complete training Hessians")

    variants = {
        str(row["id"]): row for row in protocol["preregistered_variants"]
    }
    if variant_id not in variants:
        raise ValueError("parent-CV variant is not preregistered")
    variant = variants[variant_id]
    for argument, expected in variant["runtime_parameters"].items():
        actual = getattr(args, argument)
        if isinstance(expected, float) or isinstance(actual, float):
            if float(actual) != float(expected):
                raise ValueError(f"parent-CV runtime parameter differs: {argument}")
        elif actual != expected:
            raise ValueError(f"parent-CV runtime parameter differs: {argument}")

    replay_expected = bool(variant["train800_energy_force_replay"])
    if replay_expected:
        if args.replay_cache_manifest is None:
            raise ValueError("parent-CV replay variant requires replay cache")
        if _sha256(args.replay_cache_manifest) != str(
            frozen["replay_cache_manifest_sha256"]
        ):
            raise ValueError("parent-CV replay cache differs from protocol")
    elif args.replay_cache_manifest is not None:
        raise ValueError("parent-CV no-replay control received replay data")

    folds = [
        list(map(str, fold))
        for fold in protocol["cross_validation"]["fold_assignment"]
    ]
    if not 0 <= int(fold_index) < len(folds):
        raise ValueError("parent-CV fold index is out of range")
    expected_seed = int(protocol["cross_validation"]["base_seed"]) + int(
        fold_index
    )
    if args.seed != expected_seed:
        raise ValueError("parent-CV seed differs from the frozen fold seed")
    expected_replay_seed = int(
        protocol["cross_validation"]["base_replay_seed"]
    ) + int(fold_index)
    if args.replay_seed != expected_replay_seed:
        raise ValueError("parent-CV replay seed differs from the frozen fold seed")
    parent_ids = [parent.molecule_id for parent in parents]
    flattened = [molecule_id for fold in folds for molecule_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(parent_ids):
        raise ValueError("parent-CV folds do not partition the baseline parents")
    held_ids = set(folds[int(fold_index)])
    fit = [parent for parent in parents if parent.molecule_id not in held_ids]
    held = [parent for parent in parents if parent.molecule_id in held_ids]
    if len(fit) != int(scope["parents_per_fit_fold"]) or len(held) != int(
        scope["parents_per_held_fold"]
    ):
        raise ValueError("parent-CV fit/held counts differ from protocol")
    return fit, held, {
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": _sha256(protocol_path),
        "variant_id": str(variant_id),
        "fold_index": int(fold_index),
        "fit_parent_ids": [parent.molecule_id for parent in fit],
        "evaluation_parent_ids": [parent.molecule_id for parent in held],
        "held_parent_hessian_used_for_checkpoint_selection": False,
        "held_parent_hessian_used_for_gradient": False,
        "train800_energy_force_replay": replay_expected,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }


def _directional_hvp_loss(
    weighted_prediction: torch.Tensor,
    weighted_target: torch.Tensor,
    reference_hessian: torch.Tensor,
    directions: torch.Tensor,
    *,
    hessian_weight: float,
    relative_fraction: float,
    absolute_scale: float,
    reference_floor: float,
) -> torch.Tensor:
    difference = (weighted_prediction - weighted_target) / hessian_weight
    error_hvp = torch.einsum("ij,dj->di", difference, directions)
    reference_hvp = torch.einsum("ij,dj->di", reference_hessian, directions)
    absolute = torch.mean(error_hvp.square(), dim=1) / absolute_scale**2
    reference_norm = torch.linalg.vector_norm(reference_hvp, dim=1)
    denominator = torch.clamp(reference_norm, min=reference_floor)
    relative = torch.sum(error_hvp.square(), dim=1) / denominator.square()
    return torch.mean(
        relative_fraction * relative + (1.0 - relative_fraction) * absolute
    )


def _low_mode_spectral_loss(
    predicted_hessian: torch.Tensor,
    reference_hessian: torch.Tensor,
    directions: torch.Tensor,
    *,
    reference_floor: float,
    wrong_curvature_multiplier: float,
) -> torch.Tensor:
    predicted = torch.einsum("di,ij,dj->d", directions, predicted_hessian, directions)
    reference = torch.einsum("di,ij,dj->d", directions, reference_hessian, directions)
    scale = torch.clamp(torch.abs(reference), min=reference_floor)
    relative_squared = ((predicted - reference) / scale).square()
    signed_prediction = torch.sign(reference) * predicted
    wrong_curvature = torch.relu(-signed_prediction).square() / scale.square()
    return torch.mean(relative_squared + wrong_curvature_multiplier * wrong_curvature)


def _aggregate_parent_hvp_losses(
    parent_losses: torch.Tensor,
    *,
    tail_fraction: float,
    tail_weight: float,
) -> torch.Tensor:
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("hvp_tail_fraction must be in (0, 1]")
    if tail_weight < 0.0:
        raise ValueError("hvp_tail_weight must be non-negative")
    mean_loss = torch.mean(parent_losses)
    if tail_weight == 0.0:
        return mean_loss
    tail_count = max(1, int(math.ceil(tail_fraction * parent_losses.numel())))
    tail_loss = torch.mean(torch.topk(parent_losses, tail_count).values)
    return mean_loss + tail_weight * tail_loss


def _losses(model, parents, args, *, sample_directions: bool = True):
    energy_terms = []
    force_terms = []
    hessian_terms = []
    spectrum_terms = []
    predictions = []
    for parent in parents:
        energy, force, weighted_hessian = model.derivatives(
            parent.descriptor,
            parent.jacobian,
            parent.weighted_feature_hessian,
            parent.hessian_weight,
            extensivity_scale=(float(parent.natoms) if args.atom_count_extensive else 1.0),
        )
        energy_terms.append(
            ((energy - parent.energy_target) / args.energy_scale) ** 2
        )
        force_terms.append(
            torch.mean(((force - parent.force_target) / args.force_scale) ** 2)
        )
        if parent.train_directions is None:
            hessian_terms.append(
                torch.sum((weighted_hessian - parent.weighted_hessian_target) ** 2)
            )
        else:
            directions = parent.train_directions
            if sample_directions and directions.shape[0] > args.directions_per_parent:
                indices = torch.randperm(directions.shape[0], device=directions.device)[
                    : args.directions_per_parent
                ]
                directions = directions[indices]
            hessian_terms.append(
                _directional_hvp_loss(
                    weighted_hessian,
                    parent.weighted_hessian_target,
                    parent.pbe_hessian_tensor,
                    directions,
                    hessian_weight=parent.hessian_weight,
                    relative_fraction=args.hvp_relative_loss_fraction,
                    absolute_scale=args.absolute_hvp_scale,
                    reference_floor=args.hvp_reference_floor,
                )
            )
        if (
            parent.train_low_mode_directions is not None
            and parent.train_low_mode_directions.numel() > 0
        ):
            predicted_total_hessian = (
                parent.baseline_hessian_tensor
                + weighted_hessian / parent.hessian_weight
            )
            spectrum_terms.append(
                _low_mode_spectral_loss(
                    predicted_total_hessian,
                    parent.pbe_hessian_tensor,
                    parent.train_low_mode_directions,
                    reference_floor=args.spectrum_reference_floor,
                    wrong_curvature_multiplier=args.wrong_curvature_multiplier,
                )
            )
        else:
            spectrum_terms.append(energy * 0.0)
        predictions.append((energy, force, weighted_hessian))
    energy_loss = torch.mean(torch.stack(energy_terms))
    force_loss = torch.mean(torch.stack(force_terms))
    hessian_loss = _aggregate_parent_hvp_losses(
        torch.stack(hessian_terms),
        tail_fraction=args.hvp_tail_fraction,
        tail_weight=args.hvp_tail_weight,
    )
    spectrum_loss = torch.mean(torch.stack(spectrum_terms))
    total = (
        args.lambda_energy * energy_loss
        + args.lambda_force * force_loss
        + args.lambda_hessian * hessian_loss
        + args.lambda_spectrum * spectrum_loss
        + args.lambda_parameter * (
            torch.mean(model.weight * model.weight)
            + torch.mean(model.output * model.output)
        )
    )
    return total, energy_loss, force_loss, hessian_loss, spectrum_loss, predictions


def _evaluate(model, parents, args):
    with torch.no_grad():
        total, energy_loss, force_loss, hessian_loss, spectrum_loss, predictions = _losses(
            model, parents, args, sample_directions=False
        )
    rows = []
    for parent, (energy, force, weighted_hessian) in zip(
        parents, predictions, strict=True
    ):
        correction = weighted_hessian.detach().cpu().numpy() / parent.hessian_weight
        predicted_hessian = parent.baseline_hessian + correction
        predicted_force = parent.baseline_force + force.detach().cpu().numpy().reshape(
            parent.baseline_force.shape
        )
        predicted_energy = parent.baseline_energy + float(energy.detach().cpu())
        row = {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
                "baseline_energy_abs_error_hartree": abs(
                    parent.baseline_energy - parent.pbe_energy
                ),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(predicted_force - parent.pbe_force))
                ),
                "baseline_force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(parent.baseline_force - parent.pbe_force))
                ),
                **hessian_metrics(predicted_hessian, parent.pbe_hessian),
                "predicted_energy": predicted_energy,
                "predicted_force": predicted_force,
                "predicted_hessian": predicted_hessian,
            }
        if parent.train_directions is not None:
            difference = predicted_hessian - parent.pbe_hessian
            for role, direction_tensor in (
                ("train", parent.train_directions),
                ("heldout", parent.heldout_directions),
            ):
                directions = direction_tensor.detach().cpu().numpy()
                error_hvp = np.einsum("ij,dj->di", difference, directions)
                reference_hvp = np.einsum(
                    "ij,dj->di", parent.pbe_hessian, directions
                )
                row[f"{role}_hvp_mae"] = float(np.mean(np.abs(error_hvp)))
                row[f"{role}_hvp_rmse"] = float(np.sqrt(np.mean(error_hvp**2)))
                row[f"{role}_hvp_relative_frobenius"] = float(
                    np.linalg.norm(error_hvp)
                    / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
                )
        rows.append(row)
    scalar = {
        "total_loss": float(total.detach().cpu()),
        "energy_loss": float(energy_loss.detach().cpu()),
        "force_loss": float(force_loss.detach().cpu()),
        "hessian_loss": float(hessian_loss.detach().cpu()),
        "spectrum_loss": float(spectrum_loss.detach().cpu()),
        "median_relative_frobenius": float(
            np.median([row["relative_frobenius"] for row in rows])
        ),
        "max_relative_frobenius": float(
            max(row["relative_frobenius"] for row in rows)
        ),
    }
    if rows and "train_hvp_relative_frobenius" in rows[0]:
        for role in ("train", "heldout"):
            values = [row[f"{role}_hvp_relative_frobenius"] for row in rows]
            scalar[f"median_{role}_hvp_relative_frobenius"] = float(np.median(values))
            scalar[f"max_{role}_hvp_relative_frobenius"] = float(max(values))
    return scalar, rows


def _parent_cv_distribution(rows: list[dict[str, object]]) -> dict[str, float]:
    relative = np.asarray(
        [float(row["relative_frobenius"]) for row in rows], dtype=np.float64
    )
    energy = np.asarray(
        [float(row["energy_abs_error_hartree"]) for row in rows], dtype=np.float64
    )
    source_energy = np.asarray(
        [float(row["baseline_energy_abs_error_hartree"]) for row in rows],
        dtype=np.float64,
    )
    force = np.asarray(
        [float(row["force_mae_hartree_per_bohr"]) for row in rows], dtype=np.float64
    )
    source_force = np.asarray(
        [float(row["baseline_force_mae_hartree_per_bohr"]) for row in rows],
        dtype=np.float64,
    )
    asymmetry = np.asarray(
        [
            float(row["antisymmetric_over_symmetric_frobenius"])
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "median_relative_frobenius": float(np.median(relative)),
        "p90_relative_frobenius": float(np.quantile(relative, 0.9)),
        "max_relative_frobenius": float(np.max(relative)),
        "fraction_relative_frobenius_at_or_below_0_10": float(
            np.mean(relative <= 0.10)
        ),
        "fraction_relative_frobenius_at_or_below_0_15": float(
            np.mean(relative <= 0.15)
        ),
        "fraction_relative_frobenius_at_or_below_0_20": float(
            np.mean(relative <= 0.20)
        ),
        "max_antisymmetric_over_symmetric_frobenius": float(np.max(asymmetry)),
        "energy_abs_error_median_hartree": float(np.median(energy)),
        "source_energy_abs_error_median_hartree": float(np.median(source_energy)),
        "energy_median_ratio_to_source": float(
            np.median(energy) / max(np.median(source_energy), np.finfo(float).tiny)
        ),
        "force_mae_median_hartree_per_bohr": float(np.median(force)),
        "source_force_mae_median_hartree_per_bohr": float(
            np.median(source_force)
        ),
        "force_median_ratio_to_source": float(
            np.median(force) / max(np.median(source_force), np.finfo(float).tiny)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if args.hvp_warmup_steps < 0:
        raise ValueError("hvp_warmup_steps must be non-negative")
    if args.replay_eval_size <= 0:
        raise ValueError("replay_eval_size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents, initial_linear, metadata = _prepare_parents(args, device)
    fit_parents, evaluation_parents, parent_cv = _parent_cv_partition(parents, args)
    if args.deep_hidden_size > 0:
        model = DeepSmoothDescriptorResidual(
            initial_linear.numel(),
            args.hidden_size,
            args.deep_hidden_size,
            initial_linear,
            args.seed,
        ).to(device)
    else:
        model = SmoothDescriptorResidual(
            initial_linear.numel(), args.hidden_size, initial_linear, args.seed
        ).to(device)
    resume_payload = None
    if args.checkpoint is not None:
        resume_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if (
            args.feature_inventory_manifest is not None
            and resume_payload.get("feature_inventory_manifest_sha256")
            != metadata["feature_inventory_manifest_sha256"]
        ):
            raise ValueError("checkpoint feature inventory hash mismatch")
        model.load_state_dict(resume_payload["state_dict"])
    replay_cache = None
    replay_generator = torch.Generator(device="cpu").manual_seed(
        args.replay_seed if args.replay_seed is not None else args.seed + 104729
    )
    replay_eval_indices = None
    if args.replay_cache_manifest is not None:
        if args.replay_batch_size <= 0:
            raise ValueError("--replay-batch-size must be positive with replay")
        replay_cache = FrozenReplayCache(
            args.replay_cache_manifest,
            initial_linear.numel(),
            cpu_cache_size=args.replay_cpu_cache_size,
        )
        if args.require_replay_schema_checkpoint_match and args.checkpoint is not None:
            source_matches = (
                _sha256(args.checkpoint)
                == replay_cache.manifest["schema_checkpoint_sha256"]
            )
            resume_matches = (
                resume_payload is not None
                and resume_payload.get("replay_cache_manifest_sha256")
                == replay_cache.manifest_sha256
            )
            if not (source_matches or resume_matches):
                raise ValueError(
                    "checkpoint is neither the schema source nor a resume from this replay cache"
                )
        eval_size = min(args.replay_eval_size, len(replay_cache))
        replay_eval_indices = torch.randperm(
            len(replay_cache), generator=replay_generator
        )[:eval_size]
    elif args.lambda_replay_energy != 0.0 or args.lambda_replay_force != 0.0:
        raise ValueError("replay loss weights require --replay-cache-manifest")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.resume_optimizer:
        if resume_payload is None:
            raise ValueError("--resume-optimizer requires --checkpoint")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    log_path = args.output_dir / "training_metrics.jsonl"
    best_total = float("inf")
    best_step = -1
    with log_path.open("w") as log_handle:
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            (
                total,
                energy_loss,
                force_loss,
                hessian_loss,
                spectrum_loss,
                _,
            ) = _losses(model, fit_parents, args)
            hvp_warmup_multiplier = (
                min(1.0, step / args.hvp_warmup_steps)
                if args.hvp_warmup_steps > 0
                else 1.0
            )
            if hvp_warmup_multiplier != 1.0:
                total = total - (
                    (1.0 - hvp_warmup_multiplier)
                    * args.lambda_hessian
                    * hessian_loss
                )
            replay_energy_loss = energy_loss * 0.0
            replay_force_loss = force_loss * 0.0
            if replay_cache is not None:
                replay_indices = _sample_replay_indices(
                    len(replay_cache), args.replay_batch_size, replay_generator
                )
                replay_energy_loss, replay_force_loss = replay_cache.batch_losses(
                    model,
                    replay_indices,
                    device,
                    energy_scale=args.energy_scale,
                    force_scale=args.force_scale,
                    atom_count_extensive=args.atom_count_extensive,
                )
                total = (
                    total
                    + args.lambda_replay_energy * replay_energy_loss
                    + args.lambda_replay_force * replay_force_loss
                )
            should_log = step % args.log_interval == 0 or step == args.steps
            weighted_terms = {
                "energy": args.lambda_energy * energy_loss
                + args.lambda_replay_energy * replay_energy_loss,
                "force": args.lambda_force * force_loss
                + args.lambda_replay_force * replay_force_loss,
                "hvp": (
                    hvp_warmup_multiplier * args.lambda_hessian * hessian_loss
                ),
            }
            if args.lambda_spectrum != 0.0:
                weighted_terms["spectrum"] = args.lambda_spectrum * spectrum_loss
            gradient_diagnostics = {}
            if should_log:
                named_parameters = list(model.named_parameters())
                gradient_diagnostics = parameter_gradient_diagnostics(
                    weighted_terms,
                    [parameter for _, parameter in named_parameters],
                    parameter_names=[name for name, _ in named_parameters],
                )
            balance_diagnostics = {}
            if step > 0:
                if args.pcgrad:
                    parameter_penalty = args.lambda_parameter * (
                        torch.mean(model.weight * model.weight)
                        + torch.mean(model.output * model.output)
                    )
                    replay_loss = (
                        weighted_terms["energy"]
                        + weighted_terms["force"]
                        + parameter_penalty
                    )
                    curvature_loss = weighted_terms["hvp"] + weighted_terms.get(
                        "spectrum", spectrum_loss * 0.0
                    )
                    balance_diagnostics = assign_two_task_pcgrad(
                        replay_loss,
                        curvature_loss,
                        model.parameters(),
                        max_second_to_first_norm_ratio=(
                            args.pcgrad_max_curvature_to_replay_norm_ratio
                            if math.isfinite(
                                args.pcgrad_max_curvature_to_replay_norm_ratio
                            )
                            else None
                        ),
                    )
                else:
                    total.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    .detach()
                    .cpu()
                )
                optimizer.step()
            else:
                gradient_norm = math.nan
            if should_log:
                scalar, _ = _evaluate(model, fit_parents, args)
                replay_eval_energy = math.nan
                replay_eval_force = math.nan
                selection_total = scalar["total_loss"]
                if replay_cache is not None:
                    with torch.no_grad():
                        replay_eval_energy_tensor, replay_eval_force_tensor = (
                            replay_cache.batch_losses(
                                model,
                                replay_eval_indices,
                                device,
                                energy_scale=args.energy_scale,
                                force_scale=args.force_scale,
                                atom_count_extensive=args.atom_count_extensive,
                            )
                        )
                    replay_eval_energy = float(replay_eval_energy_tensor.cpu())
                    replay_eval_force = float(replay_eval_force_tensor.cpu())
                    selection_total += (
                        args.lambda_replay_energy * replay_eval_energy
                        + args.lambda_replay_force * replay_eval_force
                    )
                row = {
                    "step": step,
                    "gradient_norm": gradient_norm,
                    "hvp_warmup_multiplier": hvp_warmup_multiplier,
                    "sampled_replay_energy_loss": float(
                        replay_energy_loss.detach().cpu()
                    ),
                    "sampled_replay_force_loss": float(
                        replay_force_loss.detach().cpu()
                    ),
                    "fixed_replay_eval_energy_loss": replay_eval_energy,
                    "fixed_replay_eval_force_loss": replay_eval_force,
                    "checkpoint_selection_total_loss": selection_total,
                    "replay_disk_load_count": (
                        replay_cache.disk_load_count if replay_cache is not None else 0
                    ),
                    "replay_cpu_cache_hit_count": (
                        replay_cache.cache_hit_count if replay_cache is not None else 0
                    ),
                    **gradient_diagnostics,
                    **balance_diagnostics,
                    **scalar,
                    "wall_time_s": time.perf_counter() - started,
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                torch.save(
                    {
                        "step": step,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": vars(args),
                        "test100_accessed": False,
                        "parent_cv": parent_cv,
                        "replay_cache_manifest_sha256": (
                            replay_cache.manifest_sha256
                            if replay_cache is not None
                            else None
                        ),
                        "active_schema_manifest_sha256": (
                            replay_cache.manifest["schema_manifest_sha256"]
                            if replay_cache is not None
                            else None
                        ),
                        "feature_inventory_manifest_sha256": metadata[
                            "feature_inventory_manifest_sha256"
                        ],
                    },
                    args.output_dir / "last.ckpt",
                )
                if selection_total < best_total:
                    best_total = selection_total
                    best_step = step
                    torch.save(
                        {
                            "step": step,
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": vars(args),
                            "test100_accessed": False,
                            "parent_cv": parent_cv,
                            "replay_cache_manifest_sha256": (
                                replay_cache.manifest_sha256
                                if replay_cache is not None
                                else None
                            ),
                            "active_schema_manifest_sha256": (
                                replay_cache.manifest["schema_manifest_sha256"]
                                if replay_cache is not None
                                else None
                            ),
                            "feature_inventory_manifest_sha256": metadata[
                                "feature_inventory_manifest_sha256"
                            ],
                        },
                        args.output_dir / "best.ckpt",
                    )

    best_payload = torch.load(
        args.output_dir / "best.ckpt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_payload["state_dict"])
    scalar, fit_rows = _evaluate(model, fit_parents, args)
    if parent_cv is None:
        evaluation_scalar = scalar
        evaluation_rows = fit_rows
        rows = fit_rows
    else:
        evaluation_scalar, evaluation_rows = _evaluate(
            model, evaluation_parents, args
        )
        rows = [
            *({"parent_cv_role": "fit", **row} for row in fit_rows),
            *({"parent_cv_role": "held", **row} for row in evaluation_rows),
        ]
    final_replay = None
    if replay_cache is not None:
        with torch.no_grad():
            final_replay_energy, final_replay_force = replay_cache.batch_losses(
                model,
                replay_eval_indices,
                device,
                energy_scale=args.energy_scale,
                force_scale=args.force_scale,
                atom_count_extensive=args.atom_count_extensive,
            )
        final_replay = {
            "fixed_eval_count": int(replay_eval_indices.numel()),
            "energy_loss": float(final_replay_energy.cpu()),
            "force_loss": float(final_replay_force.cpu()),
            "disk_load_count": replay_cache.disk_load_count,
            "cpu_cache_hit_count": replay_cache.cache_hit_count,
        }
    csv_rows = []
    for row in rows:
        array_row = dict(row)
        predicted_force = array_row.pop("predicted_force")
        predicted_hessian = array_row.pop("predicted_hessian")
        np.savez_compressed(
            args.output_dir / f"{row['molecule_id']}_result.npz",
            predicted_force=predicted_force,
            predicted_hessian=predicted_hessian,
            pbe_hessian=next(
                parent.pbe_hessian
                for parent in parents
                if parent.molecule_id == row["molecule_id"]
            ),
        )
        csv_rows.append(array_row)
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "definition": (
            "shared float64 Softplus nonlinear invariant descriptor scalar residual; "
            "energy, force, and Hessian are analytic derivatives of one scalar"
        ),
        "architecture": (
            "one_hidden_plus_deep_residual"
            if args.deep_hidden_size > 0
            else "one_hidden"
        ),
        "source_split_sha256": metadata["source_split_sha256"],
        "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "design_provenance": _design_provenance(args.design_root),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "feature_metadata": metadata["stages"],
        "feature_hessian_mixed_partials_symmetrized": metadata[
            "feature_hessian_mixed_partials_symmetrized"
        ],
        "direction_manifest": metadata["direction_manifest"],
        "direction_manifest_sha256": metadata["direction_manifest_sha256"],
        "feature_inventory_manifest": metadata["feature_inventory_manifest"],
        "feature_inventory_manifest_sha256": metadata[
            "feature_inventory_manifest_sha256"
        ],
        "zero_linear_initialization": metadata["zero_linear_initialization"],
        "hidden_size": args.hidden_size,
        "deep_hidden_size": args.deep_hidden_size,
        "atom_count_extensive": args.atom_count_extensive,
        "seed": args.seed,
        "source_checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "best_step": best_step,
        "best_total_loss": best_total,
        "replay_cache": (
            {
                "manifest": args.replay_cache_manifest.resolve().as_posix(),
                "manifest_sha256": replay_cache.manifest_sha256,
                "schema_manifest_sha256": replay_cache.manifest[
                    "schema_manifest_sha256"
                ],
                "schema_checkpoint_sha256": replay_cache.manifest[
                    "schema_checkpoint_sha256"
                ],
                "record_count": len(replay_cache),
                "batch_size": args.replay_batch_size,
                "cpu_cache_size": args.replay_cpu_cache_size,
                "final_fixed_eval": final_replay,
            }
            if replay_cache is not None
            else None
        ),
        "loss_weights": {
            "energy": args.lambda_energy,
            "force": args.lambda_force,
            "hessian": args.lambda_hessian,
            "spectrum": args.lambda_spectrum,
            "parameter": args.lambda_parameter,
            "replay_energy": args.lambda_replay_energy,
            "replay_force": args.lambda_replay_force,
            "hvp_warmup_steps": args.hvp_warmup_steps,
            "hvp_tail_fraction": args.hvp_tail_fraction,
            "hvp_tail_weight": args.hvp_tail_weight,
        },
        "pcgrad": args.pcgrad,
        "pcgrad_max_curvature_to_replay_norm_ratio": (
            args.pcgrad_max_curvature_to_replay_norm_ratio
        ),
        "final": scalar,
        "parent_cv": parent_cv,
        "parent_cv_held_final": (
            evaluation_scalar if parent_cv is not None else None
        ),
        "parent_cv_held_distribution": (
            _parent_cv_distribution(evaluation_rows)
            if parent_cv is not None
            else None
        ),
        "per_parent": csv_rows,
        "stage1_relative_gate_passed": all(
            row["relative_frobenius"] <= args.relative_frobenius_gate
            for row in fit_rows
        ),
        "stage1_asymmetry_gate_passed": all(
            row["antisymmetric_over_symmetric_frobenius"]
            <= args.asymmetry_ratio_gate
            for row in fit_rows
        ),
        "stage1_energy_gate_passed": bool(
            np.median([row["energy_abs_error_hartree"] for row in fit_rows])
            <= args.energy_median_gate
            and max(row["energy_abs_error_hartree"] for row in fit_rows)
            <= args.energy_max_gate
        ),
        "stage1_force_gate_passed": bool(
            np.median([row["force_mae_hartree_per_bohr"] for row in fit_rows])
            <= args.force_median_gate
            and max(row["force_mae_hartree_per_bohr"] for row in fit_rows)
            <= args.force_max_gate
        ),
        "stage1_gate_thresholds": {
            "relative_frobenius_max": args.relative_frobenius_gate,
            "asymmetry_ratio_max": args.asymmetry_ratio_gate,
            "energy_abs_error_median_max_hartree": args.energy_median_gate,
            "energy_abs_error_all_parent_max_hartree": args.energy_max_gate,
            "force_mae_median_max_hartree_per_bohr": args.force_median_gate,
            "force_mae_all_parent_max_hartree_per_bohr": args.force_max_gate,
        },
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    result["stage1_shared_gate_passed"] = bool(
        result["stage1_relative_gate_passed"]
        and result["stage1_asymmetry_gate_passed"]
        and result["stage1_energy_gate_passed"]
        and result["stage1_force_gate_passed"]
    )
    if parent_cv is not None:
        held = result["parent_cv_held_distribution"]
        result["parent_cv_held_gate"] = {
            "median_relative_frobenius_le_0p10": held[
                "median_relative_frobenius"
            ]
            <= 0.10,
            "fraction_relative_frobenius_le_0p15_ge_0p80": held[
                "fraction_relative_frobenius_at_or_below_0_15"
            ]
            >= 0.80,
            "p90_relative_frobenius_le_0p20": held[
                "p90_relative_frobenius"
            ]
            <= 0.20,
            "asymmetry_max_le_0p005": held[
                "max_antisymmetric_over_symmetric_frobenius"
            ]
            <= 0.005,
            "energy_median_no_more_than_5pct_worse_than_source": held[
                "energy_median_ratio_to_source"
            ]
            <= 1.05,
            "force_median_no_more_than_5pct_worse_than_source": held[
                "force_median_ratio_to_source"
            ]
            <= 1.05,
        }
        result["parent_cv_held_gate"]["passed"] = all(
            result["parent_cv_held_gate"].values()
        )
    if args.direction_manifest is not None:
        result["stage2_training_direction_gate_passed"] = all(
            row["train_hvp_relative_frobenius"]
            <= args.training_direction_relative_gate
            for row in csv_rows
        )
        result["stage2_heldout_direction_gate_passed"] = all(
            row["heldout_hvp_relative_frobenius"]
            <= args.heldout_direction_relative_gate
            for row in csv_rows
        )
        result["stage2_full_hessian_gate_passed"] = bool(
            scalar["median_relative_frobenius"]
            <= args.full_hessian_median_relative_gate
        )
        result["stage2_direction_generalization_gate_passed"] = bool(
            result["stage2_training_direction_gate_passed"]
            and result["stage2_heldout_direction_gate_passed"]
            and result["stage2_full_hessian_gate_passed"]
            and result["stage1_asymmetry_gate_passed"]
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--feature-inventory-manifest", type=Path)
    parser.add_argument("--initial-root", type=Path)
    parser.add_argument("--zero-linear-initialization", action="store_true")
    parser.add_argument("--direction-manifest", type=Path)
    parser.add_argument("--parent-cv-protocol", type=Path)
    parser.add_argument("--parent-cv-variant-id")
    parser.add_argument("--parent-cv-fold-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--replay-cache-manifest", type=Path)
    parser.add_argument("--replay-batch-size", type=int, default=0)
    parser.add_argument("--replay-eval-size", type=int, default=32)
    parser.add_argument("--replay-cpu-cache-size", type=int, default=256)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument(
        "--require-replay-schema-checkpoint-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--deep-hidden-size", type=int, default=0)
    parser.add_argument("--atom-count-extensive", action="store_true")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--energy-scale", type=float, default=0.1)
    parser.add_argument("--force-scale", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=1.0)
    parser.add_argument("--lambda-hessian", type=float, default=1.0)
    parser.add_argument("--lambda-spectrum", type=float, default=0.0)
    parser.add_argument("--lambda-parameter", type=float, default=1.0e-8)
    parser.add_argument("--lambda-replay-energy", type=float, default=0.0)
    parser.add_argument("--lambda-replay-force", type=float, default=0.0)
    parser.add_argument("--hvp-warmup-steps", type=int, default=0)
    parser.add_argument("--pcgrad", action="store_true")
    parser.add_argument(
        "--pcgrad-max-curvature-to-replay-norm-ratio",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("--column-norm-relative-cutoff", type=float, default=1.0e-8)
    parser.add_argument(
        "--feature-scale-mode",
        choices=("column_norm", "floored_column_norm", "unit"),
        default="column_norm",
    )
    parser.add_argument("--feature-scale-floor", type=float, default=0.0)
    parser.add_argument("--norm-chunk-rows", type=int, default=1024)
    parser.add_argument("--absolute-hessian-scale", type=float, default=0.1)
    parser.add_argument("--relative-loss-fraction", type=float, default=0.5)
    parser.add_argument("--directions-per-parent", type=int, default=4)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-loss-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    parser.add_argument("--hvp-tail-fraction", type=float, default=0.2)
    parser.add_argument("--hvp-tail-weight", type=float, default=0.0)
    parser.add_argument("--spectrum-reference-floor", type=float, default=0.01)
    parser.add_argument("--wrong-curvature-multiplier", type=float, default=2.0)
    parser.add_argument("--training-direction-relative-gate", type=float, default=0.05)
    parser.add_argument("--heldout-direction-relative-gate", type=float, default=0.15)
    parser.add_argument("--full-hessian-median-relative-gate", type=float, default=0.15)
    parser.add_argument("--relative-frobenius-gate", type=float, default=0.05)
    parser.add_argument("--asymmetry-ratio-gate", type=float, default=0.005)
    parser.add_argument("--energy-median-gate", type=float, default=float("inf"))
    parser.add_argument("--energy-max-gate", type=float, default=float("inf"))
    parser.add_argument("--force-median-gate", type=float, default=float("inf"))
    parser.add_argument("--force-max-gate", type=float, default=float("inf"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
