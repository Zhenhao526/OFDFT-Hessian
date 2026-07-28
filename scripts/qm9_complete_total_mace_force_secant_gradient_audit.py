#!/usr/bin/env python3
"""Audit E/F/HVP parameter-gradient scales for a frozen scalar MACE checkpoint."""

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

from mldft.ml.models.components.local_mace_scalar_residual import LocalMACEScalarResidual

try:
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        _build_model_and_optimizer,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )
    from scripts.qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )
    from scripts.qm9_complete_total_mace_force_secant_capacity import (
        _parent_loss_multi_direction,
        accumulated_full_basis_task_gradients,
    )
except ModuleNotFoundError:
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        _build_model_and_optimizer,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )
    from qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )
    from qm9_complete_total_mace_force_secant_capacity import (
        _parent_loss_multi_direction,
        accumulated_full_basis_task_gradients,
    )


def parse_direction_groups(value: str) -> list[list[int]]:
    groups = [
        [int(index) for index in group.split(",") if index]
        for group in value.split(";")
        if group
    ]
    if not groups or any(not group or min(group) < 0 for group in groups):
        raise ValueError("direction groups must contain nonnegative indices")
    if any(len(set(group)) != len(group) for group in groups):
        raise ValueError("direction indices within a group must be unique")
    return groups


def flatten_task_gradient(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= torch.finfo(first.dtype).tiny:
        return 0.0
    return float((torch.dot(first, second) / denominator).detach().cpu())


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "formal")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    parents = select_audit_parents(parents, args.parent_id)
    model, _, architecture = _build_model_and_optimizer(arm, device)
    if not isinstance(model, LocalMACEScalarResidual):
        raise ValueError("gradient audit is restricted to scalar MACE")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("protocol_sha256") != _sha256(args.protocol):
        raise ValueError("checkpoint/protocol hash mismatch")
    if checkpoint.get("arm_id") != args.arm_id:
        raise ValueError("checkpoint arm mismatch")
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not freeze Test100")
    model.load_state_dict(checkpoint["state_dict"])
    model.train()
    parameters = model.trainable_parameters()
    training = protocol["training"]
    rows = []
    for parent in parents:
        direction_groups = (
            [list(range(parent.positions.numel()))]
            if args.accumulated_full_basis
            else args.direction_groups
        )
        for direction_indices in direction_groups:
            if max(direction_indices) >= parent.positions.numel():
                raise ValueError("direction index exceeds parent coordinate count")
            if args.accumulated_full_basis:
                gradients, terms = accumulated_full_basis_task_gradients(
                    model, parent, parameters, training
                )
                tasks = {
                    "energy": float(training["lambda_energy"]) * terms["energy_loss"],
                    "force": float(training["lambda_force"]) * terms["force_loss"],
                    "hvp": float(training["lambda_hvp"]) * terms["hvp_loss"],
                }
            else:
                _, terms = _parent_loss_multi_direction(
                    model, parent, direction_indices, training
                )
                tasks = {
                    "energy": float(training["lambda_energy"]) * terms["energy_loss"],
                    "force": float(training["lambda_force"]) * terms["force_loss"],
                    "hvp": float(training["lambda_hvp"]) * terms["hvp_loss"],
                }
                gradients = {
                    name: flatten_task_gradient(loss, parameters, retain_graph=True).detach()
                    for name, loss in tasks.items()
                }
            norms = {
                name: float(torch.linalg.vector_norm(value).cpu())
                for name, value in gradients.items()
            }
            rows.append(
                {
                    "molecule_id": parent.molecule_id,
                    "direction_indices": ",".join(map(str, direction_indices)),
                    "direction_count": len(direction_indices),
                    "energy_loss": float(tasks["energy"].detach().cpu()),
                    "force_loss": float(tasks["force"].detach().cpu()),
                    "hvp_loss": float(tasks["hvp"].detach().cpu()),
                    "energy_gradient_norm": norms["energy"],
                    "force_gradient_norm": norms["force"],
                    "hvp_gradient_norm": norms["hvp"],
                    "cosine_energy_force": gradient_cosine(
                        gradients["energy"], gradients["force"]
                    ),
                    "cosine_energy_hvp": gradient_cosine(
                        gradients["energy"], gradients["hvp"]
                    ),
                    "cosine_force_hvp": gradient_cosine(
                        gradients["force"], gradients["hvp"]
                    ),
                }
            )
            del gradients, tasks, terms
            if device.type == "cuda":
                torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with (args.output_dir / "per_direction_gradients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    numeric_keys = [
        "energy_gradient_norm",
        "force_gradient_norm",
        "hvp_gradient_norm",
        "cosine_energy_force",
        "cosine_energy_hvp",
        "cosine_force_hvp",
    ]
    aggregate = {
        key: {
            "median": float(np.median([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in numeric_keys
    }
    result = {
        "definition": "Frozen-checkpoint parameter gradients of normalized scalar energy, scalar-derived force, and scalar-force-secant HVP tasks.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "arm_id": args.arm_id,
        "architecture": architecture,
        "selected_parent_ids": [parent.molecule_id for parent in parents],
        "direction_groups": args.direction_groups,
        "accumulated_full_basis": args.accumulated_full_basis,
        "aggregate": aggregate,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--direction-indices", default="0,6,12,18,24,30,36,42")
    parser.add_argument("--direction-groups")
    parser.add_argument("--accumulated-full-basis", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        if args.direction_groups:
            args.direction_groups = parse_direction_groups(args.direction_groups)
        else:
            args.direction_groups = [
                [int(value)]
                for value in args.direction_indices.split(",")
                if value
            ]
            if not args.direction_groups or min(group[0] for group in args.direction_groups) < 0:
                raise ValueError("direction indices must be nonnegative")
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    run(parse_args())
