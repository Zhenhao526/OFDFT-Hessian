#!/usr/bin/env python3
"""Measure correction amplitude and direction at a scalar MACE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        _build_model_and_optimizer,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )
    from scripts.qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )
except ModuleNotFoundError:
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        _build_model_and_optimizer,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _sha256,
    )
    from qm9_complete_total_local_scalar_jacobian_range_audit import (
        select_audit_parents,
    )


def correction_geometry(
    predicted: np.ndarray, source: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    target = reference - source
    correction = predicted - source
    target_norm = float(np.linalg.norm(target))
    correction_norm = float(np.linalg.norm(correction))
    denominator = max(target_norm * correction_norm, np.finfo(float).tiny)
    return {
        "target_correction_frobenius": target_norm,
        "learned_correction_frobenius": correction_norm,
        "learned_over_target_norm": correction_norm
        / max(target_norm, np.finfo(float).tiny),
        "learned_target_cosine": float(np.sum(correction * target) / denominator),
        "residual_frobenius": float(np.linalg.norm(predicted - reference)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "formal")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    parents = select_audit_parents(parents, args.parent_id)
    model, _, architecture = _build_model_and_optimizer(arm, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("protocol_sha256") != _sha256(args.protocol):
        raise ValueError("checkpoint protocol hash drift")
    if checkpoint.get("arm_id") != args.arm_id:
        raise ValueError("checkpoint arm drift")
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not certify frozen Test100")
    selected = [parent.molecule_id for parent in parents]
    if [str(value) for value in checkpoint.get("selected_parent_ids", [])] != selected:
        raise ValueError("checkpoint parent list drift")
    model.load_state_dict(checkpoint["state_dict"])
    evaluation, rows, arrays = _evaluate(
        model, parents, protocol["capacity_gate"], anchor_energy_force=False
    )
    per_parent = []
    for row in rows:
        molecule_id = str(row["molecule_id"])
        payload = arrays[molecule_id]
        per_parent.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(row["natoms"]),
                "hessian_relative_frobenius": float(row["relative_frobenius"]),
                **correction_geometry(
                    payload["predicted_hessian"],
                    payload["source_hessian_symmetric"],
                    payload["pbe_hessian"],
                ),
            }
        )
    result = {
        "definition": "Geometry of the learned scalar-Hessian correction relative to the PBE-minus-source correction.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "arm_id": args.arm_id,
        "architecture": architecture,
        "evaluation": evaluation,
        "per_parent": per_parent,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "opened_label_paths": provenance["opened_label_paths"],
        "opened_capacity_arrays": provenance["opened_capacity_arrays"],
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
