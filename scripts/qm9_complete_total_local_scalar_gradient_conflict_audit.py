#!/usr/bin/env python3
"""Audit separate energy, force, and Hessian parameter gradients on stable5."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
    _build_model_and_optimizer,
    _evaluate,
    _load_protocol,
    _load_selected_parents,
    _parent_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_gradient_statistics(
    task_losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> dict[str, Any]:
    gradients = {
        name: torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for name, loss in task_losses.items()
    }
    norms: dict[str, torch.Tensor] = {}
    for name, task_gradients in gradients.items():
        norm_squared = task_losses[name].new_zeros(())
        for gradient in task_gradients:
            if gradient is not None:
                norm_squared = norm_squared + torch.sum(gradient * gradient)
        norms[name] = torch.sqrt(
            norm_squared.clamp_min(torch.finfo(norm_squared.dtype).tiny)
        )

    pairwise = {}
    names = list(task_losses)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            dot = task_losses[first_name].new_zeros(())
            for first_gradient, second_gradient in zip(
                gradients[first_name], gradients[second_name], strict=True
            ):
                if first_gradient is not None and second_gradient is not None:
                    dot = dot + torch.sum(first_gradient * second_gradient)
            denominator = (norms[first_name] * norms[second_name]).clamp_min(
                torch.finfo(dot.dtype).tiny
            )
            pairwise[f"{first_name}_vs_{second_name}"] = {
                "dot": float(dot.detach().cpu()),
                "cosine": float((dot / denominator).detach().cpu()),
                "conflict": bool((dot < 0).detach().cpu()),
            }
    return {
        "losses": {
            name: float(loss.detach().cpu()) for name, loss in task_losses.items()
        },
        "gradient_norms": {
            name: float(norm.detach().cpu()) for name, norm in norms.items()
        },
        "pairwise": pairwise,
    }


def _parent_task_losses(model, parent, training: dict[str, Any]) -> dict[str, torch.Tensor]:
    _, terms = _parent_loss(model, parent, training)
    return {
        "energy": float(training["lambda_energy"]) * terms["energy_loss"],
        "force": float(training["lambda_force"]) * terms["force_loss"],
        "hessian": float(training["lambda_hessian"]) * terms["hessian_loss"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "formal")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    model, _, architecture = _build_model_and_optimizer(arm, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("protocol_sha256") != _sha256(args.protocol):
        raise ValueError("checkpoint protocol hash drift")
    if checkpoint.get("arm_id") != args.arm_id:
        raise ValueError("checkpoint arm mismatch")
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not certify frozen Test100")
    model.load_state_dict(checkpoint["state_dict"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    training = protocol["training"]

    per_parent = []
    aggregate_terms: dict[str, list[torch.Tensor]] = {
        "energy": [],
        "force": [],
        "hessian": [],
    }
    for parent in parents:
        tasks = _parent_task_losses(model, parent, training)
        stats = task_gradient_statistics(tasks, parameters)
        per_parent.append({"molecule_id": parent.molecule_id, **stats})
        for name, loss in tasks.items():
            aggregate_terms[name].append(loss)

    aggregate_tasks = {
        name: torch.mean(torch.stack(losses))
        for name, losses in aggregate_terms.items()
    }
    aggregate = task_gradient_statistics(aggregate_tasks, parameters)
    del aggregate_tasks, aggregate_terms
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final, _, _ = _evaluate(
        model,
        parents,
        protocol["capacity_gate"],
        anchor_energy_force=bool(training.get("anchor_energy_force", True)),
    )
    result = {
        "definition": "Separate stable5 parameter-gradient audit for scalar-derived E/F/H; read-only best checkpoint evaluation.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "architecture": architecture,
        "aggregate": aggregate,
        "per_parent": per_parent,
        "checkpoint_metrics": final,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
