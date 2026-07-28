#!/usr/bin/env python3
"""Run deterministic full-batch LBFGS as a stable5 scalar-capacity ceiling audit."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
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
        _parent_loss,
        _periodic_evaluation_metrics,
        _sha256,
        _write_rows,
    )
except ModuleNotFoundError:
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        _build_model_and_optimizer,
        _evaluate,
        _load_protocol,
        _load_selected_parents,
        _parent_loss,
        _periodic_evaluation_metrics,
        _sha256,
        _write_rows,
    )


SUPPORTED_AUDIT_PROTOCOLS = {
    "qm9_complete_total_local_scalar_post_v4_diagnostics_v1",
    "qm9_complete_total_bounded_equivariant_lbfgs_v1",
    "qm9_complete_total_mace_lbfgs_v1",
}

STABLE5_AUDIT_PROTOCOLS = {
    "qm9_complete_total_bounded_equivariant_lbfgs_v1",
    "qm9_complete_total_mace_lbfgs_v1",
}


def ceiling_objective(
    terms: dict[str, torch.Tensor],
    training: dict[str, Any],
    objective: str,
) -> torch.Tensor:
    hessian = float(training["lambda_hessian"]) * terms["hessian_loss"]
    regularization = float(training["lambda_parameter"]) * terms["parameter_loss"]
    if objective == "hessian_only":
        return hessian + regularization
    if objective == "joint":
        return (
            float(training["lambda_energy"]) * terms["energy_loss"]
            + float(training["lambda_force"]) * terms["force_loss"]
            + hessian
            + regularization
        )
    raise ValueError(f"unsupported LBFGS ceiling objective: {objective}")


def ceiling_selection_score(final: dict[str, Any], objective: str) -> float:
    if objective == "joint":
        return float(final["selection_score"])
    if objective == "hessian_only":
        distribution = final["hessian_relative_frobenius"]
        return float(distribution["median"] + distribution["max"])
    raise ValueError(f"unsupported LBFGS ceiling objective: {objective}")


def validate_audit_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    manifest = json.loads(args.audit_manifest.read_text())
    protocol_id = manifest.get("protocol_id")
    if protocol_id not in SUPPORTED_AUDIT_PROTOCOLS:
        raise ValueError("unexpected LBFGS audit protocol")
    if protocol_id in STABLE5_AUDIT_PROTOCOLS:
        parent_scope = manifest.get("parent_scope")
        if parent_scope not in {"stable5_fit_only", "stable5_one_parent_diagnostic"}:
            raise ValueError("scalar LBFGS audit is not stable5-only")
        if manifest.get("validation_accessed") is not False:
            raise ValueError("scalar LBFGS audit does not freeze validation")
        parent_id = getattr(args, "parent_id", None)
        if parent_scope == "stable5_fit_only" and parent_id is not None:
            raise ValueError("full-stable5 LBFGS audit cannot select one parent")
        if parent_scope == "stable5_one_parent_diagnostic":
            allowed = [str(value) for value in manifest.get("diagnostic_parent_ids", [])]
            if parent_id is None or str(parent_id) not in allowed:
                raise ValueError("one-parent LBFGS diagnostic parent is not frozen")
    if manifest.get("test100_accessed") is not False:
        raise ValueError("LBFGS audit protocol does not freeze Test100")
    if int(manifest.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("LBFGS audit protocol Test100 count is nonzero")
    expected = {
        "source_protocol": args.source_protocol.resolve().as_posix(),
        "source_protocol_sha256": _sha256(args.source_protocol),
        "source_arm_id": args.source_arm_id,
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"LBFGS audit manifest {key} mismatch")
    configuration = manifest.get("lbfgs", {}).get(args.objective)
    if not isinstance(configuration, dict):
        raise ValueError("LBFGS objective is not frozen in audit manifest")
    numeric = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "history_size": args.history_size,
        "max_evaluations_per_iteration": args.max_evaluations_per_iteration,
        "tolerance_grad": args.tolerance_grad,
        "tolerance_change": args.tolerance_change,
        "log_interval": args.log_interval,
    }
    for key, value in numeric.items():
        if float(configuration.get(key, math.nan)) != float(value):
            raise ValueError(f"LBFGS audit manifest {key} mismatch")
    if args.objective == "hessian_only" and bool(
        configuration.get("parent_cv_authorization_allowed")
    ):
        raise ValueError("Hessian-only audit cannot authorize parent-CV")
    if manifest.get("parent_scope") == "stable5_one_parent_diagnostic" and bool(
        configuration.get("parent_cv_authorization_allowed")
    ):
        raise ValueError("one-parent LBFGS diagnostic cannot authorize parent-CV")
    return manifest, _sha256(args.audit_manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    audit_manifest, audit_manifest_sha256 = validate_audit_manifest(args)
    protocol, arm = _load_protocol(args.source_protocol, args.source_arm_id, "formal")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    if args.parent_id is not None:
        parents = [parent for parent in parents if parent.molecule_id == args.parent_id]
        if len(parents) != 1:
            raise ValueError("requested diagnostic parent is absent from stable5")
    model, _, architecture = _build_model_and_optimizer(arm, device)
    source = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    if source.get("protocol_sha256") != _sha256(args.source_protocol):
        raise ValueError("source checkpoint protocol hash drift")
    if source.get("arm_id") != args.source_arm_id:
        raise ValueError("source checkpoint arm mismatch")
    if source.get("test100_accessed") is not False:
        raise ValueError("source checkpoint does not certify frozen Test100")
    model.load_state_dict(source["state_dict"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=args.learning_rate,
        max_iter=1,
        max_eval=args.max_evaluations_per_iteration,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
    )
    training = protocol["training"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    best_score = math.inf
    best_iteration = -1
    closure_calls = 0
    latest_terms: dict[str, float] = {}

    with metrics_path.open("w") as handle:
        for iteration in range(args.iterations + 1):
            if iteration % args.log_interval == 0 or iteration == args.iterations:
                final, rows, arrays = _evaluate(
                    model,
                    parents,
                    protocol["capacity_gate"],
                    anchor_energy_force=bool(
                        training.get("anchor_energy_force", True)
                    ),
                )
                selection_score = ceiling_selection_score(final, args.objective)
                row = {
                    "iteration": iteration,
                    "wall_time_s": time.perf_counter() - started,
                    "closure_calls": closure_calls,
                    "lbfgs_selection_score": selection_score,
                    "objective": args.objective,
                    "diagnostic_parent_id": args.parent_id,
                    "fit_parent_ids": [parent.molecule_id for parent in parents],
                    **{f"latest_{key}": value for key, value in latest_terms.items()},
                    **_periodic_evaluation_metrics(final),
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                if not all(
                    math.isfinite(float(value))
                    for value in row.values()
                    if isinstance(value, (int, float))
                ):
                    raise FloatingPointError("non-finite LBFGS capacity metric")
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                checkpoint = {
                    "iteration": iteration,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "source_protocol": args.source_protocol.resolve().as_posix(),
                    "source_protocol_sha256": _sha256(args.source_protocol),
                    "source_arm_id": args.source_arm_id,
                    "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
                    "source_checkpoint_sha256": _sha256(args.source_checkpoint),
                    "objective": args.objective,
                    "audit_manifest": args.audit_manifest.resolve().as_posix(),
                    "audit_manifest_sha256": audit_manifest_sha256,
                    "test100_accessed": False,
                    "test100_evaluations_used": 0,
                    **provenance,
                }
                torch.save(checkpoint, args.output_dir / "last.ckpt")
                if selection_score < best_score:
                    best_score = selection_score
                    best_iteration = iteration
                    torch.save(checkpoint, args.output_dir / "best.ckpt")
                    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
                    for molecule_id, payload in arrays.items():
                        np.savez_compressed(
                            args.output_dir / f"{molecule_id}_result.npz", **payload
                        )
            if iteration == args.iterations:
                break

            def closure() -> torch.Tensor:
                nonlocal closure_calls, latest_terms
                optimizer.zero_grad(set_to_none=True)
                losses = [_parent_loss(model, parent, training) for parent in parents]
                terms = {
                    key: torch.mean(torch.stack([value[1][key] for value in losses]))
                    for key in losses[0][1]
                }
                objective = ceiling_objective(terms, training, args.objective)
                if not bool(torch.isfinite(objective).detach().cpu()):
                    raise FloatingPointError("non-finite LBFGS objective")
                objective.backward()
                closure_calls += 1
                latest_terms = {
                    key: float(value.detach().cpu()) for key, value in terms.items()
                }
                latest_terms["objective"] = float(objective.detach().cpu())
                return objective

            optimizer.step(closure)

    best = torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    final, rows, arrays = _evaluate(
        model,
        parents,
        protocol["capacity_gate"],
        anchor_energy_force=bool(training.get("anchor_energy_force", True)),
    )
    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
    for molecule_id, payload in arrays.items():
        np.savez_compressed(args.output_dir / f"{molecule_id}_result.npz", **payload)
    result = {
        "definition": (
            "Deterministic LBFGS scalar-capacity audit over full stable5 or one explicitly "
            "registered stable5 diagnostic parent. A one-parent or Hessian-only result can "
            "never authorize parent-CV."
        ),
        "objective": args.objective,
        "diagnostic_parent_id": args.parent_id,
        "fit_parent_ids": [parent.molecule_id for parent in parents],
        "source_protocol": args.source_protocol.resolve().as_posix(),
        "source_protocol_sha256": _sha256(args.source_protocol),
        "source_arm_id": args.source_arm_id,
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "audit_manifest": args.audit_manifest.resolve().as_posix(),
        "audit_manifest_sha256": audit_manifest_sha256,
        "audit_protocol_id": audit_manifest["protocol_id"],
        "architecture": architecture,
        "iterations": args.iterations,
        "closure_calls": closure_calls,
        "best_iteration": best_iteration,
        "best_selection_score": best_score,
        "final": final,
        "per_parent": rows,
        "capacity_gate_passed": bool(final["gate"]["passed"]),
        "shared_stable5_capacity_gate_passed": bool(
            args.parent_id is None and final["gate"]["passed"]
        ),
        "parent_cv_authorized": bool(
            args.parent_id is None
            and args.objective == "joint"
            and final["gate"]["passed"]
        ),
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
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-arm-id", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-id")
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--objective", choices=("joint", "hessian_only"), required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=25)
    parser.add_argument("--max-evaluations-per-iteration", type=int, default=5)
    parser.add_argument("--tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--tolerance-change", type=float, default=1e-12)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iterations <= 0 or args.log_interval <= 0:
        parser.error("iterations and log interval must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
