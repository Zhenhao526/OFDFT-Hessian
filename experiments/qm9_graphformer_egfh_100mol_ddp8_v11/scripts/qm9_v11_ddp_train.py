#!/usr/bin/env python3
"""Float64 DDP8 QM9 E/G/F/H trainer with explicit high-order gradient reduction.

This v11 entry point deliberately reuses the audited scalar-energy, density/KKT
response and Torch-autograd DQC/libcint primitives from the canonical v10
runner.  There are no independent force or Hessian heads.  DDP wraps the
Graphformer, while parameter gradients produced by ``torch.autograd.grad`` are
explicitly SUM-all-reduced in float64 and divided by WORLD_SIZE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

import qm9_complete_total_capacity_train as core


COMPONENTS = ("energy", "gradient", "force", "hessian")
REQUIRED_CANONICAL_NAMESPACE_FIELDS = frozenset(
    {
        "analytic_response_constraint_tolerance",
        "analytic_response_damping",
        "analytic_response_residual_tolerance",
        "center_electron_number_residual_max",
        "charge",
        "connect_lagrange_multiplier_response",
        "convergence_tolerance",
        "density_response_unroll_lr",
        "density_response_unroll_steps",
        "density_strict_threshold",
        "implicit_density_parameter_response",
        "implicit_response_damping",
        "implicit_response_diagonal_probes",
        "implicit_response_max_iterations",
        "implicit_response_solver",
        "implicit_response_tolerance",
        "implicit_response_warm_start",
        "initialization",
        "integral_cache_entries",
        "integral_derivative_backend",
        "integral_derivative_step",
        "integral_derivative_workers",
        "integral_directional_second_step",
        "lbfgs_history_size",
        "lbfgs_max_iterations",
        "lbfgs_refine",
        "lbfgs_tolerance",
        "lr",
        "max_cycle",
        "model_geometry_derivative",
        "model_geometry_fd_richardson",
        "model_geometry_fd_step",
        "momentum",
        "negative_integrated_density_penalty_weight",
        "newton_damping",
        "newton_diagonal_probes",
        "newton_krylov_tolerance",
        "newton_max_iterations",
        "newton_max_krylov_iterations",
        "newton_refine",
        "newton_tolerance",
        "optimizer",
        "output_dir",
        "transform_device",
        "fallback_max_cycle",
        "fallback_convergence_tolerance",
    }
)
EXPECTED_CANONICAL_RUNTIME_BINDINGS = {
    "connect_lagrange_multiplier_response": True,
    "density_response_unroll_steps": 0,
    "density_response_unroll_lr": 1.0e-3,
    "implicit_density_parameter_response": True,
    "implicit_response_tolerance": 3.0e-5,
    "implicit_response_max_iterations": 2500,
    "implicit_response_damping": 1.0e-8,
    "implicit_response_diagonal_probes": 0,
    "implicit_response_solver": "direct",
    "implicit_response_warm_start": True,
}

if len(REQUIRED_CANONICAL_NAMESPACE_FIELDS) != 46:
    raise RuntimeError("canonical namespace registry must contain exactly 46 fields")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def distributed_identity() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if enabled:
        dist.init_process_group(backend="nccl")
        if world_size != 8:
            raise RuntimeError(f"v11 requires WORLD_SIZE=8, found {world_size}")
    return enabled, rank, local_rank, world_size


def all_reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(world_size)
    return result


def reduce_optional_gradients(
    local: list[torch.Tensor | None],
    parameters: list[torch.nn.Parameter],
    world_size: int,
) -> list[torch.Tensor | None]:
    reduced: list[torch.Tensor | None] = []
    for gradient, parameter in zip(local, parameters, strict=True):
        used = torch.tensor(
            0 if gradient is None else 1,
            dtype=torch.int64,
            device=parameter.device,
        )
        if world_size > 1:
            dist.all_reduce(used, op=dist.ReduceOp.SUM)
        if int(used.item()) == 0:
            reduced.append(None)
            continue
        value = (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient.detach().clone()
        )
        if world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(world_size)
        reduced.append(value)
    return reduced


def gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(torch.sum(gradient.detach() ** 2).cpu())
    return math.sqrt(total)


def gradient_dot(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None]
) -> float:
    total = 0.0
    for lhs, rhs in zip(left, right, strict=True):
        if lhs is not None and rhs is not None:
            total += float(torch.sum(lhs.detach() * rhs.detach()).cpu())
    return total


def gradient_cosine(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None]
) -> float:
    denominator = gradient_norm(left) * gradient_norm(right)
    return gradient_dot(left, right) / denominator if denominator else math.nan


def add_gradients(
    collections: list[list[torch.Tensor | None]],
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    result: list[torch.Tensor | None] = []
    for index, parameter in enumerate(parameters):
        values = [item[index] for item in collections if item[index] is not None]
        result.append(None if not values else sum(values, torch.zeros_like(parameter)))
    return result


def accumulate_gradients(
    accumulator: list[torch.Tensor | None],
    incoming: list[torch.Tensor | None],
    scale: float,
) -> None:
    for index, gradient in enumerate(incoming):
        if gradient is None:
            continue
        cpu_value = gradient.detach().cpu() * scale
        if accumulator[index] is None:
            accumulator[index] = cpu_value.clone()
        else:
            accumulator[index].add_(cpu_value)


def density_namespace(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> argparse.Namespace:
    registration = protocol["canonical_runtime_namespace"]
    runtime = registration["fields"]
    if int(registration["required_field_count"]) != 46:
        raise RuntimeError("protocol canonical namespace field-count drift")
    if runtime != EXPECTED_CANONICAL_RUNTIME_BINDINGS:
        raise RuntimeError("protocol canonical runtime binding drift")
    source_summary = Path(registration["source_summary"])
    if sha256(source_summary) != registration["source_summary_sha256"]:
        raise RuntimeError("canonical runtime value-source SHA256 mismatch")
    audit_report = Path(registration["audit_report"])
    if sha256(audit_report) != registration["audit_report_sha256"]:
        raise RuntimeError("canonical namespace audit-report SHA256 mismatch")
    values = vars(args).copy()
    values.update(
        {
            "initialization": "label_reference",
            "optimizer": "adam",
            "lr": 1.0e-3,
            "max_cycle": 1000,
            "convergence_tolerance": 1.0e-2,
            "momentum": 0.9,
            "fallback_optimizer": "adam",
            "fallback_lr": 3.0e-4,
            "fallback_max_cycle": 10000,
            "fallback_convergence_tolerance": 1.0e-5,
            "fallback_always": True,
            "lbfgs_refine": True,
            "lbfgs_tolerance": args.density_solver_target,
            "lbfgs_max_iterations": 500,
            "lbfgs_history_size": 50,
            "newton_refine": True,
            "newton_tolerance": args.density_solver_target,
            "newton_max_iterations": 6,
            "newton_max_krylov_iterations": 200,
            "newton_krylov_tolerance": 1.0e-10,
            "newton_diagonal_probes": 8,
            "newton_damping": 1.0e-8,
            # Bind the v11 protocol's strict density-solver target to the
            # canonical runner field consumed by _relax.  This is the same
            # physical gate, not a relaxed fallback threshold.
            "density_strict_threshold": args.density_solver_target,
            "connect_lagrange_multiplier_response": runtime[
                "connect_lagrange_multiplier_response"
            ],
            "density_response_unroll_steps": runtime[
                "density_response_unroll_steps"
            ],
            "density_response_unroll_lr": runtime["density_response_unroll_lr"],
            "implicit_density_parameter_response": runtime[
                "implicit_density_parameter_response"
            ],
            "implicit_response_tolerance": runtime[
                "implicit_response_tolerance"
            ],
            "implicit_response_max_iterations": runtime[
                "implicit_response_max_iterations"
            ],
            "implicit_response_damping": runtime["implicit_response_damping"],
            "implicit_response_diagonal_probes": runtime[
                "implicit_response_diagonal_probes"
            ],
            "implicit_response_solver": runtime["implicit_response_solver"],
            "implicit_response_warm_start": runtime[
                "implicit_response_warm_start"
            ],
            "response_predictor_fast_refine": False,
            "response_predictor_fast_refine_threshold": 5.0e-6,
            "negative_integrated_density_penalty_weight": 0.0,
            "integral_derivative_backend": "torch_autograd_dqc",
            "integral_derivative_step": 0.0,
            "integral_directional_second_step": 0.0,
            "max_xc_memory": 4000,
            "normalize_initial_guess": True,
            "ks_basis": "sto-3g",
            "model_geometry_derivative": "autograd",
            "model_geometry_fd_step": None,
            "model_geometry_fd_richardson": False,
            "optimization_trace_dir": args.output_dir / "density_traces",
            "coefficients_dir": args.output_dir / "density_coefficients",
        }
    )
    return argparse.Namespace(**values)


def load_local_molecules(
    direction_manifest: dict[str, Any],
    molecule_ids: list[str],
    protocol: dict[str, Any],
) -> list[core.MoleculeState]:
    by_id = {
        str(row["molecule_id"]): row for row in direction_manifest["parents"]
    }
    missing = sorted(set(molecule_ids).difference(by_id))
    if missing:
        raise RuntimeError(f"direction manifest lacks molecules: {missing}")
    molecules = [core._load_molecule(by_id[mid], 0, by_id[mid], "all") for mid in molecule_ids]
    for molecule in molecules:
        core._symmetrize_canonical_reference_hessian(
            molecule,
            antisymmetric_over_symmetric_frobenius_max=float(
                protocol["gates"][
                    "reference_hessian_antisymmetric_over_symmetric_fro_max"
                ]
            ),
        )
        core._assert_canonical_internal_basis(molecule, protocol["directions"])
    return molecules


def component_losses(
    context: Any,
    cache: core.IntegralBundleCache,
    molecule: core.MoleculeState,
    protocol: dict[str, Any],
    update_index: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if molecule.base is None:
        raise RuntimeError(f"missing strict center density for {molecule.molecule_id}")
    loss_config = protocol["loss"]
    replay = core._evaluate_point_graph(
        context,
        cache,
        molecule,
        core._label_density_replay_point(molecule),
        create_graph=True,
        attach_density_parameter_response=False,
    )
    canonical = core._canonical_structures25_replay_terms(replay, molecule, loss_config)
    core._assert_canonical_replay_closures(
        canonical,
        protocol["canonical_replay_closure_gates"],
        molecule.molecule_id,
    )
    direction = core._hutchinson_direction(
        molecule, step=update_index, seed=seed
    )
    hvp, density_norms, response, center = (
        core._analytic_relaxed_direction_prediction_with_center(
            context, cache, molecule, direction, create_graph=True
        )
    )
    response_fraction = float(response["response_correction_fraction_of_relaxed_norm"])
    cancellation = float(response["cancellation_index"])
    if response_fraction > float(protocol["gates"]["response_correction_fraction_max"]):
        raise RuntimeError("response correction fraction gate failed")
    if cancellation > float(protocol["gates"]["cancellation_index_max"]):
        raise RuntimeError("response cancellation gate failed")
    maximum_density = max(float(value.detach().cpu()) for value in density_norms)
    if maximum_density > float(protocol["gates"]["center_projected_density_gradient_max"]):
        raise RuntimeError(
            f"density stationarity gate failed for {molecule.molecule_id}: {maximum_density}"
        )
    target_hvp = torch.as_tensor(
        direction.target_hvp, dtype=hvp.dtype, device=hvp.device
    )
    internal_basis = torch.as_tensor(
        np.stack(
            [
                item.vector.reshape(-1)
                for item in sorted(molecule.directions, key=lambda item: item.index)
            ]
        ),
        dtype=hvp.dtype,
        device=hvp.device,
    )
    hessian_loss = (
        core.hutchinson_internal_projected_frobenius_squared_loss(
            hvp,
            target_hvp,
            internal_basis,
            reduction="mean_internal_matrix",
        )
        / float(loss_config["hvp"]["absolute_scale_hartree_per_bohr2"]) ** 2
    )
    force_loss = core.mixed_absolute_relative_l1(
        center.force,
        torch.as_tensor(
            molecule.pbe_force,
            dtype=center.force.dtype,
            device=center.force.device,
        ),
        absolute_scale=float(loss_config["force"]["absolute_scale_hartree_per_bohr"]),
        relative_floor=float(
            loss_config["force"]["relative_rms_floor_hartree_per_bohr"]
        ),
        relative_fraction=float(loss_config["force"]["relative_fraction"]),
    )
    losses = {
        "energy": canonical["energy_loss"],
        "gradient": canonical["gradient_loss"],
        "force": force_loss,
        "hessian": hessian_loss,
    }
    diagnostics = {
        "molecule_id": molecule.molecule_id,
        "hvp_direction_index": int(direction.index),
        "density_projected_gradient_norm": maximum_density,
        "response_stationarity_residual": float(response["response_stationarity_residual"]),
        "response_constraint_residual": float(response["response_constraint_residual"]),
        "center_electron_number_residual_abs": float(
            response["center_electron_number_residual_abs"]
        ),
        "response_correction_fraction": response_fraction,
        "cancellation_index": cancellation,
        "center_graph_seconds": float(response["center_graph_seconds"]),
        "kkt_solve_seconds": float(response["kkt_solve_seconds"]),
        "relaxed_hvp_seconds": float(response["relaxed_hvp_seconds"]),
    }
    return losses, diagnostics


def collect_rng_state(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def checkpoint(
    args: argparse.Namespace,
    context: Any,
    bare_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    local_molecules: list[core.MoleculeState],
    rank: int,
    world_size: int,
    global_step: int,
) -> None:
    rng_state = collect_rng_state(rank)
    gathered_rng: list[Any] | None = [None] * world_size if rank == 0 else None
    local_density = core._center_density_checkpoint_state(
        local_molecules, global_step, args.density_solver_target
    )
    gathered_density: list[Any] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(rng_state, gathered_rng, dst=0)
        dist.gather_object(local_density, gathered_density, dst=0)
    else:
        gathered_rng = [rng_state]
        gathered_density = [local_density]
    if rank != 0:
        return
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    model_state = context.model.state_dict()
    clean_state = {
        key.replace("net.module.", "net."): value.detach().cpu()
        for key, value in model_state.items()
    }
    source["state_dict"] = clean_state
    merged_density: dict[str, Any] = {"parameter_step": global_step, "entries": {}}
    for state in gathered_density or []:
        merged_density["entries"].update(state.get("entries", {}))
    source["qm9_v11_ddp8"] = {
        "protocol_sha256": sha256(args.protocol),
        "split_manifest_sha256": sha256(args.split_manifest),
        "schedule_sha256": sha256(args.schedule),
        "direction_manifest_sha256": sha256(args.direction_manifest),
        "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "global_step": global_step,
        "epoch": global_step // 10,
        "optimizer_state_dict": optimizer.state_dict(),
        "per_rank_rng_state": gathered_rng,
        "sampler_state": {"next_global_update": global_step},
        "hvp_direction_sequence_state": {
            "seed": args.seed,
            "next_global_update": global_step,
        },
        "per_molecule_density_predictor_state": merged_density,
        "model_owner": "single_scalar_graphformer",
        "force_or_hessian_heads": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    output = args.output_dir / "checkpoints" / f"step_{global_step:07d}.ckpt"
    atomic_torch_save(output, source)


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "calibration", "pilot", "train"], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lambda-h", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--reference-global-batch", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--integral-cache-entries", type=int, default=4)
    parser.add_argument("--integral-derivative-workers", type=int, default=1)
    parser.add_argument("--density-solver-target", type=float, default=5.0e-9)
    parser.add_argument("--analytic-response-damping", type=float, default=0.0)
    parser.add_argument("--analytic-response-residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--analytic-response-constraint-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--center-electron-number-residual-max", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    enabled, rank, local_rank, world_size = distributed_identity()
    if enabled and args.reference_global_batch:
        raise ValueError("reference-global-batch is single-process only")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    protocol = yaml.safe_load(args.protocol.read_text())
    split = json.loads(args.split_manifest.read_text())
    direction_manifest = json.loads(args.direction_manifest.read_text())
    schedule = json.loads(args.schedule.read_text())
    if protocol["protocol_id"] != "qm9_graphformer_egfh_100mol_ddp8_v11":
        raise RuntimeError("protocol identity mismatch")
    if sha256(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise RuntimeError("released checkpoint SHA256 mismatch")
    if schedule["split_manifest_sha256"] != sha256(args.split_manifest):
        raise RuntimeError("schedule/split binding mismatch")
    train_ids = [str(row["molecule_id"]) for row in split["splits"]["train"]]
    if len(train_ids) != 80 or len(set(train_ids)) != 80:
        raise RuntimeError("frozen Train80 cardinality failure")
    scheduled = [mid for batch in schedule["batches"] for mid in batch]
    if set(scheduled) != set(train_ids) or len(scheduled) != 80:
        raise RuntimeError("schedule does not cover frozen Train80")
    if args.reference_global_batch:
        local_ids = list(schedule["batches"][0])
    else:
        schedule_rank = rank if enabled else 0
        local_ids = list(schedule["rank_molecules"][str(schedule_rank)])

    density_args = density_namespace(args, protocol)
    missing_density_fields = sorted(
        REQUIRED_CANONICAL_NAMESPACE_FIELDS.difference(vars(density_args))
    )
    if missing_density_fields:
        raise RuntimeError(
            "density namespace missing canonical fields: "
            + ", ".join(missing_density_fields)
        )
    for field, expected in EXPECTED_CANONICAL_RUNTIME_BINDINGS.items():
        if getattr(density_args, field) != expected:
            raise RuntimeError(f"canonical runtime binding mismatch: {field}")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    run_spec = core._parse_run(
        f"released_article_qm9_scalar_graphformer={args.source_run_dir}={args.source_checkpoint}"
    )
    context = core._load_context(run_spec, density_args, device)
    context.model.to(torch.float64)
    bare_net = context.model.net
    if bare_net.__class__.__name__ != "Graphformer":
        raise RuntimeError(f"expected Graphformer, found {bare_net.__class__.__name__}")
    parameter_count = sum(parameter.numel() for parameter in bare_net.parameters())
    if parameter_count != 18692586:
        raise RuntimeError(f"Graphformer parameter-count drift: {parameter_count}")
    ddp_net = DDP(
        bare_net,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    ) if enabled else None
    if ddp_net is not None:
        # High-order component gradients are obtained with autograd.grad and
        # reduced explicitly below; disable reducer hooks but keep every model
        # forward routed through the DDP wrapper.
        ddp_net.require_backward_grad_sync = False
        context.model.net = ddp_net
    named_parameters = list(
        (ddp_net if ddp_net is not None else bare_net).named_parameters()
    )
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    # Keep tensor keys identical between the serial reference and DDP probe.
    parameter_names = [
        name.removeprefix("module.")
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    local_molecules = load_local_molecules(direction_manifest, local_ids, protocol)
    cache = core.IntegralBundleCache(context, density_args)
    weights = {
        "energy": float(protocol["loss"]["initial_weights"]["lambda_E"]),
        "gradient": float(protocol["loss"]["initial_weights"]["lambda_G"]),
        "force": float(protocol["loss"]["initial_weights"]["lambda_F"]),
        "hessian": float(args.lambda_h),
    }

    if rank == 0:
        atomic_json(args.output_dir / "run_registration.json", {
            "mode": args.mode,
            "protocol_sha256": sha256(args.protocol),
            "split_manifest_sha256": sha256(args.split_manifest),
            "direction_manifest_sha256": sha256(args.direction_manifest),
            "schedule_sha256": sha256(args.schedule),
            "source_checkpoint_sha256": sha256(args.source_checkpoint),
            "world_size": world_size,
            "ddp": enabled,
            "reference_global_batch": args.reference_global_batch,
            "dtype": "float64",
            "amp": False,
            "tf32": False,
            "learning_rate": args.learning_rate,
            "lambda_H": args.lambda_h,
            "fresh_adamw": True,
            "canonical_namespace_required_field_count": len(
                REQUIRED_CANONICAL_NAMESPACE_FIELDS
            ),
            "canonical_runtime_bindings": EXPECTED_CANONICAL_RUNTIME_BINDINGS,
            "test_accessed": False,
        })

    calibration_egf = [None for _ in parameters]
    calibration_h = [None for _ in parameters]
    per_molecule_calibration: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update in range(args.steps):
        update_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        if args.reference_global_batch:
            active = local_molecules
        else:
            active = [local_molecules[update % len(local_molecules)]]
        local_component_gradients: dict[str, list[torch.Tensor | None]] = {
            name: [None for _ in parameters] for name in COMPONENTS
        }
        local_loss_sums = {name: 0.0 for name in COMPONENTS}
        local_diagnostics: list[dict[str, Any]] = []
        for molecule in active:
            molecule_gradient_norms: dict[str, float] = {}
            refresh = core._refresh_base_densities(
                context,
                [molecule],
                density_args,
                refresh_index=update + 1,
                parameter_step=update,
            )
            if len(refresh) != 1:
                raise RuntimeError("density refresh cardinality drift")
            losses, diagnostics = component_losses(
                context, cache, molecule, protocol, update, args.seed
            )
            local_diagnostics.append(diagnostics)
            for component_index, name in enumerate(COMPONENTS):
                weighted_loss = weights[name] * losses[name] / len(active)
                gradients = torch.autograd.grad(
                    weighted_loss,
                    parameters,
                    retain_graph=component_index < len(COMPONENTS) - 1,
                    allow_unused=True,
                )
                molecule_gradient_norms[name] = gradient_norm(list(gradients))
                for parameter_index, gradient in enumerate(gradients):
                    if gradient is None:
                        continue
                    detached = gradient.detach()
                    current = local_component_gradients[name][parameter_index]
                    if current is None:
                        local_component_gradients[name][parameter_index] = detached.clone()
                    else:
                        current.add_(detached)
                local_loss_sums[name] += float(losses[name].detach().cpu()) / len(active)
            per_molecule_calibration.append({
                "global_update": update,
                "rank": rank,
                "molecule_id": molecule.molecule_id,
                **{f"raw_{name}_loss": float(losses[name].detach().cpu()) for name in COMPONENTS},
                **{
                    f"weighted_{name}_gradient_norm": molecule_gradient_norms[name]
                    for name in COMPONENTS
                },
            })
            del losses
            torch.cuda.empty_cache()

        global_gradients = {
            name: reduce_optional_gradients(
                local_component_gradients[name], parameters, world_size
            )
            for name in COMPONENTS
        }
        global_losses = {
            name: float(
                all_reduce_mean(
                    torch.tensor(local_loss_sums[name], dtype=torch.float64, device=device),
                    world_size,
                ).cpu()
            )
            for name in COMPONENTS
        }
        egf_gradients = add_gradients(
            [global_gradients[name] for name in ("energy", "gradient", "force")],
            parameters,
        )
        total_gradients = add_gradients(
            [global_gradients[name] for name in COMPONENTS], parameters
        )
        if args.mode == "calibration":
            # Calibration is defined on the raw synchronized component
            # gradients.  Accumulate before the diagnostic clipping pass,
            # which scales ``parameter.grad`` in-place.
            accumulate_gradients(calibration_egf, egf_gradients, 1.0 / args.steps)
            accumulate_gradients(
                calibration_h, global_gradients["hessian"], 1.0 / args.steps
            )
        for parameter, gradient in zip(parameters, total_gradients, strict=True):
            parameter.grad = gradient
        clip = core._clip_parameter_gradients_with_diagnostics(
            parameters, args.gradient_clip_norm
        )
        if args.mode != "calibration":
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        rank_timing = {
            "rank": rank,
            "molecule_ids": [molecule.molecule_id for molecule in active],
            "elapsed_seconds": time.perf_counter() - update_started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "diagnostics": local_diagnostics,
        }
        gathered_timing: list[Any] | None = [None] * world_size if rank == 0 else None
        if world_size > 1:
            dist.gather_object(rank_timing, gathered_timing, dst=0)
        else:
            gathered_timing = [rank_timing]
        if rank == 0:
            row: dict[str, Any] = {
                "global_update": update + 1,
                **{f"raw_{name}_loss": global_losses[name] for name in COMPONENTS},
                **{f"weighted_{name}_loss": weights[name] * global_losses[name] for name in COMPONENTS},
                **{f"weighted_{name}_gradient_norm": gradient_norm(global_gradients[name]) for name in COMPONENTS},
                "aggregate_egf_gradient_norm": gradient_norm(egf_gradients),
                "aggregate_egfh_gradient_norm": gradient_norm(total_gradients),
                "cosine_E_G": gradient_cosine(global_gradients["energy"], global_gradients["gradient"]),
                "cosine_E_F": gradient_cosine(global_gradients["energy"], global_gradients["force"]),
                "cosine_E_H": gradient_cosine(global_gradients["energy"], global_gradients["hessian"]),
                "cosine_G_F": gradient_cosine(global_gradients["gradient"], global_gradients["force"]),
                "cosine_G_H": gradient_cosine(global_gradients["gradient"], global_gradients["hessian"]),
                "cosine_F_H": gradient_cosine(global_gradients["force"], global_gradients["hessian"]),
                "gradient_clip_coefficient": float(clip["gradient_clip_scale"]),
                "gradient_clipped": bool(clip["gradient_clipping_active"]),
                "step_seconds_max_rank": max(item["elapsed_seconds"] for item in gathered_timing or []),
                "rank_timings_json": json.dumps(gathered_timing, sort_keys=True),
            }
            write_row(args.output_dir / "training_curve.csv", row)

        if args.mode == "probe" and update == 0:
            probe = {
                "losses": global_losses,
                "gradients": {
                    name: {
                        parameter_name: (
                            None if gradient is None else gradient.detach().cpu()
                        )
                        for parameter_name, gradient in zip(
                            parameter_names, global_gradients[name], strict=True
                        )
                    }
                    for name in COMPONENTS
                },
                "updated_parameters": {
                    name: parameter.detach().cpu()
                    for name, parameter in zip(parameter_names, parameters, strict=True)
                },
            }
            if rank == 0:
                atomic_torch_save(args.output_dir / "consistency_probe.pt", probe)

        if (
            args.mode in {"pilot", "train"}
            and args.checkpoint_interval > 0
            and (update + 1) % args.checkpoint_interval == 0
        ):
            checkpoint(
                args,
                context,
                bare_net,
                optimizer,
                local_molecules,
                rank,
                world_size,
                update + 1,
            )
        if world_size > 1:
            dist.barrier()

    if args.mode == "calibration":
        local_rows = per_molecule_calibration
        gathered_rows: list[Any] | None = [None] * world_size if rank == 0 else None
        if world_size > 1:
            dist.gather_object(local_rows, gathered_rows, dst=0)
        else:
            gathered_rows = [local_rows]
        if rank == 0:
            egf_norm = gradient_norm(calibration_egf)
            h_probe_norm = gradient_norm(calibration_h)
            if not math.isfinite(egf_norm) or not math.isfinite(h_probe_norm):
                raise RuntimeError("non-finite calibration gradient norm")
            if h_probe_norm <= 0.0:
                raise RuntimeError("zero Hessian calibration gradient norm")
            formal_lambda_h = min(
                1.0,
                max(1.0e-4, args.lambda_h * 0.25 * egf_norm / h_probe_norm),
            )
            artifact = {
                "artifact_id": "qm9_v11_train80_ddp8_lambda_h_calibration_v1",
                "protocol_sha256": sha256(args.protocol),
                "split_manifest_sha256": sha256(args.split_manifest),
                "schedule_sha256": sha256(args.schedule),
                "direction_manifest_sha256": sha256(args.direction_manifest),
                "source_checkpoint_sha256": sha256(args.source_checkpoint),
                "probe_lambda_H": args.lambda_h,
                "target_weighted_h_to_egf_gradient_ratio": 0.25,
                "aggregate_weighted_egf_gradient_norm": egf_norm,
                "aggregate_weighted_h_probe_gradient_norm": h_probe_norm,
                "formal_lambda_H": formal_lambda_h,
                "per_molecule": [row for rows in gathered_rows or [] for row in rows],
                "validation_accessed": False,
                "test_accessed": False,
            }
            atomic_json(args.output_dir / "lambda_h_calibration.json", artifact)
            (args.output_dir / "lambda_h_calibration.sha256").write_text(
                sha256(args.output_dir / "lambda_h_calibration.json") + "\n"
            )

    if rank == 0:
        atomic_json(args.output_dir / "summary.json", {
            "mode": args.mode,
            "steps": args.steps,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "complete",
            "validation_accessed": False,
            "test_accessed": False,
        })
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.abort()
            except Exception:
                pass
        raise
