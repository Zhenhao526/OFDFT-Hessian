#!/usr/bin/env python3
"""Fit a train-only local-additive scalar correction on frozen Stage-2 directions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mldft.ml.models.components.local_body_order_residual import (
    BodyOrderTopology,
    LocalBodyOrderResidual,
    build_body_order_topology,
)
from mldft.ofdft.complete_total_training import (
    assign_two_task_pcgrad,
    parameter_gradient_diagnostics,
)

try:
    from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:
    from qm9_complete_total_geometry_shared_capacity import _load_parents
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


@dataclass
class LocalParent:
    molecule_id: str
    natoms: int
    positions: torch.Tensor
    topology: BodyOrderTopology
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    source_energy: float
    source_force: np.ndarray
    source_hessian: np.ndarray
    train_directions: torch.Tensor
    heldout_directions: torch.Tensor
    low_mode_directions: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_device_topology(topology: BodyOrderTopology, device: torch.device) -> BodyOrderTopology:
    return BodyOrderTopology(
        **{
            name: getattr(topology, name).to(device=device)
            for name in topology.__dataclass_fields__
        }
    )


def _load_data(args: argparse.Namespace, device: torch.device) -> tuple[list[LocalParent], dict[str, Any]]:
    baseline_parents, baseline_manifest = _load_parents(args.baseline_manifest)
    source_summary_path = args.source_run_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary.get("test100_accessed") is not False:
        raise ValueError("source run does not certify frozen Test100")
    source_rows = {
        str(row["molecule_id"]): row for row in source_summary["per_parent"]
    }
    expected_ids = {parent.molecule_id for parent in baseline_parents}
    if set(source_rows) != expected_ids:
        raise ValueError("source run and baseline manifest parent IDs differ")

    direction_manifest = json.loads(args.direction_manifest.read_text())
    if direction_manifest.get("test100_accessed") is not False:
        raise ValueError("direction manifest does not certify frozen Test100")
    direction_rows = {
        str(row["molecule_id"]): row for row in direction_manifest["directions"]
    }
    missing_direction_ids = sorted(expected_ids - set(direction_rows))
    if missing_direction_ids:
        raise ValueError(
            f"direction manifest omits baseline parents: {missing_direction_ids}"
        )

    parents = []
    for parent in baseline_parents:
        molecule_id = parent.molecule_id
        result_path = args.source_run_dir / f"{molecule_id}_result.npz"
        with np.load(result_path) as payload:
            source_force = np.asarray(payload["predicted_force"], dtype=np.float64)
            source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            saved_reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if not np.allclose(saved_reference, parent.pbe_hessian, atol=0.0, rtol=0.0):
            raise ValueError(f"PBE Hessian mismatch for {molecule_id}")
        direction_row = direction_rows[molecule_id]
        direction_path = Path(direction_row["direction_path"])
        with np.load(direction_path) as payload:
            directions = np.asarray(payload["directions"], dtype=np.float64)
            roles = np.asarray(payload["roles"]).astype(str)
            kinds = np.asarray(payload["kinds"]).astype(str)
        if directions.shape[1] != parent.force.size:
            raise ValueError(f"direction shape mismatch for {molecule_id}")
        atomic_numbers = torch.as_tensor(parent.atomic_numbers, dtype=torch.long)
        parents.append(
            LocalParent(
                molecule_id=molecule_id,
                natoms=int(parent.atomic_numbers.size),
                positions=torch.as_tensor(
                    parent.positions_bohr, dtype=torch.float64, device=device
                ),
                topology=_to_device_topology(
                    build_body_order_topology(
                        atomic_numbers,
                        torch.as_tensor(parent.positions_bohr, dtype=torch.float64),
                        bond_scale=args.bond_scale,
                    ),
                    device,
                ),
                pbe_energy=parent.pbe_energy,
                pbe_force=parent.pbe_force.copy(),
                pbe_hessian=parent.pbe_hessian.copy(),
                source_energy=float(source_rows[molecule_id]["predicted_energy"]),
                source_force=source_force.copy(),
                source_hessian=source_hessian.copy(),
                train_directions=torch.as_tensor(
                    directions[roles == "train"], dtype=torch.float64, device=device
                ),
                heldout_directions=torch.as_tensor(
                    directions[roles == "heldout"], dtype=torch.float64, device=device
                ),
                low_mode_directions=torch.as_tensor(
                    directions[(roles == "train") & (kinds == "low_mode")],
                    dtype=torch.float64,
                    device=device,
                ),
            )
        )
    metadata = {
        "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "source_run_dir": args.source_run_dir.resolve().as_posix(),
        "source_summary": source_summary_path.resolve().as_posix(),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_checkpoint_sha256": _sha256(args.source_run_dir / "best.ckpt"),
        "direction_manifest": args.direction_manifest.resolve().as_posix(),
        "direction_manifest_sha256": _sha256(args.direction_manifest),
        "source_split_sha256": baseline_manifest["source_split_sha256"],
    }
    if args.parent_limit > 0:
        parents = parents[: args.parent_limit]
    metadata["selected_parent_ids"] = [parent.molecule_id for parent in parents]
    metadata["parent_limit"] = args.parent_limit
    return parents, metadata


def _relative_hvp_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_scale: float,
    relative_fraction: float,
    reference_floor: float,
) -> torch.Tensor:
    error = predicted - reference
    absolute = torch.mean((error / absolute_scale) ** 2)
    error_norm_squared = torch.sum(error * error, dim=1)
    reference_norm_squared = torch.sum(reference * reference, dim=1)
    relative = torch.mean(
        error_norm_squared
        / torch.clamp(reference_norm_squared, min=reference_floor**2)
    )
    return (1.0 - relative_fraction) * absolute + relative_fraction * relative


def _parameter_penalty(model: torch.nn.Module) -> torch.Tensor:
    return torch.mean(
        torch.stack([torch.mean(parameter * parameter) for parameter in model.parameters()])
    )


def _task_gradient_norm(
    loss: torch.Tensor, parameters: list[torch.Tensor]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    norm_squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            norm_squared = norm_squared + torch.sum(gradient * gradient)
    return torch.sqrt(norm_squared.clamp_min(torch.finfo(loss.dtype).tiny))


def _parent_losses(
    model: LocalBodyOrderResidual,
    parent: LocalParent,
    directions: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    energy, correction_force, correction_hvp = model.energy_force_hvp(
        parent.positions,
        parent.topology,
        directions.reshape(-1, parent.natoms, 3),
        reference_positions_bohr=(
            parent.positions if args.anchor_energy_force_at_parent else None
        ),
    )
    predicted_energy = energy + parent.source_energy
    predicted_force = correction_force + torch.as_tensor(
        parent.source_force, dtype=torch.float64, device=parent.positions.device
    )
    source_hessian = torch.as_tensor(
        parent.source_hessian, dtype=torch.float64, device=parent.positions.device
    )
    pbe_hessian = torch.as_tensor(
        parent.pbe_hessian, dtype=torch.float64, device=parent.positions.device
    )
    flat_directions = directions.reshape(directions.shape[0], -1)
    total_hvp = correction_hvp.reshape(directions.shape[0], -1) + torch.einsum(
        "ij,dj->di", source_hessian, flat_directions
    )
    reference_hvp = torch.einsum("ij,dj->di", pbe_hessian, flat_directions)
    energy_loss = ((predicted_energy - parent.pbe_energy) / args.energy_scale) ** 2
    force_loss = torch.mean(
        (
            (predicted_force - torch.as_tensor(parent.pbe_force, device=parent.positions.device))
            / args.force_scale
        )
        ** 2
    )
    hvp_loss = _relative_hvp_loss(
        total_hvp,
        reference_hvp,
        absolute_scale=args.absolute_hvp_scale,
        relative_fraction=args.hvp_relative_loss_fraction,
        reference_floor=args.hvp_reference_floor,
    )
    rayleigh_prediction = torch.sum(total_hvp * flat_directions, dim=1)
    rayleigh_reference = torch.sum(reference_hvp * flat_directions, dim=1)
    signed = torch.sign(rayleigh_reference) * rayleigh_prediction
    spectrum_loss = torch.mean(
        ((rayleigh_prediction - rayleigh_reference) / args.spectrum_reference_floor) ** 2
        + args.wrong_curvature_multiplier
        * torch.relu(-signed).square()
        / args.spectrum_reference_floor**2
    )
    return energy_loss, force_loss, hvp_loss, spectrum_loss


def _sample_batch(
    parents: list[LocalParent], args: argparse.Namespace, generator: torch.Generator
) -> list[tuple[LocalParent, torch.Tensor]]:
    if args.parent_batch_size <= len(parents):
        indices = torch.randperm(len(parents), generator=generator)[
            : args.parent_batch_size
        ]
    else:
        indices = torch.randint(
            len(parents), (args.parent_batch_size,), generator=generator
        )
    result = []
    for raw_index in indices:
        parent = parents[int(raw_index)]
        count = min(args.directions_per_parent, parent.train_directions.shape[0])
        selected = torch.randperm(
            parent.train_directions.shape[0], generator=generator
        )[:count].to(device=parent.positions.device)
        result.append((parent, parent.train_directions[selected]))
    return result


def _evaluate(
    model: LocalBodyOrderResidual,
    parents: list[LocalParent],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows = []
    arrays = {}
    for parent in parents:
        energy, force, correction_hessian = model.energy_force_hessian(
            parent.positions,
            parent.topology,
            create_parameter_graph=False,
            reference_positions_bohr=(
                parent.positions if args.anchor_energy_force_at_parent else None
            ),
        )
        predicted_energy = parent.source_energy + float(energy.detach().cpu())
        predicted_force = parent.source_force + force.detach().cpu().numpy()
        predicted_hessian = parent.source_hessian + correction_hessian.detach().cpu().numpy()
        row: dict[str, Any] = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(predicted_force - parent.pbe_force))
            ),
            "predicted_energy": predicted_energy,
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        difference = predicted_hessian - parent.pbe_hessian
        for role, direction_tensor in (
            ("train", parent.train_directions),
            ("heldout", parent.heldout_directions),
        ):
            directions = direction_tensor.detach().cpu().numpy()
            error_hvp = np.einsum("ij,dj->di", difference, directions)
            reference_hvp = np.einsum("ij,dj->di", parent.pbe_hessian, directions)
            row[f"{role}_hvp_mae"] = float(np.mean(np.abs(error_hvp)))
            row[f"{role}_hvp_rmse"] = float(np.sqrt(np.mean(error_hvp**2)))
            row[f"{role}_hvp_relative_frobenius"] = float(
                np.linalg.norm(error_hvp)
                / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
            )
        rows.append(row)
        arrays[parent.molecule_id] = {
            "predicted_force": predicted_force,
            "predicted_hessian": predicted_hessian,
            "pbe_hessian": parent.pbe_hessian,
        }
    scalar = {}
    for key in (
        "relative_frobenius",
        "train_hvp_relative_frobenius",
        "heldout_hvp_relative_frobenius",
        "energy_abs_error_hartree",
        "force_mae_hartree_per_bohr",
    ):
        values = np.asarray([float(row[key]) for row in rows])
        scalar[f"median_{key}"] = float(np.median(values))
        scalar[f"max_{key}"] = float(np.max(values))
        scalar[f"p90_{key}"] = float(np.quantile(values, 0.9))
    scalar["curvature_selection_score"] = (
        scalar["median_relative_frobenius"]
        + scalar["median_heldout_hvp_relative_frobenius"]
    )
    gate_penalty = (
        max(
            0.0,
            scalar["median_energy_abs_error_hartree"] - args.energy_median_gate,
        )
        / args.energy_median_gate
        + max(
            0.0,
            scalar["max_energy_abs_error_hartree"] - args.energy_max_gate,
        )
        / args.energy_max_gate
        + max(
            0.0,
            scalar["median_force_mae_hartree_per_bohr"] - args.force_median_gate,
        )
        / args.force_median_gate
        + max(
            0.0,
            scalar["max_force_mae_hartree_per_bohr"] - args.force_max_gate,
        )
        / args.force_max_gate
    )
    scalar["energy_force_gate_penalty"] = gate_penalty
    scalar["selection_score"] = scalar["curvature_selection_score"] + gate_penalty
    return scalar, rows, arrays


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.parent_limit < 0:
        raise ValueError("parent_limit must be non-negative")
    if (
        args.gradnorm_target_ratio <= 0.0
        or args.gradnorm_min_scale <= 0.0
        or args.gradnorm_max_scale < args.gradnorm_min_scale
    ):
        raise ValueError("invalid GradNorm target or scale bounds")
    if any(
        gate <= 0.0
        for gate in (
            args.energy_median_gate,
            args.energy_max_gate,
            args.force_median_gate,
            args.force_max_gate,
        )
    ):
        raise ValueError("energy and force selection gates must be positive")
    if args.learning_rate <= 0.0 or (
        args.torsion_learning_rate is not None
        and args.torsion_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents, provenance = _load_data(args, device)
    model = LocalBodyOrderResidual(
        pair_hidden_size=args.pair_hidden_size,
        triplet_hidden_size=args.triplet_hidden_size,
        torsion_hidden_size=args.torsion_hidden_size,
        seed=args.seed,
        canceling_output=args.canceling_output,
        cutoff_bohr=args.cutoff_bohr,
    ).to(device)
    torsion_learning_rate = (
        args.learning_rate
        if args.torsion_learning_rate is None
        else args.torsion_learning_rate
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [*model.pair.parameters(), *model.triplet.parameters()],
                "lr": args.learning_rate,
            },
            {"params": model.torsion.parameters(), "lr": torsion_learning_rate},
        ]
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 104729)
    best_score = float("inf")
    best_step = -1
    log_path = args.output_dir / "training_metrics.jsonl"
    with log_path.open("w") as log_handle:
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = _sample_batch(parents, args, generator)
            terms = [_parent_losses(model, parent, directions, args) for parent, directions in batch]
            energy_loss, force_loss, hvp_loss, spectrum_loss = [
                torch.mean(torch.stack(values)) for values in zip(*terms, strict=True)
            ]
            warmup = min(1.0, step / args.hvp_warmup_steps) if args.hvp_warmup_steps else 1.0
            replay_task = args.lambda_energy * energy_loss + args.lambda_force * force_loss
            curvature_task = warmup * (
                args.lambda_hessian * hvp_loss + args.lambda_spectrum * spectrum_loss
            )
            penalty = args.lambda_parameter * _parameter_penalty(model)
            gradnorm_scale = curvature_task.new_ones(())
            replay_gradient_norm = curvature_task.new_tensor(float("nan"))
            curvature_gradient_norm = curvature_task.new_tensor(float("nan"))
            if args.gradnorm_balance and step > 0:
                parameters = list(model.parameters())
                replay_gradient_norm = _task_gradient_norm(
                    replay_task + penalty, parameters
                )
                curvature_gradient_norm = _task_gradient_norm(
                    curvature_task, parameters
                )
                target_ratio = args.gradnorm_target_ratio * warmup
                requested_scale = (
                    target_ratio
                    * replay_gradient_norm
                    / curvature_gradient_norm
                )
                gradnorm_scale = torch.clamp(
                    requested_scale,
                    min=args.gradnorm_min_scale,
                    max=args.gradnorm_max_scale,
                ).detach()
            balanced_curvature_task = gradnorm_scale * curvature_task
            total = replay_task + balanced_curvature_task + penalty
            should_log = step % args.log_interval == 0 or step == args.steps
            diagnostics = {}
            if should_log:
                names = list(model.named_parameters())
                diagnostics = parameter_gradient_diagnostics(
                    {
                        "energy_force": replay_task,
                        "hvp_spectrum": balanced_curvature_task,
                    },
                    [parameter for _, parameter in names],
                    parameter_names=[name for name, _ in names],
                )
            balance = {}
            if step > 0:
                if args.pcgrad:
                    balance = assign_two_task_pcgrad(
                        replay_task + penalty,
                        balanced_curvature_task,
                        model.parameters(),
                        max_second_to_first_norm_ratio=args.pcgrad_max_curvature_to_replay_norm_ratio,
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
                scalar, rows, arrays = _evaluate(model, parents, args)
                row = {
                    "step": step,
                    "sampled_energy_loss": float(energy_loss.detach().cpu()),
                    "sampled_force_loss": float(force_loss.detach().cpu()),
                    "sampled_hvp_loss": float(hvp_loss.detach().cpu()),
                    "sampled_spectrum_loss": float(spectrum_loss.detach().cpu()),
                    "hvp_warmup_multiplier": warmup,
                    "gradnorm/curvature_scale": float(gradnorm_scale.detach().cpu()),
                    "gradnorm/replay_gradient_norm": float(
                        replay_gradient_norm.detach().cpu()
                    ),
                    "gradnorm/raw_curvature_gradient_norm": float(
                        curvature_gradient_norm.detach().cpu()
                    ),
                    "gradient_norm": gradient_norm,
                    "wall_time_s": time.perf_counter() - started,
                    "gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                    **diagnostics,
                    **balance,
                    **scalar,
                }
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                payload = {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args),
                    "test100_accessed": False,
                    **provenance,
                }
                torch.save(payload, args.output_dir / "last.ckpt")
                if scalar["selection_score"] < best_score:
                    best_score = scalar["selection_score"]
                    best_step = step
                    torch.save(payload, args.output_dir / "best.ckpt")
                    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
                    for molecule_id, values in arrays.items():
                        np.savez_compressed(
                            args.output_dir / f"{molecule_id}_result.npz", **values
                        )
    best = torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    scalar, rows, arrays = _evaluate(model, parents, args)
    _write_rows(args.output_dir / "per_parent_metrics.csv", rows)
    for molecule_id, values in arrays.items():
        np.savez_compressed(args.output_dir / f"{molecule_id}_result.npz", **values)
    result: dict[str, Any] = {
        "definition": "train-only pair-plus-triplet local-additive float64 scalar correction with locked antisymmetric output pairs and optional reference-geometry Taylor anchoring; force and Hessian are exact derivatives of that scalar",
        "architecture": {
            "pair_hidden_size": args.pair_hidden_size,
            "triplet_hidden_size": args.triplet_hidden_size,
            "torsion_hidden_size": args.torsion_hidden_size,
            "bond_scale": args.bond_scale,
            "torsion_learning_rate": torsion_learning_rate,
            "canceling_output": args.canceling_output,
            "locked_antisymmetric_output_pairs": True,
            "anchor_energy_force_at_parent": args.anchor_energy_force_at_parent,
            "cutoff_bohr": args.cutoff_bohr,
        },
        "best_step": best_step,
        "best_selection_score": best_score,
        "final": scalar,
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
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
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--parent-batch-size", type=int, default=1)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--directions-per-parent", type=int, default=2)
    parser.add_argument("--pair-hidden-size", type=int, default=32)
    parser.add_argument("--triplet-hidden-size", type=int, default=64)
    parser.add_argument("--torsion-hidden-size", type=int, default=64)
    parser.add_argument("--bond-scale", type=float, default=1.25)
    parser.add_argument("--canceling-output", type=float, default=1e-3)
    parser.add_argument("--cutoff-bohr", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--torsion-learning-rate", type=float)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--hvp-warmup-steps", type=int, default=1000)
    parser.add_argument("--energy-scale", type=float, default=0.1)
    parser.add_argument("--force-scale", type=float, default=0.05)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-loss-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    parser.add_argument("--spectrum-reference-floor", type=float, default=0.01)
    parser.add_argument("--wrong-curvature-multiplier", type=float, default=2.0)
    parser.add_argument("--lambda-energy", type=float, default=1.0)
    parser.add_argument("--lambda-force", type=float, default=1.0)
    parser.add_argument("--lambda-hessian", type=float, default=1.0)
    parser.add_argument("--lambda-spectrum", type=float, default=0.1)
    parser.add_argument("--lambda-parameter", type=float, default=1e-8)
    parser.add_argument("--energy-median-gate", type=float, default=1e-3)
    parser.add_argument("--energy-max-gate", type=float, default=2e-3)
    parser.add_argument("--force-median-gate", type=float, default=1e-3)
    parser.add_argument("--force-max-gate", type=float, default=3e-3)
    parser.add_argument("--anchor-energy-force-at-parent", action="store_true")
    parser.add_argument("--pcgrad", action="store_true")
    parser.add_argument("--gradnorm-balance", action="store_true")
    parser.add_argument("--gradnorm-target-ratio", type=float, default=1.0)
    parser.add_argument("--gradnorm-min-scale", type=float, default=1e-3)
    parser.add_argument("--gradnorm-max-scale", type=float, default=1e3)
    parser.add_argument(
        "--pcgrad-max-curvature-to-replay-norm-ratio", type=float, default=10.0
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
