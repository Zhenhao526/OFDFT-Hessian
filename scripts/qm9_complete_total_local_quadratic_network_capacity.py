#!/usr/bin/env python3
"""Train a shared local-environment quadratic scalar on frozen train HVPs."""

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

from mldft.ml.models.components.local_quadratic_residual import (
    InternalCoordinateSpec,
    LocalQuadraticCoefficientNetwork,
    assemble_network_hessian,
    build_cross_coordinate_pairs,
    build_internal_coordinate_specs,
    coordinate_feature_matrix,
    cross_coordinate_feature_matrix,
    global_cross_coordinate_feature_matrix,
    internal_coordinate_values_and_jacobian,
    network_hvp,
)

try:
    from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:
    from qm9_complete_total_geometry_shared_capacity import _load_parents
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


@dataclass
class Parent:
    molecule_id: str
    natoms: int
    specs: list[InternalCoordinateSpec]
    jacobian: torch.Tensor
    coordinate_features: torch.Tensor
    cross_features: torch.Tensor
    cross_first: torch.Tensor
    cross_second: torch.Tensor
    fit_directions: torch.Tensor
    calibration_directions: torch.Tensor
    train_directions: torch.Tensor
    heldout_directions: torch.Tensor
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: torch.Tensor
    source_energy: float
    source_force: np.ndarray
    source_hessian: torch.Tensor
    initial_diagonal: torch.Tensor | None = None
    initial_cross: torch.Tensor | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mixed_hvp_loss(
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


def _load_data(args: argparse.Namespace, device: torch.device) -> tuple[list[Parent], dict[str, Any]]:
    baseline, baseline_manifest = _load_parents(args.baseline_manifest)
    source_summary_path = args.source_run_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary.get("test100_accessed") is not False:
        raise ValueError("source run does not certify frozen Test100")
    source_rows = {
        str(row["molecule_id"]): row for row in source_summary["per_parent"]
    }
    direction_manifest = json.loads(args.direction_manifest.read_text())
    if direction_manifest.get("test100_accessed") is not False:
        raise ValueError("direction manifest does not certify frozen Test100")
    direction_rows = {
        str(row["molecule_id"]): row for row in direction_manifest["directions"]
    }
    expected = {parent.molecule_id for parent in baseline}
    if set(source_rows) != expected or not expected.issubset(direction_rows):
        raise ValueError("baseline, source, and direction parent IDs differ")

    parents = []
    for state in baseline:
        molecule_id = state.molecule_id
        with np.load(args.source_run_dir / f"{molecule_id}_result.npz") as payload:
            source_force = np.asarray(payload["predicted_force"], dtype=np.float64)
            source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            saved_reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if not np.array_equal(saved_reference, state.pbe_hessian):
            raise ValueError(f"PBE Hessian mismatch for {molecule_id}")
        with np.load(direction_rows[molecule_id]["direction_path"]) as payload:
            directions = np.asarray(payload["directions"], dtype=np.float64)
            roles = np.asarray(payload["roles"]).astype(str)
        train_directions = directions[roles == "train"]
        calibration_indices = np.arange(train_directions.shape[0]) % args.calibration_stride == args.calibration_offset
        if not np.any(calibration_indices) or np.all(calibration_indices):
            raise ValueError(f"invalid fit/calibration split for {molecule_id}")
        positions = torch.as_tensor(state.positions_bohr, dtype=torch.float64)
        specs = build_internal_coordinate_specs(
            torch.as_tensor(state.atomic_numbers, dtype=torch.long),
            positions,
            bond_scale=args.bond_scale,
        )
        values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
        coordinate_features = coordinate_feature_matrix(specs, values, jacobian)
        cross_first, cross_second, overlap = build_cross_coordinate_pairs(
            specs, local_only=not args.cross_all_pairs
        )
        if args.global_cross_features:
            cross_features = global_cross_coordinate_feature_matrix(
                coordinate_features,
                jacobian,
                cross_first,
                cross_second,
                overlap,
                specs,
                positions,
            )
        else:
            cross_features = cross_coordinate_feature_matrix(
                coordinate_features,
                jacobian,
                cross_first,
                cross_second,
                overlap,
                specs,
            )
        parents.append(
            Parent(
                molecule_id=molecule_id,
                natoms=int(state.atomic_numbers.size),
                specs=specs,
                jacobian=jacobian.to(device=device),
                coordinate_features=coordinate_features.to(device=device),
                cross_features=cross_features.to(device=device),
                cross_first=cross_first.to(device=device),
                cross_second=cross_second.to(device=device),
                fit_directions=torch.as_tensor(
                    train_directions[~calibration_indices], dtype=torch.float64, device=device
                ),
                calibration_directions=torch.as_tensor(
                    train_directions[calibration_indices], dtype=torch.float64, device=device
                ),
                train_directions=torch.as_tensor(
                    train_directions, dtype=torch.float64, device=device
                ),
                heldout_directions=torch.as_tensor(
                    directions[roles == "heldout"], dtype=torch.float64, device=device
                ),
                pbe_energy=state.pbe_energy,
                pbe_force=state.pbe_force.copy(),
                pbe_hessian=torch.as_tensor(
                    state.pbe_hessian, dtype=torch.float64, device=device
                ),
                source_energy=float(source_rows[molecule_id]["predicted_energy"]),
                source_force=source_force,
                source_hessian=torch.as_tensor(
                    source_hessian, dtype=torch.float64, device=device
                ),
            )
        )
    metadata = {
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "source_summary": str(source_summary_path.resolve()),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_checkpoint_sha256": _sha256(args.source_run_dir / "best.ckpt"),
        "direction_manifest": str(args.direction_manifest.resolve()),
        "direction_manifest_sha256": _sha256(args.direction_manifest),
        "source_split_sha256": baseline_manifest["source_split_sha256"],
        "selected_parent_ids": [parent.molecule_id for parent in parents],
    }
    return parents, metadata


def _coefficients(
    model: LocalQuadraticCoefficientNetwork,
    parent: Parent,
    coefficient_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    diagonal, cross = model(parent.coordinate_features, parent.cross_features)
    if parent.initial_diagonal is not None:
        diagonal = diagonal - parent.initial_diagonal
    if parent.initial_cross is not None:
        cross = cross - parent.initial_cross
    return coefficient_scale * diagonal, coefficient_scale * cross


def _predicted_hvp(
    model: LocalQuadraticCoefficientNetwork,
    parent: Parent,
    directions: torch.Tensor,
    coefficient_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    diagonal, cross = _coefficients(model, parent, coefficient_scale)
    correction = network_hvp(
        parent.jacobian,
        directions,
        diagonal,
        cross,
        parent.cross_first,
        parent.cross_second,
    )
    total = directions @ parent.source_hessian.T + correction
    reference = directions @ parent.pbe_hessian.T
    return total, reference


def _role_loss(
    model: LocalQuadraticCoefficientNetwork,
    parents: list[Parent],
    role: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    losses = []
    for parent in parents:
        directions = getattr(parent, f"{role}_directions")
        predicted, reference = _predicted_hvp(
            model, parent, directions, args.coefficient_scale
        )
        losses.append(
            mixed_hvp_loss(
                predicted,
                reference,
                absolute_scale=args.absolute_hvp_scale,
                relative_fraction=args.hvp_relative_fraction,
                reference_floor=args.hvp_reference_floor,
            )
        )
    return torch.mean(torch.stack(losses))


def _sample_fit_loss(
    model: LocalQuadraticCoefficientNetwork,
    parents: list[Parent],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    parent_indices = torch.randperm(len(parents), generator=generator)[
        : min(args.parent_batch_size, len(parents))
    ]
    losses = []
    for raw_index in parent_indices:
        parent = parents[int(raw_index)]
        count = min(args.directions_per_parent, parent.fit_directions.shape[0])
        indices = torch.randperm(
            parent.fit_directions.shape[0], generator=generator
        )[:count].to(device=parent.fit_directions.device)
        directions = parent.fit_directions[indices]
        predicted, reference = _predicted_hvp(
            model, parent, directions, args.coefficient_scale
        )
        losses.append(
            mixed_hvp_loss(
                predicted,
                reference,
                absolute_scale=args.absolute_hvp_scale,
                relative_fraction=args.hvp_relative_fraction,
                reference_floor=args.hvp_reference_floor,
            )
        )
    return torch.mean(torch.stack(losses))


def _parameter_norm(model: torch.nn.Module) -> torch.Tensor:
    return torch.sqrt(
        torch.stack([torch.sum(parameter.square()) for parameter in model.parameters()]).sum()
    )


def _evaluate(
    model: LocalQuadraticCoefficientNetwork,
    parents: list[Parent],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows = []
    arrays = {}
    with torch.no_grad():
        for parent in parents:
            diagonal, cross = _coefficients(model, parent, args.coefficient_scale)
            correction = assemble_network_hessian(
                parent.jacobian,
                diagonal,
                cross,
                parent.cross_first,
                parent.cross_second,
            )
            predicted = parent.source_hessian + correction
            predicted_numpy = predicted.detach().cpu().numpy()
            reference_numpy = parent.pbe_hessian.detach().cpu().numpy()
            difference = predicted_numpy - reference_numpy
            row: dict[str, Any] = {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "energy_abs_error_hartree": abs(parent.source_energy - parent.pbe_energy),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(parent.source_force - parent.pbe_force))
                ),
                "diagonal_coefficient_l2": float(torch.linalg.vector_norm(diagonal).cpu()),
                "cross_coefficient_l2": float(torch.linalg.vector_norm(cross).cpu()),
                **hessian_metrics(predicted_numpy, reference_numpy),
            }
            for role in ("train", "heldout"):
                directions = getattr(parent, f"{role}_directions").detach().cpu().numpy()
                error_hvp = np.einsum("ij,dj->di", difference, directions, optimize=True)
                reference_hvp = np.einsum(
                    "ij,dj->di", reference_numpy, directions, optimize=True
                )
                row[f"{role}_hvp_mae"] = float(np.mean(np.abs(error_hvp)))
                row[f"{role}_hvp_rmse"] = float(np.sqrt(np.mean(error_hvp**2)))
                row[f"{role}_hvp_relative_frobenius"] = float(
                    np.linalg.norm(error_hvp)
                    / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
                )
            rows.append(row)
            arrays[parent.molecule_id] = {
                "predicted_hessian": predicted_numpy,
                "pbe_hessian": reference_numpy,
                "correction_hessian": correction.detach().cpu().numpy(),
                "predicted_force": parent.source_force,
                "pbe_force": parent.pbe_force,
            }
    scalar = {}
    for key in (
        "relative_frobenius",
        "train_hvp_relative_frobenius",
        "heldout_hvp_relative_frobenius",
        "energy_abs_error_hartree",
        "force_mae_hartree_per_bohr",
        "antisymmetric_over_symmetric_frobenius",
    ):
        values = np.asarray([float(row[key]) for row in rows])
        scalar[f"median_{key}"] = float(np.median(values))
        scalar[f"p90_{key}"] = float(np.quantile(values, 0.9))
        scalar[f"max_{key}"] = float(np.max(values))
    return scalar, rows, arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parents, provenance = _load_data(args, device)
    model = LocalQuadraticCoefficientNetwork(
        parents[0].coordinate_features.shape[1],
        parents[0].cross_features.shape[1],
        hidden_size=args.hidden_size,
        seed=args.seed,
        zero_output=False,
    ).to(device)
    with torch.no_grad():
        for parent in parents:
            initial_diagonal, initial_cross = model(
                parent.coordinate_features, parent.cross_features
            )
            parent.initial_diagonal = initial_diagonal.detach().clone()
            parent.initial_cross = initial_cross.detach().clone()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 104729)
    best_calibration = float("inf")
    best_step = -1
    log_path = args.output_dir / "training_metrics.jsonl"
    with log_path.open("w") as log_handle:
        for step in range(args.steps + 1):
            if step > 0:
                optimizer.zero_grad(set_to_none=True)
                fit_loss = _sample_fit_loss(model, parents, args, generator)
                fit_loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    .detach()
                    .cpu()
                )
                optimizer.step()
            else:
                fit_loss = _sample_fit_loss(model, parents, args, generator)
                gradient_norm = math.nan
            should_log = step % args.log_interval == 0 or step == args.steps
            if should_log:
                with torch.no_grad():
                    full_fit_loss = _role_loss(model, parents, "fit", args)
                    calibration_loss = _role_loss(model, parents, "calibration", args)
                    row = {
                        "step": step,
                        "sampled_fit_loss": float(fit_loss.detach().cpu()),
                        "full_fit_loss": float(full_fit_loss.detach().cpu()),
                        "calibration_loss": float(calibration_loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "parameter_norm": float(_parameter_norm(model).detach().cpu()),
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
                payload = {
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args),
                    "calibration_loss": row["calibration_loss"],
                    "test100_accessed": False,
                    **provenance,
                }
                torch.save(payload, args.output_dir / "last.ckpt")
                if row["calibration_loss"] < best_calibration:
                    best_calibration = row["calibration_loss"]
                    best_step = step
                    torch.save(payload, args.output_dir / "best.ckpt")
    best = torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    final, rows, arrays = _evaluate(model, parents, args)
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for molecule_id, values in arrays.items():
        np.savez_compressed(args.output_dir / f"{molecule_id}_result.npz", **values)
    result = {
        "definition": (
            "shared continuous local-environment coefficient network owning a reference-"
            "anchored bond/angle/torsion quadratic scalar; force and Hessian are exact "
            "derivatives of that scalar"
        ),
        "formal_limit": (
            "reference-local Stage-2 capacity model; it requires a later globally "
            "transferable scalar/replay implementation before unseen-parent promotion"
        ),
        "selection_definition": (
            "checkpoint selected only by a deterministic subset of original train "
            "directions; original held-out directions evaluated once after freezing"
        ),
        "hidden_size": args.hidden_size,
        "coefficient_scale": args.coefficient_scale,
        "cross_all_pairs": args.cross_all_pairs,
        "global_cross_features": args.global_cross_features,
        "cross_pair_counts": {
            parent.molecule_id: int(parent.cross_first.numel()) for parent in parents
        },
        "zero_correction_initialization": (
            "trainable smooth coefficient network minus its frozen seeded initial function"
        ),
        "best_step": best_step,
        "best_calibration_loss": best_calibration,
        "final": final,
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "energy_force_anchor_exact_by_construction": True,
        "heldout_directions_used_for_selection": False,
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
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bond-scale", type=float, default=1.25)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--coefficient-scale", type=float, default=0.1)
    parser.add_argument("--cross-all-pairs", action="store_true")
    parser.add_argument("--global-cross-features", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--parent-batch-size", type=int, default=4)
    parser.add_argument("--directions-per-parent", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--calibration-offset", type=int, default=4)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
