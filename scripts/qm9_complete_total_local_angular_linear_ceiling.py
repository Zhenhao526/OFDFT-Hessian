#!/usr/bin/env python3
"""Solve the stable5 E/F/full-Hessian ceiling of the local angular feature map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mldft.ml.models.components.local_angular_scalar_residual import (
    LocalAngularScalarResidual,
)

try:
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )
except ModuleNotFoundError:
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _chunked_vector_jet(
    feature_function: Any,
    flat: torch.Tensor,
    *,
    feature_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a vector value/Jacobian/Hessian without a quadratic output basis."""
    if feature_chunk_size <= 0:
        raise ValueError("feature_chunk_size must be positive")
    features = feature_function(flat)
    jacobian_chunks = []
    hessian_chunks = []
    for feature_start in range(0, features.numel(), feature_chunk_size):
        feature_stop = min(feature_start + feature_chunk_size, features.numel())

        def chunk_function(flat_positions: torch.Tensor) -> torch.Tensor:
            return feature_function(flat_positions)[feature_start:feature_stop]

        chunk_jacobian_function = torch.func.jacrev(chunk_function)
        jacobian_chunks.append(chunk_jacobian_function(flat))
        hessian_chunks.append(torch.func.jacfwd(chunk_jacobian_function)(flat))
    return (
        features,
        torch.cat(jacobian_chunks, dim=0),
        torch.cat(hessian_chunks, dim=0),
    )


def _feature_jet(
    model: LocalAngularScalarResidual,
    parent: CapacityParent,
    *,
    feature_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    element_index = model._element_index(parent.atomic_numbers)
    coordinate_count = parent.positions.numel()

    def feature_function(flat_positions: torch.Tensor) -> torch.Tensor:
        positions = flat_positions.reshape(parent.positions.shape)
        atomic = model.network.atomic_features(
            positions, element_index, parent.topology
        )
        return atomic.sum(dim=0)

    flat = parent.positions.detach().reshape(-1)
    features, jacobian, hessian = _chunked_vector_jet(
        feature_function,
        flat,
        feature_chunk_size=feature_chunk_size,
    )
    if jacobian.shape != (features.numel(), coordinate_count):
        raise AssertionError("unexpected angular feature Jacobian shape")
    if hessian.shape != (features.numel(), coordinate_count, coordinate_count):
        raise AssertionError("unexpected angular feature Hessian shape")
    if not all(torch.all(torch.isfinite(value)) for value in (features, jacobian, hessian)):
        raise FloatingPointError(f"non-finite feature jet for {parent.molecule_id}")
    return features, jacobian, hessian


def _design_blocks(
    parent: CapacityParent,
    features: torch.Tensor,
    jacobian: torch.Tensor,
    feature_hessian: torch.Tensor,
    *,
    energy_scale: float,
    force_scale: float,
    hessian_scale: float,
    relative_fraction: float,
    hessian_reference_floor: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    coordinate_count = parent.positions.numel()
    target_energy = features.new_tensor(parent.pbe_energy - parent.source_energy)
    target_force = features.new_tensor(parent.pbe_force - parent.source_force).reshape(-1)
    target_hessian = features.new_tensor(
        parent.pbe_hessian - parent.source_hessian_symmetric
    )
    hessian_design = feature_hessian.permute(1, 2, 0).reshape(
        coordinate_count * coordinate_count, -1
    )
    absolute_weight = math.sqrt(1.0 - relative_fraction) / (
        hessian_scale * coordinate_count
    )
    relative_weight = math.sqrt(relative_fraction) / max(
        float(torch.linalg.vector_norm(features.new_tensor(parent.pbe_hessian))),
        hessian_reference_floor,
    )
    designs = [
        features[None, :] / energy_scale,
        -jacobian.T / (force_scale * math.sqrt(coordinate_count)),
        absolute_weight * hessian_design,
        relative_weight * hessian_design,
    ]
    targets = [
        target_energy.reshape(1) / energy_scale,
        target_force / (force_scale * math.sqrt(coordinate_count)),
        absolute_weight * target_hessian.reshape(-1),
        relative_weight * target_hessian.reshape(-1),
    ]
    return designs, targets


def _solve(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    column_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    column_norm = torch.linalg.vector_norm(design, dim=0)
    active = column_norm > column_floor
    if not bool(torch.any(active)):
        raise ValueError("angular design has no active columns")
    scale = column_norm[active]
    normalized = design[:, active] / scale[None, :]
    if ridge > 0.0:
        identity = torch.eye(
            normalized.shape[1], dtype=normalized.dtype, device=normalized.device
        )
        normalized = torch.cat((normalized, math.sqrt(ridge) * identity), dim=0)
        target = torch.cat(
            (target, torch.zeros(normalized.shape[1], dtype=target.dtype, device=target.device))
        )
    solution = torch.linalg.lstsq(normalized, target).solution
    coefficients = torch.zeros(design.shape[1], dtype=design.dtype, device=design.device)
    coefficients[active] = solution / scale
    residual = design @ coefficients - target[: design.shape[0]]
    return coefficients, {
        "active_feature_count": int(torch.sum(active)),
        "design_row_count": int(design.shape[0]),
        "design_column_count": int(design.shape[1]),
        "design_residual_relative": float(
            torch.linalg.vector_norm(residual)
            / torch.clamp(torch.linalg.vector_norm(target[: design.shape[0]]), min=1e-30)
        ),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "column_norm_min_active": float(torch.min(scale)),
        "column_norm_max": float(torch.max(column_norm)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "feature_jet_progress.jsonl"
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "smoke")
    if str(arm["architecture"]) != "local_angular_symmetry_network":
        raise ValueError("linear ceiling requires the local angular architecture")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    parents, provenance = _load_selected_parents(protocol, device)
    model = LocalAngularScalarResidual(
        hidden_size=int(arm["hidden_size"]),
        radial_size=int(arm["radial_size"]),
        angular_order=int(arm["angular_order"]),
        cutoff_bohr=float(arm["cutoff_bohr"]),
        seed=int(arm["seed"]),
        activation=str(arm["activation"]),
        radial_feature_scale=float(arm["radial_feature_scale"]),
        angular_feature_scale=float(arm["angular_feature_scale"]),
    ).to(device)
    training = protocol["training"]
    designs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    jets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    jet_seconds: dict[str, float] = {}
    with progress_path.open("w") as progress_handle:
        for parent_index, parent in enumerate(parents, start=1):
            jet_started = time.perf_counter()
            jet = _feature_jet(
                model,
                parent,
                feature_chunk_size=args.feature_chunk_size,
            )
            jets[parent.molecule_id] = jet
            jet_seconds[parent.molecule_id] = time.perf_counter() - jet_started
            parent_design, parent_target = _design_blocks(
                parent,
                *jet,
                energy_scale=float(training["energy_scale"]),
                force_scale=float(training["force_scale"]),
                hessian_scale=float(training["absolute_hessian_scale"]),
                relative_fraction=float(training["relative_loss_fraction"]),
                hessian_reference_floor=float(training["hessian_reference_floor"]),
            )
            designs.extend(parent_design)
            targets.extend(parent_target)
            progress = {
                "event": "feature_jet_complete",
                "parent_index": parent_index,
                "parent_count": len(parents),
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "coordinate_count": parent.positions.numel(),
                "feature_count": jet[0].numel(),
                "feature_chunk_size": args.feature_chunk_size,
                "feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
                "elapsed_wall_time_s": time.perf_counter() - started,
                "gpu_memory_allocated_mb": (
                    torch.cuda.memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
                "gpu_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            progress_handle.write(json.dumps(progress, sort_keys=True) + "\n")
            progress_handle.flush()
            print(json.dumps(progress, sort_keys=True), flush=True)
    design = torch.cat(designs, dim=0)
    target = torch.cat(targets, dim=0)
    print(
        json.dumps(
            {
                "event": "linear_solve_start",
                "design_shape": list(design.shape),
                "elapsed_wall_time_s": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    coefficients, solve = _solve(
        design,
        target,
        ridge=args.ridge,
        column_floor=args.column_floor,
    )

    rows = []
    for parent in parents:
        features, jacobian, feature_hessian = jets[parent.molecule_id]
        correction_energy = torch.dot(features, coefficients)
        correction_force = -(jacobian.T @ coefficients).reshape(parent.positions.shape)
        correction_hessian = torch.einsum("fij,f->ij", feature_hessian, coefficients)
        predicted_energy = parent.source_energy + float(correction_energy)
        predicted_force = parent.source_force + correction_force.detach().cpu().numpy()
        predicted_hessian = (
            parent.source_hessian_symmetric
            + correction_hessian.detach().cpu().numpy()
        )
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "source_energy_abs_error_hartree": abs(parent.source_energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(predicted_force - parent.pbe_force))
            ),
            "source_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.source_force - parent.pbe_force))
            ),
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_force=predicted_force,
            predicted_hessian=predicted_hessian,
            pbe_hessian=parent.pbe_hessian,
            source_hessian_symmetric=parent.source_hessian_symmetric,
        )
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "coefficients": coefficients.detach().cpu(),
            "protocol": args.protocol.resolve().as_posix(),
            "protocol_sha256": _sha256(args.protocol),
            "arm_id": args.arm_id,
            "test100_accessed": False,
            **provenance,
        },
        args.output_dir / "linear_ceiling.pt",
    )
    hessian_distribution = _distribution(rows, "relative_frobenius")
    energy_distribution = _distribution(rows, "energy_abs_error_hartree")
    source_energy_distribution = _distribution(rows, "source_energy_abs_error_hartree")
    force_distribution = _distribution(rows, "force_mae_hartree_per_bohr")
    source_force_distribution = _distribution(rows, "source_force_mae_hartree_per_bohr")
    energy_ratio = energy_distribution["median"] / max(
        source_energy_distribution["median"], np.finfo(float).tiny
    )
    force_ratio = force_distribution["median"] / max(
        source_force_distribution["median"], np.finfo(float).tiny
    )
    gate = protocol["capacity_gate"]
    checks = {
        "training_median_relative_frobenius": hessian_distribution["median"]
        <= float(gate["training_median_relative_frobenius_max"]),
        "training_all_parent_relative_frobenius": hessian_distribution["max"]
        <= float(gate["training_all_parent_relative_frobenius_max"]),
        "energy_median_ratio_to_source": energy_ratio
        <= float(gate["energy_median_ratio_to_source_max"]),
        "force_median_ratio_to_source": force_ratio
        <= float(gate["force_median_ratio_to_source_max"]),
        "antisymmetric_over_symmetric_frobenius": max(
            float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
        )
        <= float(gate["antisymmetric_over_symmetric_frobenius_max"]),
    }
    result = {
        "definition": "Column-normalized linear E/F/full-Hessian ceiling of the frozen local angular scalar feature map.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "arm_id": args.arm_id,
        "ridge": args.ridge,
        "column_floor": args.column_floor,
        "solve": solve,
        "hessian_relative_frobenius": hessian_distribution,
        "energy_abs_error_hartree": energy_distribution,
        "force_mae_hartree_per_bohr": force_distribution,
        "energy_median_ratio_to_source": energy_ratio,
        "force_median_ratio_to_source": force_ratio,
        "gate": {**checks, "passed": all(checks.values())},
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "parent_cv_design_authorized": False,
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-id", default="M1_local_angular_tanh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--column-floor", type=float, default=1e-10)
    parser.add_argument("--feature-chunk-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
