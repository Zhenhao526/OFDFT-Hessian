#!/usr/bin/env python3
"""Audit a nonlinear local random-feature scalar ceiling on frozen stable5."""

from __future__ import annotations

import argparse
import csv
import gc
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
    from scripts.qm9_complete_total_local_angular_four_body_linear_ceiling import (
        _build_feature_plans,
        _normal_equation_solve,
    )
    from scripts.qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _design_blocks,
        _distribution,
    )
    from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )
except ModuleNotFoundError:
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from qm9_complete_total_local_angular_four_body_linear_ceiling import (
        _build_feature_plans,
        _normal_equation_solve,
    )
    from qm9_complete_total_local_angular_linear_ceiling import (
        _chunked_vector_jet,
        _design_blocks,
        _distribution,
    )
    from qm9_complete_total_local_scalar_full_hessian_capacity import (
        CapacityParent,
        _load_protocol,
        _load_selected_parents,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _central_random_features(
    environment: torch.Tensor,
    element_index: torch.Tensor,
    *,
    inverse_rms: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    activation_scale: torch.Tensor,
    central_element_indices: tuple[int, ...],
) -> torch.Tensor:
    normalized = environment * inverse_rms[None, :]
    hidden = torch.tanh(
        (normalized @ projection + bias[None, :])
        * activation_scale[None, :]
    )
    return torch.cat(
        [
            hidden[element_index == central_index].sum(dim=0)
            for central_index in central_element_indices
        ]
    )


def _random_feature_columns(
    *,
    base_feature_count: int,
    width: int,
    central_element_indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                base_feature_count + central_index * width,
                base_feature_count + (central_index + 1) * width,
                dtype=torch.long,
                device=device,
            )
            for central_index in central_element_indices
        ]
    )


def _environment_inverse_rms(
    parents: list[CapacityParent],
    model: LocalAngularScalarResidual,
    *,
    floor: float,
) -> tuple[torch.Tensor, int]:
    if floor <= 0.0:
        raise ValueError("RMS floor must be positive")
    sum_squared = None
    atom_count = 0
    with torch.no_grad():
        for parent in parents:
            element_index = model._element_index(parent.atomic_numbers)
            atomic = model.network.atomic_features(
                parent.positions, element_index, parent.topology
            )
            environment = atomic[:, model.network.element_count :]
            contribution = torch.sum(environment * environment, dim=0)
            sum_squared = (
                contribution if sum_squared is None else sum_squared + contribution
            )
            atom_count += int(environment.shape[0])
    if sum_squared is None or atom_count == 0:
        raise ValueError("no atoms available for environment normalization")
    rms = torch.sqrt(sum_squared / atom_count)
    active = rms > floor
    inverse_rms = torch.where(active, 1.0 / rms, torch.zeros_like(rms))
    return inverse_rms, int(torch.sum(active))


def _random_kernel_parameters(
    *,
    input_size: int,
    active_input_count: int,
    width_per_scale: int,
    scales: tuple[float, ...],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_size <= 0 or active_input_count <= 0 or width_per_scale <= 0:
        raise ValueError("random-kernel dimensions must be positive")
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("random-kernel activation scales must be positive")
    width = width_per_scale * len(scales)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        (input_size, width), dtype=dtype, generator=generator
    ) / math.sqrt(active_input_count)
    bias = torch.rand(width, dtype=dtype, generator=generator) - 0.5
    activation_scale = torch.repeat_interleave(
        torch.as_tensor(scales, dtype=dtype), width_per_scale
    )
    return tuple(
        value.to(device=device) for value in (projection, bias, activation_scale)
    )


def _random_feature_jet(
    model: LocalAngularScalarResidual,
    parent: CapacityParent,
    *,
    inverse_rms: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    activation_scale: torch.Tensor,
    central_element_indices: tuple[int, ...],
    feature_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    element_index = model._element_index(parent.atomic_numbers)

    def feature_function(flat_positions: torch.Tensor) -> torch.Tensor:
        positions = flat_positions.reshape(parent.positions.shape)
        atomic = model.network.atomic_features(
            positions, element_index, parent.topology
        )
        return _central_random_features(
            atomic[:, model.network.element_count :],
            element_index,
            inverse_rms=inverse_rms,
            projection=projection,
            bias=bias,
            activation_scale=activation_scale,
            central_element_indices=central_element_indices,
        )

    return _chunked_vector_jet(
        feature_function,
        parent.positions.detach().reshape(-1),
        feature_chunk_size=feature_chunk_size,
    )


def _load_base_jet(
    cache_dir: Path,
    parent: CapacityParent,
    *,
    protocol_sha256: str,
    arm_id: str,
    angular_feature_mode: str,
    global_feature_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = cache_dir / f"{parent.molecule_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_sha256": protocol_sha256,
        "arm_id": arm_id,
        "molecule_id": parent.molecule_id,
        "angular_feature_mode": angular_feature_mode,
        "global_feature_count": global_feature_count,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"base feature-jet cache mismatch for {parent.molecule_id}: "
            f"expected={expected}, observed={observed}"
        )
    return tuple(payload[key] for key in ("features", "jacobian", "hessian"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol, arm = _load_protocol(args.protocol, args.arm_id, "smoke")
    prerequisite_path = Path(protocol["inputs"]["base_exact_summary"])
    if _sha256(prerequisite_path) != str(
        protocol["inputs"]["base_exact_summary_sha256"]
    ):
        raise ValueError("v8 exact-summary prerequisite hash drift")
    prerequisite = json.loads(prerequisite_path.read_text())
    if prerequisite.get("test100_accessed") is not False:
        raise ValueError("v8 prerequisite does not freeze Test100")
    if prerequisite["solve"]["converged"] is not True:
        raise ValueError("v9 requires a converged v8 exact solve")
    if prerequisite["gate"]["passed"] is not False:
        raise ValueError("v9 is unnecessary after a passed v8 gate")

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
    base_settings = protocol["base_feature_definition"]
    plans, angular_feature_count, global_four_body_keys = _build_feature_plans(
        parents, model, base_settings
    )
    base_feature_count = angular_feature_count + len(global_four_body_keys)
    if base_feature_count != int(protocol["inputs"]["base_global_feature_count"]):
        raise ValueError("v8 base global feature count drift")

    kernel = protocol["random_feature_kernel"]
    inverse_rms, active_environment_count = _environment_inverse_rms(
        parents, model, floor=float(kernel["environment_rms_floor"])
    )
    scales = tuple(float(value) for value in kernel["activation_scales"])
    projection, bias, activation_scale = _random_kernel_parameters(
        input_size=int(inverse_rms.numel()),
        active_input_count=active_environment_count,
        width_per_scale=int(kernel["width_per_scale"]),
        scales=scales,
        seed=int(kernel["seed"]),
        dtype=torch.float64,
        device=device,
    )
    random_width = int(projection.shape[1])
    global_feature_count = base_feature_count + model.network.element_count * random_width
    base_cache_dir = Path(protocol["inputs"]["base_feature_jet_cache_dir"])
    base_cache_protocol = str(protocol["inputs"]["base_feature_jet_protocol_sha256"])
    random_cache_dir = args.output_dir / "random_feature_jets"
    random_cache_dir.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = _sha256(args.protocol)
    training = protocol["training"]
    designs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    jets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    jet_seconds: dict[str, float] = {}
    progress_path = args.output_dir / "feature_jet_progress.jsonl"
    with progress_path.open("w") as progress_handle:
        for parent_index, parent in enumerate(parents, start=1):
            plan = plans[parent.molecule_id]
            base_jet = _load_base_jet(
                base_cache_dir,
                parent,
                protocol_sha256=base_cache_protocol,
                arm_id=args.arm_id,
                angular_feature_mode=plan.angular_feature_mode,
                global_feature_count=base_feature_count,
            )
            cache_path = random_cache_dir / f"{parent.molecule_id}.pt"
            jet_started = time.perf_counter()
            cache_hit = cache_path.is_file()
            if cache_hit:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                expected = {
                    "protocol_sha256": protocol_sha256,
                    "molecule_id": parent.molecule_id,
                    "random_width": random_width,
                }
                if {key: cached.get(key) for key in expected} != expected:
                    raise ValueError(f"random feature-jet cache mismatch: {cache_path}")
                random_jet = tuple(
                    cached[key] for key in ("features", "jacobian", "hessian")
                )
            else:
                chunk_size = int(kernel["feature_chunk_size"])
                if parent.natoms >= int(kernel["large_parent_min_natoms"]):
                    chunk_size = int(kernel["large_parent_feature_chunk_size"])
                random_jet = tuple(
                    value.detach().cpu()
                    for value in _random_feature_jet(
                        model,
                        parent,
                        inverse_rms=inverse_rms,
                        projection=projection,
                        bias=bias,
                        activation_scale=activation_scale,
                        central_element_indices=plan.central_element_indices,
                        feature_chunk_size=chunk_size,
                    )
                )
                temporary_path = cache_path.with_suffix(".tmp")
                torch.save(
                    {
                        "protocol_sha256": protocol_sha256,
                        "molecule_id": parent.molecule_id,
                        "random_width": random_width,
                        "features": random_jet[0],
                        "jacobian": random_jet[1],
                        "hessian": random_jet[2],
                    },
                    temporary_path,
                )
                temporary_path.replace(cache_path)
            jet_seconds[parent.molecule_id] = time.perf_counter() - jet_started
            combined_jet = tuple(
                torch.cat((base_value, random_value), dim=0)
                for base_value, random_value in zip(base_jet, random_jet, strict=True)
            )
            jets[parent.molecule_id] = combined_jet
            random_columns = _random_feature_columns(
                base_feature_count=base_feature_count,
                width=random_width,
                central_element_indices=plan.central_element_indices,
                device=torch.device("cpu"),
            )
            global_columns = torch.cat(
                (plan.global_columns.detach().cpu(), random_columns)
            )
            local_designs, parent_targets = _design_blocks(
                parent,
                *combined_jet,
                energy_scale=float(training["energy_scale"]),
                force_scale=float(training["force_scale"]),
                hessian_scale=float(training["absolute_hessian_scale"]),
                relative_fraction=float(training["relative_loss_fraction"]),
                hessian_reference_floor=float(training["hessian_reference_floor"]),
            )
            for local_design in local_designs:
                designs.append(
                    torch.zeros(
                        (local_design.shape[0], global_feature_count),
                        dtype=local_design.dtype,
                    ).index_copy(1, global_columns, local_design)
                )
            targets.extend(parent_targets)
            row = {
                "event": "nonlinear_random_feature_jet_complete",
                "parent_index": parent_index,
                "parent_count": len(parents),
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "base_local_feature_count": int(base_jet[0].numel()),
                "random_local_feature_count": int(random_jet[0].numel()),
                "global_feature_count": global_feature_count,
                "random_feature_cache_hit": cache_hit,
                "random_feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
                "elapsed_wall_time_s": time.perf_counter() - started,
                "gpu_peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            progress_handle.write(json.dumps(row, sort_keys=True) + "\n")
            progress_handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    design = torch.cat(designs, dim=0)
    target = torch.cat(targets, dim=0)
    designs.clear()
    targets.clear()
    gc.collect()
    design = design.to(device)
    target = target.to(device)
    print(
        json.dumps(
            {
                "event": "nonlinear_random_feature_solve_start",
                "design_shape": list(design.shape),
                "elapsed_wall_time_s": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    coefficients, solve = _normal_equation_solve(
        design,
        target,
        ridge=float(kernel["ridge"]),
        column_floor=float(kernel["column_floor"]),
    )
    coefficients = coefficients.detach().cpu()
    del design, target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for parent in parents:
        plan = plans[parent.molecule_id]
        features, jacobian, feature_hessian = jets[parent.molecule_id]
        random_columns = _random_feature_columns(
            base_feature_count=base_feature_count,
            width=random_width,
            central_element_indices=plan.central_element_indices,
            device=torch.device("cpu"),
        )
        global_columns = torch.cat(
            (plan.global_columns.detach().cpu(), random_columns)
        )
        local_coefficients = coefficients[global_columns]
        correction_energy = torch.dot(features, local_coefficients)
        correction_force = -(jacobian.T @ local_coefficients).reshape(
            parent.positions.shape
        )
        correction_hessian = torch.einsum(
            "fij,f->ij", feature_hessian, local_coefficients
        )
        predicted_energy = parent.source_energy + float(correction_energy)
        predicted_force = parent.source_force + correction_force.numpy()
        predicted_hessian = (
            parent.source_hessian_symmetric + correction_hessian.numpy()
        )
        metric = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "random_feature_jet_wall_time_s": jet_seconds[parent.molecule_id],
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "source_energy_abs_error_hartree": abs(
                parent.source_energy - parent.pbe_energy
            ),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(predicted_force - parent.pbe_force))
            ),
            "source_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.source_force - parent.pbe_force))
            ),
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
        }
        rows.append(metric)
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
    torch.save(
        {
            "coefficients": coefficients,
            "environment_inverse_rms": inverse_rms.detach().cpu(),
            "projection": projection.detach().cpu(),
            "bias": bias.detach().cpu(),
            "activation_scale": activation_scale.detach().cpu(),
            "protocol": args.protocol.resolve().as_posix(),
            "protocol_sha256": protocol_sha256,
            "arm_id": args.arm_id,
            "test100_accessed": False,
            **provenance,
        },
        args.output_dir / "random_feature_ceiling.pt",
    )
    result = {
        "definition": (
            "Exact linear-output ceiling of a central-element-resolved nonlinear local "
            "tanh random-feature scalar appended to the frozen v8 angular/four-body scalar."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "arm_id": args.arm_id,
        "base_global_feature_count": base_feature_count,
        "random_width_per_central_element": random_width,
        "global_feature_count": global_feature_count,
        "active_environment_feature_count": active_environment_count,
        "random_feature_kernel": kernel,
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
