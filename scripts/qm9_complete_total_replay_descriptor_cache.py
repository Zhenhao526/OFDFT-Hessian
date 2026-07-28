#!/usr/bin/env python3
"""Cache frozen scalar descriptor values/Jacobians for train800 E/F replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


DESCRIPTOR_SETTING_NAMES = (
    "three_body_center_min",
    "three_body_center_max",
    "three_body_center_count",
    "three_body_sigma",
    "three_body_angular_order",
    "four_body_center_min",
    "four_body_center_max",
    "four_body_center_count",
    "four_body_sigma",
    "four_body_torsion_order",
    "four_body_bond_scale",
)

try:
    from mldft.ml.models.components.three_body_geometry_residual import (
        build_triplet_groups,
        make_three_body_feature_function,
    )
    from scripts.qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )
except ModuleNotFoundError:
    from mldft.ml.models.components.three_body_geometry_residual import (
        build_triplet_groups,
        make_three_body_feature_function,
    )
    from qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("replay baseline success CSV is empty")
    return rows


def _load_schema(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("active feature schema does not freeze Test100")
    stages = {}
    for row in manifest["stages"]:
        schema_path = Path(row["schema"])
        if _sha256(schema_path) != row["schema_sha256"]:
            raise ValueError(f"active schema hash mismatch: {schema_path}")
        with np.load(schema_path) as payload:
            stages[str(row["stage"])] = {
                "feature_keys": np.asarray(payload["feature_keys"], dtype=np.int64),
                "column_norms": np.asarray(payload["column_norms"], dtype=np.float64),
            }
    if set(stages) != {"three_body", "four_body"}:
        raise ValueError("active feature schema must contain three_body and four_body")
    return manifest, stages


def _validate_descriptor_settings(
    schema_manifest: dict[str, Any], settings: dict[str, Any]
) -> None:
    expected = schema_manifest.get("descriptor_settings")
    if expected is None:
        raise ValueError("active feature schema is missing descriptor_settings")
    mismatches = {
        name: {"schema": expected.get(name), "runtime": settings.get(name)}
        for name in DESCRIPTOR_SETTING_NAMES
        if expected.get(name) != settings.get(name)
    }
    if mismatches:
        raise ValueError(
            "runtime descriptor settings do not match the active feature schema: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _map_energy_gradient(
    local_keys: list[tuple[int, ...]],
    local_energy: np.ndarray,
    local_gradient: np.ndarray,
    active_keys: np.ndarray,
    active_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    global_keys = [tuple(int(value) for value in row) for row in active_keys]
    key_to_index = {key: index for index, key in enumerate(global_keys)}
    descriptor = np.zeros(len(global_keys), dtype=np.float64)
    jacobian = np.zeros((local_gradient.shape[0], len(global_keys)), dtype=np.float64)
    matched = 0
    for local_index, key in enumerate(local_keys):
        global_index = key_to_index.get(tuple(int(value) for value in key))
        if global_index is None:
            continue
        norm = float(active_norms[global_index])
        descriptor[global_index] = local_energy[local_index] / norm
        jacobian[:, global_index] = local_gradient[:, local_index] / norm
        matched += 1
    return descriptor, jacobian, {
        "local_feature_count": len(local_keys),
        "matched_active_feature_count": matched,
        "matched_active_feature_fraction": (
            matched / len(local_keys) if local_keys else 1.0
        ),
    }


def _feature_function(
    stage: str,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, list[tuple[int, ...]]]:
    if stage == "three_body":
        groups = build_triplet_groups(atomic_numbers)
        centers = np.linspace(
            args.three_body_center_min,
            args.three_body_center_max,
            args.three_body_center_count,
        )
        if not groups:
            return None, []
        return make_three_body_feature_function(
            groups,
            centers,
            args.three_body_sigma,
            args.three_body_angular_order,
        )
    groups, _ = build_bonded_chain_groups(
        atomic_numbers, positions_bohr, args.four_body_bond_scale
    )
    if not groups:
        return None, []
    centers = np.linspace(
        args.four_body_center_min,
        args.four_body_center_max,
        args.four_body_center_count,
    )
    return make_four_body_feature_function(
        groups,
        centers,
        args.four_body_sigma,
        args.four_body_torsion_order,
    )


def _energy_gradient(
    function: Callable[[torch.Tensor], torch.Tensor] | None,
    positions_bohr: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if function is None:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((positions_bohr.size, 0), dtype=np.float64),
        )
    positions = torch.as_tensor(
        positions_bohr, dtype=torch.float64, device=device
    )
    energy = function(positions)
    jacobian = torch.func.jacfwd(function)(positions).reshape(energy.numel(), -1)
    return energy.detach().cpu().numpy(), jacobian.detach().cpu().numpy().T


def _task_hash(
    row: dict[str, str], schema_sha256: str, settings: dict[str, Any]
) -> str:
    payload = {
        "molecule_id": row["molecule_id"],
        "sample_id": int(row["sample_id"]),
        "baseline_array_sha256": row["baseline_array_sha256"],
        "schema_manifest_sha256": schema_sha256,
        "settings": settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    task_rows = _rows(args.task_csv)
    schema_sha = _sha256(args.schema_manifest)
    schema_manifest, stages = _load_schema(args.schema_manifest)
    settings = {name: getattr(args, name) for name in DESCRIPTOR_SETTING_NAMES}
    _validate_descriptor_settings(schema_manifest, settings)
    task_csv_sha = _sha256(args.task_csv)
    selected = [
        (index, row)
        for index, row in enumerate(task_rows)
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task_index, row in selected:
        molecule_id = str(row["molecule_id"])
        sample_id = int(row["sample_id"])
        output_dir = (
            args.output_dir
            / f"task_{task_index:04d}_{molecule_id}_{sample_id:07d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        task_sha = _task_hash(row, schema_sha, settings)
        if summary_path.is_file():
            prior = json.loads(summary_path.read_text())
            cache_path = Path(prior.get("descriptor_cache", ""))
            if (
                prior.get("success") is True
                and prior.get("task_sha256") == task_sha
                and cache_path.is_file()
                and _sha256(cache_path) == prior.get("descriptor_cache_sha256")
            ):
                records.append(prior)
                continue
        task_started = time.perf_counter()
        summary: dict[str, Any] = {
            "task_index": task_index,
            "molecule_id": molecule_id,
            "sample_id": sample_id,
            "task_sha256": task_sha,
            "task_csv": args.task_csv.resolve().as_posix(),
            "task_csv_sha256": task_csv_sha,
            "schema_manifest": args.schema_manifest.resolve().as_posix(),
            "schema_manifest_sha256": schema_sha,
            "descriptor_settings": settings,
            "baseline_array": row["baseline_array"],
            "baseline_array_sha256": row["baseline_array_sha256"],
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "success": False,
        }
        try:
            baseline_path = Path(row["baseline_array"])
            if _sha256(baseline_path) != row["baseline_array_sha256"]:
                raise ValueError(f"baseline array hash mismatch: {baseline_path}")
            with np.load(baseline_path) as payload:
                atomic_numbers = np.asarray(payload["atomic_numbers"], dtype=np.int64)
                positions = np.asarray(payload["positions_bohr"], dtype=np.float64)
                pbe_energy = float(payload["pbe_total_energy"])
                pbe_force = np.asarray(payload["pbe_force"], dtype=np.float64)
                baseline_energy = float(payload["baseline_total_energy"])
                baseline_force = np.asarray(
                    payload["baseline_total_force"], dtype=np.float64
                )
            descriptors = []
            jacobians = []
            coverage = {}
            for stage_name in ("three_body", "four_body"):
                function, local_keys = _feature_function(
                    stage_name, atomic_numbers, positions, args
                )
                local_energy, local_gradient = _energy_gradient(
                    function, positions, device
                )
                descriptor, jacobian, local_coverage = _map_energy_gradient(
                    local_keys,
                    local_energy,
                    local_gradient,
                    stages[stage_name]["feature_keys"],
                    stages[stage_name]["column_norms"],
                )
                descriptors.append(descriptor)
                jacobians.append(jacobian)
                for key, value in local_coverage.items():
                    coverage[f"{stage_name}_{key}"] = value
            descriptor = np.concatenate(descriptors)
            jacobian = np.concatenate(jacobians, axis=1)
            energy_target = pbe_energy - baseline_energy
            force_target = pbe_force - baseline_force
            finite = bool(
                np.all(np.isfinite(descriptor))
                and np.all(np.isfinite(jacobian))
                and np.isfinite(energy_target)
                and np.all(np.isfinite(force_target))
            )
            if not finite:
                raise FloatingPointError("non-finite replay descriptor or target")
            cache_path = output_dir / "descriptor_cache.npz"
            np.savez_compressed(
                cache_path,
                descriptor=descriptor,
                descriptor_jacobian=jacobian,
                energy_target=np.asarray(energy_target, dtype=np.float64),
                force_target=force_target,
                atomic_numbers=atomic_numbers,
                positions_bohr=positions,
                baseline_energy=np.asarray(baseline_energy, dtype=np.float64),
                baseline_force=baseline_force,
                pbe_energy=np.asarray(pbe_energy, dtype=np.float64),
                pbe_force=pbe_force,
            )
            summary.update(
                success=True,
                finite=True,
                natoms=int(atomic_numbers.size),
                coordinate_count=int(positions.size),
                active_feature_count=int(descriptor.size),
                descriptor_cache=cache_path.resolve().as_posix(),
                descriptor_cache_sha256=_sha256(cache_path),
                energy_target_hartree=energy_target,
                force_target_mae_hartree_per_bohr=float(
                    np.mean(np.abs(force_target))
                ),
                **coverage,
            )
        except Exception as error:
            summary.update(
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
        summary["wall_time_s"] = time.perf_counter() - task_started
        summary["max_rss_mb"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
        summary["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        records.append(summary)

    result = {
        "definition": "sharded train800 frozen scalar descriptor E/F replay cache",
        "task_csv": args.task_csv.resolve().as_posix(),
        "task_csv_sha256": task_csv_sha,
        "schema_manifest": args.schema_manifest.resolve().as_posix(),
        "schema_manifest_sha256": schema_sha,
        "schema_checkpoint_sha256": schema_manifest["checkpoint_sha256"],
        "active_feature_count": int(schema_manifest["active_feature_count"]),
        "descriptor_settings": settings,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_task_count": len(selected),
        "success_count": sum(bool(row.get("success")) for row in records),
        "failure_count": sum(not bool(row.get("success")) for row in records),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "wall_time_s": time.perf_counter() - started,
    }
    shard_summary = args.output_dir / f"shard_{args.shard_index:04d}_summary.json"
    shard_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failure_count"]:
        raise RuntimeError(
            f"descriptor cache shard has {result['failure_count']} failed tasks"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--schema-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--three-body-center-min", type=float, default=0.5)
    parser.add_argument("--three-body-center-max", type=float, default=8.0)
    parser.add_argument("--three-body-center-count", type=int, default=6)
    parser.add_argument("--three-body-sigma", type=float, default=0.5)
    parser.add_argument("--three-body-angular-order", type=int, default=4)
    parser.add_argument("--four-body-center-min", type=float, default=1.0)
    parser.add_argument("--four-body-center-max", type=float, default=4.0)
    parser.add_argument("--four-body-center-count", type=int, default=4)
    parser.add_argument("--four-body-sigma", type=float, default=0.75)
    parser.add_argument("--four-body-torsion-order", type=int, default=5)
    parser.add_argument("--four-body-bond-scale", type=float, default=1.35)
    return parser.parse_args()


if __name__ == "__main__":
    cache(parse_args())
