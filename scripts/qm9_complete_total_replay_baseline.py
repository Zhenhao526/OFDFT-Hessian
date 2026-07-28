#!/usr/bin/env python3
"""Generate strict complete-total density-relaxed E/F replay baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qm9_hessian_density_relaxed_eval import _load_context, _parse_run
from qm9_total_ofdft_force_audit import _evaluate_point
from mldft.ofdft.conservative_force import prepare_fixed_geometry_scalar_energy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"molecule_id", "sample_id", "label_path", "label_sha256"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"task CSV lacks fields {sorted(required)}")
    keys = [(row["molecule_id"], int(row["sample_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate molecule/sample tasks")
    return rows


def _select_task_rows(
    rows: list[dict[str, str]],
    *,
    task_index: int | None,
    shard_index: int | None,
    shard_count: int | None,
) -> list[tuple[int, dict[str, str]]]:
    if task_index is not None:
        if shard_index is not None or shard_count is not None:
            raise ValueError("task-index and sharding are mutually exclusive")
        if task_index < 0 or task_index >= len(rows):
            raise ValueError(f"task index {task_index} outside [0, {len(rows)})")
        return [(task_index, rows[task_index])]
    if shard_index is None or shard_count is None:
        raise ValueError("both shard-index and shard-count are required")
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index/count")
    return [
        (index, row)
        for index, row in enumerate(rows)
        if index % shard_count == shard_index
    ]


def _load_label(path: Path) -> dict[str, Any]:
    root = zarr.open(path, mode="r")
    energy_trace = np.asarray(root["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        root["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    return {
        "atomic_numbers": np.asarray(
            root["geometry/atomic_numbers"], dtype=np.int64
        ),
        "positions_bohr": np.asarray(root["geometry/atom_pos"], dtype=np.float64),
        "pbe_total_energy": float(energy_trace[has_energy][-1]),
        "pbe_force": np.asarray(
            root["metadata/pbe_derivatives/forces"], dtype=np.float64
        ),
        "label_coefficients": np.asarray(
            root["of_labels/spatial/coeffs"][-1], dtype=np.float64
        ),
    }


def _output_dir(root: Path, task_index: int, molecule_id: str, sample_id: int) -> Path:
    return root / f"task_{task_index:04d}_{molecule_id}_{sample_id:07d}"


def _is_completed(path: Path, expected_task_sha256: str) -> bool:
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("success")
        and payload.get("test100_accessed") is False
        and payload.get("task_sha256") == expected_task_sha256
    )


def _task_sha256(row: dict[str, str], run_spec: str, protocol_sha256: str) -> str:
    payload = {
        "molecule_id": row["molecule_id"],
        "sample_id": int(row["sample_id"]),
        "label_sha256": row["label_sha256"],
        "run": run_spec,
        "protocol_sha256": protocol_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _save_trace(path: Path, trace: Any) -> None:
    np.savez_compressed(
        path,
        cycle=np.arange(len(trace.gradient_norms), dtype=np.int64),
        total_energy=np.asarray(trace.total_energies, dtype=np.float64),
        projected_density_gradient_norm=np.asarray(
            trace.gradient_norms, dtype=np.float64
        ),
    )


def _fixed_geometry_rebuild_diagnostics(
    context: Any,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    final_coefficients: torch.Tensor,
    charge: int,
) -> dict[str, float]:
    """Re-evaluate the final physical density through the ordinary geometry builder."""
    evaluator = prepare_fixed_geometry_scalar_energy(
        context.sample_generator,
        context.functional_factory,
        atomic_numbers,
        positions_bohr,
        charge=charge,
    )
    coefficients = final_coefficients.detach().to(
        device=evaluator.sample.coeffs.device, dtype=evaluator.sample.coeffs.dtype
    )
    coefficients = coefficients.clone().requires_grad_(True)
    total_energy = evaluator(coefficients)
    gradient = torch.autograd.grad(total_energy, coefficients)[0]
    normalization = evaluator.normalization_untransformed.to(
        device=gradient.device, dtype=gradient.dtype
    )
    multiplier = -torch.dot(normalization, gradient) / torch.dot(
        normalization, normalization
    )
    projected = gradient + multiplier * normalization
    transformation = evaluator.sample.transformation_matrix.to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    inverse = evaluator.sample.inv_transformation_matrix.to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    transformed = transformation @ coefficients
    coefficient_roundtrip = inverse @ transformed
    transformed_identity = transformation @ inverse
    physical_identity = inverse @ transformation
    return {
        "fixed_rebuild_total_energy": float(total_energy.detach().cpu()),
        "fixed_rebuild_projected_density_gradient_norm": float(
            torch.linalg.vector_norm(projected).detach().cpu()
        ),
        "fixed_rebuild_electron_number": float(
            torch.dot(normalization, coefficients).detach().cpu()
        ),
        "fixed_rebuild_coefficient_roundtrip_relative_error": float(
            (
                torch.linalg.vector_norm(coefficient_roundtrip - coefficients)
                / torch.clamp(torch.linalg.vector_norm(coefficients), min=1.0e-30)
            )
            .detach()
            .cpu()
        ),
        "fixed_rebuild_transformed_identity_max_abs_error": float(
            torch.max(
                torch.abs(
                    transformed_identity
                    - torch.eye(
                        transformed_identity.shape[0],
                        dtype=transformed_identity.dtype,
                        device=transformed_identity.device,
                    )
                )
            )
            .detach()
            .cpu()
        ),
        "fixed_rebuild_physical_identity_max_abs_error": float(
            torch.max(
                torch.abs(
                    physical_identity
                    - torch.eye(
                        physical_identity.shape[0],
                        dtype=physical_identity.dtype,
                        device=physical_identity.device,
                    )
                )
            )
            .detach()
            .cpu()
        ),
    }


def _evaluate_task(
    context: Any,
    args: argparse.Namespace,
    row: dict[str, str],
    task_index: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    molecule_id = str(row["molecule_id"])
    sample_id = int(row["sample_id"])
    output = _output_dir(args.output_dir, task_index, molecule_id, sample_id)
    task_sha = _task_sha256(row, args.run, protocol_sha256)
    if _is_completed(output, task_sha):
        return {
            "task_index": task_index,
            "molecule_id": molecule_id,
            "sample_id": sample_id,
            "status": "cached",
            "success": True,
            "output_dir": output.as_posix(),
        }

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "definition": (
            "strict complete-total density-relaxed frozen-EGF energy/force baseline"
        ),
        "task_index": task_index,
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "task_sha256": task_sha,
        "protocol_sha256": protocol_sha256,
        "label_path": str(Path(row["label_path"]).resolve()),
        "label_sha256": row["label_sha256"],
        "run": args.run,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "success": False,
    }
    try:
        label_path = Path(row["label_path"])
        if args.verify_label_hash and _sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"label hash mismatch: {label_path}")
        label = _load_label(label_path)
        initial_coefficients = torch.as_tensor(
            label["label_coefficients"], dtype=torch.float64, device=context.model.device
        )
        point = _evaluate_point(
            context,
            label["atomic_numbers"],
            label["positions_bohr"],
            args.charge,
            args,
            base_coeffs=None,
            need_force=True,
            initial_coeffs=initial_coefficients,
            initialization_mode_override="label_reference",
        )
        trace = point.pop("trace")
        metadata = point.pop("optimization_metadata")
        final_coefficients = point.pop("final_coeffs")
        total_force = np.asarray(point.pop("total_force"), dtype=np.float64)
        incomplete_force = np.asarray(point.pop("incomplete_force"), dtype=np.float64)
        total_energy = float(point["tensor_total_energy"])
        if args.audit_fixed_geometry_rebuild:
            rebuild = _fixed_geometry_rebuild_diagnostics(
                context,
                label["atomic_numbers"],
                label["positions_bohr"],
                final_coefficients,
                args.charge,
            )
            rebuild["fixed_rebuild_legacy_energy_difference"] = (
                rebuild["fixed_rebuild_total_energy"]
                - float(point["legacy_total_energy"])
            )
            rebuild["fixed_rebuild_tensor_energy_difference"] = (
                rebuild["fixed_rebuild_total_energy"] - total_energy
            )
            point.update(rebuild)
        force_error = total_force - label["pbe_force"]
        final_norm = float(metadata["final_gradient_norm"])
        tensor_norm = float(point["tensor_projected_gradient_norm"])
        strict_converged = bool(
            metadata["converged"]
            and final_norm < args.strict_threshold
            and tensor_norm < args.strict_threshold
        )
        finite = bool(
            np.isfinite(total_energy)
            and np.all(np.isfinite(total_force))
            and np.all(np.isfinite(final_coefficients.detach().cpu().numpy()))
        )
        arrays_path = output / "baseline.npz"
        np.savez_compressed(
            arrays_path,
            atomic_numbers=label["atomic_numbers"],
            positions_bohr=label["positions_bohr"],
            pbe_total_energy=np.asarray(label["pbe_total_energy"], dtype=np.float64),
            pbe_force=label["pbe_force"],
            baseline_total_energy=np.asarray(total_energy, dtype=np.float64),
            baseline_total_force=total_force,
            baseline_incomplete_force=incomplete_force,
            final_density_coefficients=final_coefficients.detach().cpu().numpy(),
        )
        trace_path = output / "density_trace.npz"
        _save_trace(trace_path, trace)
        summary.update(
            {
                "success": bool(strict_converged and finite),
                "strict_converged": strict_converged,
                "finite": finite,
                "natoms": int(label["atomic_numbers"].size),
                "baseline_total_energy_hartree": total_energy,
                "pbe_total_energy_hartree": label["pbe_total_energy"],
                "energy_abs_error_hartree": abs(
                    total_energy - label["pbe_total_energy"]
                ),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(force_error))
                ),
                "force_rmse_hartree_per_bohr": float(
                    np.sqrt(np.mean(force_error**2))
                ),
                "final_projected_density_gradient_norm": final_norm,
                "tensor_projected_density_gradient_norm": tensor_norm,
                "cycles": int(metadata["cycles"]),
                "first_stage_cycles": int(metadata["first_stage_cycles"]),
                "fallback_cycles": int(metadata["fallback_cycles"]),
                "lbfgs_closure_evaluations": int(
                    metadata["lbfgs_closure_evaluations"]
                ),
                "newton_energy_evaluations": int(
                    metadata["newton_energy_evaluations"]
                ),
                "newton_krylov_iterations": metadata[
                    "newton_krylov_iterations"
                ],
                "baseline_array": arrays_path.resolve().as_posix(),
                "baseline_array_sha256": _sha256(arrays_path),
                "density_trace": trace_path.resolve().as_posix(),
                "density_trace_sha256": _sha256(trace_path),
                **{
                    key: value
                    for key, value in point.items()
                    if isinstance(value, (str, int, float, bool, type(None)))
                },
            }
        )
    except Exception as error:  # Persist every failure for deterministic recovery.
        summary.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    summary["wall_time_s"] = time.perf_counter() - started
    summary["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated(context.model.device) / 1024.0**2
        if context.model.device.type == "cuda"
        else None
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return {
        "task_index": task_index,
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "status": "complete" if summary["success"] else "failed",
        "success": summary["success"],
        "strict_converged": summary.get("strict_converged", False),
        "cycles": summary.get("cycles"),
        "wall_time_s": summary["wall_time_s"],
        "output_dir": output.as_posix(),
        "error": summary.get("error"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    protocol_sha = _sha256(args.protocol)
    rows = _task_rows(args.task_csv)
    selected = _select_task_rows(
        rows,
        task_index=args.task_index,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(_parse_run(args.run), args, device)
    results = []
    for local_index, (task_index, row) in enumerate(selected, start=1):
        result = _evaluate_task(context, args, row, task_index, protocol_sha)
        results.append(result)
        print(
            f"[{local_index}/{len(selected)}] task={task_index} "
            f"{result['molecule_id']}.{result['sample_id']:07d} "
            f"status={result['status']} wall={result.get('wall_time_s', 0):.2f}s",
            flush=True,
        )
    shard_name = (
        f"task_{args.task_index:04d}"
        if args.task_index is not None
        else f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    )
    summary = {
        "definition": "Stage-3 strict complete-total E/F replay baseline shard",
        "task_csv": args.task_csv.resolve().as_posix(),
        "task_csv_sha256": _sha256(args.task_csv),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha,
        "run": args.run,
        "task_count": len(results),
        "success_count": sum(bool(row["success"]) for row in results),
        "cached_count": sum(row["status"] == "cached" for row in results),
        "failure_count": sum(not bool(row["success"]) for row in results),
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else None
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "results": results,
    }
    (args.output_dir / f"{shard_name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--verify-label-hash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--max-cycle", type=int, default=1000)
    parser.add_argument("--convergence-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fallback-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--fallback-lr", type=float, default=3.0e-4)
    parser.add_argument("--fallback-max-cycle", type=int, default=10000)
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--fallback-always", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lbfgs-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lbfgs-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=500)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--newton-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--newton-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--newton-max-iterations", type=int, default=6)
    parser.add_argument("--newton-max-krylov-iterations", type=int, default=200)
    parser.add_argument("--newton-krylov-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--newton-diagonal-probes", type=int, default=8)
    parser.add_argument("--newton-damping", type=float, default=1.0e-8)
    parser.add_argument("--strict-threshold", type=float, default=1.0e-8)
    parser.add_argument(
        "--audit-fixed-geometry-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--integral-derivative-step", type=float, default=1.0e-4)
    parser.add_argument("--integral-derivative-workers", type=int, default=4)
    parser.add_argument("--model-geometry-derivative", choices=["autograd", "numerical"], default="autograd")
    parser.add_argument("--model-geometry-fd-step", type=float, default=1.0e-6)
    parser.add_argument("--model-geometry-fd-richardson", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
