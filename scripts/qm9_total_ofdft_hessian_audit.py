#!/usr/bin/env python3
"""Build a strict density-relaxed total-OFDFT Hessian from scalar-derived forces."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from qm9_hessian_density_relaxed_eval import (
    _compare,
    _load_context,
    _load_geometry,
    _parse_run,
    _save_optimization_trace,
)
from qm9_total_ofdft_force_audit import _evaluate_point, _save_final_coefficients


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _extract_point(
    point: dict[str, Any],
    args: argparse.Namespace,
    context: Any,
    molecule_id: str,
    coordinate: int | None,
    point_name: str,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, float, dict[str, Any]]:
    coefficients = point.pop("final_coeffs")
    trace = point.pop("trace")
    metadata = point.pop("optimization_metadata")
    force = point.pop("total_force")
    point.pop("incomplete_force", None)
    trace_file = _save_optimization_trace(
        args.output_dir / "optimization_traces",
        context,
        molecule_id,
        args.sample_id,
        coordinate,
        point_name,
        trace,
        metadata,
    )
    coefficients_file = None
    if args.save_final_coefficients:
        coefficients_file = _save_final_coefficients(
            args.output_dir
            / "final_coefficients"
            / f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_{point_name}.npz",
            coefficients,
            atomic_numbers,
            positions_bohr,
        )
    row = {
        "run": context.spec.name,
        "molecule_id": molecule_id,
        "coordinate": coordinate,
        "point": point_name,
        "trace_file": trace_file,
        "final_coefficients_file": coefficients_file,
        **point,
        **metadata,
    }
    return coefficients, force, float(point["legacy_total_energy"]), row


def _reference_path(args: argparse.Namespace, molecule_id: str) -> Path:
    return (
        args.reference_dir
        / f"pbe_hessian_{molecule_id}_{args.sample_id:07d}.npz"
    )


def _load_label_coefficients(
    dataset_dir: Path, molecule_id: str, sample_id: int, device: torch.device
) -> torch.Tensor:
    root = zarr.open(
        dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip",
        mode="r",
    )
    coefficients = np.asarray(root["of_labels/spatial/coeffs"][-1], dtype=np.float64)
    return torch.as_tensor(coefficients, dtype=torch.float64, device=device)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    molecule_ids = [value for item in args.molecules for value in item.split(",") if value]
    point_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for context in contexts:
        for molecule_id in molecule_ids:
            molecule_started = time.perf_counter()
            atomic_numbers, positions = _load_geometry(
                args.dataset_dir, molecule_id, args.sample_id
            )
            reference_file = _reference_path(args, molecule_id)
            if not reference_file.exists():
                raise FileNotFoundError(reference_file)
            reference_hessian = np.load(reference_file)["pbe_hessian"]
            base_initial_coefficients = None
            if args.base_initialization == "label_reference":
                base_initial_coefficients = _load_label_coefficients(
                    args.dataset_dir, molecule_id, args.sample_id, device
                )
            base_point = _evaluate_point(
                context,
                atomic_numbers,
                positions,
                args.charge,
                args,
                base_coeffs=None,
                need_force=True,
                initial_coeffs=base_initial_coefficients,
                initialization_mode_override=args.base_initialization,
            )
            base_coefficients, _, base_energy, base_row = _extract_point(
                base_point,
                args,
                context,
                molecule_id,
                None,
                "base",
                atomic_numbers,
                positions,
            )
            point_rows.append(base_row)

            columns: list[np.ndarray] = []
            energy_curvatures: list[float] = []
            n_coordinates = int(positions.size)
            flat_positions = positions.reshape(-1)
            for coordinate in range(n_coordinates):
                forces: dict[str, np.ndarray] = {}
                energies: dict[str, float] = {}
                plus_coefficients: torch.Tensor | None = None
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    displaced = flat_positions.copy()
                    displaced[coordinate] += sign * args.displacement
                    initialization_coefficients = base_coefficients
                    if (
                        side == "minus"
                        and args.pair_response_extrapolation
                        and plus_coefficients is not None
                    ):
                        initialization_coefficients = (
                            2.0 * base_coefficients - plus_coefficients
                        )
                    point = _evaluate_point(
                        context,
                        atomic_numbers,
                        displaced.reshape(positions.shape),
                        args.charge,
                        args,
                        base_coeffs=initialization_coefficients,
                        need_force=True,
                    )
                    coefficients, force, energy, row = _extract_point(
                        point,
                        args,
                        context,
                        molecule_id,
                        coordinate,
                        f"coord_{coordinate:03d}_{side}",
                        atomic_numbers,
                        displaced.reshape(positions.shape),
                    )
                    if side == "plus":
                        plus_coefficients = coefficients
                    forces[side] = force.reshape(-1)
                    energies[side] = energy
                    point_rows.append(row)
                columns.append(
                    -(forces["plus"] - forces["minus"])
                    / (2.0 * args.displacement)
                )
                energy_curvatures.append(
                    (energies["plus"] - 2.0 * base_energy + energies["minus"])
                    / args.displacement**2
                )
                print(
                    context.spec.name,
                    molecule_id,
                    f"coordinate={coordinate + 1}/{n_coordinates}",
                    flush=True,
                )

            hessian = np.stack(columns, axis=1)
            symmetric_hessian = 0.5 * (hessian + hessian.T)
            antisymmetric_hessian = 0.5 * (hessian - hessian.T)
            diagonal_force = np.diag(hessian)
            diagonal_energy = np.asarray(energy_curvatures)
            displaced_rows = [
                row
                for row in point_rows
                if row["run"] == context.spec.name
                and row["molecule_id"] == molecule_id
                and row["point"] != "base"
            ]
            strict_tolerance = (
                args.newton_tolerance
                if args.newton_refine
                else args.fallback_convergence_tolerance
            )
            comparison = _compare(hessian, reference_hessian)
            hessian_path = (
                args.output_dir
                / f"{context.spec.name}_{molecule_id}_{args.sample_id:07d}_total_hessian.npz"
            )
            row = {
                "run": context.spec.name,
                "molecule_id": molecule_id,
                "sample_id": args.sample_id,
                "success": True,
                "hessian_npz": hessian_path.as_posix(),
                "natoms": int(atomic_numbers.size),
                "n_coordinates": n_coordinates,
                "displacement_bohr": args.displacement,
                "strict_tolerance": strict_tolerance,
                "strict_points": sum(
                    float(item["final_gradient_norm"]) < strict_tolerance
                    for item in displaced_rows
                ),
                "total_displaced_points": len(displaced_rows),
                "max_displaced_gradient_norm": max(
                    float(item["final_gradient_norm"]) for item in displaced_rows
                ),
                "mean_displaced_cycles": float(
                    np.mean([float(item["cycles"]) for item in displaced_rows])
                ),
                "force_vs_energy_diagonal_mae": float(
                    np.mean(np.abs(diagonal_force - diagonal_energy))
                ),
                "force_vs_energy_diagonal_rmse": float(
                    np.sqrt(np.mean((diagonal_force - diagonal_energy) ** 2))
                ),
                "force_vs_energy_diagonal_max_abs": float(
                    np.max(np.abs(diagonal_force - diagonal_energy))
                ),
                "wall_time_s": time.perf_counter() - molecule_started,
                **comparison,
            }
            metric_rows.append(row)
            np.savez_compressed(
                hessian_path,
                atomic_numbers=atomic_numbers,
                positions_bohr=positions,
                total_ofdft_hessian=hessian,
                symmetric_hessian=symmetric_hessian,
                antisymmetric_hessian=antisymmetric_hessian,
                pbe_hessian=reference_hessian,
                force_diagonal_curvature=diagonal_force,
                relaxed_energy_diagonal_curvature=diagonal_energy,
            )
            print(
                context.spec.name,
                molecule_id,
                f"strict={row['strict_points']}/{row['total_displaced_points']}",
                f"rel_fro={row['relative_fro_error']:.3e}",
                f"asym_ratio={row['antisymmetric_over_symmetric_fro']:.3e}",
                flush=True,
            )

    result = {
        "definition": (
            "Density-relaxed total-OFDFT Hessian H=-dF/dR from centered finite differences "
            "of the complete scalar-derived force. Each displaced density is independently "
            "optimized and the learned-model geometry derivative uses transform autograd."
        ),
        "runs": args.run,
        "molecules": molecule_ids,
        "sample_id": args.sample_id,
        "displacement_bohr": args.displacement,
        "integral_derivative_step_bohr": args.integral_derivative_step,
        "model_geometry_derivative": args.model_geometry_derivative,
        "base_initialization": args.base_initialization,
        "reference_dir": str(args.reference_dir),
        "metric_rows": metric_rows,
        "point_rows": point_rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "metrics.csv", metric_rows)
    _write_csv(args.output_dir / "points.csv", point_rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--molecules", action="append", required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--integral-derivative-step", type=float, default=1e-4)
    parser.add_argument("--integral-derivative-workers", type=int, default=1)
    parser.add_argument(
        "--model-geometry-derivative",
        choices=["autograd", "numerical"],
        default="autograd",
    )
    parser.add_argument("--model-geometry-fd-step", type=float, default=1e-6)
    parser.add_argument(
        "--model-geometry-fd-richardson",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-final-coefficients", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pair-response-extrapolation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument(
        "--base-initialization",
        choices=["configured", "label_reference"],
        default="configured",
    )
    parser.add_argument(
        "--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--max-cycle", type=int, default=1000)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fallback-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--fallback-max-cycle", type=int, default=10000)
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--fallback-lr", type=float, default=3e-4)
    parser.add_argument(
        "--fallback-always", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--lbfgs-refine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lbfgs-tolerance", type=float, default=1e-8)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=200)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument(
        "--newton-refine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--newton-tolerance", type=float, default=1e-8)
    parser.add_argument("--newton-max-iterations", type=int, default=3)
    parser.add_argument("--newton-max-krylov-iterations", type=int, default=100)
    parser.add_argument("--newton-krylov-tolerance", type=float, default=1e-10)
    parser.add_argument("--newton-diagonal-probes", type=int, default=4)
    parser.add_argument("--newton-damping", type=float, default=1e-8)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
