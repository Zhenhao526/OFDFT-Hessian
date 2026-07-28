#!/usr/bin/env python3
"""Re-solve cached shared geometry designs with stable task-scaled objectives."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

try:
    from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qm9_complete_total_geometry_shared_capacity import _load_parents
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


def _column_squared_norms(matrix: np.ndarray, chunk_rows: int) -> np.ndarray:
    squared = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk_rows):
        block = np.asarray(matrix[start : start + chunk_rows])
        squared += np.einsum("ij,ij->j", block, block, optimize=True)
    return squared


def solve_preconditioned_lsqr(
    design: np.ndarray,
    target: np.ndarray,
    *,
    anchor_design: np.ndarray | None,
    anchor_target: np.ndarray | None,
    anchor_weights: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
    condition_limit: float,
    chunk_rows: int,
    initial_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a column-preconditioned matrix-free least-squares objective."""
    column_squared = _column_squared_norms(design, chunk_rows)
    if anchor_design is not None:
        if anchor_target is None or anchor_weights is None:
            raise ValueError("anchor target and weights are required with anchor design")
        weighted_anchor = anchor_design * anchor_weights[:, None]
        column_squared += np.einsum(
            "ij,ij->j", weighted_anchor, weighted_anchor, optimize=True
        )
        combined_target = np.concatenate((target, anchor_target * anchor_weights))
        anchor_rows = anchor_design.shape[0]
    else:
        weighted_anchor = None
        combined_target = target
        anchor_rows = 0
    maximum_norm = float(np.sqrt(np.max(column_squared)))
    floor = max(maximum_norm * 1.0e-12, np.finfo(np.float64).tiny)
    column_norms = np.sqrt(column_squared).clip(min=floor)
    inverse_norms = 1.0 / column_norms
    hessian_rows = design.shape[0]

    def matvec(preconditioned: np.ndarray) -> np.ndarray:
        coefficients = preconditioned * inverse_norms
        hessian = design @ coefficients
        if weighted_anchor is None:
            return hessian
        return np.concatenate((hessian, weighted_anchor @ coefficients))

    def rmatvec(residual: np.ndarray) -> np.ndarray:
        gradient = design.T @ residual[:hessian_rows]
        if weighted_anchor is not None:
            gradient += weighted_anchor.T @ residual[hessian_rows:]
        return gradient * inverse_norms

    operator = LinearOperator(
        shape=(hessian_rows + anchor_rows, design.shape[1]),
        dtype=np.float64,
        matvec=matvec,
        rmatvec=rmatvec,
    )
    initial_preconditioned = None
    if initial_coefficients is not None:
        initial_coefficients = np.asarray(initial_coefficients, dtype=np.float64)
        if initial_coefficients.shape != (design.shape[1],):
            raise ValueError("initial coefficients have incompatible shape")
        initial_preconditioned = initial_coefficients * column_norms
    solution = lsqr(
        operator,
        combined_target,
        atol=tolerance,
        btol=tolerance,
        conlim=condition_limit,
        iter_lim=max_iterations,
        show=False,
        x0=initial_preconditioned,
    )
    coefficients = solution[0] * inverse_norms
    residual = matvec(solution[0]) - combined_target
    return coefficients, {
        "lsqr_stop_code": int(solution[1]),
        "lsqr_iterations": int(solution[2]),
        "initial_target_norm": float(np.linalg.norm(combined_target)),
        "final_residual_norm": float(np.linalg.norm(residual)),
        "relative_objective_residual": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(combined_target), np.finfo(float).tiny)
        ),
        "lsqr_operator_norm": float(solution[5]),
        "lsqr_condition_estimate": float(solution[6]),
        "lsqr_normal_residual_norm": float(solution[7]),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "column_norm_min": float(np.min(column_norms)),
        "column_norm_median": float(np.median(column_norms)),
        "column_norm_max": float(np.max(column_norms)),
        "warm_started": initial_coefficients is not None,
    }


def _anchor_weights(parents, energy_scale: float, force_scale: float) -> np.ndarray:
    if energy_scale <= 0 or force_scale <= 0:
        raise ValueError("anchor scales must be positive")
    parent_count = len(parents)
    force_count = sum(parent.force.size for parent in parents)
    rows = []
    for parent in parents:
        rows.append(1.0 / (energy_scale * np.sqrt(parent_count)))
        rows.extend(
            [1.0 / (force_scale * np.sqrt(force_count))] * parent.force.size
        )
    return np.asarray(rows, dtype=np.float64)


def _stage_target(parents, relative_fraction: float, absolute_scale: float):
    rows = []
    weights = []
    anchor = []
    for parent in parents:
        count = parent.hessian.size
        relative = 1.0 / max(np.linalg.norm(parent.pbe_hessian), np.finfo(float).tiny)
        absolute = 1.0 / (absolute_scale * np.sqrt(count))
        weight = np.sqrt(
            relative_fraction * relative**2
            + (1.0 - relative_fraction) * absolute**2
        )
        rows.append((parent.pbe_hessian - parent.hessian).reshape(-1) * weight)
        weights.append(weight)
        anchor.append(parent.pbe_energy - parent.energy)
        anchor.extend((-(parent.pbe_force - parent.force)).reshape(-1))
    return np.concatenate(rows), weights, np.asarray(anchor, dtype=np.float64)


def _run_stage(parents, stage: str, args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    source = args.design_root / stage
    design = np.load(source / "weighted_hessian_design.npy", mmap_mode="r")
    constraints = np.load(source / "anchor_constraints.npy", mmap_mode="r")
    target, parent_weights, anchor_target = _stage_target(
        parents, args.relative_loss_fraction, args.absolute_hessian_scale
    )
    if args.anchor_mode == "soft":
        anchor_design = constraints
        anchor_weights = _anchor_weights(
            parents, args.energy_anchor_scale, args.force_anchor_scale
        )
    else:
        anchor_design = None
        anchor_weights = None
        anchor_target = None
    initial_coefficients = None
    if args.initial_root is not None:
        with np.load(args.initial_root / stage / "shared_coefficients.npz") as payload:
            initial_coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    coefficients, solver = solve_preconditioned_lsqr(
        design,
        target,
        anchor_design=anchor_design,
        anchor_target=anchor_target,
        anchor_weights=anchor_weights,
        tolerance=args.lsqr_tolerance,
        max_iterations=args.lsqr_max_iterations,
        condition_limit=args.lsqr_condition_limit,
        chunk_rows=args.norm_chunk_rows,
        initial_coefficients=initial_coefficients,
    )
    stage_dir = args.output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    with np.load(source / "shared_coefficients.npz") as payload:
        feature_keys = np.asarray(payload["feature_keys"], dtype=np.int64)
    np.savez_compressed(
        stage_dir / "shared_coefficients.npz",
        coefficients=coefficients,
        feature_keys=feature_keys,
    )

    rows = []
    hessian_offset = 0
    constraint_offset = 0
    for parent, weight in zip(parents, parent_weights, strict=True):
        hessian_rows = parent.hessian.size
        constraint_rows = 1 + parent.force.size
        hessian_correction = (
            design[hessian_offset : hessian_offset + hessian_rows] @ coefficients
            / weight
        ).reshape(parent.hessian.shape)
        local_constraints = constraints[
            constraint_offset : constraint_offset + constraint_rows
        ]
        energy_correction = float(local_constraints[0] @ coefficients)
        force_correction = -(
            local_constraints[1:] @ coefficients
        ).reshape(parent.force.shape)
        parent.hessian += hessian_correction
        parent.energy += energy_correction
        parent.force += force_correction
        row = {
            "stage": stage,
            "molecule_id": parent.molecule_id,
            "natoms": int(parent.atomic_numbers.size),
            "energy_abs_error_hartree": abs(parent.energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.force - parent.pbe_force))
            ),
            **hessian_metrics(parent.hessian, parent.pbe_hessian),
        }
        rows.append(row)
        np.savez_compressed(
            stage_dir / f"{parent.molecule_id}_result.npz",
            predicted_hessian=parent.hessian,
            pbe_hessian=parent.pbe_hessian,
            predicted_force=parent.force,
            pbe_force=parent.pbe_force,
            predicted_energy=parent.energy,
            pbe_energy=parent.pbe_energy,
            correction_hessian=hessian_correction,
        )
        hessian_offset += hessian_rows
        constraint_offset += constraint_rows
    with (stage_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": stage,
        "per_parent": rows,
        "median_relative_frobenius": float(
            np.median([row["relative_frobenius"] for row in rows])
        ),
        "max_relative_frobenius": float(
            max(row["relative_frobenius"] for row in rows)
        ),
        "all_below_0p05": all(row["relative_frobenius"] <= 0.05 for row in rows),
        "solver": solver,
        "wall_time_s": time.perf_counter() - started,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    parents, manifest = _load_parents(args.baseline_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = [_run_stage(parents, stage, args) for stage in ("three_body", "four_body")]
    final = stages[-1]
    result = {
        "definition": "column-preconditioned cached shared conservative scalar capacity solve",
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "design_root": str(args.design_root.resolve()),
        "initial_root": (
            str(args.initial_root.resolve()) if args.initial_root is not None else None
        ),
        "source_split_sha256": manifest["source_split_sha256"],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "anchor_mode": args.anchor_mode,
        "energy_anchor_scale_hartree": (
            args.energy_anchor_scale if args.anchor_mode == "soft" else None
        ),
        "force_anchor_scale_hartree_per_bohr": (
            args.force_anchor_scale if args.anchor_mode == "soft" else None
        ),
        "stages": stages,
        "stage1_shared_gate_passed": bool(final["all_below_0p05"]),
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path)
    parser.add_argument("--anchor-mode", choices=("none", "soft"), default="soft")
    parser.add_argument("--energy-anchor-scale", type=float, default=0.03)
    parser.add_argument("--force-anchor-scale", type=float, default=0.03)
    parser.add_argument("--absolute-hessian-scale", type=float, default=0.1)
    parser.add_argument("--relative-loss-fraction", type=float, default=0.5)
    parser.add_argument("--lsqr-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--lsqr-max-iterations", type=int, default=5000)
    parser.add_argument("--lsqr-condition-limit", type=float, default=1.0e12)
    parser.add_argument("--norm-chunk-rows", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
