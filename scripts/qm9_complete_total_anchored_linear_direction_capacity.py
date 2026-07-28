#!/usr/bin/env python3
"""Fit an anchored conservative linear descriptor residual to frozen train HVPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    source_energy: float
    source_force: np.ndarray
    source_hessian: np.ndarray
    train_directions: np.ndarray
    heldout_directions: np.ndarray
    row_offset: int
    hessian_weight: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_hessian_weight(
    reference_hessian: np.ndarray,
    *,
    relative_fraction: float,
    absolute_scale: float,
) -> float:
    row_count = reference_hessian.size
    relative_weight = 1.0 / max(
        float(np.linalg.norm(reference_hessian)), np.finfo(float).tiny
    )
    absolute_weight = 1.0 / (absolute_scale * np.sqrt(row_count))
    return float(
        np.sqrt(
            relative_fraction * relative_weight**2
            + (1.0 - relative_fraction) * absolute_weight**2
        )
    )


def mixed_hvp_row_weights(
    reference_hvp: np.ndarray,
    *,
    parent_count: int,
    absolute_scale: float,
    relative_fraction: float,
    reference_floor: float,
) -> np.ndarray:
    """Return component weights exactly matching the mixed per-parent HVP loss."""
    reference_hvp = np.asarray(reference_hvp, dtype=np.float64)
    if reference_hvp.ndim != 2 or reference_hvp.size == 0:
        raise ValueError("reference_hvp must be a non-empty (directions, coordinates) array")
    if parent_count <= 0 or absolute_scale <= 0.0 or reference_floor <= 0.0:
        raise ValueError("parent count and HVP scales must be positive")
    if not 0.0 <= relative_fraction <= 1.0:
        raise ValueError("relative_fraction must be between zero and one")
    direction_count, coordinate_count = reference_hvp.shape
    reference_norm_squared = np.einsum(
        "di,di->d", reference_hvp, reference_hvp, optimize=True
    )
    absolute_squared = (1.0 - relative_fraction) / (
        parent_count * direction_count * coordinate_count * absolute_scale**2
    )
    relative_squared = relative_fraction / (
        parent_count
        * direction_count
        * np.maximum(reference_norm_squared, reference_floor**2)
    )
    return np.sqrt(absolute_squared + relative_squared[:, None]) * np.ones(
        (1, coordinate_count), dtype=np.float64
    )


def conjugate_gradient_normal_equations(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Solve ``(A.T A + ridge I)x=A.T y`` without forming the Gram matrix."""
    if design.ndim != 2 or target.shape != (design.shape[0],):
        raise ValueError("incompatible design and target shapes")
    if ridge < 0.0 or tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("invalid CG controls")
    right = design.T @ target

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return design.T @ (design @ vector) + ridge * vector

    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    initial_norm = torch.sqrt(residual_squared).clamp_min(
        torch.finfo(design.dtype).tiny
    )
    converged = bool(float(initial_norm.detach().cpu()) == 0.0)
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        action = matvec(direction)
        denominator = torch.dot(direction, action)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise RuntimeError("normal-equation operator is not positive definite")
        alpha = residual_squared / denominator
        solution = solution + alpha * direction
        new_residual = residual - alpha * action
        new_squared = torch.dot(new_residual, new_residual)
        relative_residual = torch.sqrt(new_squared) / initial_norm
        if not bool(torch.isfinite(relative_residual)):
            raise RuntimeError("CG produced a non-finite residual")
        if float(relative_residual.detach().cpu()) <= tolerance:
            residual = new_residual
            residual_squared = new_squared
            converged = True
            break
        beta = new_squared / residual_squared
        direction = new_residual + beta * direction
        residual = new_residual
        residual_squared = new_squared
    final_relative = float(
        (torch.sqrt(residual_squared) / initial_norm).detach().cpu()
    )
    objective_residual = design @ solution - target
    return solution, {
        "iterations": iteration,
        "converged": converged,
        "normal_relative_residual": final_relative,
        "weighted_objective_residual_norm": float(
            torch.linalg.vector_norm(objective_residual).detach().cpu()
        ),
        "weighted_target_norm": float(torch.linalg.vector_norm(target).detach().cpu()),
    }


def _load_data(args: argparse.Namespace) -> tuple[list[Parent], dict[str, Any]]:
    baseline, manifest = _load_parents(args.baseline_manifest)
    baseline_by_id = {parent.molecule_id: parent for parent in baseline}
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
    design_summary_path = args.design_root / "summary.json"
    design_summary = json.loads(design_summary_path.read_text())
    design_order = [str(row["molecule_id"]) for row in design_summary["baseline"]]
    expected = set(baseline_by_id)
    if set(design_order) != expected or set(source_rows) != expected:
        raise ValueError("baseline, source, and descriptor-design parent IDs differ")
    if not expected.issubset(direction_rows):
        raise ValueError("direction manifest omits one or more baseline parents")

    parents = []
    row_offset = 0
    for molecule_id in design_order:
        state = baseline_by_id[molecule_id]
        result_path = args.source_run_dir / f"{molecule_id}_result.npz"
        with np.load(result_path) as payload:
            source_force = np.asarray(payload["predicted_force"], dtype=np.float64)
            source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            saved_reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if not np.array_equal(saved_reference, state.pbe_hessian):
            raise ValueError(f"PBE Hessian mismatch for {molecule_id}")
        with np.load(direction_rows[molecule_id]["direction_path"]) as payload:
            directions = np.asarray(payload["directions"], dtype=np.float64)
            roles = np.asarray(payload["roles"]).astype(str)
        coordinate_count = state.force.size
        if directions.shape[1] != coordinate_count:
            raise ValueError(f"direction shape mismatch for {molecule_id}")
        parents.append(
            Parent(
                molecule_id=molecule_id,
                natoms=int(state.atomic_numbers.size),
                pbe_energy=state.pbe_energy,
                pbe_force=state.pbe_force.copy(),
                pbe_hessian=state.pbe_hessian.copy(),
                source_energy=float(source_rows[molecule_id]["predicted_energy"]),
                source_force=source_force,
                source_hessian=source_hessian,
                train_directions=directions[roles == "train"],
                heldout_directions=directions[roles == "heldout"],
                row_offset=row_offset,
                hessian_weight=_descriptor_hessian_weight(
                    state.pbe_hessian,
                    relative_fraction=args.design_relative_fraction,
                    absolute_scale=args.design_absolute_hessian_scale,
                ),
            )
        )
        row_offset += coordinate_count**2
    metadata = {
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "source_summary": str(source_summary_path.resolve()),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_checkpoint_sha256": _sha256(args.source_run_dir / "best.ckpt"),
        "direction_manifest": str(args.direction_manifest.resolve()),
        "direction_manifest_sha256": _sha256(args.direction_manifest),
        "design_summary": str(design_summary_path.resolve()),
        "design_summary_sha256": _sha256(design_summary_path),
        "source_split_sha256": manifest["source_split_sha256"],
        "selected_parent_ids": design_order,
    }
    return parents, metadata


def _load_stage_designs(args: argparse.Namespace) -> list[tuple[str, np.ndarray, int]]:
    result = []
    for stage in args.stages:
        root = args.design_root / stage
        summary = json.loads((root / "summary.json").read_text())
        design = np.load(root / "weighted_hessian_design.npy", mmap_mode="r")
        feature_count = int(summary["shared_feature_count"])
        if design.shape[1] != feature_count:
            raise ValueError(f"feature count mismatch in {stage}")
        result.append((stage, design, feature_count))
    return result


def _symmetrized_local_basis(
    weighted_design: np.ndarray,
    parent: Parent,
) -> np.ndarray:
    coordinate_count = parent.pbe_hessian.shape[0]
    row_count = coordinate_count**2
    local = np.asarray(
        weighted_design[parent.row_offset : parent.row_offset + row_count],
        dtype=np.float64,
    ).reshape(coordinate_count, coordinate_count, -1)
    local = local / parent.hessian_weight
    return 0.5 * (local + np.swapaxes(local, 0, 1))


def _build_weighted_direction_design(
    parents: list[Parent],
    stages: list[tuple[str, np.ndarray, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    total_rows = sum(parent.train_directions.size for parent in parents)
    total_features = sum(feature_count for _, _, feature_count in stages)
    design = torch.empty(
        (total_rows, total_features), dtype=torch.float64, device=device
    )
    target = torch.empty(total_rows, dtype=torch.float64, device=device)
    row_offset = 0
    build_rows = []
    for parent in parents:
        started = time.perf_counter()
        directions = parent.train_directions
        coordinate_count = parent.pbe_hessian.shape[0]
        residual = parent.pbe_hessian - parent.source_hessian
        correction_target = np.einsum(
            "ij,dj->di", residual, directions, optimize=True
        )
        reference_hvp = np.einsum(
            "ij,dj->di", parent.pbe_hessian, directions, optimize=True
        )
        weights = mixed_hvp_row_weights(
            reference_hvp,
            parent_count=len(parents),
            absolute_scale=args.absolute_hvp_scale,
            relative_fraction=args.hvp_relative_fraction,
            reference_floor=args.hvp_reference_floor,
        )
        parent_rows = directions.size
        target[row_offset : row_offset + parent_rows] = torch.from_numpy(
            (correction_target * weights).reshape(-1)
        ).to(device=device)
        feature_offset = 0
        for stage, weighted_design, feature_count in stages:
            local = _symmetrized_local_basis(weighted_design, parent)
            directional = np.einsum(
                "ijm,dj->dim", local, directions, optimize=True
            )
            block = (directional * weights[:, :, None]).reshape(
                parent_rows, feature_count
            )
            design[
                row_offset : row_offset + parent_rows,
                feature_offset : feature_offset + feature_count,
            ] = torch.from_numpy(block).to(device=device)
            feature_offset += feature_count
        build_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "train_direction_count": int(directions.shape[0]),
                "coordinate_count": coordinate_count,
                "weighted_row_count": parent_rows,
                "weighted_row_start": row_offset,
                "weighted_row_stop": row_offset + parent_rows,
                "build_wall_time_s": time.perf_counter() - started,
            }
        )
        row_offset += parent_rows
    return design, target, build_rows


def _calibration_mask(
    row_count: int,
    build_rows: list[dict[str, Any]],
    *,
    stride: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Freeze every ``stride``-th train direction for ridge selection only."""
    if stride < 2 or not 0 <= offset < stride:
        raise ValueError("invalid calibration stride or offset")
    mask = torch.zeros(row_count, dtype=torch.bool, device=device)
    for row in build_rows:
        coordinate_count = int(row["coordinate_count"])
        direction_count = int(row["train_direction_count"])
        start = int(row["weighted_row_start"])
        for direction_index in range(offset, direction_count, stride):
            first = start + direction_index * coordinate_count
            mask[first : first + coordinate_count] = True
    if not bool(torch.any(mask)) or bool(torch.all(mask)):
        raise ValueError("calibration split must leave non-empty fit and calibration rows")
    return mask


def _solve_normalized(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    column_norm_relative_floor: float,
    cg_tolerance: float,
    cg_max_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    column_norm = torch.linalg.vector_norm(design, dim=0)
    floor = torch.max(column_norm) * column_norm_relative_floor
    active = column_norm > floor
    if not bool(torch.any(active)):
        raise RuntimeError("all descriptor columns are numerically inactive")
    normalized = design[:, active] / column_norm[active]
    normalized_solution, solver = conjugate_gradient_normal_equations(
        normalized,
        target,
        ridge=ridge,
        tolerance=cg_tolerance,
        max_iterations=cg_max_iterations,
    )
    coefficients = torch.zeros(
        design.shape[1], dtype=design.dtype, device=design.device
    )
    coefficients[active] = normalized_solution / column_norm[active]
    solver["active_feature_count"] = int(torch.count_nonzero(active).detach().cpu())
    return coefficients, active, column_norm, solver


def _select_ridge_on_train_calibration(
    design: torch.Tensor,
    target: torch.Tensor,
    build_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, Any]]]:
    calibration = _calibration_mask(
        design.shape[0],
        build_rows,
        stride=args.calibration_stride,
        offset=args.calibration_offset,
        device=design.device,
    )
    fit = ~calibration
    fit_design = design[fit]
    fit_target = target[fit]
    calibration_design = design[calibration]
    calibration_target = target[calibration]
    rows = []
    for ridge in args.ridge_grid:
        started = time.perf_counter()
        coefficients, _, _, solver = _solve_normalized(
            fit_design,
            fit_target,
            ridge=ridge,
            column_norm_relative_floor=args.column_norm_relative_floor,
            cg_tolerance=args.cg_tolerance,
            cg_max_iterations=args.cg_max_iterations,
        )
        fit_residual = fit_design @ coefficients - fit_target
        calibration_residual = calibration_design @ coefficients - calibration_target
        rows.append(
            {
                "ridge": ridge,
                "fit_weighted_squared_error": float(
                    torch.sum(fit_residual.square()).detach().cpu()
                ),
                "calibration_weighted_squared_error": float(
                    torch.sum(calibration_residual.square()).detach().cpu()
                ),
                "coefficient_norm": float(
                    torch.linalg.vector_norm(coefficients).detach().cpu()
                ),
                "wall_time_s": time.perf_counter() - started,
                **solver,
            }
        )
    selected = min(
        rows,
        key=lambda row: (row["calibration_weighted_squared_error"], row["ridge"]),
    )
    del fit_design, fit_target, calibration_design, calibration_target
    return float(selected["ridge"]), rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for key in (
        "relative_frobenius",
        "train_hvp_relative_frobenius",
        "heldout_hvp_relative_frobenius",
        "energy_abs_error_hartree",
        "force_mae_hartree_per_bohr",
        "antisymmetric_over_symmetric_frobenius",
    ):
        values = np.asarray([float(row[key]) for row in rows])
        result[f"median_{key}"] = float(np.median(values))
        result[f"p90_{key}"] = float(np.quantile(values, 0.9))
        result[f"max_{key}"] = float(np.max(values))
    return result


def _evaluate(
    parents: list[Parent],
    stages: list[tuple[str, np.ndarray, int]],
    coefficients: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    coefficient_offset = 0
    stage_coefficients = []
    for stage, design, feature_count in stages:
        stage_coefficients.append(
            (stage, design, coefficients[coefficient_offset : coefficient_offset + feature_count])
        )
        coefficient_offset += feature_count
    for parent in parents:
        correction = np.zeros_like(parent.source_hessian)
        for _, design, stage_coefficient in stage_coefficients:
            local = _symmetrized_local_basis(design, parent)
            correction += np.einsum(
                "ijm,m->ij", local, stage_coefficient, optimize=True
            )
        predicted = parent.source_hessian + correction
        difference = predicted - parent.pbe_hessian
        row: dict[str, Any] = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "energy_abs_error_hartree": abs(parent.source_energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.source_force - parent.pbe_force))
            ),
            **hessian_metrics(predicted, parent.pbe_hessian),
        }
        for role, directions in (
            ("train", parent.train_directions),
            ("heldout", parent.heldout_directions),
        ):
            error_hvp = np.einsum("ij,dj->di", difference, directions, optimize=True)
            reference_hvp = np.einsum(
                "ij,dj->di", parent.pbe_hessian, directions, optimize=True
            )
            row[f"{role}_hvp_mae"] = float(np.mean(np.abs(error_hvp)))
            row[f"{role}_hvp_rmse"] = float(np.sqrt(np.mean(error_hvp**2)))
            row[f"{role}_hvp_relative_frobenius"] = float(
                np.linalg.norm(error_hvp)
                / max(np.linalg.norm(reference_hvp), np.finfo(float).tiny)
            )
        rows.append(row)
        np.savez_compressed(
            output_dir / f"{parent.molecule_id}_result.npz",
            predicted_hessian=predicted,
            pbe_hessian=parent.pbe_hessian,
            correction_hessian=correction,
            predicted_force=parent.source_force,
            pbe_force=parent.pbe_force,
        )
    return _aggregate(rows), rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    parents, provenance = _load_data(args)
    stages = _load_stage_designs(args)
    design, target, build_rows = _build_weighted_direction_design(
        parents, stages, args, device
    )
    ridge_selection = []
    selected_ridge = args.ridge
    if args.ridge_grid:
        selected_ridge, ridge_selection = _select_ridge_on_train_calibration(
            design, target, build_rows, args
        )
    coefficients_tensor, active, column_norm, solver = _solve_normalized(
        design,
        target,
        ridge=selected_ridge,
        column_norm_relative_floor=args.column_norm_relative_floor,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
    )
    coefficients = coefficients_tensor.detach().cpu().numpy()
    del design, target
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final, rows = _evaluate(parents, stages, coefficients, args.output_dir)
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "coefficients.npz",
        coefficients=coefficients,
        active_columns=active.detach().cpu().numpy(),
        column_norms=column_norm.detach().cpu().numpy(),
        stages=np.asarray([stage for stage, _, _ in stages]),
        stage_feature_counts=np.asarray([count for _, _, count in stages]),
    )
    result = {
        "definition": (
            "train-direction-only shared linear three-/four-body descriptor scalar; "
            "its value and first Taylor term are removed independently at each reference "
            "geometry, preserving source E/F while retaining a conservative Hessian correction"
        ),
        "formal_limit": (
            "reference-geometry anchoring is a local curvature-capacity audit, not yet one "
            "globally transferable total-energy functional"
        ),
        "stages": [stage for stage, _, _ in stages],
        "feature_count": int(coefficients.size),
        "active_feature_count": int(torch.count_nonzero(active).detach().cpu()),
        "ridge": selected_ridge,
        "ridge_selection_definition": (
            "deterministic within-train-direction calibration; original held-out "
            "directions are never used for ridge selection"
            if args.ridge_grid
            else "fixed before evaluation"
        ),
        "ridge_selection": ridge_selection,
        "calibration_stride": args.calibration_stride if args.ridge_grid else None,
        "calibration_offset": args.calibration_offset if args.ridge_grid else None,
        "column_norm_relative_floor": args.column_norm_relative_floor,
        "descriptor_basis_symmetrized": True,
        "energy_force_anchor_exact_by_construction": True,
        "solver": solver,
        "build_rows": build_rows,
        "final": final,
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
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
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["three_body", "four_body"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--ridge-grid", type=float, nargs="*", default=[])
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--calibration-offset", type=int, default=4)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int, default=1000)
    parser.add_argument("--column-norm-relative-floor", type=float, default=1e-12)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    parser.add_argument("--design-relative-fraction", type=float, default=0.5)
    parser.add_argument("--design-absolute-hessian-scale", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
