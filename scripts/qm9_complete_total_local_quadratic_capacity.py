#!/usr/bin/env python3
"""Fit a conservative local internal-coordinate quadratic scalar to train HVPs."""

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

from mldft.ml.models.components.local_quadratic_residual import (
    QuadraticTerm,
    assemble_quadratic_hessian,
    build_internal_coordinate_specs,
    build_quadratic_terms,
    internal_coordinate_values_and_jacobian,
)

try:
    from scripts.qm9_complete_total_anchored_linear_direction_capacity import (
        _select_ridge_on_train_calibration,
        _solve_normalized,
        mixed_hvp_row_weights,
    )
    from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:
    from qm9_complete_total_anchored_linear_direction_capacity import (
        _select_ridge_on_train_calibration,
        _solve_normalized,
        mixed_hvp_row_weights,
    )
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
    jacobian: np.ndarray
    terms: list[QuadraticTerm]
    coordinate_kind_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(args: argparse.Namespace) -> tuple[list[Parent], dict[str, Any]]:
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
        positions = torch.as_tensor(state.positions_bohr, dtype=torch.float64)
        specs = build_internal_coordinate_specs(
            torch.as_tensor(state.atomic_numbers, dtype=torch.long),
            positions,
            bond_scale=args.bond_scale,
        )
        values, jacobian = internal_coordinate_values_and_jacobian(positions, specs)
        if not bool(torch.all(torch.isfinite(jacobian))):
            raise ValueError(f"non-finite internal-coordinate Jacobian for {molecule_id}")
        terms = build_quadratic_terms(
            specs,
            values,
            include_cross_terms=args.include_cross_terms,
            cross_all_pairs=args.cross_all_pairs,
            use_environment_types=args.use_environment_types,
        )
        if args.parent_specific_keys:
            terms = [
                QuadraticTerm(
                    key=f"{molecule_id}|{term.key}",
                    first=term.first,
                    second=term.second,
                    scale=term.scale,
                )
                for term in terms
            ]
        kind_counts: dict[str, int] = {}
        for spec in specs:
            kind_counts[spec.kind] = kind_counts.get(spec.kind, 0) + 1
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
                jacobian=jacobian.detach().cpu().numpy(),
                terms=terms,
                coordinate_kind_counts=kind_counts,
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


def _directional_block(
    parent: Parent,
    directions: np.ndarray,
    key_to_index: dict[str, int],
) -> np.ndarray:
    direction_projection = directions @ parent.jacobian.T
    row_count = directions.size
    block = np.zeros((row_count, len(key_to_index)), dtype=np.float64)
    coordinate_count = parent.jacobian.shape[1]
    for term in parent.terms:
        first_vector = parent.jacobian[term.first]
        if term.first == term.second:
            product = (
                direction_projection[:, term.first, None]
                * first_vector[None, :]
            )
        else:
            second_vector = parent.jacobian[term.second]
            product = (
                direction_projection[:, term.second, None]
                * first_vector[None, :]
                + direction_projection[:, term.first, None]
                * second_vector[None, :]
            )
        block[:, key_to_index[term.key]] += (term.scale * product).reshape(
            -1
        )
    if block.shape[0] != directions.shape[0] * coordinate_count:
        raise RuntimeError("internal directional design shape mismatch")
    return block


def _build_design(
    parents: list[Parent],
    key_to_index: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    total_rows = sum(parent.train_directions.size for parent in parents)
    design = torch.empty(
        (total_rows, len(key_to_index)), dtype=torch.float64, device=device
    )
    target = torch.empty(total_rows, dtype=torch.float64, device=device)
    build_rows = []
    row_offset = 0
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
        block = _directional_block(parent, directions, key_to_index)
        parent_rows = directions.size
        design[row_offset : row_offset + parent_rows] = torch.from_numpy(
            block * weights.reshape(-1, 1)
        ).to(device=device)
        target[row_offset : row_offset + parent_rows] = torch.from_numpy(
            (correction_target * weights).reshape(-1)
        ).to(device=device)
        build_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "coordinate_count": coordinate_count,
                "internal_coordinate_count": int(parent.jacobian.shape[0]),
                "quadratic_term_count": len(parent.terms),
                "coordinate_kind_counts": parent.coordinate_kind_counts,
                "train_direction_count": int(directions.shape[0]),
                "weighted_row_count": parent_rows,
                "weighted_row_start": row_offset,
                "weighted_row_stop": row_offset + parent_rows,
                "build_wall_time_s": time.perf_counter() - started,
            }
        )
        row_offset += parent_rows
    return design, target, build_rows


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
    keys: list[str],
    coefficients: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    coefficient_lookup = dict(zip(keys, coefficients.tolist(), strict=True))
    rows = []
    for parent in parents:
        correction = assemble_quadratic_hessian(
            torch.from_numpy(parent.jacobian),
            parent.terms,
            coefficient_lookup,
        ).numpy()
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


def _run_per_parent(
    parents: list[Parent],
    provenance: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    started: float,
) -> dict[str, Any]:
    rows = []
    fit_summaries = []
    total_feature_count = 0
    total_active_count = 0
    for parent in parents:
        keys = sorted({term.key for term in parent.terms})
        key_to_index = {key: index for index, key in enumerate(keys)}
        design, target, build_rows = _build_design(
            [parent], key_to_index, args, device
        )
        ridge_selection = []
        selected_ridge = args.ridge
        if args.ridge_grid:
            selected_ridge, ridge_selection = _select_ridge_on_train_calibration(
                design, target, build_rows, args
            )
        coefficient_tensor, active, column_norm, solver = _solve_normalized(
            design,
            target,
            ridge=selected_ridge,
            column_norm_relative_floor=args.column_norm_relative_floor,
            cg_tolerance=args.cg_tolerance,
            cg_max_iterations=args.cg_max_iterations,
        )
        coefficients = coefficient_tensor.detach().cpu().numpy()
        _, parent_rows = _evaluate(
            [parent], keys, coefficients, args.output_dir
        )
        rows.extend(parent_rows)
        active_count = int(torch.count_nonzero(active).detach().cpu())
        total_feature_count += len(keys)
        total_active_count += active_count
        fit_summaries.append(
            {
                "molecule_id": parent.molecule_id,
                "feature_count": len(keys),
                "active_feature_count": active_count,
                "ridge": selected_ridge,
                "ridge_selection": ridge_selection,
                "solver": solver,
                "build": build_rows[0],
            }
        )
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_coefficients.npz",
            coefficients=coefficients,
            feature_keys=np.asarray(keys),
            active_columns=active.detach().cpu().numpy(),
            column_norms=column_norm.detach().cpu().numpy(),
        )
        del design, target, coefficient_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "definition": (
            "parent-specific reference-anchored internal-coordinate quadratic scalar "
            "oracle fit only to each parent's train directions"
        ),
        "formal_limit": (
            "capacity/identifiability oracle only: parent-specific coefficients are not "
            "a transferable model and cannot advance to unseen-parent validation"
        ),
        "include_cross_terms": args.include_cross_terms,
        "cross_all_pairs": args.cross_all_pairs,
        "use_environment_types": args.use_environment_types,
        "parent_specific_keys": args.parent_specific_keys,
        "solve_per_parent": True,
        "bond_scale": args.bond_scale,
        "feature_count_sum": total_feature_count,
        "active_feature_count_sum": total_active_count,
        "ridge_selection_definition": (
            "independent deterministic calibration split within each parent's original "
            "train directions; original held-out directions remain untouched"
        ),
        "per_parent_fits": fit_summaries,
        "final": _aggregate(rows),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    parents, provenance = _load_data(args)
    if args.solve_per_parent:
        if not args.parent_specific_keys:
            raise ValueError("solve_per_parent requires parent_specific_keys")
        return _run_per_parent(parents, provenance, args, device, started)
    keys = sorted({term.key for parent in parents for term in parent.terms})
    key_to_index = {key: index for index, key in enumerate(keys)}
    design, target, build_rows = _build_design(parents, key_to_index, args, device)
    ridge_selection = []
    selected_ridge = args.ridge
    if args.ridge_grid:
        selected_ridge, ridge_selection = _select_ridge_on_train_calibration(
            design, target, build_rows, args
        )
    coefficient_tensor, active, column_norm, solver = _solve_normalized(
        design,
        target,
        ridge=selected_ridge,
        column_norm_relative_floor=args.column_norm_relative_floor,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
    )
    coefficients = coefficient_tensor.detach().cpu().numpy()
    final, rows = _evaluate(parents, keys, coefficients, args.output_dir)
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "coefficients.npz",
        coefficients=coefficients,
        feature_keys=np.asarray(keys),
        active_columns=active.detach().cpu().numpy(),
        column_norms=column_norm.detach().cpu().numpy(),
    )
    result = {
        "definition": (
            "reference-anchored local internal-coordinate quadratic scalar with "
            "bond/angle/torsion diagonal force constants and optional parity-even "
            "shared local cross couplings; E/F/Hessian are derivatives of one scalar"
        ),
        "formal_limit": (
            "this is a reference-local curvature-capacity model, not yet one globally "
            "transferable nonlinear total-energy functional"
        ),
        "include_cross_terms": args.include_cross_terms,
        "cross_all_pairs": args.cross_all_pairs,
        "use_environment_types": args.use_environment_types,
        "parent_specific_keys": args.parent_specific_keys,
        "solve_per_parent": False,
        "bond_scale": args.bond_scale,
        "feature_count": len(keys),
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
        "energy_force_anchor_exact_by_construction": True,
        "parity_odd_cross_terms_excluded": True,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bond-scale", type=float, default=1.25)
    parser.add_argument("--include-cross-terms", action="store_true")
    parser.add_argument("--cross-all-pairs", action="store_true")
    parser.add_argument("--use-environment-types", action="store_true")
    parser.add_argument("--parent-specific-keys", action="store_true")
    parser.add_argument("--solve-per-parent", action="store_true")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--ridge-grid", type=float, nargs="*", default=[])
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--calibration-offset", type=int, default=4)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int, default=1000)
    parser.add_argument("--column-norm-relative-floor", type=float, default=1e-12)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
