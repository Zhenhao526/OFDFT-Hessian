#!/usr/bin/env python3
"""Fit shared conservative three-/four-body residuals on frozen Stage-1 parents."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zarr
from scipy.sparse.linalg import LinearOperator, lsqr

try:
    from scripts.qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from mldft.ml.models.components.three_body_geometry_residual import (
        build_triplet_groups,
        make_three_body_feature_function,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qm9_complete_total_geometry_four_body_capacity import (
        build_bonded_chain_groups,
        make_four_body_feature_function,
    )
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics
    from mldft.ml.models.components.three_body_geometry_residual import (
        build_triplet_groups,
        make_three_body_feature_function,
    )


@dataclass
class ParentState:
    molecule_id: str
    atomic_numbers: np.ndarray
    positions_bohr: np.ndarray
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    energy: float
    force: np.ndarray
    hessian: np.ndarray


@dataclass
class FeatureDesign:
    molecule_id: str
    keys: list[tuple[int, ...]]
    energy: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    metadata: dict[str, object]


def _load_parents(manifest_path: Path) -> tuple[list[ParentState], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("baseline manifest does not certify frozen Test100")
    parents = []
    for row in manifest["parents"]:
        label = zarr.open(row["label_path"], mode="r")
        atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
        pbe_force = np.asarray(
            label["metadata/pbe_derivatives/forces"], dtype=np.float64
        )
        energy_trace = np.asarray(label["ks_labels/energies/e_tot"], dtype=np.float64)
        has_energy = np.asarray(
            label["ks_labels/energies/has_energy_label"], dtype=np.bool_
        )
        with np.load(row["capacity_array"]) as payload:
            predicted_hessian = np.asarray(
                payload["predicted_hessian"], dtype=np.float64
            )
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
            predicted_force = np.asarray(
                payload["predicted_base_force"], dtype=np.float64
            )
        parents.append(
            ParentState(
                molecule_id=str(row["molecule_id"]),
                atomic_numbers=atomic_numbers,
                positions_bohr=positions,
                pbe_energy=float(energy_trace[has_energy][-1]),
                pbe_force=pbe_force,
                pbe_hessian=pbe_hessian,
                energy=float(row["baseline_total_energy_hartree"]),
                force=predicted_force,
                hessian=predicted_hessian,
            )
        )
    return parents, manifest


def _feature_derivative_design(
    feature_function: Callable[[torch.Tensor], torch.Tensor],
    positions_numpy: np.ndarray,
    derivative_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = torch.tensor(positions_numpy, dtype=torch.float64)
    energy = feature_function(positions)
    jacobian = torch.func.jacfwd(feature_function)(positions).reshape(
        energy.numel(), -1
    )
    columns = []
    for coordinate in range(positions.numel()):
        direction = torch.zeros_like(positions).reshape(-1)
        direction[coordinate] = 1.0
        direction = direction.reshape_as(positions)
        plus = torch.func.jacfwd(feature_function)(
            positions + derivative_step * direction
        ).reshape(energy.numel(), -1)
        minus = torch.func.jacfwd(feature_function)(
            positions - derivative_step * direction
        ).reshape(energy.numel(), -1)
        columns.append(((plus - minus) / (2.0 * derivative_step)).T)
    hessian = torch.stack(columns, dim=1)
    return (
        energy.detach().cpu().numpy(),
        jacobian.detach().cpu().numpy().T,
        hessian.detach().cpu().numpy().reshape(-1, energy.numel()),
    )


def _build_design(
    parent: ParentState,
    stage: str,
    args: argparse.Namespace,
) -> FeatureDesign:
    started = time.perf_counter()
    if stage == "three_body":
        groups = build_triplet_groups(parent.atomic_numbers)
        centers = np.linspace(
            args.three_body_center_min,
            args.three_body_center_max,
            args.three_body_center_count,
        )
        function, keys = make_three_body_feature_function(
            groups,
            centers,
            args.three_body_sigma,
            args.three_body_angular_order,
        )
        metadata = {"group_count": len(groups), "bond_count": None}
    elif stage == "four_body":
        groups, bonds = build_bonded_chain_groups(
            parent.atomic_numbers, parent.positions_bohr, args.four_body_bond_scale
        )
        centers = np.linspace(
            args.four_body_center_min,
            args.four_body_center_max,
            args.four_body_center_count,
        )
        function, keys = make_four_body_feature_function(
            groups,
            centers,
            args.four_body_sigma,
            args.four_body_torsion_order,
        )
        metadata = {"group_count": len(groups), "bond_count": len(bonds)}
    else:
        raise ValueError(f"unsupported stage: {stage}")
    energy, gradient, hessian = _feature_derivative_design(
        function, parent.positions_bohr, args.derivative_step
    )
    metadata.update(
        {
            "feature_count": len(keys),
            "build_wall_time_s": time.perf_counter() - started,
        }
    )
    return FeatureDesign(
        molecule_id=parent.molecule_id,
        keys=keys,
        energy=energy,
        gradient=gradient,
        hessian=hessian,
        metadata=metadata,
    )


def constrained_lsqr(
    design: np.ndarray,
    target: np.ndarray,
    constraints: np.ndarray,
    constraint_target: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    svd_tolerance: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a large shared constrained least-squares problem without a null basis."""
    constraint_u, singular, constraint_vt = np.linalg.svd(
        constraints, full_matrices=False
    )
    threshold = svd_tolerance * max(constraints.shape) * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > threshold))
    row_basis = constraint_vt[:rank]

    def project(vector: np.ndarray) -> np.ndarray:
        return vector - row_basis.T @ (row_basis @ vector)

    particular = row_basis.T @ (
        (constraint_u[:, :rank].T @ constraint_target) / singular[:rank]
    )
    residual_target = target - design @ particular
    operator = LinearOperator(
        shape=design.shape,
        dtype=np.float64,
        matvec=lambda vector: design @ project(vector),
        rmatvec=lambda vector: project(design.T @ vector),
    )
    solution = lsqr(
        operator,
        residual_target,
        atol=tolerance,
        btol=tolerance,
        iter_lim=max_iterations,
        show=False,
    )
    coefficients = particular + project(solution[0])
    return coefficients, {
        "constraint_rank": rank,
        "lsqr_stop_code": int(solution[1]),
        "lsqr_iterations": int(solution[2]),
        "lsqr_residual_norm": float(solution[3]),
        "lsqr_normal_residual_norm": float(solution[4]),
        "lsqr_operator_norm": float(solution[5]),
        "lsqr_condition_estimate": float(solution[6]),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "constraint_residual_max_abs": float(
            np.max(np.abs(constraints @ coefficients - constraint_target))
        ),
    }


def _fit_shared_stage(
    parents: list[ParentState],
    designs: list[FeatureDesign],
    stage: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[tuple[int, ...]], dict[str, object]]:
    key_union = sorted({key for design in designs for key in design.keys})
    key_to_index = {key: index for index, key in enumerate(key_union)}
    hessian_row_count = sum(parent.hessian.size for parent in parents)
    constraint_row_count = sum(1 + parent.force.size for parent in parents)
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    weighted_design = np.lib.format.open_memmap(
        stage_dir / "weighted_hessian_design.npy",
        mode="w+",
        dtype=np.float64,
        shape=(hessian_row_count, len(key_union)),
    )
    weighted_design[:] = 0.0
    weighted_target = np.zeros(hessian_row_count, dtype=np.float64)
    constraints = np.lib.format.open_memmap(
        stage_dir / "anchor_constraints.npy",
        mode="w+",
        dtype=np.float64,
        shape=(constraint_row_count, len(key_union)),
    )
    constraints[:] = 0.0
    constraint_target = np.zeros(constraint_row_count, dtype=np.float64)
    hessian_offset = 0
    constraint_offset = 0
    for parent, local in zip(parents, designs, strict=True):
        indices = np.asarray([key_to_index[key] for key in local.keys], dtype=np.int64)
        local_rows = parent.hessian.size
        reference_norm = max(np.linalg.norm(parent.pbe_hessian), np.finfo(float).tiny)
        absolute_weight = 1.0 / (
            args.absolute_hessian_scale * np.sqrt(local_rows)
        )
        relative_weight = 1.0 / reference_norm
        weight = np.sqrt(
            args.relative_loss_fraction * relative_weight**2
            + (1.0 - args.relative_loss_fraction) * absolute_weight**2
        )
        row_indices = np.arange(hessian_offset, hessian_offset + local_rows)
        weighted_design[np.ix_(row_indices, indices)] = local.hessian * weight
        weighted_target[row_indices] = (
            parent.pbe_hessian - parent.hessian
        ).reshape(-1) * weight
        constraint_rows = 1 + parent.force.size
        local_constraint = np.concatenate(
            (local.energy[None, :], local.gradient), axis=0
        )
        constraint_indices = np.arange(
            constraint_offset, constraint_offset + constraint_rows
        )
        constraints[np.ix_(constraint_indices, indices)] = local_constraint
        if stage == "three_body":
            constraint_target[constraint_indices] = np.concatenate(
                (
                    np.asarray([parent.pbe_energy - parent.energy]),
                    -(parent.pbe_force - parent.force).reshape(-1),
                )
            )
        hessian_offset += local_rows
        constraint_offset += constraint_rows
    weighted_design.flush()
    constraints.flush()
    np.save(stage_dir / "feature_keys.npy", np.asarray(key_union, dtype=np.int64))
    if bool(getattr(args, "design_only", False)):
        summary = {
            "stage": stage,
            "shared_feature_count": len(key_union),
            "parent_count": len(parents),
            "design_only": True,
            "weighted_hessian_design": str(
                (stage_dir / "weighted_hessian_design.npy").resolve()
            ),
            "anchor_constraints": str(
                (stage_dir / "anchor_constraints.npy").resolve()
            ),
        }
        (stage_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return np.zeros(len(key_union)), key_union, summary
    coefficients, solver = constrained_lsqr(
        weighted_design,
        weighted_target,
        constraints,
        constraint_target,
        tolerance=args.lsqr_tolerance,
        max_iterations=args.lsqr_max_iterations,
        svd_tolerance=args.svd_tolerance,
    )
    np.savez(
        stage_dir / "shared_coefficients.npz",
        coefficients=coefficients,
        feature_keys=np.asarray(key_union, dtype=np.int64),
    )

    rows = []
    for parent, local in zip(parents, designs, strict=True):
        indices = np.asarray([key_to_index[key] for key in local.keys], dtype=np.int64)
        local_coefficients = coefficients[indices]
        energy_correction = float(local.energy @ local_coefficients)
        force_correction = -(local.gradient @ local_coefficients).reshape(
            parent.force.shape
        )
        hessian_correction = (local.hessian @ local_coefficients).reshape(
            parent.hessian.shape
        )
        parent.energy += energy_correction
        parent.force += force_correction
        parent.hessian += hessian_correction
        metrics = hessian_metrics(parent.hessian, parent.pbe_hessian)
        row = {
            "stage": stage,
            "molecule_id": parent.molecule_id,
            "natoms": int(parent.atomic_numbers.size),
            "energy_abs_error_hartree": abs(parent.energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.force - parent.pbe_force))
            ),
            **metrics,
            **local.metadata,
        }
        rows.append(row)
        np.savez(
            stage_dir / f"{parent.molecule_id}_result.npz",
            predicted_hessian=parent.hessian,
            pbe_hessian=parent.pbe_hessian,
            predicted_force=parent.force,
            pbe_force=parent.pbe_force,
            predicted_energy=parent.energy,
            pbe_energy=parent.pbe_energy,
            correction_hessian=hessian_correction,
        )
    with (stage_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": stage,
        "shared_feature_count": len(key_union),
        "parent_count": len(parents),
        "per_parent": rows,
        "median_relative_frobenius": float(
            np.median([row["relative_frobenius"] for row in rows])
        ),
        "max_relative_frobenius": float(
            max(row["relative_frobenius"] for row in rows)
        ),
        "all_below_0p05": all(row["relative_frobenius"] <= 0.05 for row in rows),
        "solver": solver,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return coefficients, key_union, summary


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    parents, baseline_manifest = _load_parents(args.baseline_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    for parent in parents:
        baseline_rows.append(
            {
                "stage": "baseline",
                "molecule_id": parent.molecule_id,
                "natoms": int(parent.atomic_numbers.size),
                "energy_abs_error_hartree": abs(parent.energy - parent.pbe_energy),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(parent.force - parent.pbe_force))
                ),
                **hessian_metrics(parent.hessian, parent.pbe_hessian),
            }
        )

    stage_summaries = []
    design_manifest = []
    for stage in ("three_body", "four_body"):
        designs = []
        for parent in parents:
            design = _build_design(parent, stage, args)
            designs.append(design)
            design_manifest.append(
                {"stage": stage, "molecule_id": parent.molecule_id, **design.metadata}
            )
        _, _, summary = _fit_shared_stage(
            parents, designs, stage, args.output_dir, args
        )
        stage_summaries.append(summary)

    final = stage_summaries[-1]
    result = {
        "definition": (
            "shared train-only conservative scalar three-body plus parity-even four-body "
            "capacity fit"
        ),
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "source_split_sha256": baseline_manifest["source_split_sha256"],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_ids": [parent.molecule_id for parent in parents],
        "baseline": baseline_rows,
        "stages": stage_summaries,
        "design_manifest": design_manifest,
        "design_only": args.design_only,
        "stage1_shared_gate_passed": bool(
            not args.design_only and final["all_below_0p05"]
        ),
        "wall_time_s": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--derivative-step", type=float, default=1.0e-3)
    parser.add_argument("--absolute-hessian-scale", type=float, default=0.1)
    parser.add_argument("--relative-loss-fraction", type=float, default=0.5)
    parser.add_argument("--lsqr-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--lsqr-max-iterations", type=int, default=1000)
    parser.add_argument("--svd-tolerance", type=float, default=1.0e-10)
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
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="Build descriptor derivative matrices without fitting full-Hessian coefficients.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
