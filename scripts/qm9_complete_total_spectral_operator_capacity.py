#!/usr/bin/env python3
"""Fit a frozen-direction equivariant spectral Hessian operator as an anchored scalar."""

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

from mldft.ml.models.components.spectral_hessian_residual import (
    equivariant_block_operator_basis,
    spectral_operator_basis,
    symmetric_hvp_interpolant,
)
from scripts.qm9_complete_total_anchored_linear_direction_capacity import mixed_hvp_row_weights
from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics


@dataclass
class Parent:
    molecule_id: str
    natoms: int
    atomic_numbers: np.ndarray
    pbe_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    source_energy: float
    source_force: np.ndarray
    source_hessian: np.ndarray
    fit_directions: np.ndarray
    calibration_directions: np.ndarray
    train_directions: np.ndarray
    heldout_directions: np.ndarray
    operator_basis: np.ndarray
    conditioned_basis_count: int
    block_basis_count: int
    spectral_scale: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(args: argparse.Namespace) -> tuple[list[Parent], dict[str, Any], list[str]]:
    baseline, baseline_manifest = _load_parents(args.baseline_manifest)
    source_from_baseline = bool(getattr(args, "source_from_baseline", False))
    source_run_dir = getattr(args, "source_run_dir", None)
    if source_from_baseline == (source_run_dir is not None):
        raise ValueError("choose exactly one of source_run_dir or source_from_baseline")
    source_summary_path = None
    source_summary = None
    source_rows: dict[str, dict[str, Any]] = {}
    if not source_from_baseline:
        source_summary_path = source_run_dir / "summary.json"
        source_summary = json.loads(source_summary_path.read_text())
        if source_summary.get("test100_accessed") is not False:
            raise ValueError("source run does not certify frozen Test100")
        source_rows = {
            str(row["molecule_id"]): row for row in source_summary["per_parent"]
        }
    direction_manifest = json.loads(args.direction_manifest.read_text())
    if direction_manifest.get("test100_accessed") is not False:
        raise ValueError("direction manifest does not certify frozen Test100")
    direction_rows = {str(row["molecule_id"]): row for row in direction_manifest["directions"]}
    expected = {parent.molecule_id for parent in baseline}
    if (
        (not source_from_baseline and set(source_rows) != expected)
        or not expected.issubset(direction_rows)
    ):
        raise ValueError("baseline, source, and direction parent IDs differ")

    parents = []
    basis_names: list[str] | None = None
    for state in baseline:
        molecule_id = state.molecule_id
        if source_from_baseline:
            source_energy = float(state.energy)
            source_force = np.asarray(state.force, dtype=np.float64).copy()
            source_hessian = np.asarray(state.hessian, dtype=np.float64).copy()
        else:
            with np.load(source_run_dir / f"{molecule_id}_result.npz") as payload:
                source_force = np.asarray(payload["predicted_force"], dtype=np.float64)
                source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
                saved_reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
            if not np.array_equal(saved_reference, state.pbe_hessian):
                raise ValueError(f"PBE Hessian mismatch for {molecule_id}")
            source_energy = float(source_rows[molecule_id]["predicted_energy"])
        with np.load(direction_rows[molecule_id]["direction_path"]) as payload:
            directions = np.asarray(payload["directions"], dtype=np.float64)
            roles = np.asarray(payload["roles"]).astype(str)
            external_basis = np.asarray(payload["external_basis"], dtype=np.float64)
        train = directions[roles == "train"]
        calibration = np.arange(train.shape[0]) % args.calibration_stride == args.calibration_offset
        if not np.any(calibration) or np.all(calibration):
            raise ValueError(f"invalid fit/calibration split for {molecule_id}")
        matrices, names, _, scale = spectral_operator_basis(
            torch.from_numpy(source_hessian),
            torch.from_numpy(external_basis),
            polynomial_degree=args.polynomial_degree,
            rbf_centers=tuple(args.rbf_centers),
            rbf_width=args.rbf_width,
        )
        conditioned_basis_count = int(matrices.shape[0])
        block_basis_count = 0
        if args.include_block_basis:
            block_matrices, block_names = equivariant_block_operator_basis(
                torch.from_numpy(source_hessian),
                torch.as_tensor(state.atomic_numbers, dtype=torch.long),
                torch.as_tensor(state.positions_bohr, dtype=torch.float64),
                torch.from_numpy(external_basis),
            )
            matrices = torch.cat((matrices, block_matrices), dim=0)
            names = [*names, *block_names]
            block_basis_count = int(block_matrices.shape[0])
        if basis_names is None:
            basis_names = names
        elif basis_names != names:
            raise ValueError("inconsistent spectral basis names")
        parents.append(
            Parent(
                molecule_id=molecule_id,
                natoms=int(state.atomic_numbers.size),
                atomic_numbers=np.asarray(state.atomic_numbers, dtype=np.int64),
                pbe_energy=state.pbe_energy,
                pbe_force=state.pbe_force.copy(),
                pbe_hessian=state.pbe_hessian.copy(),
                source_energy=source_energy,
                source_force=source_force,
                source_hessian=source_hessian,
                fit_directions=train[~calibration],
                calibration_directions=train[calibration],
                train_directions=train,
                heldout_directions=directions[roles == "heldout"],
                operator_basis=matrices.numpy(),
                conditioned_basis_count=conditioned_basis_count,
                block_basis_count=block_basis_count,
                spectral_scale=scale,
            )
        )
    metadata: dict[str, Any] = {
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "source_definition": (
            "strict_complete_total_original_A_baseline_manifest"
            if source_from_baseline
            else "frozen_geometry_residual_source_run"
        ),
        "direction_manifest": str(args.direction_manifest.resolve()),
        "direction_manifest_sha256": _sha256(args.direction_manifest),
        "source_split_sha256": baseline_manifest["source_split_sha256"],
        "selected_parent_ids": [parent.molecule_id for parent in parents],
    }
    if source_from_baseline:
        metadata["source_baseline_manifest_sha256"] = _sha256(args.baseline_manifest)
        metadata["source_checkpoint_sha256"] = None
    else:
        assert source_summary_path is not None
        metadata.update(
            {
                "source_summary": str(source_summary_path.resolve()),
                "source_summary_sha256": _sha256(source_summary_path),
                "source_checkpoint_sha256": _sha256(source_run_dir / "best.ckpt"),
            }
        )
    return parents, metadata, basis_names or []


def _parent_feature_matrix(parents: list[Parent]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    raw_rows = []
    for parent in parents:
        fractions = [np.mean(parent.atomic_numbers == value) for value in (6, 7, 8, 9)]
        raw_rows.append(
            [
                parent.natoms / 20.0,
                *fractions,
                np.mean(parent.atomic_numbers) / 10.0,
                np.log(max(parent.spectral_scale, np.finfo(float).tiny)),
            ]
        )
    raw = np.asarray(raw_rows, dtype=np.float64)
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (raw - mean) / scale
    features = np.concatenate((np.ones((len(parents), 1)), normalized), axis=1)
    names = ["constant", "natoms", "fraction_C", "fraction_N", "fraction_O", "fraction_F", "mean_Z", "log_spectral_scale"]
    return features, names, {"mean": mean.tolist(), "scale": scale.tolist()}


def block_parent_conditioned_basis(
    block_basis: np.ndarray,
    parent_features: np.ndarray,
    *,
    pair_type_count: int = 15,
    radial_count: int = 4,
    pair_transform_count: int = 8,
    element_count: int = 5,
    diagonal_transform_count: int = 6,
) -> np.ndarray:
    """Low-rank chemistry conditioning without a full feature tensor product."""
    block_basis = np.asarray(block_basis, dtype=np.float64)
    parent_features = np.asarray(parent_features, dtype=np.float64)
    pair_count = pair_type_count * radial_count * pair_transform_count
    diagonal_count = element_count * diagonal_transform_count
    if block_basis.shape[0] != pair_count + diagonal_count:
        raise ValueError("block basis does not match the registered layout")
    if parent_features.ndim != 1 or parent_features.size < 2:
        raise ValueError("parent_features must include constant and nonconstant channels")
    matrix_shape = block_basis.shape[1:]
    pair = block_basis[:pair_count].reshape(
        pair_type_count, radial_count, pair_transform_count, *matrix_shape
    )
    diagonal = block_basis[pair_count:].reshape(
        element_count, diagonal_transform_count, *matrix_shape
    )
    pair_type_aggregate = np.sum(pair, axis=0).reshape(-1, *matrix_shape)
    radial_aggregate = np.sum(pair, axis=1).reshape(-1, *matrix_shape)
    element_aggregate = np.sum(diagonal, axis=0).reshape(-1, *matrix_shape)
    shared = np.concatenate(
        (pair_type_aggregate, radial_aggregate, element_aggregate), axis=0
    )
    return np.concatenate(
        [feature * shared for feature in parent_features[1:]], axis=0
    )


def _combined_basis(
    parent: Parent,
    features: np.ndarray,
    *,
    block_parent_conditioning: bool,
) -> np.ndarray:
    conditioned = np.einsum(
        "f,mij->fmij",
        features,
        parent.operator_basis[: parent.conditioned_basis_count],
        optimize=True,
    ).reshape(
        features.size * parent.conditioned_basis_count,
        parent.source_hessian.shape[0],
        parent.source_hessian.shape[1],
    )
    unconditioned = parent.operator_basis[parent.conditioned_basis_count :]
    parts = [conditioned, unconditioned]
    if block_parent_conditioning:
        if parent.block_basis_count == 0:
            raise ValueError("block parent conditioning requires the block basis")
        parts.append(block_parent_conditioned_basis(unconditioned, features))
    return np.concatenate(parts, axis=0)


def _weighted_system(
    parents: list[Parent],
    parent_features: np.ndarray,
    role: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = []
    targets = []
    for index, parent in enumerate(parents):
        directions = getattr(parent, f"{role}_directions")
        basis = _combined_basis(
            parent,
            parent_features[index],
            block_parent_conditioning=args.block_parent_conditioning,
        )
        residual = parent.pbe_hessian - parent.source_hessian
        target_hvp = np.einsum("ij,dj->di", residual, directions, optimize=True)
        reference_hvp = np.einsum("ij,dj->di", parent.pbe_hessian, directions, optimize=True)
        weights = mixed_hvp_row_weights(
            reference_hvp,
            parent_count=len(parents),
            absolute_scale=args.absolute_hvp_scale,
            relative_fraction=args.hvp_relative_fraction,
            reference_floor=args.hvp_reference_floor,
        )
        directional = np.einsum("mij,dj->dim", basis, directions, optimize=True)
        blocks.append((directional * weights[:, :, None]).reshape(-1, basis.shape[0]))
        targets.append((target_hvp * weights).reshape(-1))
    return torch.from_numpy(np.concatenate(blocks)), torch.from_numpy(np.concatenate(targets))


def _solve(design: torch.Tensor, target: torch.Tensor, ridge: float) -> tuple[torch.Tensor, dict[str, Any]]:
    norms = torch.linalg.vector_norm(design, dim=0)
    active = norms > torch.max(norms) * 1e-12
    normalized = design[:, active] / norms[active]
    gram = normalized.T @ normalized
    right = normalized.T @ target
    solution = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype), right)
    coefficients = torch.zeros(design.shape[1], dtype=design.dtype)
    coefficients[active] = solution / norms[active]
    residual = design @ coefficients - target
    return coefficients, {
        "active_feature_count": int(torch.count_nonzero(active)),
        "weighted_squared_error": float(torch.sum(residual.square())),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
    }


def _relative_hvp(predicted: np.ndarray, reference: np.ndarray, directions: np.ndarray) -> float:
    error = np.einsum("ij,dj->di", predicted - reference, directions, optimize=True)
    target = np.einsum("ij,dj->di", reference, directions, optimize=True)
    return float(np.linalg.norm(error) / max(np.linalg.norm(target), np.finfo(float).tiny))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents, provenance, basis_names = _load_data(args)
    parent_features, parent_feature_names, feature_normalization = _parent_feature_matrix(parents)
    fit_design, fit_target = _weighted_system(parents, parent_features, "fit", args)
    calibration_design, calibration_target = _weighted_system(parents, parent_features, "calibration", args)
    ridge_rows = []
    for ridge in args.ridge_grid:
        coefficients, solver = _solve(fit_design, fit_target, ridge)
        calibration_residual = calibration_design @ coefficients - calibration_target
        ridge_rows.append(
            {
                "ridge": ridge,
                "calibration_weighted_squared_error": float(torch.sum(calibration_residual.square())),
                **solver,
            }
        )
    selected = min(ridge_rows, key=lambda row: (row["calibration_weighted_squared_error"], row["ridge"]))
    train_design, train_target = _weighted_system(parents, parent_features, "train", args)
    coefficients, final_solver = _solve(train_design, train_target, float(selected["ridge"]))
    coefficients_numpy = coefficients.numpy()

    rows = []
    source_rows = []
    for index, parent in enumerate(parents):
        basis = _combined_basis(
            parent,
            parent_features[index],
            block_parent_conditioning=args.block_parent_conditioning,
        )
        correction = np.einsum("m,mij->ij", coefficients_numpy, basis, optimize=True)
        predicted = parent.source_hessian + correction
        interpolation = np.zeros_like(predicted)
        if args.exact_train_interpolant:
            symmetric_predicted = 0.5 * (predicted + predicted.T)
            target_hvp = np.einsum(
                "ij,dj->di",
                parent.pbe_hessian - symmetric_predicted,
                parent.train_directions,
                optimize=True,
            )
            interpolation = symmetric_hvp_interpolant(
                torch.from_numpy(parent.train_directions),
                torch.from_numpy(target_hvp),
            ).numpy()
            predicted = predicted + interpolation
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": parent.natoms,
            "energy_abs_error_hartree": abs(parent.source_energy - parent.pbe_energy),
            "force_mae_hartree_per_bohr": float(np.mean(np.abs(parent.source_force - parent.pbe_force))),
            "interpolant_relative_frobenius": float(
                np.linalg.norm(interpolation)
                / max(np.linalg.norm(parent.pbe_hessian), np.finfo(float).tiny)
            ),
            "train_hvp_relative_frobenius": _relative_hvp(predicted, parent.pbe_hessian, parent.train_directions),
            "heldout_hvp_relative_frobenius": _relative_hvp(predicted, parent.pbe_hessian, parent.heldout_directions),
            **hessian_metrics(predicted, parent.pbe_hessian),
        }
        source_row = {
            "molecule_id": parent.molecule_id,
            "train_hvp_relative_frobenius": _relative_hvp(parent.source_hessian, parent.pbe_hessian, parent.train_directions),
            "heldout_hvp_relative_frobenius": _relative_hvp(parent.source_hessian, parent.pbe_hessian, parent.heldout_directions),
            **hessian_metrics(parent.source_hessian, parent.pbe_hessian),
        }
        rows.append(row)
        source_rows.append(source_row)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_hessian=predicted,
            source_hessian=parent.source_hessian,
            pbe_hessian=parent.pbe_hessian,
            correction_hessian=correction,
            train_interpolation_hessian=interpolation,
            predicted_force=parent.source_force,
            pbe_force=parent.pbe_force,
        )

    def aggregate(local_rows: list[dict[str, Any]]) -> dict[str, float]:
        result = {}
        for key in ("relative_frobenius", "train_hvp_relative_frobenius", "heldout_hvp_relative_frobenius", "antisymmetric_over_symmetric_frobenius"):
            values = np.asarray([float(row[key]) for row in local_rows])
            result[f"median_{key}"] = float(np.median(values))
            result[f"p90_{key}"] = float(np.quantile(values, 0.9))
            result[f"max_{key}"] = float(np.max(values))
        return result

    final = aggregate(rows)
    source = aggregate(source_rows)
    gate = {
        "train_all_parent_max_le_0p05": final["max_train_hvp_relative_frobenius"] <= 0.05,
        "heldout_all_parent_max_le_0p15": final["max_heldout_hvp_relative_frobenius"] <= 0.15,
        "full_hessian_median_le_0p15": final["median_relative_frobenius"] <= 0.15,
    }
    gate["passed"] = all(gate.values())
    with (args.output_dir / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "coefficients.npz", coefficients=coefficients_numpy)
    result = {
        "definition": "reference-anchored Cartesian scalar quadratic whose symmetric correction is a chemistry-conditioned matrix function of the source complete-total Hessian",
        "formal_limit": "Stage-2 operator-capacity audit requiring the source Hessian at the reference geometry; not yet a standalone transferable OFDFT functional",
        "scalar_energy_owner": True,
        "energy_force_anchor_exact_by_construction": True,
        "operator_equivariance": "orthogonal coordinate similarity, including rigid rotation and atom permutation",
        "basis_names": basis_names,
        "conditioned_spectral_basis_count": parents[0].conditioned_basis_count,
        "include_block_basis": args.include_block_basis,
        "block_parent_conditioning": args.block_parent_conditioning,
        "exact_train_interpolant": args.exact_train_interpolant,
        "exact_train_interpolant_definition": (
            "minimum-rank symmetric completion from train HVP labels; fixes observed P-P/P-Q "
            "blocks while the shared operator supplies unseen Q-Q response"
            if args.exact_train_interpolant
            else None
        ),
        "block_parent_conditioning_definition": (
            "nonconstant parent chemistry channels multiply pair-type-aggregated, "
            "radial-aggregated, and element-aggregated block operators"
            if args.block_parent_conditioning
            else None
        ),
        "parent_feature_names": parent_feature_names,
        "feature_normalization": feature_normalization,
        "feature_count": int(coefficients.numel()),
        "operator_config": {
            "polynomial_degree": args.polynomial_degree,
            "rbf_centers": args.rbf_centers,
            "rbf_width": args.rbf_width,
        },
        "ridge_selection_definition": "deterministic subset of original train directions only; original held-out directions read once after freezing",
        "ridge_selection": ridge_rows,
        "selected_ridge": float(selected["ridge"]),
        "final_solver": final_solver,
        "source": source,
        "final": final,
        "stage2_gate": gate,
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **provenance,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run-dir", type=Path)
    source.add_argument("--source-from-baseline", action="store_true")
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polynomial-degree", type=int, default=5)
    parser.add_argument("--rbf-centers", type=float, nargs="+", default=[-0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5])
    parser.add_argument("--rbf-width", type=float, default=0.5)
    parser.add_argument("--include-block-basis", action="store_true")
    parser.add_argument("--block-parent-conditioning", action="store_true")
    parser.add_argument("--exact-train-interpolant", action="store_true")
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1.0, 100.0])
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--calibration-offset", type=int, default=4)
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
