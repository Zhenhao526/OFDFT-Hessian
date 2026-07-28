#!/usr/bin/env python3
"""Train-only parent-held-out CV for bounded spectral/block Hessian operators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
from scripts.qm9_complete_total_spectral_operator_capacity import (
    Parent,
    _combined_basis,
    _load_data,
    _solve,
    _weighted_system,
)


PARENT_FEATURE_NAMES = [
    "constant",
    "natoms",
    "fraction_C",
    "fraction_N",
    "fraction_O",
    "fraction_F",
    "mean_Z",
    "log_spectral_scale",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_parent_feature(parent: Parent) -> np.ndarray:
    fractions = [np.mean(parent.atomic_numbers == value) for value in (6, 7, 8, 9)]
    return np.asarray(
        [
            parent.natoms / 20.0,
            *fractions,
            np.mean(parent.atomic_numbers) / 10.0,
            np.log(max(parent.spectral_scale, np.finfo(float).tiny)),
        ],
        dtype=np.float64,
    )


def _feature_normalization(parents: list[Parent]) -> dict[str, np.ndarray]:
    raw = np.stack([_raw_parent_feature(parent) for parent in parents])
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-12] = 1.0
    return {"mean": mean, "scale": scale}


def _transform_parent_feature(
    parent: Parent,
    normalization: dict[str, np.ndarray],
    *,
    transform: str,
    transform_scale: float,
) -> np.ndarray:
    z = (
        _raw_parent_feature(parent) - normalization["mean"]
    ) / normalization["scale"]
    if transform == "standard":
        transformed = z
    elif transform == "tanh":
        if transform_scale <= 0.0:
            raise ValueError("tanh transform_scale must be positive")
        transformed = np.tanh(z / transform_scale)
    elif transform == "clip":
        if transform_scale <= 0.0:
            raise ValueError("clip transform_scale must be positive")
        transformed = np.clip(z, -transform_scale, transform_scale)
    elif transform == "constant":
        transformed = np.zeros_like(z)
    else:
        raise ValueError(f"unsupported parent feature transform: {transform}")
    return np.concatenate(([1.0], transformed))


def _folds(parents: list[Parent], fold_count: int) -> list[list[int]]:
    if fold_count < 2 or fold_count > len(parents):
        raise ValueError("fold_count must be between 2 and parent count")
    order = sorted(
        range(len(parents)),
        key=lambda index: (
            -parents[index].natoms,
            tuple(-int(np.sum(parents[index].atomic_numbers == z)) for z in (9, 8, 7, 6, 1)),
            parents[index].molecule_id,
        ),
    )
    result = [[] for _ in range(fold_count)]
    for position, parent_index in enumerate(order):
        cycle, offset = divmod(position, fold_count)
        fold_index = offset if cycle % 2 == 0 else fold_count - 1 - offset
        result[fold_index].append(parent_index)
    return result


def _cap_correction(
    correction: np.ndarray,
    source_hessian: np.ndarray,
    max_ratio: float,
) -> tuple[np.ndarray, float]:
    if not np.isfinite(max_ratio):
        return correction, 1.0
    if max_ratio <= 0.0:
        raise ValueError("max_correction_to_source must be positive")
    correction_norm = float(np.linalg.norm(correction))
    limit = max_ratio * float(np.linalg.norm(source_hessian))
    scale = min(1.0, limit / max(correction_norm, np.finfo(float).tiny))
    return scale * correction, scale


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    relative = np.asarray([float(row["relative_frobenius"]) for row in rows])
    asymmetry = np.asarray(
        [float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows]
    )
    return {
        "mean_relative_frobenius": float(np.mean(relative)),
        "median_relative_frobenius": float(np.median(relative)),
        "p90_relative_frobenius": float(np.quantile(relative, 0.9)),
        "max_relative_frobenius": float(np.max(relative)),
        "fraction_relative_frobenius_at_or_below_0_10": float(
            np.mean(relative <= 0.10)
        ),
        "fraction_relative_frobenius_at_or_below_0_15": float(
            np.mean(relative <= 0.15)
        ),
        "fraction_relative_frobenius_at_or_below_0_20": float(
            np.mean(relative <= 0.20)
        ),
        "max_antisymmetric_over_symmetric_frobenius": float(np.max(asymmetry)),
    }


def _gate(metrics: dict[str, float]) -> dict[str, bool]:
    result = {
        "median_relative_frobenius_le_0p10": (
            metrics["median_relative_frobenius"] <= 0.10
        ),
        "fraction_relative_frobenius_le_0p15_ge_0p80": (
            metrics["fraction_relative_frobenius_at_or_below_0_15"] >= 0.80
        ),
        "p90_relative_frobenius_le_0p20": (
            metrics["p90_relative_frobenius"] <= 0.20
        ),
        "asymmetry_max_le_0p005": (
            metrics["max_antisymmetric_over_symmetric_frobenius"] <= 0.005
        ),
    }
    result["passed"] = all(result.values())
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    scope = protocol["scope"]
    if scope.get("validation_parent_access_allowed") is not False:
        raise ValueError("protocol must prohibit validation-parent access")
    if scope.get("test100_accessed") is not False:
        raise ValueError("protocol must freeze Test100")
    if int(scope.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("protocol records Test100 access")
    frozen = protocol["frozen_inputs"]
    if _sha256(args.baseline_manifest) != frozen["baseline_manifest_sha256"]:
        raise ValueError("baseline manifest hash differs from protocol")
    if _sha256(args.direction_manifest) != frozen["direction_manifest_sha256"]:
        raise ValueError("direction manifest hash differs from protocol")
    if not args.source_from_baseline or args.source_run_dir is not None:
        raise ValueError("formal parent CV requires the original-A baseline source")
    if args.fold_count != int(scope["fold_count"]):
        raise ValueError("fold count differs from protocol")
    if [float(value) for value in args.ridge_grid] != [
        float(value) for value in protocol["cross_validation"]["ridge_grid"]
    ]:
        raise ValueError("ridge grid differs from protocol")
    operator = protocol["operator"]
    if args.polynomial_degree != int(
        operator["source_hessian_spectral_polynomial_degree"]
    ):
        raise ValueError("spectral polynomial degree differs from protocol")
    if not np.array_equal(
        np.asarray(args.rbf_centers, dtype=np.float64),
        np.asarray(operator["source_hessian_spectral_rbf_centers"], dtype=np.float64),
    ):
        raise ValueError("spectral RBF centers differ from protocol")
    if args.rbf_width != float(operator["source_hessian_spectral_rbf_width"]):
        raise ValueError("spectral RBF width differs from protocol")
    variants = {
        str(variant["id"]): variant for variant in protocol["preregistered_variants"]
    }
    if args.variant_id not in variants:
        raise ValueError("variant ID is not preregistered")
    variant = variants[args.variant_id]
    expected = (
        str(variant["parent_feature_transform"]),
        float(variant["parent_feature_transform_scale"]),
        bool(variant["block_parent_conditioning"]),
        float(variant["max_correction_to_source"]),
    )
    actual = (
        args.parent_feature_transform,
        args.parent_feature_transform_scale,
        args.block_parent_conditioning,
        args.max_correction_to_source,
    )
    if actual != expected:
        raise ValueError("runtime variant settings differ from protocol")
    return protocol


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _validate_protocol(args)
    parents, provenance, basis_names = _load_data(args)
    folds = _folds(parents, args.fold_count)
    rows_by_ridge: dict[float, list[dict[str, Any]]] = {
        float(ridge): [] for ridge in args.ridge_grid
    }
    artifacts_by_ridge: dict[float, dict[str, dict[str, Any]]] = {
        float(ridge): {} for ridge in args.ridge_grid
    }
    fold_rows = []

    for fold_index, held_indices in enumerate(folds):
        held_set = set(held_indices)
        train_parents = [
            parent for index, parent in enumerate(parents) if index not in held_set
        ]
        held_parents = [parents[index] for index in held_indices]
        normalization = _feature_normalization(train_parents)
        train_features = np.stack(
            [
                _transform_parent_feature(
                    parent,
                    normalization,
                    transform=args.parent_feature_transform,
                    transform_scale=args.parent_feature_transform_scale,
                )
                for parent in train_parents
            ]
        )
        held_features = np.stack(
            [
                _transform_parent_feature(
                    parent,
                    normalization,
                    transform=args.parent_feature_transform,
                    transform_scale=args.parent_feature_transform_scale,
                )
                for parent in held_parents
            ]
        )
        train_design, train_target = _weighted_system(
            train_parents, train_features, "train", args
        )
        held_bases = [
            _combined_basis(
                parent,
                held_features[index],
                block_parent_conditioning=args.block_parent_conditioning,
            )
            for index, parent in enumerate(held_parents)
        ]
        for ridge in args.ridge_grid:
            coefficients, solver = _solve(train_design, train_target, float(ridge))
            coefficients_numpy = coefficients.numpy()
            fold_rows.append(
                {
                    "fold": fold_index,
                    "ridge": float(ridge),
                    "train_parent_count": len(train_parents),
                    "held_parent_count": len(held_parents),
                    **solver,
                }
            )
            for local_index, parent in enumerate(held_parents):
                correction = np.einsum(
                    "m,mij->ij", coefficients_numpy, held_bases[local_index], optimize=True
                )
                correction, cap_scale = _cap_correction(
                    correction,
                    parent.source_hessian,
                    args.max_correction_to_source,
                )
                predicted = parent.source_hessian + correction
                metrics = hessian_metrics(predicted, parent.pbe_hessian)
                source_metrics = hessian_metrics(
                    parent.source_hessian, parent.pbe_hessian
                )
                rows_by_ridge[float(ridge)].append(
                    {
                        "fold": fold_index,
                        "molecule_id": parent.molecule_id,
                        "natoms": parent.natoms,
                        "ridge": float(ridge),
                        "energy_abs_error_hartree": abs(
                            parent.source_energy - parent.pbe_energy
                        ),
                        "force_mae_hartree_per_bohr": float(
                            np.mean(np.abs(parent.source_force - parent.pbe_force))
                        ),
                        "source_relative_frobenius": source_metrics[
                            "relative_frobenius"
                        ],
                        "relative_frobenius_improvement": (
                            source_metrics["relative_frobenius"]
                            - metrics["relative_frobenius"]
                        ),
                        "correction_cap_scale": cap_scale,
                        "parent_feature_max_abs": float(
                            np.max(np.abs(held_features[local_index][1:]))
                        ),
                        **metrics,
                    }
                )
                artifacts_by_ridge[float(ridge)][parent.molecule_id] = {
                    "predicted_hessian": predicted,
                    "correction_hessian": correction,
                    "correction_cap_scale": cap_scale,
                }

    ridge_rows = []
    for ridge in args.ridge_grid:
        metrics = _aggregate(rows_by_ridge[float(ridge)])
        ridge_rows.append(
            {
                "ridge": float(ridge),
                **metrics,
                **{f"gate_{key}": value for key, value in _gate(metrics).items()},
            }
        )
    selected = min(
        ridge_rows,
        key=lambda row: (
            -row["fraction_relative_frobenius_at_or_below_0_15"],
            row["p90_relative_frobenius"],
            row["median_relative_frobenius"],
            -row["ridge"],
        ),
    )
    selected_ridge = float(selected["ridge"])
    selected_rows = rows_by_ridge[selected_ridge]
    selected_metrics = _aggregate(selected_rows)
    selected_gate = _gate(selected_metrics)

    normalization = _feature_normalization(parents)
    parent_features = np.stack(
        [
            _transform_parent_feature(
                parent,
                normalization,
                transform=args.parent_feature_transform,
                transform_scale=args.parent_feature_transform_scale,
            )
            for parent in parents
        ]
    )
    full_design, full_target = _weighted_system(parents, parent_features, "train", args)
    coefficients, final_solver = _solve(full_design, full_target, selected_ridge)

    for row in selected_rows:
        molecule_id = str(row["molecule_id"])
        parent = next(item for item in parents if item.molecule_id == molecule_id)
        artifact = artifacts_by_ridge[selected_ridge][molecule_id]
        np.savez_compressed(
            args.output_dir / f"{molecule_id}_result.npz",
            predicted_energy=np.asarray(parent.source_energy),
            predicted_force=parent.source_force,
            predicted_hessian=artifact["predicted_hessian"],
            source_hessian=parent.source_hessian,
            correction_hessian=artifact["correction_hessian"],
            correction_cap_scale=np.asarray(artifact["correction_cap_scale"]),
            pbe_energy=np.asarray(parent.pbe_energy),
            pbe_force=parent.pbe_force,
            pbe_hessian=parent.pbe_hessian,
        )

    _write_csv(args.output_dir / "per_parent_metrics.csv", selected_rows)
    _write_csv(args.output_dir / "ridge_cross_validation.csv", ridge_rows)
    _write_csv(args.output_dir / "fold_solver_metrics.csv", fold_rows)
    np.savez_compressed(
        args.output_dir / "coefficients.npz",
        coefficients=coefficients.numpy(),
        parent_feature_mean=normalization["mean"],
        parent_feature_scale=normalization["scale"],
    )
    result = {
        "definition": (
            "Five-fold train-only parent-held-out evaluation of a reference-anchored "
            "scalar spectral/block Hessian operator; held parents never receive HVP completion."
        ),
        "formal_limit": (
            "Local reference-geometry scalar capacity model requiring a source Hessian; "
            "not yet a standalone global OFDFT functional."
        ),
        "scalar_energy_owner": True,
        "variant_id": args.variant_id,
        "energy_force_anchor_exact_by_construction": True,
        "held_parent_hvp_labels_read_for_fit": False,
        "exact_train_interpolant": False,
        "fold_assignment": [
            [parents[index].molecule_id for index in fold] for fold in folds
        ],
        "fold_count": args.fold_count,
        "parent_feature_names": PARENT_FEATURE_NAMES,
        "parent_feature_transform": args.parent_feature_transform,
        "parent_feature_transform_scale": args.parent_feature_transform_scale,
        "block_parent_conditioning": args.block_parent_conditioning,
        "max_correction_to_source": args.max_correction_to_source,
        "ridge_selection_definition": (
            "train20 parent-held-out full-Hessian metrics only: maximize fraction <=0.15, "
            "then minimize P90, median, and prefer larger ridge on an exact tie"
        ),
        "ridge_cross_validation": ridge_rows,
        "selected_ridge": selected_ridge,
        "selected_cross_parent_metrics": selected_metrics,
        "cross_parent_gate": selected_gate,
        "source_cross_parent_metrics": _aggregate(
            [
                {
                    **row,
                    **hessian_metrics(
                        next(
                            parent.source_hessian
                            for parent in parents
                            if parent.molecule_id == row["molecule_id"]
                        ),
                        next(
                            parent.pbe_hessian
                            for parent in parents
                            if parent.molecule_id == row["molecule_id"]
                        ),
                    ),
                }
                for row in selected_rows
            ]
        ),
        "parent_win_fraction_vs_source": float(
            np.mean(
                [float(row["relative_frobenius_improvement"]) > 0.0 for row in selected_rows]
            )
        ),
        "final_solver": final_solver,
        "final_parent_feature_normalization": {
            "mean": normalization["mean"].tolist(),
            "scale": normalization["scale"].tolist(),
        },
        "basis_names": basis_names,
        "conditioned_spectral_basis_count": parents[0].conditioned_basis_count,
        "feature_count": int(coefficients.numel()),
        "operator_config": {
            "polynomial_degree": args.polynomial_degree,
            "rbf_centers": args.rbf_centers,
            "rbf_width": args.rbf_width,
        },
        "per_parent": selected_rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(args.protocol),
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
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run-dir", type=Path)
    source.add_argument("--source-from-baseline", action="store_true")
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--calibration-offset", type=int, default=4)
    parser.add_argument("--polynomial-degree", type=int, default=5)
    parser.add_argument(
        "--rbf-centers",
        type=float,
        nargs="+",
        default=[-0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5],
    )
    parser.add_argument("--rbf-width", type=float, default=0.5)
    parser.add_argument("--include-block-basis", action="store_true")
    parser.add_argument("--block-parent-conditioning", action="store_true")
    parser.add_argument(
        "--parent-feature-transform",
        choices=("standard", "tanh", "clip", "constant"),
        default="standard",
    )
    parser.add_argument("--parent-feature-transform-scale", type=float, default=2.0)
    parser.add_argument("--max-correction-to-source", type=float, default=float("inf"))
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 1.0, 100.0]
    )
    parser.add_argument("--absolute-hvp-scale", type=float, default=0.1)
    parser.add_argument("--hvp-relative-fraction", type=float, default=0.5)
    parser.add_argument("--hvp-reference-floor", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
