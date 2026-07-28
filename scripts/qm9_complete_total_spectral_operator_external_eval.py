#!/usr/bin/env python3
"""Evaluate a frozen shared spectral/block Hessian operator on unseen parents."""

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
import torch
import yaml

from mldft.ml.models.components.spectral_hessian_residual import (
    equivariant_block_operator_basis,
    spectral_operator_basis,
)
from scripts.prepare_qm9_complete_total_direction_generalization import _external_basis
from scripts.qm9_complete_total_geometry_shared_capacity import _load_parents
from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
from scripts.qm9_complete_total_spectral_operator_capacity import (
    _combined_basis,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p80": float(np.quantile(array, 0.8)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _parent_features(
    atomic_numbers: np.ndarray,
    spectral_scale: float,
    normalization: dict[str, list[float]],
) -> np.ndarray:
    fractions = [np.mean(atomic_numbers == value) for value in (6, 7, 8, 9)]
    raw = np.asarray(
        [
            atomic_numbers.size / 20.0,
            *fractions,
            np.mean(atomic_numbers) / 10.0,
            np.log(max(spectral_scale, np.finfo(float).tiny)),
        ],
        dtype=np.float64,
    )
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    if raw.shape != mean.shape or raw.shape != scale.shape or np.any(scale <= 0.0):
        raise ValueError("invalid frozen parent-feature normalization")
    return np.concatenate(([1.0], (raw - mean) / scale))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(
    args: argparse.Namespace,
    parent_ids: list[str],
    baseline_sha256: str,
    source_checkpoint_sha256: str,
    candidate_summary_sha256: str,
    coefficient_sha256: str,
) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    scope = protocol["scope"]
    if scope.get("test100_accessed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(scope.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("protocol records Test100 access")
    if [str(value) for value in scope["validation_parent_ids"]] != parent_ids:
        raise ValueError("validation parent IDs differ from the preregistered protocol")
    frozen = protocol["frozen_inputs"]
    if baseline_sha256 != frozen["validation_baseline_manifest_sha256"]:
        raise ValueError("baseline manifest hash differs from the protocol")
    registered_arms = {
        (
            frozen["h512_source_checkpoint_sha256"],
            frozen["stage2_candidate_summary_sha256"],
            frozen["shared_coefficients_sha256"],
        ),
        (
            frozen["replay_source_checkpoint_sha256"],
            frozen["replay_source_candidate_summary_sha256"],
            frozen["replay_source_shared_coefficients_sha256"],
        ),
    }
    if (
        source_checkpoint_sha256,
        candidate_summary_sha256,
        coefficient_sha256,
    ) not in registered_arms:
        raise ValueError("source, candidate, and coefficients are not a registered arm")
    evaluation = protocol["evaluation"]
    if evaluation.get("exact_parent_hvp_interpolant_allowed") is not False:
        raise ValueError("protocol must prohibit per-parent HVP interpolation")
    if evaluation.get("per_parent_fitting_allowed") is not False:
        raise ValueError("protocol must prohibit per-parent fitting")
    if evaluation.get("validation_hvp_labels_allowed") is not False:
        raise ValueError("protocol must prohibit validation HVP labels")
    if int(evaluation["polynomial_degree"]) != args.polynomial_degree:
        raise ValueError("polynomial degree differs from the protocol")
    if not np.array_equal(
        np.asarray(evaluation["rbf_centers"], dtype=np.float64),
        np.asarray(args.rbf_centers, dtype=np.float64),
    ):
        raise ValueError("RBF centers differ from the protocol")
    if float(evaluation["rbf_width"]) != args.rbf_width:
        raise ValueError("RBF width differs from the protocol")
    return protocol


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    parents, baseline_manifest = _load_parents(args.baseline_manifest)
    source_summary_path = args.source_run_dir / "summary.json"
    candidate_summary_path = args.candidate_run_dir / "summary.json"
    coefficient_path = args.candidate_run_dir / "coefficients.npz"
    source_summary = json.loads(source_summary_path.read_text())
    candidate_summary = json.loads(candidate_summary_path.read_text())
    for name, manifest in (
        ("baseline", baseline_manifest),
        ("source", source_summary),
        ("candidate", candidate_summary),
    ):
        if manifest.get("test100_accessed") is not False:
            raise ValueError(f"{name} does not certify frozen Test100")
        if int(manifest.get("test100_evaluations_used", 0)) != 0:
            raise ValueError(f"{name} records Test100 access")
    if source_summary.get("checkpoint_sha256") != candidate_summary.get(
        "source_checkpoint_sha256"
    ):
        raise ValueError("external source checkpoint differs from candidate source")
    if not candidate_summary.get("include_block_basis"):
        raise ValueError("candidate does not contain the registered block basis")
    if not candidate_summary.get("block_parent_conditioning"):
        raise ValueError("candidate does not contain registered parent conditioning")

    coefficients = np.asarray(np.load(coefficient_path)["coefficients"], dtype=np.float64)
    if coefficients.size != int(candidate_summary["feature_count"]):
        raise ValueError("coefficient count differs from candidate summary")
    expected_basis_names = list(candidate_summary["basis_names"])
    expected_parent_feature_names = [
        "constant",
        "natoms",
        "fraction_C",
        "fraction_N",
        "fraction_O",
        "fraction_F",
        "mean_Z",
        "log_spectral_scale",
    ]
    if candidate_summary["parent_feature_names"] != expected_parent_feature_names:
        raise ValueError("unsupported candidate parent-feature schema")
    baseline_sha256 = _sha256(args.baseline_manifest)
    candidate_summary_sha256 = _sha256(candidate_summary_path)
    coefficient_sha256 = _sha256(coefficient_path)
    protocol = _validate_protocol(
        args,
        [parent.molecule_id for parent in parents],
        baseline_sha256,
        str(source_summary["checkpoint_sha256"]),
        candidate_summary_sha256,
        coefficient_sha256,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for parent in parents:
        parent_started = time.perf_counter()
        source_path = args.source_run_dir / f"{parent.molecule_id}_result.npz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with np.load(source_path) as payload:
            source_energy = float(payload["predicted_energy"])
            source_force = np.asarray(payload["predicted_force"], dtype=np.float64)
            source_hessian = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            source_pbe = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if not np.array_equal(source_pbe, parent.pbe_hessian):
            raise ValueError(f"PBE Hessian mismatch for {parent.molecule_id}")

        external = _external_basis(parent.positions_bohr)
        spectral, spectral_names, _, spectral_scale = spectral_operator_basis(
            torch.from_numpy(source_hessian),
            torch.from_numpy(external),
            polynomial_degree=args.polynomial_degree,
            rbf_centers=tuple(args.rbf_centers),
            rbf_width=args.rbf_width,
        )
        block, block_names = equivariant_block_operator_basis(
            torch.from_numpy(source_hessian),
            torch.as_tensor(parent.atomic_numbers, dtype=torch.long),
            torch.as_tensor(parent.positions_bohr, dtype=torch.float64),
            torch.from_numpy(external),
        )
        names = [*spectral_names, *block_names]
        if names != expected_basis_names:
            raise ValueError("reconstructed operator basis differs from candidate")
        features = _parent_features(
            np.asarray(parent.atomic_numbers, dtype=np.int64),
            spectral_scale,
            candidate_summary["feature_normalization"],
        )
        operator_basis = np.concatenate((spectral.numpy(), block.numpy()), axis=0)

        # _combined_basis only requires these registered attributes. Keeping the
        # reconstruction here avoids reading any unseen-parent HVP labels.
        holder = argparse.Namespace(
            operator_basis=operator_basis,
            conditioned_basis_count=int(spectral.shape[0]),
            block_basis_count=int(block.shape[0]),
            source_hessian=source_hessian,
        )
        basis = _combined_basis(
            holder,
            features,
            block_parent_conditioning=True,
        )
        if basis.shape[0] != coefficients.size:
            raise ValueError("external combined basis has the wrong feature count")
        correction = np.einsum("m,mij->ij", coefficients, basis, optimize=True)
        predicted_hessian = source_hessian + correction
        source_metrics = hessian_metrics(source_hessian, parent.pbe_hessian)
        predicted_metrics = hessian_metrics(predicted_hessian, parent.pbe_hessian)
        pbe_frobenius = max(
            float(np.linalg.norm(parent.pbe_hessian)), np.finfo(float).tiny
        )
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": int(parent.atomic_numbers.size),
            "baseline_energy_abs_error_hartree": abs(
                parent.energy - parent.pbe_energy
            ),
            "baseline_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(parent.force - parent.pbe_force))
            ),
            "source_energy_abs_error_hartree": abs(source_energy - parent.pbe_energy),
            "source_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(source_force - parent.pbe_force))
            ),
            "source_relative_frobenius": source_metrics["relative_frobenius"],
            "source_frobenius_over_pbe_frobenius": float(
                np.linalg.norm(source_hessian) / pbe_frobenius
            ),
            "correction_frobenius_over_pbe_frobenius": float(
                np.linalg.norm(correction) / pbe_frobenius
            ),
            "spectral_scale": float(spectral_scale),
            "parent_feature_max_abs_z": float(np.max(np.abs(features[1:]))),
            "parent_feature_l2_z": float(np.linalg.norm(features[1:])),
            "relative_frobenius": predicted_metrics["relative_frobenius"],
            "relative_frobenius_improvement": (
                source_metrics["relative_frobenius"]
                - predicted_metrics["relative_frobenius"]
            ),
            "relative_frobenius_ratio_to_source": (
                predicted_metrics["relative_frobenius"]
                / max(source_metrics["relative_frobenius"], np.finfo(float).tiny)
            ),
            "wall_time_s": time.perf_counter() - parent_started,
            **predicted_metrics,
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_energy=np.asarray(source_energy),
            predicted_force=source_force,
            predicted_hessian=predicted_hessian,
            source_hessian=source_hessian,
            correction_hessian=correction,
            pbe_energy=np.asarray(parent.pbe_energy),
            pbe_force=parent.pbe_force,
            pbe_hessian=parent.pbe_hessian,
        )

    csv_path = args.output_dir / "per_parent_metrics.csv"
    _write_csv(csv_path, rows)
    relative = [float(row["relative_frobenius"]) for row in rows]
    source_relative = [float(row["source_relative_frobenius"]) for row in rows]
    distribution = _quantiles(relative)
    source_distribution = _quantiles(source_relative)
    fraction_0p15 = float(np.mean(np.asarray(relative) <= 0.15))
    max_asymmetry = max(
        float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
    )
    baseline_energy = _quantiles(
        [float(row["baseline_energy_abs_error_hartree"]) for row in rows]
    )
    source_energy = _quantiles(
        [float(row["source_energy_abs_error_hartree"]) for row in rows]
    )
    baseline_force = _quantiles(
        [float(row["baseline_force_mae_hartree_per_bohr"]) for row in rows]
    )
    source_force = _quantiles(
        [float(row["source_force_mae_hartree_per_bohr"]) for row in rows]
    )
    registered_gates = protocol["gates"]
    energy_median_limit = 1.0 + float(
        registered_gates["energy_median_regression_max_fraction"]
    )
    energy_p90_limit = 1.0 + float(
        registered_gates["energy_p90_regression_max_fraction"]
    )
    force_median_limit = 1.0 + float(
        registered_gates["force_median_regression_max_fraction"]
    )
    force_p90_limit = 1.0 + float(
        registered_gates["force_p90_regression_max_fraction"]
    )
    gate = {
        "median_relative_frobenius_le_0p10": distribution["median"] <= 0.10,
        "fraction_relative_frobenius_le_0p15_ge_0p80": fraction_0p15 >= 0.80,
        "p90_relative_frobenius_le_0p20": distribution["p90"] <= 0.20,
        "asymmetry_max_le_0p005": max_asymmetry <= 0.005,
        "energy_median_regression_le_0p05": (
            source_energy["median"] <= energy_median_limit * baseline_energy["median"]
        ),
        "energy_p90_regression_le_0p05": (
            source_energy["p90"] <= energy_p90_limit * baseline_energy["p90"]
        ),
        "force_median_regression_le_0p05": (
            source_force["median"] <= force_median_limit * baseline_force["median"]
        ),
        "force_p90_regression_le_0p05": (
            source_force["p90"] <= force_p90_limit * baseline_force["p90"]
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "definition": (
            "Frozen shared spectral/block operator evaluated on independent validation "
            "parents without per-parent HVP fitting or symmetric completion."
        ),
        "formal_limit": (
            "Reference-geometry anchored scalar capacity operator requiring a source "
            "complete-total Hessian; not a standalone transferable OFDFT functional."
        ),
        "scalar_energy_owner": True,
        "energy_force_anchor_exact_by_construction": True,
        "per_parent_hvp_labels_read": False,
        "exact_train_interpolant_applied": False,
        "candidate_training_used_exact_interpolant": bool(
            candidate_summary.get("exact_train_interpolant")
        ),
        "parent_count": len(rows),
        "source_hessian_relative_frobenius": source_distribution,
        "hessian_relative_frobenius": distribution,
        "fraction_relative_frobenius_at_or_below_0_10": float(
            np.mean(np.asarray(relative) <= 0.10)
        ),
        "fraction_relative_frobenius_at_or_below_0_15": fraction_0p15,
        "fraction_relative_frobenius_at_or_below_0_20": float(
            np.mean(np.asarray(relative) <= 0.20)
        ),
        "parent_win_fraction_vs_source": float(
            np.mean(np.asarray(relative) < np.asarray(source_relative))
        ),
        "baseline_energy_abs_error_hartree": baseline_energy,
        "energy_abs_error_hartree": source_energy,
        "baseline_force_mae_hartree_per_bohr": baseline_force,
        "force_mae_hartree_per_bohr": source_force,
        "max_antisymmetric_over_symmetric_frobenius": max_asymmetry,
        "stage3_validation_gate": gate,
        "operator_config": {
            "polynomial_degree": args.polynomial_degree,
            "rbf_centers": args.rbf_centers,
            "rbf_width": args.rbf_width,
            "block_parent_conditioning": True,
        },
        "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
        "baseline_manifest_sha256": baseline_sha256,
        "source_summary": source_summary_path.resolve().as_posix(),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_checkpoint_sha256": source_summary["checkpoint_sha256"],
        "candidate_summary": candidate_summary_path.resolve().as_posix(),
        "candidate_summary_sha256": candidate_summary_sha256,
        "candidate_coefficients": coefficient_path.resolve().as_posix(),
        "candidate_coefficients_sha256": coefficient_sha256,
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(args.protocol),
        "source_split_sha256": baseline_manifest.get("source_split_sha256"),
        "per_parent_metrics": csv_path.resolve().as_posix(),
        "per_parent_metrics_sha256": _sha256(csv_path),
        "per_parent": rows,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polynomial-degree", type=int, default=5)
    parser.add_argument(
        "--rbf-centers",
        type=float,
        nargs="+",
        default=[-0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5],
    )
    parser.add_argument("--rbf-width", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
