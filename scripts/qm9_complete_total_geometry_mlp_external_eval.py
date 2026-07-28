#!/usr/bin/env python3
"""Evaluate a frozen scalar geometry residual on independent complete-total baselines."""

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

try:
    from scripts.qm9_complete_total_geometry_mlp_capacity import (
        DeepSmoothDescriptorResidual,
        SmoothDescriptorResidual,
    )
    from scripts.qm9_complete_total_replay_descriptor_cache import _load_schema
    from scripts.qm9_complete_total_geometry_shared_capacity import (
        FeatureDesign,
        _build_design,
        _load_parents,
    )
    from scripts.qm9_complete_total_geometry_three_body_capacity import hessian_metrics
except ModuleNotFoundError:
    from qm9_complete_total_geometry_mlp_capacity import (
        DeepSmoothDescriptorResidual,
        SmoothDescriptorResidual,
    )
    from qm9_complete_total_replay_descriptor_cache import _load_schema
    from qm9_complete_total_geometry_shared_capacity import (
        FeatureDesign,
        _build_design,
        _load_parents,
    )
    from qm9_complete_total_geometry_three_body_capacity import hessian_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_local_design(
    local: FeatureDesign,
    active_keys: np.ndarray,
    active_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    keys = [tuple(int(value) for value in row) for row in np.asarray(active_keys)]
    key_to_index = {key: index for index, key in enumerate(keys)}
    if len(key_to_index) != len(keys):
        raise ValueError("active training feature keys are not unique")
    descriptor = np.zeros(len(keys), dtype=np.float64)
    jacobian = np.zeros((local.gradient.shape[0], len(keys)), dtype=np.float64)
    hessian = np.zeros((local.hessian.shape[0], len(keys)), dtype=np.float64)
    matched = 0
    for local_index, key in enumerate(local.keys):
        global_index = key_to_index.get(tuple(int(value) for value in key))
        if global_index is None:
            continue
        norm = float(active_norms[global_index])
        descriptor[global_index] = local.energy[local_index] / norm
        jacobian[:, global_index] = local.gradient[:, local_index] / norm
        hessian[:, global_index] = local.hessian[:, local_index] / norm
        matched += 1
    return descriptor, jacobian, hessian, {
        "local_feature_count": len(local.keys),
        "matched_active_feature_count": matched,
        "matched_active_feature_fraction": (
            matched / len(local.keys) if local.keys else 1.0
        ),
    }


def _descriptor_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        derivative_step=args.derivative_step,
        three_body_center_min=args.three_body_center_min,
        three_body_center_max=args.three_body_center_max,
        three_body_center_count=args.three_body_center_count,
        three_body_sigma=args.three_body_sigma,
        three_body_angular_order=args.three_body_angular_order,
        four_body_center_min=args.four_body_center_min,
        four_body_center_max=args.four_body_center_max,
        four_body_center_count=args.four_body_center_count,
        four_body_sigma=args.four_body_sigma,
        four_body_torsion_order=args.four_body_torsion_order,
        four_body_bond_scale=args.four_body_bond_scale,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p80": float(np.quantile(array, 0.8)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _prefixed_hessian_metrics(
    prefix: str, prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in hessian_metrics(prediction, reference).items()
    }


def _model_from_state(
    state: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.nn.Module, int, int, int]:
    feature_count = int(state["linear"].numel())
    hidden_size = int(state["output"].numel())
    deep_keys = {"deep_weight", "deep_bias", "deep_output"}
    present_deep_keys = deep_keys.intersection(state)
    if present_deep_keys and present_deep_keys != deep_keys:
        raise ValueError("checkpoint has an incomplete deep residual state")
    if present_deep_keys:
        deep_hidden_size = int(state["deep_output"].numel())
        model = DeepSmoothDescriptorResidual(
            feature_count,
            hidden_size,
            deep_hidden_size,
            torch.zeros(feature_count, dtype=torch.float64),
            seed=0,
        )
    else:
        deep_hidden_size = 0
        model = SmoothDescriptorResidual(
            feature_count,
            hidden_size,
            torch.zeros(feature_count, dtype=torch.float64),
            seed=0,
        )
    model = model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model, feature_count, hidden_size, deep_hidden_size


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    parents, baseline_manifest = _load_parents(args.baseline_manifest)
    if baseline_manifest.get("test100_accessed") is not False:
        raise ValueError("external baseline manifest does not freeze Test100")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("geometry-residual checkpoint does not freeze Test100")
    state = checkpoint["state_dict"]
    device = torch.device(args.device)
    model, feature_count, hidden_size, deep_hidden_size = _model_from_state(
        state, device
    )
    checkpoint_atom_count_extensive = bool(
        checkpoint.get("config", {}).get("atom_count_extensive", False)
    )

    schema_sha = _sha256(args.active_schema_manifest)
    schema_manifest, schema_stages = _load_schema(args.active_schema_manifest)
    if int(schema_manifest["active_feature_count"]) != feature_count:
        raise ValueError("active schema/checkpoint feature count mismatch")
    checkpoint_matches_schema_source = (
        _sha256(args.checkpoint) == schema_manifest["checkpoint_sha256"]
    )
    checkpoint_inherits_schema = (
        checkpoint.get("active_schema_manifest_sha256") == schema_sha
    )
    checkpoint_config = checkpoint.get("config", {})
    try:
        checkpoint_scale_floor = float(checkpoint_config["feature_scale_floor"])
    except (KeyError, TypeError, ValueError):
        checkpoint_scale_floor = float("nan")
    checkpoint_matches_inventory_definition = (
        checkpoint.get("feature_inventory_manifest_sha256")
        == schema_manifest["inventory_manifest_sha256"]
        and checkpoint_config.get("feature_scale_mode")
        == schema_manifest["feature_scale_mode"]
        and checkpoint_scale_floor == float(schema_manifest["feature_scale_floor"])
    )
    if not (
        checkpoint_matches_schema_source
        or checkpoint_inherits_schema
        or checkpoint_matches_inventory_definition
    ):
        raise ValueError("checkpoint is not bound to the frozen active feature schema")
    stages = [
        {
            "stage": stage,
            "keys": schema_stages[stage]["feature_keys"],
            "norms": schema_stages[stage]["column_norms"],
        }
        for stage in ("three_body", "four_body")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_args = _descriptor_args(args)
    rows = []
    for parent in parents:
        parent_started = time.perf_counter()
        descriptors = []
        jacobians = []
        hessians = []
        coverage: dict[str, Any] = {}
        for stage in stages:
            local = _build_design(parent, str(stage["stage"]), descriptor_args)
            descriptor, jacobian, hessian, local_coverage = _map_local_design(
                local, np.asarray(stage["keys"]), np.asarray(stage["norms"])
            )
            coordinate_count = parent.force.size
            hessian = hessian.reshape(coordinate_count, coordinate_count, -1)
            hessian = 0.5 * (hessian + hessian.transpose(1, 0, 2))
            descriptors.append(descriptor)
            jacobians.append(jacobian)
            hessians.append(hessian.reshape(coordinate_count**2, -1))
            for key, value in local_coverage.items():
                coverage[f"{stage['stage']}_{key}"] = value

        descriptor_tensor = torch.as_tensor(
            np.concatenate(descriptors), dtype=torch.float64, device=device
        )
        jacobian_tensor = torch.as_tensor(
            np.concatenate(jacobians, axis=1), dtype=torch.float64, device=device
        )
        hessian_tensor = torch.as_tensor(
            np.concatenate(hessians, axis=1), dtype=torch.float64, device=device
        )
        with torch.no_grad():
            energy, force, hessian = model.derivatives(
                descriptor_tensor,
                jacobian_tensor,
                hessian_tensor,
                hessian_weight=1.0,
                extensivity_scale=(
                    float(parent.atomic_numbers.size)
                    if checkpoint_atom_count_extensive
                    else 1.0
                ),
            )
            linear_energy = torch.dot(model.linear, descriptor_tensor)
            linear_force = -(jacobian_tensor @ model.linear)
            linear_hessian = (hessian_tensor @ model.linear).reshape(
                parent.force.size, parent.force.size
            )
            atom_scale = float(parent.atomic_numbers.size)
            extensive_energy, extensive_force, extensive_hessian = model.derivatives(
                descriptor_tensor / atom_scale,
                jacobian_tensor / atom_scale,
                hessian_tensor / atom_scale,
                hessian_weight=1.0,
            )
            extensive_energy = atom_scale * extensive_energy
            extensive_force = atom_scale * extensive_force
            extensive_hessian = atom_scale * extensive_hessian
            activation = model.weight @ descriptor_tensor + model.bias
            extensive_activation = (
                model.weight @ (descriptor_tensor / atom_scale) + model.bias
            )
        predicted_energy = parent.energy + float(energy.cpu())
        predicted_force = parent.force + force.cpu().numpy().reshape(parent.force.shape)
        predicted_hessian = parent.hessian + hessian.cpu().numpy()
        linear_predicted_energy = parent.energy + float(linear_energy.cpu())
        linear_predicted_force = parent.force + linear_force.cpu().numpy().reshape(
            parent.force.shape
        )
        linear_predicted_hessian = parent.hessian + linear_hessian.cpu().numpy()
        extensive_predicted_energy = parent.energy + float(extensive_energy.cpu())
        extensive_predicted_force = (
            parent.force + extensive_force.cpu().numpy().reshape(parent.force.shape)
        )
        extensive_predicted_hessian = (
            parent.hessian + extensive_hessian.cpu().numpy()
        )
        force_error = predicted_force - parent.pbe_force
        baseline_force_error = parent.force - parent.pbe_force
        baseline_hessian_metrics = {
            f"baseline_{key}": value
            for key, value in hessian_metrics(
                parent.hessian, parent.pbe_hessian
            ).items()
        }
        row = {
            "molecule_id": parent.molecule_id,
            "natoms": int(parent.atomic_numbers.size),
            "energy_abs_error_hartree": abs(predicted_energy - parent.pbe_energy),
            "baseline_energy_abs_error_hartree": abs(
                parent.energy - parent.pbe_energy
            ),
            "force_mae_hartree_per_bohr": float(np.mean(np.abs(force_error))),
            "baseline_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(baseline_force_error))
            ),
            "force_rmse_hartree_per_bohr": float(np.sqrt(np.mean(force_error**2))),
            "linear_only_energy_abs_error_hartree": abs(
                linear_predicted_energy - parent.pbe_energy
            ),
            "linear_only_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(linear_predicted_force - parent.pbe_force))
            ),
            "atom_extensive_energy_abs_error_hartree": abs(
                extensive_predicted_energy - parent.pbe_energy
            ),
            "atom_extensive_force_mae_hartree_per_bohr": float(
                np.mean(np.abs(extensive_predicted_force - parent.pbe_force))
            ),
            "descriptor_l2": float(torch.linalg.vector_norm(descriptor_tensor).cpu()),
            "descriptor_max_abs": float(torch.max(torch.abs(descriptor_tensor)).cpu()),
            "activation_l2": float(torch.linalg.vector_norm(activation).cpu()),
            "activation_max_abs": float(torch.max(torch.abs(activation)).cpu()),
            "atom_extensive_activation_l2": float(
                torch.linalg.vector_norm(extensive_activation).cpu()
            ),
            "atom_extensive_activation_max_abs": float(
                torch.max(torch.abs(extensive_activation)).cpu()
            ),
            "hessian_correction_frobenius": float(
                torch.linalg.matrix_norm(hessian).cpu()
            ),
            "linear_only_hessian_correction_frobenius": float(
                torch.linalg.matrix_norm(linear_hessian).cpu()
            ),
            "atom_extensive_hessian_correction_frobenius": float(
                torch.linalg.matrix_norm(extensive_hessian).cpu()
            ),
            "wall_time_s": time.perf_counter() - parent_started,
            **coverage,
            **baseline_hessian_metrics,
            **hessian_metrics(predicted_hessian, parent.pbe_hessian),
            **_prefixed_hessian_metrics(
                "linear_only", linear_predicted_hessian, parent.pbe_hessian
            ),
            **_prefixed_hessian_metrics(
                "atom_extensive", extensive_predicted_hessian, parent.pbe_hessian
            ),
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{parent.molecule_id}_result.npz",
            predicted_energy=np.asarray(predicted_energy),
            predicted_force=predicted_force,
            predicted_hessian=predicted_hessian,
            linear_only_predicted_energy=np.asarray(linear_predicted_energy),
            linear_only_predicted_force=linear_predicted_force,
            linear_only_predicted_hessian=linear_predicted_hessian,
            atom_extensive_predicted_energy=np.asarray(extensive_predicted_energy),
            atom_extensive_predicted_force=extensive_predicted_force,
            atom_extensive_predicted_hessian=extensive_predicted_hessian,
            baseline_energy=np.asarray(parent.energy),
            baseline_force=parent.force,
            baseline_hessian=parent.hessian,
            pbe_energy=np.asarray(parent.pbe_energy),
            pbe_force=parent.pbe_force,
            pbe_hessian=parent.pbe_hessian,
        )

    csv_path = args.output_dir / "per_parent_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    relative = [float(row["relative_frobenius"]) for row in rows]
    relative_summary = _quantiles(relative)
    fraction_at_or_below_0_15 = float(np.mean(np.asarray(relative) <= 0.15))
    max_asymmetry_ratio = max(
        float(row["antisymmetric_over_symmetric_frobenius"]) for row in rows
    )
    result = {
        "definition": (
            "Frozen shared scalar geometry residual evaluated on independent strict "
            "complete-total density-relaxed baselines; no external-parent fitting."
        ),
        "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
        "baseline_manifest_sha256": _sha256(args.baseline_manifest),
        "active_schema_manifest": args.active_schema_manifest.resolve().as_posix(),
        "active_schema_manifest_sha256": schema_sha,
        "schema_checkpoint_sha256": schema_manifest["checkpoint_sha256"],
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "architecture": (
            "one_hidden_plus_deep_residual" if deep_hidden_size else "one_hidden"
        ),
        "hidden_size": hidden_size,
        "deep_hidden_size": deep_hidden_size,
        "checkpoint_schema_binding": (
            "exact_schema_source"
            if checkpoint_matches_schema_source
            else "explicit_schema_hash"
            if checkpoint_inherits_schema
            else "matching_inventory_and_scale_definition"
        ),
        "checkpoint_atom_count_extensive": checkpoint_atom_count_extensive,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_count": len(rows),
        "source_split_sha256": baseline_manifest.get("source_split_sha256"),
        "baseline_hessian_relative_frobenius": _quantiles(
            [float(row["baseline_relative_frobenius"]) for row in rows]
        ),
        "hessian_relative_frobenius": relative_summary,
        "linear_only_hessian_relative_frobenius": _quantiles(
            [float(row["linear_only_relative_frobenius"]) for row in rows]
        ),
        "atom_extensive_hessian_relative_frobenius": _quantiles(
            [float(row["atom_extensive_relative_frobenius"]) for row in rows]
        ),
        "fraction_relative_frobenius_at_or_below_0_10": float(
            np.mean(np.asarray(relative) <= 0.10)
        ),
        "fraction_relative_frobenius_at_or_below_0_15": fraction_at_or_below_0_15,
        "fraction_relative_frobenius_at_or_below_0_20": float(
            np.mean(np.asarray(relative) <= 0.20)
        ),
        "energy_abs_error_hartree": _quantiles(
            [float(row["energy_abs_error_hartree"]) for row in rows]
        ),
        "baseline_energy_abs_error_hartree": _quantiles(
            [float(row["baseline_energy_abs_error_hartree"]) for row in rows]
        ),
        "force_mae_hartree_per_bohr": _quantiles(
            [float(row["force_mae_hartree_per_bohr"]) for row in rows]
        ),
        "baseline_force_mae_hartree_per_bohr": _quantiles(
            [float(row["baseline_force_mae_hartree_per_bohr"]) for row in rows]
        ),
        "active_feature_coverage": {
            stage: _quantiles(
                [
                    float(row[f"{stage}_matched_active_feature_fraction"])
                    for row in rows
                ]
            )
            for stage in ("three_body", "four_body")
        },
        "max_antisymmetric_over_symmetric_frobenius": max_asymmetry_ratio,
        "stage3_hessian_distribution_gate_passed": bool(
            relative_summary["median"] <= 0.10
            and fraction_at_or_below_0_15 >= 0.80
            and relative_summary["p90"] <= 0.20
        ),
        "numerical_asymmetry_gate_passed": bool(max_asymmetry_ratio <= 0.005),
        "per_parent_metrics": csv_path.resolve().as_posix(),
        "per_parent_metrics_sha256": _sha256(csv_path),
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--active-schema-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--derivative-step", type=float, default=1.0e-3)
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
    evaluate(parse_args())
