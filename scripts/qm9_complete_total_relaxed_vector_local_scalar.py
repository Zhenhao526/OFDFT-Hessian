#!/usr/bin/env python3
"""Fit a same-source scalar local correction to relaxed vector-HVP targets."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.qm9_complete_total_relaxed_q_local_scalar import (
    _dual_normalized_ridge_grid,
)


@dataclass
class ParentData:
    molecule_id: str
    natoms: int
    pbe_energy: float
    pbe_force: np.ndarray
    source_energy: float
    source_force: np.ndarray
    direction_index: np.ndarray
    direction_kind: np.ndarray
    role: np.ndarray
    direction: np.ndarray
    source_hvp: np.ndarray
    pbe_hvp: np.ndarray
    feature_jet_path: Path
    feature_jet_sha256: str
    feature_columns: torch.Tensor
    features: torch.Tensor
    jacobian: torch.Tensor
    hessian: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if (
        protocol.get("stage")
        != "train20_complete_total_relaxed_vector_local_scalar_diagnostic"
    ):
        raise ValueError("unexpected relaxed-vector fit stage")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    for key, expected in protocol["inputs"].items():
        if not key.endswith("_sha256"):
            continue
        source = Path(protocol["inputs"][key.removesuffix("_sha256")])
        if _sha256(source) != str(expected):
            raise ValueError(f"input hash drift: {source}")
    return protocol


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty vector-HVP stratum")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _hvp_relative(
    error: np.ndarray, reference: np.ndarray, floor: float
) -> float:
    return float(
        np.linalg.norm(error) / max(float(np.linalg.norm(reference)), floor)
    )


def _hvp_design_blocks(
    hessian: torch.Tensor,
    directions: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_scale: float,
    relative_fraction: float,
    reference_floor: float,
    absolute_target_cap: float,
    relative_target_cap: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not 0.0 < relative_fraction < 1.0:
        raise ValueError("relative HVP fraction must be strictly between zero and one")
    coordinate_count = hessian.shape[1]
    if directions.ndim != 2 or directions.shape[1] != coordinate_count:
        raise ValueError("HVP direction shape drift")
    if tuple(target.shape) != tuple(directions.shape):
        raise ValueError("HVP target shape drift")
    design = torch.einsum("fij,dj->dif", hessian, directions)
    component_normalization = math.sqrt(float(coordinate_count))
    target_rms = torch.linalg.vector_norm(target, dim=1) / component_normalization
    absolute_denominator = torch.maximum(
        target.new_full(target_rms.shape, absolute_scale),
        target_rms / absolute_target_cap,
    )
    reference_norm = torch.linalg.vector_norm(reference, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    relative_denominator = torch.maximum(
        target.new_full(reference_norm.shape, reference_floor),
        reference_norm,
    )
    relative_denominator = torch.maximum(
        relative_denominator, target_norm / relative_target_cap
    )
    absolute_weight = math.sqrt(1.0 - relative_fraction)
    relative_weight = math.sqrt(relative_fraction)
    absolute_design = (
        design
        * (absolute_weight / component_normalization)
        / absolute_denominator[:, None, None]
    ).reshape(-1, hessian.shape[0])
    absolute_target = (
        target
        * (absolute_weight / component_normalization)
        / absolute_denominator[:, None]
    ).reshape(-1)
    relative_design = (
        design
        * (relative_weight / component_normalization)
        / relative_denominator[:, None, None]
    ).reshape(-1, hessian.shape[0])
    relative_target = (
        target
        * (relative_weight / component_normalization)
        / relative_denominator[:, None]
    ).reshape(-1)
    return [absolute_design, relative_design], [
        absolute_target,
        relative_target,
    ]


def _load_parents(protocol: dict[str, Any]) -> tuple[list[ParentData], int]:
    inputs = protocol["inputs"]
    data = protocol["data"]
    selection = json.loads(Path(inputs["selection_manifest"]).read_text())
    targets = json.loads(Path(inputs["vector_target_manifest"]).read_text())
    baseline_manifest = json.loads(
        Path(inputs["source_baseline_manifest"]).read_text()
    )
    feature_manifest = json.loads(Path(inputs["feature_jet_manifest"]).read_text())
    if any(
        payload.get("test100_accessed") is not False
        for payload in (selection, targets, baseline_manifest, feature_manifest)
    ):
        raise ValueError("an input manifest opened Test100")
    if any(
        payload.get("validation_accessed") is not False
        for payload in (selection, targets, feature_manifest)
    ):
        raise ValueError("an input manifest opened validation")
    source_hash = str(inputs["source_checkpoint_sha256"])
    if (
        selection.get("source_checkpoint_sha256") != source_hash
        or targets.get("source_checkpoint_sha256") != source_hash
    ):
        raise ValueError("source checkpoint identity mismatch")
    if baseline_manifest.get("protocol_sha256") != str(
        inputs["target_protocol_sha256"]
    ):
        raise ValueError("source baseline protocol mismatch")
    if baseline_manifest["counts"] != {
        **baseline_manifest["counts"],
        "expected_tasks": int(data["parent_count"]),
        "successful_tasks": int(data["parent_count"]),
        "failed_or_missing_tasks": 0,
    }:
        raise ValueError("source baseline count mismatch")

    selection_by_id = {
        str(row["molecule_id"]): row for row in selection["parents"]
    }
    target_by_id = {
        str(row["molecule_id"]): row for row in targets["entries"]
    }
    feature_by_id = {
        str(row["molecule_id"]): row for row in feature_manifest["entries"]
    }
    baseline_rows = _read_csv(Path(inputs["source_baseline_success_csv"]))
    baseline_by_id = {str(row["molecule_id"]): row for row in baseline_rows}
    expected_ids = set(selection_by_id)
    if not (
        len(expected_ids)
        == int(data["parent_count"])
        == len(target_by_id)
        == len(baseline_by_id)
    ):
        raise ValueError("parent inventory size mismatch")
    if expected_ids != set(target_by_id) or expected_ids != set(baseline_by_id):
        raise ValueError("parent identity mismatch")

    parents = []
    global_feature_count = int(data["global_feature_count"])
    feature_protocol_hash = str(inputs["feature_protocol_sha256"])
    for molecule_id in [str(row["molecule_id"]) for row in selection["parents"]]:
        selection_row = selection_by_id[molecule_id]
        target_row = target_by_id[molecule_id]
        baseline_row = baseline_by_id[molecule_id]
        feature_row = feature_by_id[molecule_id]
        feature_path = Path(feature_row["artifact"])
        if _sha256(feature_path) != str(feature_row["artifact_sha256"]):
            raise ValueError(f"feature artifact hash drift: {molecule_id}")
        target_path = Path(target_row["sidecar"])
        if _sha256(target_path) != str(target_row["sidecar_sha256"]):
            raise ValueError(f"vector sidecar hash drift: {molecule_id}")
        baseline_path = Path(baseline_row["baseline_array"])
        if _sha256(baseline_path) != str(baseline_row["baseline_array_sha256"]):
            raise ValueError(f"source baseline hash drift: {molecule_id}")
        pbe_base_path = Path(selection_row["pbe_base_array"])
        if _sha256(pbe_base_path) != str(selection_row["pbe_base_array_sha256"]):
            raise ValueError(f"PBE base provenance drift: {molecule_id}")

        feature = torch.load(feature_path, map_location="cpu", weights_only=False)
        if (
            feature.get("protocol_sha256") != feature_protocol_hash
            or feature.get("molecule_id") != molecule_id
        ):
            raise ValueError(f"feature provenance drift: {molecule_id}")
        columns = feature["global_columns"].to(dtype=torch.long)
        if int(feature["global_feature_count"]) != global_feature_count:
            raise ValueError(f"global feature count drift: {molecule_id}")
        with np.load(target_path) as vector, np.load(
            baseline_path
        ) as base, np.load(pbe_base_path) as pbe_provenance:
            atomic_numbers = np.asarray(vector["atomic_numbers"], dtype=np.int64)
            positions = np.asarray(vector["positions_bohr"], dtype=np.float64)
            if not np.array_equal(
                atomic_numbers, np.asarray(base["atomic_numbers"], dtype=np.int64)
            ) or not np.allclose(
                positions,
                np.asarray(base["positions_bohr"], dtype=np.float64),
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(f"same-source base geometry drift: {molecule_id}")
            for key in ("atomic_numbers", "positions_bohr", "pbe_force"):
                if not np.allclose(
                    np.asarray(base[key]),
                    np.asarray(pbe_provenance[key]),
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    raise ValueError(f"PBE base data drift: {molecule_id} {key}")
            if not math.isclose(
                float(base["pbe_total_energy"]),
                float(pbe_provenance["pbe_total_energy"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"PBE energy drift: {molecule_id}")
            parent = ParentData(
                molecule_id=molecule_id,
                natoms=int(atomic_numbers.size),
                pbe_energy=float(base["pbe_total_energy"]),
                pbe_force=np.asarray(base["pbe_force"], dtype=np.float64),
                source_energy=float(base["baseline_total_energy"]),
                source_force=np.asarray(
                    base["baseline_total_force"], dtype=np.float64
                ),
                direction_index=np.asarray(
                    vector["direction_index"], dtype=np.int64
                ),
                direction_kind=np.asarray(vector["direction_kind"]).astype(str),
                role=np.asarray(vector["role"]).astype(str),
                direction=np.asarray(vector["direction"], dtype=np.float64),
                source_hvp=np.asarray(
                    vector["source_hvp_hartree_per_bohr2"], dtype=np.float64
                ),
                pbe_hvp=np.asarray(
                    vector["pbe_hvp_hartree_per_bohr2"], dtype=np.float64
                ),
                feature_jet_path=feature_path,
                feature_jet_sha256=str(feature_row["artifact_sha256"]),
                feature_columns=columns,
                features=feature["features"].to(dtype=torch.float64),
                jacobian=feature["jacobian"].to(dtype=torch.float64),
                hessian=feature["hessian"].to(dtype=torch.float64),
            )
        if parent.features.numel() != parent.feature_columns.numel():
            raise ValueError(f"feature-column mismatch: {molecule_id}")
        coordinate_count = 3 * parent.natoms
        if tuple(parent.jacobian.shape) != (
            parent.features.numel(),
            coordinate_count,
        ) or tuple(parent.hessian.shape) != (
            parent.features.numel(),
            coordinate_count,
            coordinate_count,
        ):
            raise ValueError(f"feature-jet shape drift: {molecule_id}")
        if parent.direction.shape != parent.source_hvp.shape or (
            parent.direction.shape != parent.pbe_hvp.shape
        ):
            raise ValueError(f"vector-HVP shape drift: {molecule_id}")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (parent.features, parent.jacobian, parent.hessian)
        ):
            raise FloatingPointError(f"non-finite feature jet: {molecule_id}")
        parents.append(parent)
        del feature
        gc.collect()
    return parents, global_feature_count


def _build_design(
    parents: list[ParentData], global_count: int, loss: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = []
    targets = []
    for parent in parents:
        coordinate_count = 3 * parent.natoms
        local_blocks = [
            parent.features[None, :] / float(loss["energy_scale_hartree"])
        ]
        local_targets = [
            parent.features.new_tensor(
                [parent.pbe_energy - parent.source_energy]
            )
            / float(loss["energy_scale_hartree"])
        ]
        force_scale = (
            float(loss["force_scale_hartree_per_bohr"])
            * math.sqrt(float(coordinate_count))
        )
        local_blocks.append(-parent.jacobian.T / force_scale)
        local_targets.append(
            parent.features.new_tensor(
                (parent.pbe_force - parent.source_force).reshape(-1)
            )
            / force_scale
        )
        train = parent.role == "train"
        directions = parent.features.new_tensor(
            parent.direction[train].reshape(np.count_nonzero(train), -1)
        )
        target = parent.features.new_tensor(
            (parent.pbe_hvp[train] - parent.source_hvp[train]).reshape(
                np.count_nonzero(train), -1
            )
        )
        reference = parent.features.new_tensor(
            parent.pbe_hvp[train].reshape(np.count_nonzero(train), -1)
        )
        hvp_blocks, hvp_targets = _hvp_design_blocks(
            parent.hessian,
            directions,
            target,
            reference,
            absolute_scale=float(
                loss["hvp_absolute_scale_hartree_per_bohr2"]
            ),
            relative_fraction=float(loss["hvp_relative_fraction"]),
            reference_floor=float(
                loss["hvp_reference_norm_floor_hartree_per_bohr2"]
            ),
            absolute_target_cap=float(
                loss["hvp_absolute_normalized_target_cap"]
            ),
            relative_target_cap=float(
                loss["hvp_relative_normalized_target_cap"]
            ),
        )
        local_blocks.extend(hvp_blocks)
        local_targets.extend(hvp_targets)
        for block in local_blocks:
            global_block = torch.zeros(
                (block.shape[0], global_count), dtype=torch.float64
            )
            global_block.index_copy_(1, parent.feature_columns, block)
            blocks.append(global_block)
        targets.extend(local_targets)
    design = torch.cat(blocks, dim=0)
    target = torch.cat(targets, dim=0)
    return design, target


def _evaluate(
    coefficients: torch.Tensor,
    parents: list[ParentData],
    floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    parent_rows = []
    direction_rows = []
    for parent in parents:
        local = coefficients[parent.feature_columns]
        correction_energy = float(torch.dot(parent.features, local))
        correction_force = (-parent.jacobian.T @ local).numpy()
        correction_hessian = torch.einsum(
            "fij,f->ij", parent.hessian, local
        ).numpy()
        predicted_energy = parent.source_energy + correction_energy
        predicted_force = parent.source_force + correction_force.reshape(
            parent.natoms, 3
        )
        symmetric = 0.5 * (
            correction_hessian + correction_hessian.T
        )
        antisymmetric = 0.5 * (
            correction_hessian - correction_hessian.T
        )
        held_candidate = []
        held_source = []
        for index in range(parent.direction.shape[0]):
            direction = parent.direction[index].reshape(-1)
            source = parent.source_hvp[index].reshape(-1)
            reference = parent.pbe_hvp[index].reshape(-1)
            predicted = source + correction_hessian @ direction
            error = predicted - reference
            source_error = source - reference
            relative = _hvp_relative(error, reference, floor)
            source_relative = _hvp_relative(
                source_error, reference, floor
            )
            row = {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "direction_index": int(parent.direction_index[index]),
                "direction_kind": str(parent.direction_kind[index]),
                "role": str(parent.role[index]),
                "mae_hartree_per_bohr2": float(np.mean(np.abs(error))),
                "rmse_hartree_per_bohr2": float(
                    np.sqrt(np.mean(error**2))
                ),
                "relative_l2_with_floor": relative,
                "source_mae_hartree_per_bohr2": float(
                    np.mean(np.abs(source_error))
                ),
                "source_relative_l2_with_floor": source_relative,
                "improved": relative < source_relative,
                "reference_norm_hartree_per_bohr2": float(
                    np.linalg.norm(reference)
                ),
            }
            direction_rows.append(row)
            if row["role"] == "heldout":
                held_candidate.append(relative)
                held_source.append(source_relative)
        parent_rows.append(
            {
                "molecule_id": parent.molecule_id,
                "natoms": parent.natoms,
                "energy_abs_error_hartree": abs(
                    predicted_energy - parent.pbe_energy
                ),
                "source_energy_abs_error_hartree": abs(
                    parent.source_energy - parent.pbe_energy
                ),
                "force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(predicted_force - parent.pbe_force))
                ),
                "source_force_mae_hartree_per_bohr": float(
                    np.mean(np.abs(parent.source_force - parent.pbe_force))
                ),
                "heldout_hvp_relative_error": float(np.mean(held_candidate)),
                "source_heldout_hvp_relative_error": float(
                    np.mean(held_source)
                ),
                "heldout_improved": float(np.mean(held_candidate))
                < float(np.mean(held_source)),
                "correction_hessian_asym_over_sym": float(
                    np.linalg.norm(antisymmetric)
                    / max(np.linalg.norm(symmetric), np.finfo(float).tiny)
                ),
            }
        )
    train_rows = [row for row in direction_rows if row["role"] == "train"]
    held_rows = [row for row in direction_rows if row["role"] == "heldout"]
    energy = _distribution(
        [row["energy_abs_error_hartree"] for row in parent_rows]
    )
    source_energy = _distribution(
        [row["source_energy_abs_error_hartree"] for row in parent_rows]
    )
    force = _distribution(
        [row["force_mae_hartree_per_bohr"] for row in parent_rows]
    )
    source_force = _distribution(
        [row["source_force_mae_hartree_per_bohr"] for row in parent_rows]
    )
    aggregate = {
        "train_hvp_relative_l2": _distribution(
            [row["relative_l2_with_floor"] for row in train_rows]
        ),
        "heldout_hvp_relative_l2": _distribution(
            [row["relative_l2_with_floor"] for row in held_rows]
        ),
        "heldout_hvp_component_mae_hartree_per_bohr2": _distribution(
            [row["mae_hartree_per_bohr2"] for row in held_rows]
        ),
        "heldout_hvp_component_rmse_hartree_per_bohr2": _distribution(
            [row["rmse_hartree_per_bohr2"] for row in held_rows]
        ),
        "heldout_hvp_fraction_at_or_below_0_15": float(
            np.mean(
                np.asarray(
                    [row["relative_l2_with_floor"] for row in held_rows]
                )
                <= 0.15
            )
        ),
        "heldout_hvp_improved_fraction": float(
            np.mean([bool(row["improved"]) for row in held_rows])
        ),
        "energy_abs_error_hartree": energy,
        "source_energy_abs_error_hartree": source_energy,
        "energy_median_ratio": energy["median"]
        / max(source_energy["median"], np.finfo(float).tiny),
        "force_mae_hartree_per_bohr": force,
        "source_force_mae_hartree_per_bohr": source_force,
        "force_median_ratio": force["median"]
        / max(source_force["median"], np.finfo(float).tiny),
        "correction_hessian_asym_over_sym_max": max(
            row["correction_hessian_asym_over_sym"] for row in parent_rows
        ),
    }
    return aggregate, parent_rows, direction_rows


def fit(
    protocol_path: Path, output_dir: Path, device_name: str
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(protocol_path)
    protocol_hash = _sha256(protocol_path)
    parents, global_count = _load_parents(protocol)
    design, target = _build_design(parents, global_count, protocol["loss"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    solutions = _dual_normalized_ridge_grid(
        design,
        target,
        ridges=[float(value) for value in protocol["solver"]["ridge_grid"]],
        column_floor=float(protocol["solver"]["column_floor"]),
        device=device,
    )
    floor = float(
        protocol["loss"]["hvp_reference_norm_floor_hartree_per_bohr2"]
    )
    arms = []
    for ridge in protocol["solver"]["ridge_grid"]:
        coefficients, solver = solutions[float(ridge)]
        metrics, parent_rows, direction_rows = _evaluate(
            coefficients, parents, floor
        )
        arm_id = f"ridge_{float(ridge):.0e}".replace("-", "m")
        arm_dir = output_dir / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = arm_dir / "model.pt"
        torch.save(
            {
                "coefficients": coefficients,
                "protocol": protocol_path.resolve().as_posix(),
                "protocol_sha256": protocol_hash,
                "ridge": float(ridge),
                "global_feature_count": global_count,
                "source_checkpoint_sha256": protocol["inputs"][
                    "source_checkpoint_sha256"
                ],
                "feature_jet_hashes": {
                    parent.molecule_id: parent.feature_jet_sha256
                    for parent in parents
                },
                "formal_stage3_authorized": False,
                "test100_accessed": False,
            },
            checkpoint,
        )
        for name, rows in (
            ("per_parent.json", parent_rows),
            ("per_direction.json", direction_rows),
        ):
            (arm_dir / name).write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n"
            )
        arm = {
            "arm_id": arm_id,
            "ridge": float(ridge),
            "solver": solver,
            "metrics": metrics,
            "checkpoint": checkpoint.resolve().as_posix(),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        (arm_dir / "summary.json").write_text(
            json.dumps(arm, indent=2, sort_keys=True) + "\n"
        )
        arms.append(arm)

    gates = protocol["diagnostic_gates"]
    energy_limit = float(gates["energy_median_ratio_max"])
    force_limit = float(gates["force_median_ratio_max"])

    def score(arm: dict[str, Any]) -> tuple[float, ...]:
        metrics = arm["metrics"]
        rejected = float(
            metrics["energy_median_ratio"] > energy_limit
            or metrics["force_median_ratio"] > force_limit
        )
        return (
            rejected,
            metrics["heldout_hvp_relative_l2"]["median"],
            metrics["heldout_hvp_relative_l2"]["p90"],
            metrics["train_hvp_relative_l2"]["median"],
        )

    selected = min(arms, key=score)
    metrics = selected["metrics"]
    checks = {
        "train_hvp_median": metrics["train_hvp_relative_l2"]["median"]
        <= float(gates["train_hvp_median_relative_error_max"]),
        "train_hvp_p90": metrics["train_hvp_relative_l2"]["p90"]
        <= float(gates["train_hvp_p90_relative_error_max"]),
        "heldout_hvp_median": metrics["heldout_hvp_relative_l2"]["median"]
        <= float(gates["heldout_hvp_median_relative_error_max"]),
        "heldout_hvp_p90": metrics["heldout_hvp_relative_l2"]["p90"]
        <= float(gates["heldout_hvp_p90_relative_error_max"]),
        "heldout_hvp_coverage": metrics[
            "heldout_hvp_fraction_at_or_below_0_15"
        ]
        >= float(gates["heldout_hvp_fraction_at_or_below_0_15_min"]),
        "heldout_hvp_improvement": metrics[
            "heldout_hvp_improved_fraction"
        ]
        >= float(gates["heldout_hvp_improved_fraction_min"]),
        "energy": metrics["energy_median_ratio"] <= energy_limit,
        "force": metrics["force_median_ratio"] <= force_limit,
        "symmetry": metrics["correction_hessian_asym_over_sym_max"]
        <= float(gates["correction_hessian_asym_over_sym_max"]),
    }
    result = {
        "definition": protocol["definitions"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": protocol_hash,
        "design_shape": list(design.shape),
        "arms": arms,
        "selected_arm_id": selected["arm_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
        "source_checkpoint": protocol["inputs"]["source_checkpoint"],
        "source_checkpoint_sha256": protocol["inputs"][
            "source_checkpoint_sha256"
        ],
        "same_source_energy_force_hvp": True,
        "train800_full_replay_included": False,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    fit(args.protocol, args.output_dir, args.device)
