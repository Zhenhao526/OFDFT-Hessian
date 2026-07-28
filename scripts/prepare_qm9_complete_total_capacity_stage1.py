#!/usr/bin/env python3
"""Freeze and validate the train-only Stage-1 complete-total Hessian capacity set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_ids(split_rows: list[Any]) -> set[str]:
    result = set()
    for row in split_rows:
        filename = Path(str(row[1])).name
        result.add(filename.split(".", maxsplit=1)[0])
    return result


def _load_stability(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No stability rows in {path}")
    return {row["molecule_id"]: row for row in rows}


def _read_reference(
    dataset_dir: Path,
    reference_dir: Path,
    molecule_id: str,
    sample_id: int,
) -> dict[str, Any]:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    hessian_path = reference_dir / f"pbe_hessian_{molecule_id}_{sample_id:07d}.npz"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    if not hessian_path.is_file():
        raise FileNotFoundError(hessian_path)

    root = zarr.open(label_path, mode="r")
    atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
    positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    force = np.asarray(root["metadata/pbe_derivatives/forces"], dtype=np.float64)
    energy_trace = np.asarray(root["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        root["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    labelled_energies = energy_trace[has_energy]
    if labelled_energies.size == 0:
        raise ValueError(f"{label_path} has no labelled total energy")
    with np.load(hessian_path) as payload:
        hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)

    expected_hessian_shape = (positions.size, positions.size)
    if positions.shape != (atomic_numbers.size, 3):
        raise ValueError(f"Invalid positions shape in {label_path}: {positions.shape}")
    if force.shape != positions.shape:
        raise ValueError(f"Invalid force shape in {label_path}: {force.shape}")
    if hessian.shape != expected_hessian_shape:
        raise ValueError(
            f"Invalid Hessian shape in {hessian_path}: {hessian.shape}, expected "
            f"{expected_hessian_shape}"
        )
    arrays = {"positions": positions, "force": force, "hessian": hessian}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf for {molecule_id}")
    if not np.isfinite(labelled_energies).all():
        raise ValueError(f"total energy contains NaN/Inf for {molecule_id}")

    symmetric = 0.5 * (hessian + hessian.T)
    symmetry_max_abs = float(np.max(np.abs(hessian - hessian.T)))
    symmetry_relative_frobenius = float(
        np.linalg.norm(hessian - hessian.T)
        / max(np.linalg.norm(symmetric), np.finfo(float).tiny)
    )
    return {
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": int(atomic_numbers.size),
        "ncoordinates": int(positions.size),
        "atomic_numbers": atomic_numbers.tolist(),
        "pbe_total_energy_hartree": float(labelled_energies[-1]),
        "pbe_force_rms_hartree_per_bohr": float(np.sqrt(np.mean(force**2))),
        "pbe_hessian_rms_hartree_per_bohr2": float(np.sqrt(np.mean(hessian**2))),
        "pbe_hessian_symmetry_max_abs": symmetry_max_abs,
        "pbe_hessian_symmetry_relative_frobenius": symmetry_relative_frobenius,
        "label_path": label_path.as_posix(),
        "label_sha256": _sha256(label_path),
        "pbe_hessian_path": hessian_path.as_posix(),
        "pbe_hessian_sha256": _sha256(hessian_path),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Protocol must explicitly freeze Test100")
    split_path = args.dataset_dir / "split.pkl"
    split_sha256 = _sha256(split_path)
    expected_split_hash = str(protocol["data"]["source_split_sha256"])
    if split_sha256 != expected_split_hash:
        raise ValueError(
            f"split hash mismatch: expected {expected_split_hash}, got {split_sha256}"
        )
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    parent_splits = {
        name: _parent_ids(split[name]) for name in ("train", "val", "test")
    }
    stability = _load_stability(args.stability_csv)
    requested = protocol["data"]["stage1_parents"]
    require_all_stability_directions = bool(
        protocol["data"].get("require_all_stability_directions", True)
    )
    sample_id = int(protocol["data"]["sample_id"])
    parents = []
    for entry in requested:
        molecule_id = str(entry["molecule_id"])
        memberships = [name for name, ids in parent_splits.items() if molecule_id in ids]
        if memberships != ["train"]:
            raise ValueError(
                f"Stage-1 parent {molecule_id} must be train-only, got {memberships}"
            )
        if molecule_id not in stability:
            raise ValueError(f"No stability result for {molecule_id}")
        stability_row = stability[molecule_id]
        if stability_row["parent_stable"].lower() != "true":
            raise ValueError(f"Stage-1 parent {molecule_id} is not parent-stable")
        if require_all_stability_directions and int(
            stability_row["stable_direction_count"]
        ) != int(stability_row["direction_count"]):
            raise ValueError(
                f"Stage-1 parent {molecule_id} does not pass every audited direction"
            )
        reference = _read_reference(
            args.dataset_dir, args.reference_dir, molecule_id, sample_id
        )
        if reference["natoms"] != int(entry["natoms"]):
            raise ValueError(
                f"natoms mismatch for {molecule_id}: protocol {entry['natoms']}, "
                f"data {reference['natoms']}"
            )
        parents.append(
            {
                **reference,
                "selection_reason": entry["reason"],
                "stability_direction_count": int(stability_row["direction_count"]),
                "stability_failure_reasons": stability_row["failure_reasons"],
            }
        )

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "test100_accessed": False,
        "test100_label_reads": 0,
        "split_path": split_path.as_posix(),
        "split_sha256": split_sha256,
        "stability_csv": args.stability_csv.resolve().as_posix(),
        "stability_sha256": _sha256(args.stability_csv),
        "stage1_role": protocol["data"].get(
            "stage1_role", "train-only capacity fit; never validation or model selection"
        ),
        "require_all_stability_directions": require_all_stability_directions,
        "parent_count": len(parents),
        "parents": parents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized)
    manifest_hash = hashlib.sha256(serialized.encode()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{manifest_hash}  {args.output.name}\n"
    )
    return {**payload, "manifest_sha256": manifest_hash}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = prepare(parse_args())
    print(
        json.dumps(
            {
                "parent_count": result["parent_count"],
                "parent_ids": [row["molecule_id"] for row in result["parents"]],
                "manifest_sha256": result["manifest_sha256"],
                "test100_accessed": result["test100_accessed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
