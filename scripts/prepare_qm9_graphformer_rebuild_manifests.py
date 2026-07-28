#!/usr/bin/env python3
"""Freeze new train-only parent and full internal-direction manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

from mldft.ofdft.internal_directions import (
    build_structured_internal_direction_bank,
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if (
        protocol.get("validation_access_allowed") is not False
        or protocol.get("test100_access_allowed") is not False
        or protocol["identity"]["old_original_a_identity_allowed"] is not False
    ):
        raise ValueError("New rebuild protocol identity/access boundary is invalid")
    return protocol


def _train_parent_ids(dataset_dir: Path) -> set[str]:
    with (dataset_dir / "split.pkl").open("rb") as handle:
        split = pickle.load(handle)
    if (
        split.get("train_only") is not True
        or split["val"]
        or split["test"]
        or split.get("validation_access_allowed") is not False
        or split.get("test_access_allowed") is not False
    ):
        raise ValueError("Dataset split is not a closed train-only split")
    return {str(row[1]).split(".", maxsplit=1)[0] for row in split["train"]}


def _reference_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if (
        payload.get("validation_accessed") is not False
        or payload.get("test100_accessed") is not False
        or payload.get("failed_count") != 0
    ):
        raise ValueError("PBE Hessian manifest is not closed and complete")
    return {str(row["molecule_id"]): row for row in payload["parents"]}


def parent_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _protocol(args.protocol)
    requested = [
        str(value)
        for value in protocol["parent_sets"][args.parent_set]["molecule_ids"]
    ]
    train_ids = _train_parent_ids(args.dataset_dir)
    if any(molecule_id not in train_ids for molecule_id in requested):
        raise ValueError(f"{args.parent_set} includes a non-train parent")
    references = _reference_map(args.pbe_manifest)
    parents = []
    for molecule_id in requested:
        reference = references[molecule_id]
        label_path = (
            args.dataset_dir / "labels" / f"{molecule_id}.0000000.zarr.zip"
        )
        hessian_path = Path(reference["cache_path"])
        if _sha256(hessian_path) != reference["pbe_hessian_sha256"]:
            raise ValueError(f"PBE Hessian hash drift for {molecule_id}")
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        force = np.asarray(
            root["metadata/pbe_derivatives/forces"], dtype=np.float64
        )
        energies = np.asarray(root["ks_labels/energies/e_tot"], dtype=np.float64)
        has_energy = np.asarray(
            root["ks_labels/energies/has_energy_label"], dtype=np.bool_
        )
        with np.load(hessian_path) as values:
            hessian = np.asarray(values["pbe_hessian"], dtype=np.float64)
        if hessian.shape != (positions.size, positions.size):
            raise ValueError(f"PBE Hessian shape drift for {molecule_id}")
        if not all(
            np.isfinite(value).all()
            for value in (positions, force, energies[has_energy], hessian)
        ):
            raise ValueError(f"Non-finite parent data for {molecule_id}")
        parents.append(
            {
                "molecule_id": molecule_id,
                "sample_id": 0,
                "natoms": int(atomic_numbers.size),
                "ncoordinates": int(positions.size),
                "atomic_numbers": atomic_numbers.tolist(),
                "pbe_total_energy_hartree": float(energies[has_energy][-1]),
                "pbe_force_rms_hartree_per_bohr": float(
                    np.sqrt(np.mean(force**2))
                ),
                "pbe_hessian_rms_hartree_per_bohr2": float(
                    np.sqrt(np.mean(hessian**2))
                ),
                "pbe_hessian_symmetry_max_abs": float(
                    np.max(np.abs(hessian - hessian.T))
                ),
                "label_path": label_path.resolve().as_posix(),
                "label_sha256": _sha256(label_path),
                "pbe_hessian_path": hessian_path.resolve().as_posix(),
                "pbe_hessian_sha256": _sha256(hessian_path),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "branch_id": protocol["branch_id"],
        "parent_set": args.parent_set,
        "role": "train-only capacity/audit; not validation or model selection",
        "parent_count": len(parents),
        "source_checkpoint": None,
        "old_original_a_identity_used": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "parents": parents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": args.output.resolve().as_posix(),
        "manifest_sha256": _sha256(args.output),
        "parent_count": len(parents),
    }


def direction_manifest(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _protocol(args.protocol)
    parent_payload = json.loads(args.parent_manifest.read_text())
    if (
        parent_payload["protocol_id"] != protocol["protocol_id"]
        or parent_payload["parent_set"] != args.parent_set
        or parent_payload["old_original_a_identity_used"] is not False
    ):
        raise ValueError("Parent manifest identity drift")
    settings = protocol["directions"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    kind_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for offset, parent in enumerate(parent_payload["parents"]):
        molecule_id = str(parent["molecule_id"])
        label_path = Path(parent["label_path"])
        hessian_path = Path(parent["pbe_hessian_path"])
        if _sha256(label_path) != parent["label_sha256"]:
            raise ValueError(f"Label hash drift for {molecule_id}")
        if _sha256(hessian_path) != parent["pbe_hessian_sha256"]:
            raise ValueError(f"PBE Hessian hash drift for {molecule_id}")
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        with np.load(hessian_path) as values:
            hessian = np.asarray(values["pbe_hessian"], dtype=np.float64)
        bank = build_structured_internal_direction_bank(
            atomic_numbers,
            positions,
            seed=int(settings["seed"])
            + offset * int(settings["parent_seed_stride"]),
            heldout_fraction=float(settings["partial_heldout_fraction"]),
            structured_per_kind=int(settings["structured_per_kind"]),
            bond_scale=float(settings["bond_scale"]),
        )
        pbe_hvp = np.einsum("ij,dj->di", hessian, bank.directions)
        artifact = artifact_dir / f"{molecule_id}.0000000.npz"
        np.savez_compressed(
            artifact,
            molecule_id=np.asarray(molecule_id),
            atomic_numbers=atomic_numbers,
            positions_bohr=positions,
            directions=bank.directions,
            kinds=bank.kinds,
            partial_roles=bank.partial_roles,
            pbe_hvp=pbe_hvp,
            external_basis=bank.external_basis,
            projector=bank.projector,
            external_rank=np.asarray(bank.external_rank, dtype=np.int64),
            internal_dimension=np.asarray(bank.internal_dimension, dtype=np.int64),
        )
        local_kinds = Counter(bank.kinds.tolist())
        local_roles = Counter(bank.partial_roles.tolist())
        kind_counts.update(local_kinds)
        role_counts.update(local_roles)
        entries.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(atomic_numbers.size),
                "external_rank": int(bank.external_rank),
                "internal_dimension": int(bank.internal_dimension),
                "direction_path": artifact.resolve().as_posix(),
                "direction_sha256": _sha256(artifact),
                "kind_counts": dict(sorted(local_kinds.items())),
                "partial_role_counts": dict(sorted(local_roles.items())),
                "orthonormality_max_abs": bank.orthonormality_max_abs,
                "external_overlap_max_abs": bank.external_overlap_max_abs,
                "projector_idempotence_max_abs": (
                    bank.projector_idempotence_max_abs
                ),
                "label_path": parent["label_path"],
                "label_sha256": parent["label_sha256"],
                "pbe_hessian_path": parent["pbe_hessian_path"],
                "pbe_hessian_sha256": parent["pbe_hessian_sha256"],
            }
        )
    maximum_orthonormality = max(
        row["orthonormality_max_abs"] for row in entries
    )
    maximum_external_overlap = max(
        row["external_overlap_max_abs"] for row in entries
    )
    if maximum_orthonormality > float(settings["orthonormality_max_abs"]):
        raise ValueError("Internal basis orthonormality gate failed")
    if maximum_external_overlap > float(settings["external_overlap_max_abs"]):
        raise ValueError("External-space overlap gate failed")
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "parent_set": args.parent_set,
        "parent_manifest": args.parent_manifest.resolve().as_posix(),
        "parent_manifest_sha256": _sha256(args.parent_manifest),
        "definition": (
            "Complete geometry-only orthonormal 3N-6 internal basis; PBE "
            "analytic Hessian supplies HVP targets after basis construction."
        ),
        "online_training_direction": (
            "One freshly resampled unnormalized +/-1 Rademacher probe in the "
            "complete orthonormal internal basis per step; this is the unbiased "
            "internal Hessian Frobenius-squared estimator."
        ),
        "parent_count": len(entries),
        "total_direction_count": sum(row["internal_dimension"] for row in entries),
        "kind_counts": dict(sorted(kind_counts.items())),
        "partial_role_counts": dict(sorted(role_counts.items())),
        "maximum_orthonormality_max_abs": maximum_orthonormality,
        "maximum_external_overlap_max_abs": maximum_external_overlap,
        "old_original_a_identity_used": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "parents": entries,
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": manifest.resolve().as_posix(),
        "manifest_sha256": _sha256(manifest),
        "parent_count": len(entries),
        "total_direction_count": payload["total_direction_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parents = subparsers.add_parser("parents")
    parents.add_argument("--protocol", type=Path, required=True)
    parents.add_argument("--dataset-dir", type=Path, required=True)
    parents.add_argument("--pbe-manifest", type=Path, required=True)
    parents.add_argument("--parent-set", choices=("stable5", "train20"), required=True)
    parents.add_argument("--output", type=Path, required=True)
    directions = subparsers.add_parser("directions")
    directions.add_argument("--protocol", type=Path, required=True)
    directions.add_argument("--parent-manifest", type=Path, required=True)
    directions.add_argument(
        "--parent-set", choices=("stable5", "train20"), required=True
    )
    directions.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = (
        parent_manifest(arguments)
        if arguments.command == "parents"
        else direction_manifest(arguments)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
