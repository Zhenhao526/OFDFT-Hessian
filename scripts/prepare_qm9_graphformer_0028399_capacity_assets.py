#!/usr/bin/env python3
"""Freeze a one-parent, no-held-direction Graphformer capacity-only asset set."""

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


MOLECULE_ID = "0028399"
SAMPLE_ID = 0
PROTOCOL_IDS = {
    "qm9_graphformer_0028399_full39_capacity_only_v1",
    "qm9_graphformer_0028399_full39_capacity_only_v2",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    boundary = protocol["access_boundary"]
    if (
        protocol["protocol_id"] not in PROTOCOL_IDS
        or boundary["molecule_ids"] != [MOLECULE_ID]
        or boundary["sample_ids"] != [SAMPLE_ID]
        or any(
            boundary[key] is not False
            for key in (
                "stable5_access_allowed",
                "train20_access_allowed",
                "held_direction_access_allowed",
                "validation_access_allowed",
                "test100_access_allowed",
            )
        )
    ):
        raise ValueError("capacity-only access boundary is not closed")
    if any(
        float(protocol["loss"][key]) != expected
        for key, expected in (
            ("lambda_energy", 0.0),
            ("lambda_force", 0.0),
            ("lambda_density", 0.0),
            ("lambda_hessian", 1.0),
        )
    ):
        raise ValueError("capacity-only loss is not Hessian-only")
    return protocol


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: {actual} != {expected}")


def _train_parent_ids(dataset_root: Path) -> set[str]:
    with (dataset_root / "split.pkl").open("rb") as handle:
        split = pickle.load(handle)
    if (
        split.get("train_only") is not True
        or split.get("val")
        or split.get("test")
        or split.get("validation_access_allowed") is not False
        or split.get("test_access_allowed") is not False
    ):
        raise ValueError("dataset is not the frozen train-only split")
    return {
        str(row[1]).split(".", maxsplit=1)[0]
        for row in split["train"]
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol(args.protocol)
    source = protocol["source"]
    dataset_root = Path(source["dataset_root"])
    checkpoint = Path(source["checkpoint"])
    checkpoint_registration = Path(source["checkpoint_registration"])
    dataset_manifest = Path(source["train_only_dataset_manifest"])
    hessian_path = Path(source["pbe_hessian"])
    label_path = dataset_root / "labels" / f"{MOLECULE_ID}.0000000.zarr.zip"

    _assert_hash(checkpoint, source["checkpoint_sha256"], "checkpoint")
    _assert_hash(
        checkpoint_registration,
        source["checkpoint_registration_sha256"],
        "checkpoint registration",
    )
    _assert_hash(
        dataset_manifest,
        source["train_only_dataset_manifest_sha256"],
        "train-only dataset manifest",
    )
    _assert_hash(dataset_root / "split.pkl", source["split_sha256"], "split")
    _assert_hash(label_path, source["label_sha256"], "0028399 label")
    _assert_hash(hessian_path, source["pbe_hessian_sha256"], "0028399 PBE Hessian")
    if MOLECULE_ID not in _train_parent_ids(dataset_root):
        raise ValueError("0028399 is absent from frozen train parents")

    registration = json.loads(checkpoint_registration.read_text())
    if (
        registration["checkpoint"] != checkpoint.as_posix()
        or registration["checkpoint_sha256"] != source["checkpoint_sha256"]
        or registration["selection"] != "fixed_final_step_without_validation"
        or registration["validation_accessed"] is not False
        or registration["test100_accessed"] is not False
    ):
        raise ValueError("fresh checkpoint registration boundary drift")

    root = zarr.open(label_path, mode="r")
    atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
    positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    pbe_force = np.asarray(
        root["metadata/pbe_derivatives/forces"], dtype=np.float64
    )
    energies = np.asarray(root["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        root["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    with np.load(hessian_path) as payload:
        pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
    if pbe_hessian.shape != (positions.size, positions.size):
        raise ValueError("0028399 PBE Hessian shape drift")
    if not all(
        np.isfinite(value).all()
        for value in (positions, pbe_force, energies[has_energy], pbe_hessian)
    ):
        raise ValueError("0028399 capacity asset contains non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    parent_manifest_path = args.output_dir / "parent_manifest.json"
    parent = {
        "molecule_id": MOLECULE_ID,
        "sample_id": SAMPLE_ID,
        "natoms": int(atomic_numbers.size),
        "ncoordinates": int(positions.size),
        "atomic_numbers": atomic_numbers.tolist(),
        "pbe_total_energy_hartree": float(energies[has_energy][-1]),
        "pbe_force_rms_hartree_per_bohr": float(np.sqrt(np.mean(pbe_force**2))),
        "pbe_hessian_rms_hartree_per_bohr2": float(
            np.sqrt(np.mean(pbe_hessian**2))
        ),
        "pbe_hessian_symmetry_max_abs": float(
            np.max(np.abs(pbe_hessian - pbe_hessian.T))
        ),
        "label_path": label_path.as_posix(),
        "label_sha256": _sha256(label_path),
        "pbe_hessian_path": hessian_path.as_posix(),
        "pbe_hessian_sha256": _sha256(hessian_path),
    }
    parent_manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "branch_id": protocol["branch_id"],
        "scope": "single_train_parent_capacity_only",
        "parent_count": 1,
        "molecule_ids": [MOLECULE_ID],
        "sample_ids": [SAMPLE_ID],
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "train_only_dataset_manifest": dataset_manifest.as_posix(),
        "train_only_dataset_manifest_sha256": _sha256(dataset_manifest),
        "stable5_accessed": False,
        "train20_accessed": False,
        "held_directions_present": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "parents": [parent],
    }
    parent_manifest_path.write_text(
        json.dumps(parent_manifest, indent=2, sort_keys=True) + "\n"
    )

    settings = protocol["directions"]
    bank = build_structured_internal_direction_bank(
        atomic_numbers,
        positions,
        seed=int(settings["seed"]),
        heldout_fraction=0.2,
        structured_per_kind=int(settings["structured_per_kind"]),
        bond_scale=float(settings["bond_scale"]),
    )
    if bank.internal_dimension != int(settings["expected_internal_dimension"]):
        raise ValueError(
            f"internal dimension drift: {bank.internal_dimension} != "
            f"{settings['expected_internal_dimension']}"
        )
    if bank.orthonormality_max_abs > float(settings["orthonormality_max_abs"]):
        raise ValueError("internal direction orthonormality gate failed")
    if bank.external_overlap_max_abs > float(
        settings["external_overlap_max_abs"]
    ):
        raise ValueError("internal direction rigid-motion gate failed")

    direction_dir = args.output_dir / "directions"
    direction_dir.mkdir()
    direction_path = direction_dir / f"{MOLECULE_ID}.0000000.npz"
    roles = np.full(bank.internal_dimension, "capacity_train", dtype="U14")
    pbe_hvp = np.einsum("ij,dj->di", pbe_hessian, bank.directions)
    np.savez_compressed(
        direction_path,
        molecule_id=np.asarray(MOLECULE_ID),
        atomic_numbers=atomic_numbers,
        positions_bohr=positions,
        directions=bank.directions,
        kinds=bank.kinds,
        partial_roles=roles,
        pbe_hvp=pbe_hvp,
        external_basis=bank.external_basis,
        projector=bank.projector,
        external_rank=np.asarray(bank.external_rank, dtype=np.int64),
        internal_dimension=np.asarray(bank.internal_dimension, dtype=np.int64),
    )
    direction_manifest_path = args.output_dir / "direction_manifest.json"
    direction_manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "parent_manifest": parent_manifest_path.resolve().as_posix(),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "scope": "single_train_parent_complete_internal_basis_capacity_only",
        "parent_count": 1,
        "total_direction_count": int(bank.internal_dimension),
        "roles": ["capacity_train"],
        "role_counts": {"capacity_train": int(bank.internal_dimension)},
        "kind_counts": dict(sorted(Counter(bank.kinds.tolist()).items())),
        "stable5_accessed": False,
        "train20_accessed": False,
        "held_directions_present": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "parents": [
            {
                **parent,
                "external_rank": int(bank.external_rank),
                "internal_dimension": int(bank.internal_dimension),
                "direction_path": direction_path.resolve().as_posix(),
                "direction_sha256": _sha256(direction_path),
                "orthonormality_max_abs": bank.orthonormality_max_abs,
                "external_overlap_max_abs": bank.external_overlap_max_abs,
                "projector_idempotence_max_abs": (
                    bank.projector_idempotence_max_abs
                ),
                "role_counts": {"capacity_train": int(bank.internal_dimension)},
            }
        ],
    }
    direction_manifest_path.write_text(
        json.dumps(direction_manifest, indent=2, sort_keys=True) + "\n"
    )

    registration_path = args.output_dir / "asset_registration.json"
    registration_payload = {
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_registration": checkpoint_registration.as_posix(),
        "checkpoint_registration_sha256": _sha256(checkpoint_registration),
        "parent_manifest": parent_manifest_path.resolve().as_posix(),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "direction_manifest": direction_manifest_path.resolve().as_posix(),
        "direction_manifest_sha256": _sha256(direction_manifest_path),
        "direction_artifact": direction_path.resolve().as_posix(),
        "direction_artifact_sha256": _sha256(direction_path),
        "molecule_ids": [MOLECULE_ID],
        "direction_count": int(bank.internal_dimension),
        "stable5_accessed": False,
        "train20_accessed": False,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    registration_path.write_text(
        json.dumps(registration_payload, indent=2, sort_keys=True) + "\n"
    )
    return {
        **registration_payload,
        "asset_registration": registration_path.resolve().as_posix(),
        "asset_registration_sha256": _sha256(registration_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
