#!/usr/bin/env python3
"""Freeze complete structured internal direction banks for train-only parents."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if (
        protocol.get("validation_access_allowed") is not False
        or protocol.get("test100_access_allowed") is not False
    ):
        raise ValueError("direction protocol must freeze validation and Test100")
    return protocol


def prepare(
    protocol_path: Path,
    parent_set: str,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    if parent_set not in protocol["parent_sets"]:
        raise ValueError(f"unknown parent set {parent_set}")
    parent_config = protocol["parent_sets"][parent_set]
    source_manifest_path = Path(parent_config["source_manifest"])
    source_manifest_hash = _sha256(source_manifest_path)
    if source_manifest_hash != str(parent_config["source_manifest_sha256"]):
        raise ValueError(
            f"{parent_set} source manifest hash mismatch: "
            f"{source_manifest_hash} != {parent_config['source_manifest_sha256']}"
        )
    source = json.loads(source_manifest_path.read_text())
    if (
        source.get("test100_accessed") is not False
        or int(source.get("test100_evaluations_used", 0)) != 0
        or int(source.get("test100_label_reads", 0)) != 0
    ):
        raise ValueError(f"{parent_set} source manifest accessed Test100")
    expected_ids = [str(value) for value in parent_config["molecule_ids"]]
    source_entries = {
        str(row["molecule_id"]): row for row in source["parents"]
    }
    if list(source_entries) != expected_ids:
        raise ValueError(
            f"{parent_set} parent order drift: {list(source_entries)} != {expected_ids}"
        )

    settings = protocol["directions"]
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    kind_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    maximum_orthonormality = 0.0
    maximum_external_overlap = 0.0
    for parent_offset, molecule_id in enumerate(expected_ids):
        parent = source_entries[molecule_id]
        label_path = Path(parent["label_path"])
        hessian_path = Path(parent["pbe_hessian_path"])
        if _sha256(label_path) != str(parent["label_sha256"]):
            raise ValueError(f"label hash drift for {molecule_id}")
        if _sha256(hessian_path) != str(parent["pbe_hessian_sha256"]):
            raise ValueError(f"PBE Hessian hash drift for {molecule_id}")
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(
            root["geometry/atomic_numbers"], dtype=np.int64
        )
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        with np.load(hessian_path) as payload:
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        parent_seed = int(settings["seed"]) + parent_offset * int(
            settings["parent_seed_stride"]
        )
        bank = build_structured_internal_direction_bank(
            atomic_numbers,
            positions,
            seed=parent_seed,
            heldout_fraction=float(settings["partial_heldout_fraction"]),
            structured_per_kind=int(settings["structured_per_kind"]),
            bond_scale=float(settings["bond_scale"]),
        )
        pbe_hvp = np.einsum("ij,dj->di", pbe_hessian, bank.directions)
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
        local_kind_counts = Counter(bank.kinds.tolist())
        local_role_counts = Counter(bank.partial_roles.tolist())
        kind_counts.update(local_kind_counts)
        role_counts.update(local_role_counts)
        maximum_orthonormality = max(
            maximum_orthonormality, bank.orthonormality_max_abs
        )
        maximum_external_overlap = max(
            maximum_external_overlap, bank.external_overlap_max_abs
        )
        entries.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(atomic_numbers.size),
                "coordinate_count": int(positions.size),
                "external_rank": bank.external_rank,
                "internal_dimension": bank.internal_dimension,
                "direction_path": artifact.resolve().as_posix(),
                "direction_sha256": _sha256(artifact),
                "kind_counts": dict(sorted(local_kind_counts.items())),
                "partial_role_counts": dict(sorted(local_role_counts.items())),
                "orthonormality_max_abs": bank.orthonormality_max_abs,
                "external_overlap_max_abs": bank.external_overlap_max_abs,
                "projector_idempotence_max_abs": (
                    bank.projector_idempotence_max_abs
                ),
                "label_path": label_path.resolve().as_posix(),
                "label_sha256": parent["label_sha256"],
                "pbe_hessian_path": hessian_path.resolve().as_posix(),
                "pbe_hessian_sha256": parent["pbe_hessian_sha256"],
            }
        )

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_set": parent_set,
        "source_manifest": source_manifest_path.resolve().as_posix(),
        "source_manifest_sha256": source_manifest_hash,
        "source_checkpoint": protocol["source"]["checkpoint"],
        "source_checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
        "definition": (
            "Complete geometry-only orthonormal internal basis. PBE analytic "
            "Hessian supplies targets only after the basis and partial roles are frozen."
        ),
        "scalar_owner": protocol["definitions"]["scalar_owner"],
        "complete_total_relaxed_hvp": protocol["definitions"][
            "complete_total_relaxed_hvp"
        ],
        "independent_force_or_hessian_head_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_count": len(entries),
        "total_direction_count": sum(
            int(row["internal_dimension"]) for row in entries
        ),
        "kind_counts": dict(sorted(kind_counts.items())),
        "partial_role_counts": dict(sorted(role_counts.items())),
        "maximum_orthonormality_max_abs": maximum_orthonormality,
        "maximum_external_overlap_max_abs": maximum_external_overlap,
        "parents": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        key: value for key, value in payload.items() if key != "parents"
    }
    summary["manifest"] = manifest_path.resolve().as_posix()
    summary["manifest_sha256"] = _sha256(manifest_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-set", choices=("stable5", "train20"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    print(
        json.dumps(
            prepare(
                arguments.protocol,
                arguments.parent_set,
                arguments.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )
