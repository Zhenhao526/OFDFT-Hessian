#!/usr/bin/env python3
"""Freeze complete internal bases/HVP targets for all 120 pre-split candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from mldft.ofdft.internal_directions import build_structured_internal_direction_bank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hessian-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--parent-seed-stride", type=int, default=100003)
    parser.add_argument("--structured-per-kind", type=int, default=2)
    parser.add_argument("--bond-scale", type=float, default=1.25)
    args = parser.parse_args()

    candidate_payload = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidates = candidate_payload["molecules"]
    if len(candidates) != 120:
        raise RuntimeError("candidate manifest must contain exactly 120 molecules")
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, object]] = []
    total_kind_counts: Counter[str] = Counter()
    maximum_orthonormality = 0.0
    maximum_external_overlap = 0.0
    maximum_projector_idempotence = 0.0

    for offset, candidate in enumerate(candidates):
        molecule_id = str(candidate["molecule_id"])
        label_matches = sorted((args.dataset_dir / "labels").glob(f"{molecule_id}.0000000.zarr.zip"))
        if len(label_matches) != 1:
            raise RuntimeError(f"expected exactly one label for {molecule_id}")
        label_path = label_matches[0]
        hessian_path = args.hessian_dir / f"pbe_hessian_{molecule_id}_0000000.npz"
        if not hessian_path.is_file():
            raise RuntimeError(f"missing Hessian for {molecule_id}")
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        hessian = np.asarray(np.load(hessian_path)["pbe_hessian"], dtype=np.float64)
        if hessian.shape != (positions.size, positions.size):
            raise RuntimeError(f"Hessian shape mismatch for {molecule_id}")
        hessian = 0.5 * (hessian + hessian.T)
        bank = build_structured_internal_direction_bank(
            atomic_numbers,
            positions,
            seed=args.seed + offset * args.parent_seed_stride,
            # The shared builder requires a non-empty direction-role split.
            # v11 performs its isolation at the molecule level, so these
            # builder roles are deliberately discarded below and every basis
            # direction is retained as an inventory/reference target.
            heldout_fraction=0.5,
            structured_per_kind=args.structured_per_kind,
            bond_scale=args.bond_scale,
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
            partial_roles=np.asarray(["all"] * bank.internal_dimension),
            pbe_hvp=pbe_hvp,
            external_basis=bank.external_basis,
            projector=bank.projector,
            external_rank=np.asarray(bank.external_rank, dtype=np.int64),
            internal_dimension=np.asarray(bank.internal_dimension, dtype=np.int64),
        )
        local_kind_counts = Counter(bank.kinds.tolist())
        total_kind_counts.update(local_kind_counts)
        maximum_orthonormality = max(maximum_orthonormality, bank.orthonormality_max_abs)
        maximum_external_overlap = max(maximum_external_overlap, bank.external_overlap_max_abs)
        maximum_projector_idempotence = max(
            maximum_projector_idempotence, bank.projector_idempotence_max_abs
        )
        entries.append({
            "molecule_id": molecule_id,
            "natoms": int(atomic_numbers.size),
            "coordinate_count": int(positions.size),
            "external_rank": int(bank.external_rank),
            "internal_dimension": int(bank.internal_dimension),
            "direction_path": str(artifact.resolve()),
            "direction_sha256": sha256(artifact),
            "kind_counts": dict(sorted(local_kind_counts.items())),
            "orthonormality_max_abs": float(bank.orthonormality_max_abs),
            "external_overlap_max_abs": float(bank.external_overlap_max_abs),
            "projector_idempotence_max_abs": float(bank.projector_idempotence_max_abs),
            "label_path": str(label_path.resolve()),
            "label_sha256": sha256(label_path),
            "pbe_hessian_path": str(hessian_path.resolve()),
            "pbe_hessian_sha256": sha256(hessian_path),
        })

    payload = {
        "artifact_id": "qm9_v11_candidate120_complete_internal_bases_v1",
        "purpose": "pre_split_geometry_basis_and_reference_hvp_inventory_not_model_selection",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "seed": args.seed,
        "parent_seed_stride": args.parent_seed_stride,
        "structured_per_kind": args.structured_per_kind,
        "bond_scale": args.bond_scale,
        "partial_heldout_fraction": 0.0,
        "builder_role_fraction_ignored": 0.5,
        "direction_roles": "all_basis_directions_retained; molecule_level_split_only",
        "validation_accessed_as_model_metric": False,
        "test_accessed_as_model_metric": False,
        "molecule_count": len(entries),
        "total_direction_count": sum(int(row["internal_dimension"]) for row in entries),
        "kind_counts": dict(sorted(total_kind_counts.items())),
        "maximum_orthonormality_max_abs": maximum_orthonormality,
        "maximum_external_overlap_max_abs": maximum_external_overlap,
        "maximum_projector_idempotence_max_abs": maximum_projector_idempotence,
        "parents": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, payload)
    atomic_json(args.output_dir / "summary.json", {
        key: value for key, value in payload.items() if key != "parents"
    } | {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
    })
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "molecule_count": len(entries),
        "total_direction_count": payload["total_direction_count"],
        "maximum_orthonormality_max_abs": maximum_orthonormality,
        "maximum_external_overlap_max_abs": maximum_external_overlap,
        "maximum_projector_idempotence_max_abs": maximum_projector_idempotence,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
