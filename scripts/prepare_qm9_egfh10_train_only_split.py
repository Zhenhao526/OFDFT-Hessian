#!/usr/bin/env python3
"""Register a ten-parent, train-only EGFH paired-geometry split."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import zarr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, object]:
    root = args.dataset_root.resolve()
    raw_labels = root / "labels"
    transformed = root / args.label_subdir
    raw_paths = sorted(raw_labels.glob("*.zarr.zip"))
    transformed_paths = sorted(transformed.glob("*.zarr.zip"))
    samples_per_parent = 1 + 2 * args.pairs_per_parent
    expected_labels = args.parent_count * samples_per_parent
    if len(raw_paths) != expected_labels or len(transformed_paths) != expected_labels:
        raise ValueError(
            f"EGFH requires exactly {expected_labels} raw and transformed labels: "
            f"raw={len(raw_paths)} transformed={len(transformed_paths)}"
        )
    if {path.name for path in raw_paths} != {
        path.name for path in transformed_paths
    }:
        raise ValueError("raw/transformed EGFH10 label names differ")

    parent_samples: dict[str, list[tuple[int, int, bool]]] = {}
    entries: list[list[object]] = []
    for path in raw_paths:
        payload = zarr.open(path, mode="r")
        reference = payload["metadata/reference"]
        parent = f"{int(reference['source_molecule_id'][()]):07d}"
        sample_id = int(reference["sample_id"][()])
        paired = bool(reference["paired_perturbations"][()])
        sign = int(reference["perturbation_pair_sign"][()])
        parent_samples.setdefault(parent, []).append((sample_id, sign, paired))
        entries.append(
            [
                args.dataset_name,
                path.name,
                int(payload["of_labels/n_scf_steps"][()]),
            ]
        )

    if len(parent_samples) != args.parent_count:
        raise ValueError(
            f"expected {args.parent_count} parents, found {len(parent_samples)}"
        )
    expected_samples = [(0, 0, True)]
    for pair_id in range(1, args.pairs_per_parent + 1):
        expected_samples.extend(
            [
                (2 * pair_id - 1, 1, True),
                (2 * pair_id, -1, True),
            ]
        )
    for parent, samples in parent_samples.items():
        if sorted(samples) != expected_samples:
            raise ValueError(f"invalid center/pair metadata for {parent}: {samples}")

    split = {
        "sizes": {
            "train": sum(int(entry[2]) for entry in entries),
            "val": 0,
            "test": 0,
        },
        "train": entries,
        "val": [],
        "test": [],
    }
    split_path = root / "split.pkl"
    split_path.write_bytes(pickle.dumps(split))
    manifest = {
        "definition": (
            f"{args.parent_count} train-only QM9 parents with center and "
            f"{args.pairs_per_parent} deterministic symmetric displacement "
            "pairs per parent for EGFH training."
        ),
        "dataset_name": args.dataset_name,
        "parents": sorted(parent_samples),
        "parent_count": len(parent_samples),
        "label_count": len(entries),
        "pairs_per_parent": args.pairs_per_parent,
        "samples_per_parent": samples_per_parent,
        "validation_accessed": False,
        "test100_accessed": False,
        "split_sha256": _sha256(split_path),
    }
    manifest_path = root / "egfh10_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--parent-count", type=int, default=10)
    parser.add_argument("--pairs-per-parent", type=int, default=1)
    parser.add_argument(
        "--label-subdir",
        default="labels_local_frames_global_symmetric_natrep",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
