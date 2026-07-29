#!/usr/bin/env python3
"""Build the blind-safe sidecar split for the QM9 random1000 residual experiment.

The historical random1000 identity is deterministic: QM9 parents 1..1000 are
permuted with NumPy seed 8, then split into 800/100/100 parents.  The rebuilt
train800 payload remains untouched.  Validation100 and Test100 labels live in
separate datasets, and split entries may therefore reference three dataset
directories.

By default this script opens train and validation labels only.  Test labels are
not inspected or included until ``--unlock-test`` is passed after model and
selection rules have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


SEED = 8
PARENT_COUNT = 1000
SAMPLES_PER_PARENT = 4
PARTITION_SIZES = {"train": 800, "val": 100, "test": 100}
EXPECTED_PARENT_HASHES = {
    "train": "d3ef814cf83eba4cff37f6a076c618a6016d8df4d2931286732c999cca54513d",
    "val": "4735dd45a3dcfe1f7c72d2f0db9c979d974e9fb61333d44cbb10bf7502c445f0",
    "test": "a4684bf139dcfc63d5012e3da88825ce2130b181909cf30517a05d45b717e147",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_hash(parent_ids: Iterable[int]) -> str:
    payload = "".join(f"{parent_id}\n" for parent_id in sorted(parent_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def historical_parent_split() -> dict[str, list[int]]:
    """Return the exact parent split used by the historical random1000 run."""
    parent_ids = np.arange(1, PARENT_COUNT + 1, dtype=np.int64)
    # RandomState intentionally matches the legacy ``np.random.seed`` code.
    order = np.random.RandomState(SEED).permutation(PARENT_COUNT)
    split = {
        "train": parent_ids[order[:800]].tolist(),
        "val": parent_ids[order[800:900]].tolist(),
        "test": parent_ids[order[900:]].tolist(),
    }
    for name, expected_size in PARTITION_SIZES.items():
        if len(split[name]) != expected_size:
            raise AssertionError(f"{name} parent count drifted")
        actual_hash = _parent_hash(split[name])
        if actual_hash != EXPECTED_PARENT_HASHES[name]:
            raise AssertionError(
                f"{name} identity drifted: {actual_hash} != {EXPECTED_PARENT_HASHES[name]}"
            )
    if set(split["train"]) & set(split["val"]):
        raise AssertionError("train/validation parent overlap")
    if set(split["train"]) & set(split["test"]):
        raise AssertionError("train/test parent overlap")
    if set(split["val"]) & set(split["test"]):
        raise AssertionError("validation/test parent overlap")
    return split


def _parent_from_filename(filename: str) -> int:
    return int(filename.removesuffix(".zarr.zip").split(".", 1)[0])


def _sample_from_filename(filename: str) -> int:
    stem = filename.removesuffix(".zarr.zip")
    parts = stem.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected <parent>.<sample>.zarr.zip, got {filename}")
    return int(parts[1])


def _expected_filenames(parent_ids: Iterable[int]) -> set[str]:
    return {
        f"{parent_id:07d}.{sample_id:07d}.zarr.zip"
        for parent_id in parent_ids
        for sample_id in range(SAMPLES_PER_PARENT)
    }


def _validate_train_split(
    train_root: Path,
    train_dataset_name: str,
    expected_parents: set[int],
) -> tuple[list[tuple[str, str, int]], int, str]:
    split_path = train_root / "split.pkl"
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    if split.get("train_only") is not True:
        raise ValueError("Rebuilt train800 split is not marked train_only")
    if split.get("val") or split.get("test"):
        raise ValueError("Rebuilt train800 split unexpectedly exposes held-out data")
    entries = list(split["train"])
    if any(dataset != train_dataset_name for dataset, _, _ in entries):
        raise ValueError("Unexpected dataset name in rebuilt train800 split")
    filenames = [filename for _, filename, _ in entries]
    if set(filenames) != _expected_filenames(expected_parents):
        raise ValueError("Rebuilt train800 geometry identity differs from historical split")
    actual_parents = {_parent_from_filename(filename) for filename in filenames}
    if actual_parents != expected_parents:
        raise ValueError("Rebuilt train800 parent identity differs from historical split")
    usable_samples = sum(max(0, int(n_scf_steps) - 1) for _, _, n_scf_steps in entries)
    recorded_size = int(split["sizes"]["train"])
    if usable_samples != recorded_size:
        raise ValueError(
            f"Rebuilt train800 usable size mismatch: {usable_samples} != {recorded_size}"
        )
    return entries, usable_samples, _sha256(split_path)


def _held_entries(
    held_root: Path,
    held_dataset_name: str,
    parent_ids: set[int],
) -> tuple[list[tuple[str, str, int]], int]:
    label_dir = held_root / "labels_local_frames_global_symmetric_natrep"
    expected = _expected_filenames(parent_ids)
    actual = {path.name for path in label_dir.glob("*.zarr.zip") if path.name in expected}
    missing = sorted(expected - actual)
    if missing:
        raise FileNotFoundError(
            f"Held-out transformed labels are incomplete: missing {len(missing)}; "
            f"first={missing[:3]}"
        )

    entries: list[tuple[str, str, int]] = []
    usable_samples = 0
    for filename in sorted(expected):
        path = label_dir / filename
        root = zarr.open(path, mode="r")
        n_scf_steps = int(root["of_labels/n_scf_steps"][()])
        if n_scf_steps < 2:
            raise ValueError(f"{path} has only {n_scf_steps} SCF steps")
        source_parent = _parent_from_filename(filename)
        source_sample = _sample_from_filename(filename)
        if "metadata/reference/source_molecule_id" in root:
            if int(root["metadata/reference/source_molecule_id"][()]) != source_parent:
                raise ValueError(f"Parent metadata mismatch in {path}")
            if int(root["metadata/reference/sample_id"][()]) != source_sample:
                raise ValueError(f"Sample metadata mismatch in {path}")
        entries.append((held_dataset_name, filename, n_scf_steps))
        usable_samples += n_scf_steps - 1
    return entries, usable_samples


def build(args: argparse.Namespace) -> dict:
    parent_split = historical_parent_split()
    train_entries, train_size, train_split_hash = _validate_train_split(
        args.train_root,
        args.train_dataset_name,
        set(parent_split["train"]),
    )
    val_entries, val_size = _held_entries(
        args.val_root,
        args.val_dataset_name,
        set(parent_split["val"]),
    )

    test_entries: list[tuple[str, str, int]] = []
    test_size = 0
    if args.unlock_test:
        if args.test_root is None:
            raise ValueError("--unlock-test requires --test-root")
        test_entries, test_size = _held_entries(
            args.test_root,
            args.test_dataset_name,
            set(parent_split["test"]),
        )

    output_split = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
        "sizes": {"train": train_size, "val": val_size, "test": test_size},
        "train_only": False,
        "validation_access_allowed": True,
        "test_access_allowed": bool(args.unlock_test),
        "protocol": "qm9_random1000_residual_graphformer_v1",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_name = "split_full.pkl" if args.unlock_test else "split_train_val.pkl"
    split_path = args.output_dir / split_name
    with split_path.open("wb") as handle:
        pickle.dump(output_split, handle, protocol=4)

    parent_manifest = {
        "schema_version": 1,
        "protocol": "qm9_random1000_residual_graphformer_v1",
        "historical_definition": (
            "QM9 parent IDs 1..1000; NumPy legacy permutation seed 8; "
            "800/100/100 parent split"
        ),
        "partition_sizes": PARTITION_SIZES,
        "parent_hashes": EXPECTED_PARENT_HASHES,
        "parents": {
            "train": sorted(parent_split["train"]),
            "val": sorted(parent_split["val"]),
            # Preserve test identity without opening test labels before unlock.
            "test": sorted(parent_split["test"]),
        },
        "train_source_split": str(args.train_root / "split.pkl"),
        "train_source_split_sha256": train_split_hash,
        "validation_dataset_root": str(args.val_root),
        "test_dataset_root": None if args.test_root is None else str(args.test_root),
        "test_labels_accessed": bool(args.unlock_test),
        "output_split": str(split_path),
        "output_split_sha256": _sha256(split_path),
        "usable_sample_sizes": output_split["sizes"],
    }
    manifest_path = args.output_dir / (
        "parent_manifest_full.json"
        if args.unlock_test
        else "parent_manifest_train_val.json"
    )
    manifest_path.write_text(json.dumps(parent_manifest, indent=2, sort_keys=True) + "\n")
    return {
        "split": str(split_path),
        "split_sha256": _sha256(split_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sizes": output_split["sizes"],
        "test_labels_accessed": bool(args.unlock_test),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-dataset-name",
        default="QM9PBEForceRandom1000Train800RebuildV1",
    )
    parser.add_argument(
        "--val-dataset-name",
        default="QM9PBEForceRandom1000Val100RebuildV1",
    )
    parser.add_argument(
        "--test-dataset-name",
        default="QM9PBEForceRandom1000Test100RebuildV1",
    )
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help="Inspect and include Test100 labels only after the protocol is frozen.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
