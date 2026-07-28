#!/usr/bin/env python3
"""Build a leakage-checked train-only paired-geometry augmentation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import zarr


def _parent_id(label_name: str) -> str:
    return label_name.split(".", maxsplit=1)[0].zfill(7)


def _parents(entries: list[list[Any]]) -> set[str]:
    return {_parent_id(str(entry[1])) for entry in entries}


def _safe_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def _n_scf_steps(path: Path) -> int:
    root = zarr.open(path, mode="r")
    return int(root["of_labels/n_scf_steps"][()])


def _validate_pair_metadata(path: Path, expected_parent: str) -> tuple[int, int]:
    root = zarr.open(path, mode="r")
    reference = root["metadata/reference"]
    source_parent = f"{int(reference['source_molecule_id'][()]):07d}"
    if source_parent != expected_parent:
        raise ValueError(f"Parent mismatch in {path}: {source_parent} != {expected_parent}")
    if not bool(reference["paired_perturbations"][()]):
        raise ValueError(f"Label is not marked as paired: {path}")
    sample_id = int(reference["sample_id"][()])
    pair_sign = int(reference["perturbation_pair_sign"][()])
    return sample_id, pair_sign


def _validate_split_label_paths(
    data_root: Path,
    split: dict[str, Any],
    label_subdir: str,
) -> dict[str, int]:
    source_counts: dict[str, int] = {}
    missing: list[Path] = []
    for split_name in ("train", "val", "test"):
        for dataset_name, label_name, _ in split[split_name]:
            source_counts[dataset_name] = source_counts.get(dataset_name, 0) + 1
            label_path = data_root / dataset_name / label_subdir / label_name
            if not label_path.is_file():
                missing.append(label_path)
    if missing:
        preview = [str(path) for path in missing[:10]]
        raise FileNotFoundError(f"Missing {len(missing)} split label paths: {preview}")
    return dict(sorted(source_counts.items()))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    base_root = args.base_root.resolve()
    paired_root = args.paired_root.resolve()
    output_root = args.output_root.resolve()
    base_split = pickle.loads((base_root / "split.pkl").read_bytes())

    train_parents = _parents(base_split["train"])
    val_parents = _parents(base_split["val"])
    test_parents = _parents(base_split["test"])
    if train_parents & val_parents or train_parents & test_parents or val_parents & test_parents:
        raise ValueError("Base split already has parent-molecule leakage")

    label_dir = paired_root / args.label_subdir
    pair_paths = sorted(label_dir.glob("*.zarr.zip"))
    if len(pair_paths) != args.expected_labels:
        raise ValueError(f"Expected {args.expected_labels} paired labels, found {len(pair_paths)}")

    grouped: dict[str, list[tuple[Path, int, int]]] = {}
    pair_entries: list[list[Any]] = []
    for path in pair_paths:
        parent = _parent_id(path.name)
        if parent not in train_parents:
            raise ValueError(f"Paired label parent is not in train split: {path.name}")
        if parent in val_parents or parent in test_parents:
            raise ValueError(f"Paired label leaks into validation/test: {path.name}")
        sample_id, sign = _validate_pair_metadata(path, parent)
        n_scf_steps = _n_scf_steps(path)
        grouped.setdefault(parent, []).append((path, sample_id, sign))
        pair_entries.append([paired_root.name, path.name, n_scf_steps])

    if set(grouped) != train_parents:
        missing = sorted(train_parents - set(grouped))
        extra = sorted(set(grouped) - train_parents)
        raise ValueError(f"Paired parent coverage mismatch: missing={missing[:10]} extra={extra[:10]}")
    for parent, items in grouped.items():
        sample_ids = sorted(item[1] for item in items)
        signs = sorted(item[2] for item in items)
        if sample_ids != [1, 2] or signs != [-1, 1]:
            raise ValueError(f"Invalid pair for {parent}: sample_ids={sample_ids}, signs={signs}")

    combined = {
        "sizes": dict(base_split.get("sizes", {})),
        "train": list(base_split["train"]) + pair_entries,
        "val": list(base_split["val"]),
        "test": list(base_split["test"]),
    }
    combined["sizes"]["train"] = int(base_split.get("sizes", {}).get("train", 0)) + sum(
        int(entry[2]) for entry in pair_entries
    )
    label_source_entries = _validate_split_label_paths(
        base_root.parent,
        combined,
        args.label_subdir,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split.pkl").write_bytes(pickle.dumps(combined))
    try:
        import yaml

        (output_root / "split.yaml").write_text(yaml.safe_dump(combined, sort_keys=False))
    except ImportError:
        pass
    _safe_relative_symlink(base_root / "dataset_info.yaml", output_root / "dataset_info.yaml")
    _safe_relative_symlink(base_root / "dataset_statistics", output_root / "dataset_statistics")

    split_hash = hashlib.sha256((output_root / "split.pkl").read_bytes()).hexdigest()
    manifest = {
        "definition": "Base random1000 split plus exact +/- perturbations for train parents only.",
        "base_root": str(base_root),
        "paired_root": str(paired_root),
        "output_root": str(output_root),
        "label_subdir": args.label_subdir,
        "base_entries": {key: len(base_split[key]) for key in ("train", "val", "test")},
        "combined_entries": {key: len(combined[key]) for key in ("train", "val", "test")},
        "paired_entries": len(pair_entries),
        "label_source_entries": label_source_entries,
        "paired_parents": len(grouped),
        "train_parents": len(train_parents),
        "val_parents": len(val_parents),
        "test_parents": len(test_parents),
        "parent_overlap": {
            "train_val": sorted(train_parents & val_parents),
            "train_test": sorted(train_parents & test_parents),
            "val_test": sorted(val_parents & test_parents),
        },
        "split_sha256": split_hash,
    }
    (output_root / "paired_augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--label-subdir", default="labels_local_frames_global_symmetric_natrep"
    )
    parser.add_argument("--expected-labels", type=int, default=1600)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
