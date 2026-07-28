#!/usr/bin/env python3
"""Create a raw QM9 subset containing only parents from an existing grouped train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


def _molecule_id(split_row: Any) -> int:
    if not isinstance(split_row, (list, tuple)) or len(split_row) < 2:
        raise ValueError(f"Unsupported split row: {split_row!r}")
    return int(Path(str(split_row[1])).name.split(".")[0])


def _ids(rows: list[Any]) -> set[int]:
    return {_molecule_id(row) for row in rows}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    with args.split_file.open("rb") as handle:
        split = pickle.load(handle)
    train_ids = _ids(split["train"])
    val_ids = _ids(split["val"])
    test_ids = _ids(split["test"])
    overlaps = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Parent-molecule leakage in source split: {overlaps}")

    args.output_raw_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for molecule_id in sorted(train_ids):
        source = args.source_raw_dir / f"dsgdb9nsd_{molecule_id:06d}.xyz"
        target = args.output_raw_dir / source.name
        if not source.exists():
            missing.append(molecule_id)
            continue
        if target.exists() or target.is_symlink():
            continue
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} train-parent XYZ files: {missing[:20]}")

    linked_ids = {
        int(path.stem.split("_")[1])
        for path in args.output_raw_dir.glob("dsgdb9nsd_*.xyz")
    }
    extra = sorted(linked_ids - train_ids)
    absent = sorted(train_ids - linked_ids)
    if extra or absent:
        raise RuntimeError(f"Raw subset mismatch: extra={extra[:20]}, absent={absent[:20]}")

    digest = hashlib.sha256(
        "\n".join(f"{molecule_id:07d}" for molecule_id in sorted(train_ids)).encode()
    ).hexdigest()
    result = {
        "source_split": args.split_file.resolve().as_posix(),
        "source_raw_dir": args.source_raw_dir.resolve().as_posix(),
        "output_raw_dir": args.output_raw_dir.resolve().as_posix(),
        "copy": bool(args.copy),
        "train_parent_count": len(train_ids),
        "val_parent_count": len(val_ids),
        "test_parent_count": len(test_ids),
        "parent_overlap": overlaps,
        "train_parent_ids_sha256": digest,
        "train_parent_ids": sorted(train_ids),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "train_parent_ids"}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--source-raw-dir", type=Path, required=True)
    parser.add_argument("--output-raw-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
