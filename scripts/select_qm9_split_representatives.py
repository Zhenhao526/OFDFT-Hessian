#!/usr/bin/env python3
"""Select deterministic atom-count-spanning parent molecules from a grouped split."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def _parent_id(label_name: str) -> str:
    return label_name.split(".", maxsplit=1)[0].zfill(7)


def select(args: argparse.Namespace) -> list[dict[str, Any]]:
    split = pickle.loads(args.split_file.read_bytes())
    parent_entries: dict[str, tuple[str, str]] = {}
    for dataset_name, label_name, _ in split[args.split]:
        parent_entries.setdefault(_parent_id(label_name), (dataset_name, label_name))

    rows = []
    for molecule_id, (dataset_name, label_name) in sorted(parent_entries.items()):
        path = args.data_dir / dataset_name / args.label_subdir / label_name
        root = zarr.open(path, mode="r")
        rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(np.asarray(root["geometry/atomic_numbers"]).size),
                "source_label": path.as_posix(),
            }
        )
    rows.sort(key=lambda row: (row["natoms"], row["molecule_id"]))
    if args.count > len(rows):
        raise ValueError(f"Requested {args.count} representatives from only {len(rows)} parents")

    indices = np.rint(np.linspace(0, len(rows) - 1, args.count)).astype(int)
    selected = []
    used = set()
    for target_index in indices:
        candidates = sorted(
            range(len(rows)), key=lambda idx: (abs(idx - target_index), idx)
        )
        index = next(idx for idx in candidates if idx not in used)
        used.add(index)
        selected.append({**rows[index], "sorted_parent_rank": index})
    selected.sort(key=lambda row: (row["natoms"], row["molecule_id"]))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(selected, indent=2) + "\n")
    if args.output_ids is not None:
        args.output_ids.write_text(",".join(row["molecule_id"] for row in selected) + "\n")
    print(json.dumps(selected, indent=2))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--label-subdir", default="labels_local_frames_global_symmetric_natrep"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ids", type=Path, default=None)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
