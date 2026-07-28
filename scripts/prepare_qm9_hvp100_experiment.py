#!/usr/bin/env python3
"""Freeze a stratified 100-train-parent HVP experiment split and provenance manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent_id(label_name: str) -> str:
    return label_name.split(".", maxsplit=1)[0].zfill(7)


def _composition_class(composition: str) -> str:
    counts = {int(item.split(":")[0]): int(item.split(":")[1]) for item in composition.split(";")}
    if counts.get(9, 0):
        return "F-containing"
    has_n = counts.get(7, 0) > 0
    has_o = counts.get(8, 0) > 0
    if has_n and has_o:
        return "N+O"
    if has_n:
        return "N-only"
    if has_o:
        return "O-only"
    return "CH-only"


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size) + 0.5) / values.size
    return ranks


def _allocate(groups: dict[tuple[Any, ...], list[dict[str, Any]]], count: int) -> dict[tuple[Any, ...], int]:
    keys = sorted(groups)
    if len(keys) > count:
        raise ValueError(f"{len(keys)} non-empty strata exceed requested count {count}")
    quotas = {key: 1 for key in keys}
    remaining = count - len(keys)
    if remaining <= 0:
        return quotas
    capacities = {key: len(groups[key]) - 1 for key in keys}
    total_capacity = sum(capacities.values())
    ideal = {
        key: remaining * capacities[key] / max(total_capacity, 1) for key in keys
    }
    for key in keys:
        add = min(capacities[key], int(np.floor(ideal[key])))
        quotas[key] += add
        remaining -= add
    while remaining:
        candidates = [key for key in keys if quotas[key] < len(groups[key])]
        if not candidates:
            raise RuntimeError("No remaining stratum capacity")
        key = max(
            candidates,
            key=lambda item: (ideal[item] - (quotas[item] - 1), len(groups[item]), item),
        )
        quotas[key] += 1
        remaining -= 1
    return quotas


def _select_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["difficulty_score"], row["parent_id"]))
    if count >= len(ordered):
        return ordered
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    return [ordered[int(index)] for index in indices]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    split = pickle.loads(args.source_split.read_bytes())
    with args.difficulty_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 800:
        raise ValueError(f"Expected 800 train-parent difficulty rows, found {len(rows)}")
    for row in rows:
        row["natoms"] = int(row["natoms"])
        row["force_component_mae"] = float(row["force_component_mae"])
        row["paired_hvp_component_mae"] = float(row["paired_hvp_component_mae"])
        row["composition_class"] = _composition_class(row["composition"])
    force_rank = _percentile_ranks(np.asarray([row["force_component_mae"] for row in rows]))
    hvp_rank = _percentile_ranks(np.asarray([row["paired_hvp_component_mae"] for row in rows]))
    natom_rank = _percentile_ranks(np.asarray([row["natoms"] for row in rows], dtype=float))
    for index, row in enumerate(rows):
        row["force_difficulty_percentile"] = float(force_rank[index])
        row["hvp_difficulty_percentile"] = float(hvp_rank[index])
        row["difficulty_score"] = float((force_rank[index] + hvp_rank[index]) / 2.0)
        row["difficulty_bin"] = min(3, int(4 * row["difficulty_score"]))
        row["natoms_bin"] = min(3, int(4 * natom_rank[index]))

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["natoms_bin"], row["composition_class"], row["difficulty_bin"])
        groups[key].append(row)
    quotas = _allocate(groups, args.train_parents)
    selected = [row for key in sorted(groups) for row in _select_evenly(groups[key], quotas[key])]
    selected_ids = {row["parent_id"] for row in selected}
    if len(selected_ids) != args.train_parents:
        raise RuntimeError(f"Selection produced {len(selected_ids)} unique parents")

    filtered_train = [
        entry for entry in split["train"] if _parent_id(entry[1]) in selected_ids
    ]
    expected_entries = args.train_parents * 6
    if len(filtered_train) != expected_entries:
        raise ValueError(f"Expected {expected_entries} train geometries, found {len(filtered_train)}")
    output_split = {
        "sizes": {
            "train": int(sum(int(entry[2]) for entry in filtered_train)),
            "val": int(sum(int(entry[2]) for entry in split["val"])),
            "test": int(sum(int(entry[2]) for entry in split["test"])),
        },
        "train": filtered_train,
        "val": split["val"],
        "test": split["test"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_pickle = args.output_dir / "split.pkl"
    split_yaml = args.output_dir / "split.yaml"
    split_pickle.write_bytes(pickle.dumps(output_split))
    split_yaml.write_text(yaml.safe_dump(output_split, sort_keys=False))
    selection_json = args.output_dir / "train_parent_selection.json"
    selection_json.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    ids_file = args.output_dir / "train_parent_ids.txt"
    ids_file.write_text("\n".join(sorted(selected_ids)) + "\n")

    parent_sets = {
        name: {_parent_id(entry[1]) for entry in output_split[name]}
        for name in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(parent_sets["train"] & parent_sets["val"]),
        "train_test": sorted(parent_sets["train"] & parent_sets["test"]),
        "val_test": sorted(parent_sets["val"] & parent_sets["test"]),
    }
    manifest = {
        "definition": (
            "100 parents selected only from the original random1000 train800, jointly stratified "
            "by atom-count quartile, coarse element composition and baseline force/paired-HVP "
            "difficulty quartile. All four original and both paired geometries follow the parent."
        ),
        "seed": args.seed,
        "source_split": args.source_split.as_posix(),
        "source_split_sha256": _sha256(args.source_split),
        "difficulty_csv": args.difficulty_csv.as_posix(),
        "difficulty_csv_sha256": _sha256(args.difficulty_csv),
        "train_parents": len(parent_sets["train"]),
        "train_geometries": len(filtered_train),
        "validation_parents": len(parent_sets["val"]),
        "test_parents_frozen": len(parent_sets["test"]),
        "parent_overlap": overlaps,
        "selection_sha256": _sha256(selection_json),
        "train_parent_ids_sha256": _sha256(ids_file),
        "split_pickle_sha256": _sha256(split_pickle),
        "split_yaml_sha256": _sha256(split_yaml),
        "strata": [
            {
                "natoms_bin": key[0],
                "composition_class": key[1],
                "difficulty_bin": key[2],
                "available": len(groups[key]),
                "selected": quotas[key],
            }
            for key in sorted(groups)
        ],
        "test_accessed_for_selection": False,
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Parent leakage detected: {overlaps}")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--difficulty-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-parents", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
