#!/usr/bin/env python3
"""Validate QM9 PBE force smoke-label zarr files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def parse_label_name(path: Path) -> tuple[int, int | None]:
    stem = path.name.removesuffix(".zarr.zip")
    parts = stem.split(".")
    molecule_id = int(parts[0])
    sample_id = int(parts[1]) if len(parts) > 1 else None
    return molecule_id, sample_id


def walk_arrays(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Array]]:
    arrays = []
    for key, value in sorted(group.items()):
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, zarr.Array):
            arrays.append((path, value))
        else:
            arrays.extend(walk_arrays(value, path))
    return arrays


def numeric_array_is_finite(array: zarr.Array) -> bool:
    if not np.issubdtype(array.dtype, np.number):
        return True
    data = np.asarray(array)
    return bool(np.isfinite(data).all())


def inspect_label(path: Path) -> dict[str, Any]:
    molecule_id, sample_id = parse_label_name(path)
    root = zarr.open(path, mode="r")
    failures = []

    if "geometry/atomic_numbers" not in root:
        natoms = None
        failures.append("missing geometry/atomic_numbers")
    else:
        natoms = int(root["geometry/atomic_numbers"].shape[0])

    force_shape = None
    force_norm_max = None
    force_path = "metadata/pbe_derivatives/forces"
    if force_path not in root:
        failures.append(f"missing {force_path}")
    else:
        forces = np.asarray(root[force_path])
        force_shape = tuple(int(dim) for dim in forces.shape)
        if natoms is not None and force_shape != (natoms, 3):
            failures.append(f"bad force shape {force_shape}, expected {(natoms, 3)}")
        if not np.isfinite(forces).all():
            failures.append("forces contain NaN/Inf")
        if forces.size:
            force_norm_max = float(np.linalg.norm(forces, axis=1).max())

    for key, array in walk_arrays(root):
        if not numeric_array_is_finite(array):
            failures.append(f"non-finite numeric array: {key}")

    return {
        "path": path.as_posix(),
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": natoms,
        "force_shape": force_shape,
        "force_norm_max": force_norm_max,
        "failures": failures,
    }


def expand_labels(paths: list[Path]) -> list[Path]:
    labels = []
    for path in paths:
        if path.is_dir():
            labels.extend(path.glob("*.zarr.zip"))
        else:
            labels.append(path)
    return sorted(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", type=Path, help="Label files or label directories.")
    parser.add_argument("--expected-molecules", type=int, default=100)
    parser.add_argument("--expected-samples", type=int, default=2)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = expand_labels(args.labels)
    if not labels:
        raise SystemExit("No .zarr.zip labels found.")

    records = [inspect_label(path) for path in labels]
    samples_by_molecule: dict[int, set[int | None]] = defaultdict(set)
    failures = []
    for record in records:
        samples_by_molecule[int(record["molecule_id"])].add(record["sample_id"])
        for failure in record["failures"]:
            failures.append({"path": record["path"], "failure": failure})

    expected_files = args.expected_molecules * args.expected_samples
    if len(records) < expected_files:
        failures.append(
            {
                "path": "",
                "failure": f"only {len(records)} labels found, expected at least {expected_files}",
            }
        )
    if len(samples_by_molecule) < args.expected_molecules:
        failures.append(
            {
                "path": "",
                "failure": (
                    f"only {len(samples_by_molecule)} molecule ids found, "
                    f"expected at least {args.expected_molecules}"
                ),
            }
        )
    incomplete = {
        molecule_id: sorted(-1 if sample is None else int(sample) for sample in sample_ids)
        for molecule_id, sample_ids in sorted(samples_by_molecule.items())
        if len(sample_ids) < args.expected_samples
    }
    if incomplete:
        failures.append({"path": "", "failure": f"incomplete molecule samples: {incomplete}"})

    force_norms = [
        record["force_norm_max"]
        for record in records
        if record["force_norm_max"] is not None
    ]
    summary = {
        "checked_files": len(records),
        "molecules": len(samples_by_molecule),
        "expected_molecules": args.expected_molecules,
        "expected_samples": args.expected_samples,
        "force_labels": len(force_norms),
        "force_norm_max": max(force_norms) if force_norms else None,
        "failures": failures,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_text + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
