#!/usr/bin/env python3
"""Freeze a deterministic 10x8 size-balanced Train80 DDP schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train = list(payload["splits"]["train"])
    if len(train) != 80:
        raise RuntimeError(f"expected Train80, found {len(train)}")
    if len({str(row["molecule_id"]) for row in train}) != 80:
        raise RuntimeError("Train80 molecule IDs are not unique")

    # Adjacent size groups make each simultaneous 8-rank batch homogeneous.
    # Alternating rank order prevents a rank from always receiving the largest
    # member of its group while preserving the exact 10x8 epoch definition.
    ranked = sorted(
        train,
        key=lambda row: (
            -int(row["atom_count"]),
            -int(row["heavy_atom_count"]),
            -float(row["max_pair_distance_angstrom"]),
            str(row["molecule_id"]),
        ),
    )
    batches: list[list[str]] = []
    for update in range(10):
        group = ranked[8 * update : 8 * (update + 1)]
        if update % 2:
            group = list(reversed(group))
        batches.append([str(row["molecule_id"]) for row in group])
    flat = [molecule_id for batch in batches for molecule_id in batch]
    if len(flat) != 80 or len(set(flat)) != 80:
        raise RuntimeError("schedule does not cover Train80 exactly once")

    output = {
        "artifact_id": "qm9_v11_train80_size_balanced_10x8_v1",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256(args.split_manifest),
        "world_size": 8,
        "global_updates_per_epoch": 10,
        "molecules_per_rank_per_update": 1,
        "batches": batches,
        "rank_molecules": {
            str(rank): [batches[update][rank] for update in range(10)]
            for rank in range(8)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "batch_atom_counts": [
            [int(next(row["atom_count"] for row in train if str(row["molecule_id"]) == mid)) for mid in batch]
            for batch in batches
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
