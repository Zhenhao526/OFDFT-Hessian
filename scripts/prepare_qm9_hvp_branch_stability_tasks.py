#!/usr/bin/env python3
"""Freeze molecule/direction/initialization branch tasks for the HVP stability audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


BRANCHES = (
    ("sad_continuation", "configured", "base_continuation"),
    ("label_continuation", "label_reference", "base_continuation"),
    ("sad_independent", "configured", "configured"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--molecules", type=Path)
    parser.add_argument("--subset", choices=["validation_diagnostic", "file"], default="validation_diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("The frozen protocol must forbid Test100 access")
    if args.subset == "validation_diagnostic":
        molecules = [str(value).zfill(7) for value in protocol["data"]["validation_diagnostic_ids"]]
    else:
        if args.molecules is None:
            raise ValueError("--molecules is required for subset=file")
        molecules = [line.strip().zfill(7) for line in args.molecules.read_text().splitlines() if line.strip()]
    direction_count = int(protocol["directions"]["per_parent"])
    rows = []
    for molecule_id in molecules:
        for direction_index in range(direction_count):
            for branch, base_initialization, displaced_initialization in BRANCHES:
                rows.append(
                    {
                        "task_index": len(rows),
                        "molecule_id": molecule_id,
                        "direction_index": direction_index,
                        "branch": branch,
                        "base_initialization": base_initialization,
                        "displaced_initialization": displaced_initialization,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "protocol": str(args.protocol.resolve()),
        "subset": args.subset,
        "molecule_count": len(molecules),
        "directions_per_parent": direction_count,
        "branches": [item[0] for item in BRANCHES],
        "task_count": len(rows),
        "test100_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
