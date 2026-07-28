#!/usr/bin/env python3
"""Expand frozen strict runs into independent run-by-validation-molecule tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-tsv", type=Path, required=True)
    parser.add_argument("--molecule-manifest", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()
    with args.runs_tsv.open() as handle:
        runs = list(csv.DictReader(handle, delimiter="\t"))
    molecules = json.loads(args.molecule_manifest.read_text())
    tasks = []
    for run in runs:
        for molecule in molecules:
            tasks.append({
                "task_index": len(tasks),
                **run,
                "molecule_id": molecule["molecule_id"],
                "natoms": molecule["natoms"],
            })
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(tasks)
    print(json.dumps({"runs": len(runs), "molecules": len(molecules),
                      "tasks": len(tasks), "output": str(args.output_tsv)}, indent=2))


if __name__ == "__main__":
    main()
