#!/usr/bin/env python3
"""Freeze stable-only strict complete-total HVP tasks from validation-only results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--multiseed-tasks", type=Path, required=True)
    parser.add_argument("--multiseed-analysis", type=Path, required=True)
    parser.add_argument("--stable-sidecar-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())
    analysis = json.loads(args.multiseed_analysis.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Validation protocol must prohibit Test100")
    if analysis.get("test100_accessed") is not False:
        raise ValueError("Multiseed analysis does not certify frozen Test100")
    run_rows = _read_tsv(args.multiseed_tasks)
    eligible_keys = {
        (
            group["variant"],
            float(group["curvature_weight"]),
            float(group["reference_floor"]),
        )
        for group in analysis["groups"]
        if group["variant"] != "A" and bool(group["multiseed_tier1_gate"])
    }
    selected_runs = []
    if eligible_keys:
        selected_runs = [row for row in run_rows if row["variant"] == "A"]
        selected_runs.extend(
            row
            for row in run_rows
            if (
                row["variant"],
                float(row["curvature_weight"]),
                float(row["reference_floor"]),
            )
            in eligible_keys
        )

    stable_by_direction: dict[int, list[str]] = {}
    stable_parent_ids = []
    for path in sorted(args.stable_sidecar_dir.glob("*.0000000.npz")):
        payload = np.load(path)
        if not bool(np.asarray(payload["parent_stability_eligible"]).item()):
            continue
        mask = np.asarray(payload["stability_mask"], dtype=np.bool_).reshape(-1)
        molecule_id = path.name.split(".", 1)[0].zfill(7)
        stable_parent_ids.append(molecule_id)
        reference = args.reference_dir / f"pbe_hessian_{molecule_id}_0000000.npz"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        for direction_index in np.flatnonzero(mask):
            stable_by_direction.setdefault(int(direction_index), []).append(molecule_id)

    minimum_stable_parents = int(
        protocol["strict_complete_total_hvp"]["minimum_stable_parent_count"]
    )
    stable_parent_gate = len(set(stable_parent_ids)) >= minimum_stable_parents
    if not stable_parent_gate:
        stable_by_direction = {}

    rows = []
    for run in selected_runs:
        for direction_index, molecule_ids in sorted(stable_by_direction.items()):
            for molecule_id in sorted(molecule_ids):
                rows.append(
                    {
                        "task_index": len(rows),
                        "run_name": run["run_name"],
                        "variant": run["variant"],
                        "seed": run["seed"],
                        "curvature_weight": run["curvature_weight"],
                        "reference_floor": run["reference_floor"],
                        "direction_index": direction_index,
                        "molecule_count": 1,
                        "molecules": molecule_id,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_index",
        "run_name",
        "variant",
        "seed",
        "curvature_weight",
        "reference_floor",
        "direction_index",
        "molecule_count",
        "molecules",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "definition": __doc__,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "multiseed_tasks_sha256": _sha256(args.multiseed_tasks),
        "multiseed_analysis_sha256": _sha256(args.multiseed_analysis),
        "eligible_candidate_configurations": sorted(list(eligible_keys)),
        "selected_run_count": len(selected_runs),
        "task_count": len(rows),
        "task_unit": "one run x one direction x one molecule",
        "stable_parent_count": len(set(stable_parent_ids)),
        "minimum_stable_parent_count": minimum_stable_parents,
        "stable_parent_count_gate": stable_parent_gate,
        "stable_parent_ids": sorted(set(stable_parent_ids)),
        "stable_molecule_directions": sum(len(values) for values in stable_by_direction.values()),
        "test100_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
