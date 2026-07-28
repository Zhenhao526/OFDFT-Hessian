#!/usr/bin/env python3
"""Freeze five validation-only full-Hessian parents after strict HVP promotion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-summary", type=Path, required=True)
    parser.add_argument("--strict-per-direction", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--parent-count", type=int, default=5)
    args = parser.parse_args()

    strict = json.loads(args.strict_summary.read_text())
    if strict.get("test100_accessed") is not False:
        raise ValueError("Strict summary does not certify frozen Test100")
    candidates = list(strict["frozen_candidates_for_full_hessian"])
    with args.strict_per_direction.open() as handle:
        directions = list(csv.DictReader(handle))

    tasks = []
    selected_molecules = []
    selected_runs = []
    if candidates:
        baseline_rows = [row for row in directions if row["variant"] == "A"]
        by_molecule: dict[str, list[dict]] = defaultdict(list)
        for row in baseline_rows:
            by_molecule[row["molecule_id"]].append(row)
        ranked = [
            {
                "molecule_id": molecule_id,
                "natoms": int(rows[0]["natoms"]),
                "baseline_hvp_mae": float(
                    np.mean([float(row["strict_hvp_mae"]) for row in rows])
                ),
            }
            for molecule_id, rows in by_molecule.items()
        ]
        ranked.sort(key=lambda row: row["baseline_hvp_mae"])
        candidate_by_molecule: dict[str, list[float]] = defaultdict(list)
        baseline_key = {
            (int(row["seed"]), row["molecule_id"], int(row["direction_index"])): float(
                row["strict_hvp_mae"]
            )
            for row in baseline_rows
        }
        candidate_runs = {candidate["run_name"] for candidate in candidates}
        for row in directions:
            if row["run_name"] not in candidate_runs:
                continue
            key = (int(row["seed"]), row["molecule_id"], int(row["direction_index"]))
            candidate_by_molecule[row["molecule_id"]].append(
                float(row["strict_hvp_mae"]) / max(baseline_key[key], np.finfo(float).tiny)
            )
        reasons = [
            ("smallest", min(ranked, key=lambda row: (row["natoms"], row["baseline_hvp_mae"]))),
            ("median_hvp", ranked[len(ranked) // 2]),
            ("hard_hvp", ranked[-1]),
            (
                "candidate_regression",
                max(
                    ranked,
                    key=lambda row: np.mean(candidate_by_molecule[row["molecule_id"]]),
                ),
            ),
            ("largest", max(ranked, key=lambda row: (row["natoms"], row["baseline_hvp_mae"]))),
        ]
        seen = set()
        for reason, row in reasons:
            if row["molecule_id"] not in seen:
                selected_molecules.append({**row, "reason": reason})
                seen.add(row["molecule_id"])
        for row in ranked:
            if len(selected_molecules) >= args.parent_count:
                break
            if row["molecule_id"] not in seen:
                selected_molecules.append({**row, "reason": "quantile_fill"})
                seen.add(row["molecule_id"])
        selected_molecules = selected_molecules[: args.parent_count]

        strict_runs = {row["run_name"]: row for row in strict["runs"]}
        run_map = {}
        for candidate in candidates:
            candidate_row = strict_runs[candidate["run_name"]]
            baseline = next(
                row
                for row in strict["runs"]
                if row["variant"] == "A" and int(row["seed"]) == int(candidate["seed"])
            )
            run_map[baseline["run_name"]] = {**baseline, "role": "matched_A"}
            run_map[candidate_row["run_name"]] = {**candidate_row, "role": "candidate"}
        selected_runs = list(run_map.values())
        for run in selected_runs:
            for molecule in selected_molecules:
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "role": run["role"],
                        "variant": run["variant"],
                        "curvature_weight": run["curvature_weight"],
                        "reference_floor": run["reference_floor"],
                        "seed": run["seed"],
                        "run_name": run["run_name"],
                        "molecule_id": molecule["molecule_id"],
                        "natoms": molecule["natoms"],
                        "molecule_reason": molecule["reason"],
                    }
                )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_index", "role", "variant", "curvature_weight", "reference_floor",
        "seed", "run_name", "molecule_id", "natoms", "molecule_reason",
    ]
    with args.output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    pbe_manifest = []
    for molecule in selected_molecules:
        path = args.reference_dir / f"pbe_hessian_{molecule['molecule_id']}_0000000.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        pbe_manifest.append(
            {
                "molecule_id": molecule["molecule_id"],
                "sample_id": 0,
                "natoms": molecule["natoms"],
                "success": True,
                "cache_path": path.as_posix(),
            }
        )
    pbe_manifest_path = args.output_json.with_name("full_hessian_pbe_manifest.json")
    pbe_manifest_path.write_text(json.dumps(pbe_manifest, indent=2, sort_keys=True) + "\n")
    payload = {
        "definition": __doc__,
        "test100_accessed": False,
        "candidates": candidates,
        "molecules": selected_molecules,
        "runs": selected_runs,
        "tasks": tasks,
        "pbe_manifest": pbe_manifest_path.as_posix(),
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": len(tasks), "molecules": selected_molecules}, indent=2))


if __name__ == "__main__":
    main()
