#!/usr/bin/env python3
"""Freeze validation molecules and matched-seed runs for full complete-total Hessians."""

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
    parser.add_argument("--strict-per-molecule", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=2)
    args = parser.parse_args()

    strict = json.loads(args.strict_summary.read_text())
    candidates = list(strict["frozen_candidates_for_full_hessian"])
    diagnostic_only = False
    if not candidates:
        diagnostic_only = True
        groups = [g for g in strict["groups"] if g["variant"] in {"C", "D"}]
        best = min(groups, key=lambda g: g["mean_strict_hvp_mae"])
        members = [
            row for row in strict["runs"]
            if (row["variant"], row["hvp_weight"])
            == (best["variant"], best["hvp_weight"])
        ]
        selected = sorted(members, key=lambda row: row["strict_hvp_mae"])[len(members) // 2]
        candidates = [{
            "variant": best["variant"],
            "hvp_weight": best["hvp_weight"],
            "run": selected["run"],
            "seed_selection": "diagnostic median seed; strict promotion gate failed",
        }]
    candidates = candidates[: args.max_candidates]

    with args.strict_per_molecule.open() as handle:
        molecule_rows = list(csv.DictReader(handle))
    baseline = [row for row in molecule_rows if row["variant"] == "A"]
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in baseline:
        by_molecule[row["molecule_id"]].append(row)
    ranked = []
    for molecule, rows in by_molecule.items():
        ranked.append({
            "molecule_id": molecule,
            "natoms": int(rows[0]["natoms"]),
            "baseline_mean_strict_hvp_mae": float(
                np.mean([float(row["strict_hvp_mae"]) for row in rows])
            ),
        })
    by_error = sorted(ranked, key=lambda row: row["baseline_mean_strict_hvp_mae"])
    small = min(ranked, key=lambda row: (row["natoms"], row["baseline_mean_strict_hvp_mae"]))
    median = by_error[len(by_error) // 2]
    hard = by_error[-1]
    molecules = []
    for reason, row in (("small", small), ("median", median), ("hard", hard)):
        if row["molecule_id"] not in {item["molecule_id"] for item in molecules}:
            molecules.append({**row, "reason": reason})

    strict_runs = {row["run"]: row for row in strict["runs"]}
    selected_runs: dict[str, dict] = {}
    for candidate in candidates:
        candidate_row = strict_runs[candidate["run"]]
        seed = candidate_row["seed"]
        for variant in ("A", "B"):
            matched = next(
                row for row in strict["runs"]
                if row["variant"] == variant and row["seed"] == seed
            )
            selected_runs[matched["run"]] = {**matched, "role": f"matched_{variant}"}
        selected_runs[candidate["run"]] = {**candidate_row, "role": "candidate"}

    tasks = []
    for run in selected_runs.values():
        for molecule in molecules:
            tasks.append({
                "task_index": len(tasks),
                "role": run["role"],
                "variant": run["variant"],
                "hvp_weight": run["hvp_weight"],
                "seed": run["seed"],
                "run": run["run"],
                "molecule_id": molecule["molecule_id"],
                "natoms": molecule["natoms"],
                "molecule_reason": molecule["reason"],
            })

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(tasks[0])
    with args.output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(tasks)
    payload = {
        "definition": "Validation-only full complete-total Hessian task freeze.",
        "test_accessed": False,
        "diagnostic_only": diagnostic_only,
        "candidates": candidates,
        "molecules": molecules,
        "runs": list(selected_runs.values()),
        "tasks": tasks,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": len(tasks), "diagnostic_only": diagnostic_only,
                      "molecules": molecules, "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
