#!/usr/bin/env python3
"""Build a frozen small implicit-complete-total HVP teacher subset from stable branch audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from qm9_hvp_branch_stability_analysis import BRANCHES, _load_tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_implicit_hvp(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["implicit_hvp"], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-analysis-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--input-sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-sidecar-dir", type=Path, required=True)
    parser.add_argument("--parent-limit", type=int, default=10)
    parser.add_argument("--expected-directions", type=int, default=4)
    args = parser.parse_args()
    if args.parent_limit <= 0:
        raise ValueError("parent-limit must be positive")

    with (args.stability_analysis_dir / "parent_stability.csv").open() as handle:
        stable_parents = sorted(
            row["molecule_id"]
            for row in csv.DictReader(handle)
            if row["parent_stable"].lower() == "true"
        )[: args.parent_limit]
    with (args.stability_analysis_dir / "direction_stability.csv").open() as handle:
        direction_stable = {
            (row["molecule_id"], int(row["direction_index"])): row["stable"].lower()
            == "true"
            for row in csv.DictReader(handle)
        }
    grouped, load_errors = _load_tasks(args.task_root)
    if load_errors:
        raise RuntimeError(f"Malformed branch tasks: {load_errors[:3]}")
    args.output_sidecar_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sidecar_entries = []
    for molecule_id in stable_parents:
        source = args.input_sidecar_dir / f"{molecule_id}.0000000.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source) as source_payload:
            payload = {key: np.asarray(source_payload[key]) for key in source_payload.files}
        source_directions = np.asarray(payload["direction"], dtype=np.float64)
        if source_directions.ndim == 2:
            source_directions = source_directions[None, ...]
        targets = []
        stability_mask = []
        for direction_index in range(args.expected_directions):
            stable = bool(direction_stable.get((molecule_id, direction_index), False))
            stability_mask.append(stable)
            if not stable:
                targets.append(np.zeros_like(source_directions[direction_index]))
                continue
            branches = grouped.get((molecule_id, direction_index), {})
            missing = sorted(set(BRANCHES.values()).difference(branches))
            if missing:
                raise RuntimeError(
                    f"{molecule_id} direction {direction_index} lacks branches {missing}"
                )
            branch_targets = np.stack(
                [
                    _load_implicit_hvp(branches[name]["arrays_path"])
                    for name in sorted(branches)
                ]
            )
            if not np.isfinite(branch_targets).all():
                raise ValueError(f"Non-finite implicit HVP for {molecule_id} d{direction_index}")
            target = branch_targets.mean(axis=0)
            targets.append(target)
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "direction_index": direction_index,
                    "branch_count": len(branch_targets),
                    "branch_rmse_max": max(
                        float(np.sqrt(np.mean((value - target) ** 2)))
                        for value in branch_targets
                    ),
                    "target_rms": float(np.sqrt(np.mean(target**2))),
                }
            )
        payload["implicit_complete_total_hvp_target"] = np.stack(targets)
        payload["stability_mask"] = np.asarray(stability_mask, dtype=np.bool_)
        payload["parent_stability_eligible"] = np.asarray(True, dtype=np.bool_)
        destination = args.output_sidecar_dir / source.name
        np.savez_compressed(destination, **payload)
        sidecar_entries.append(
            {
                "molecule_id": molecule_id,
                "filename": destination.name,
                "sha256": _sha256(destination),
                "stability_mask": np.asarray(stability_mask, dtype=int).tolist(),
            }
        )

    csv_path = args.output_sidecar_dir.parent / "implicit_target_directions.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "definition": (
            "Strictly stable train100 subset with branch-mean baseline OFDFT implicit "
            "complete-total HVP teacher targets. Predictions remain learned-energy "
            "fixed-density HVPs; this is not end-to-end implicit differentiation."
        ),
        "selection_rule": "lexicographically first stable parent IDs",
        "parent_limit": args.parent_limit,
        "parent_count": len(stable_parents),
        "parent_ids": stable_parents,
        "entries": sidecar_entries,
        "directions_per_parent": args.expected_directions,
        "sidecar_count": len(list(args.output_sidecar_dir.glob("*.npz"))),
        "input_sidecar_manifest_sha256": _sha256(
            args.input_sidecar_dir.parent / "manifest.json"
        ),
        "stability_summary_sha256": _sha256(
            args.stability_analysis_dir / "summary.json"
        ),
        "parent_stability_csv_sha256": _sha256(
            args.stability_analysis_dir / "parent_stability.csv"
        ),
        "direction_stability_csv_sha256": _sha256(
            args.stability_analysis_dir / "direction_stability.csv"
        ),
        "test100_accessed": False,
    }
    manifest_path = args.output_sidecar_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
