#!/usr/bin/env python3
"""Select a robust-scale stable5 candidate without reading validation parents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(value: str) -> tuple[str, float, Path]:
    try:
        name, floor_text, path_text = value.split("=", maxsplit=2)
        floor = float(floor_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=FLOOR=SUMMARY_JSON"
        ) from error
    if not name or floor <= 0.0:
        raise argparse.ArgumentTypeError("candidate name and positive floor are required")
    return name, floor, Path(path_text)


def select(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_names = set()
    for name, floor, path in args.candidate:
        if name in seen_names:
            raise ValueError(f"duplicate candidate name: {name}")
        seen_names.add(name)
        payload = json.loads(path.read_text())
        if payload.get("test100_accessed") is not False:
            raise ValueError(f"candidate does not freeze Test100: {path}")
        parents = payload["per_parent"]
        hessian = np.asarray([row["relative_frobenius"] for row in parents])
        energy = np.asarray([row["energy_abs_error_hartree"] for row in parents])
        force = np.asarray([row["force_mae_hartree_per_bohr"] for row in parents])
        finite = bool(
            all(np.all(np.isfinite(values)) for values in (hessian, energy, force))
        )
        gates = {
            "hessian_gate_passed": bool(finite and np.max(hessian) <= args.hessian_max),
            "energy_gate_passed": bool(
                finite
                and np.median(energy) <= args.energy_median_max
                and np.max(energy) <= args.energy_all_max
            ),
            "force_gate_passed": bool(
                finite
                and np.median(force) <= args.force_median_max
                and np.max(force) <= args.force_all_max
            ),
        }
        records.append(
            {
                "name": name,
                "feature_scale_floor": floor,
                "summary": path.resolve().as_posix(),
                "summary_sha256": _sha256(path),
                "parent_count": len(parents),
                "finite": finite,
                "hessian_relative_frobenius_median": float(np.median(hessian)),
                "hessian_relative_frobenius_max": float(np.max(hessian)),
                "energy_abs_error_median_hartree": float(np.median(energy)),
                "energy_abs_error_max_hartree": float(np.max(energy)),
                "force_mae_median_hartree_per_bohr": float(np.median(force)),
                "force_mae_max_hartree_per_bohr": float(np.max(force)),
                **gates,
                "all_gates_passed": bool(all(gates.values())),
            }
        )

    passing = [row for row in records if row["all_gates_passed"]]
    selected = max(passing, key=lambda row: row["feature_scale_floor"]) if passing else None
    with (args.output_dir / "candidate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "definition": "stable5-only robust feature-scale selection",
        "selection_rule": "largest feature-scale floor passing all preregistered gates",
        "gates": {
            "hessian_relative_frobenius_all_parent_max": args.hessian_max,
            "energy_abs_error_median_max_hartree": args.energy_median_max,
            "energy_abs_error_all_parent_max_hartree": args.energy_all_max,
            "force_mae_median_max_hartree_per_bohr": args.force_median_max,
            "force_mae_all_parent_max_hartree_per_bohr": args.force_all_max,
        },
        "candidates": records,
        "selected_candidate": selected,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "validation_parent_metrics_used": False,
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hessian-max", type=float, default=0.05)
    parser.add_argument("--energy-median-max", type=float, default=1.0e-3)
    parser.add_argument("--energy-all-max", type=float, default=2.0e-3)
    parser.add_argument("--force-median-max", type=float, default=1.0e-3)
    parser.add_argument("--force-all-max", type=float, default=3.0e-3)
    return parser.parse_args()


if __name__ == "__main__":
    select(parse_args())
