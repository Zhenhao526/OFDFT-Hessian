#!/usr/bin/env python3
"""Merge and validate Stage-3 complete-total replay baseline shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qm9_complete_total_replay_baseline import (
    _output_dir,
    _sha256,
    _task_rows,
    _task_sha256,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {name: None for name in ("mean", "median", "p90", "p99", "max")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _failure_reason(summary: dict[str, Any]) -> str:
    if summary.get("error"):
        return f"exception:{summary.get('error_type', 'unknown')}"
    if not summary.get("finite", False):
        return "non_finite"
    if not summary.get("strict_converged", False):
        metadata_norm = float(
            summary.get("final_projected_density_gradient_norm", math.inf)
        )
        tensor_norm = float(
            summary.get("tensor_projected_density_gradient_norm", math.inf)
        )
        if metadata_norm < 1.0e-8 and tensor_norm >= 1.0e-8:
            return "tensor_legacy_stationarity_mismatch"
        return "density_not_strict"
    return "invalid_success_flag"


def merge(args: argparse.Namespace) -> dict[str, Any]:
    protocol_sha = _sha256(args.protocol)
    tasks = _task_rows(args.task_csv)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        molecule_id = str(task["molecule_id"])
        sample_id = int(task["sample_id"])
        task_dir = _output_dir(
            args.baseline_root, task_index, molecule_id, sample_id
        )
        summary_path = task_dir / "summary.json"
        expected_task_sha = _task_sha256(task, args.run, protocol_sha)
        audit = {
            "task_index": task_index,
            "molecule_id": molecule_id,
            "sample_id": sample_id,
            "task_dir": task_dir.resolve().as_posix(),
            "status": "missing",
            "failure_reason": "missing_summary",
        }
        if not summary_path.is_file():
            failures.append(audit)
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            audit.update(
                status="invalid", failure_reason=f"invalid_summary:{type(error).__name__}"
            )
            failures.append(audit)
            continue
        reasons = []
        if summary.get("test100_accessed") is not False:
            reasons.append("test100_not_frozen")
        if summary.get("task_sha256") != expected_task_sha:
            reasons.append("task_hash_mismatch")
        if summary.get("protocol_sha256") != protocol_sha:
            reasons.append("protocol_hash_mismatch")
        if not summary.get("success"):
            reasons.append(_failure_reason(summary))
        baseline_path_value = summary.get("baseline_array")
        baseline_path = Path(baseline_path_value) if baseline_path_value else None
        if baseline_path is None or not baseline_path.is_file():
            reasons.append("missing_baseline_array")
        elif _sha256(baseline_path) != summary.get("baseline_array_sha256"):
            reasons.append("baseline_array_hash_mismatch")
        audit.update(
            {
                "status": "success" if not reasons else "failed",
                "failure_reason": ";".join(reasons),
                "strict_converged": bool(summary.get("strict_converged", False)),
                "finite": bool(summary.get("finite", False)),
                "natoms": summary.get("natoms"),
                "cycles": summary.get("cycles"),
                "wall_time_s": summary.get("wall_time_s"),
                "energy_abs_error_hartree": summary.get("energy_abs_error_hartree"),
                "force_mae_hartree_per_bohr": summary.get(
                    "force_mae_hartree_per_bohr"
                ),
                "force_rmse_hartree_per_bohr": summary.get(
                    "force_rmse_hartree_per_bohr"
                ),
                "final_projected_density_gradient_norm": summary.get(
                    "final_projected_density_gradient_norm"
                ),
                "tensor_projected_density_gradient_norm": summary.get(
                    "tensor_projected_density_gradient_norm"
                ),
                "tensor_legacy_energy_difference": summary.get(
                    "tensor_legacy_energy_difference"
                ),
                "baseline_array": baseline_path_value,
                "baseline_array_sha256": summary.get("baseline_array_sha256"),
                "summary_path": summary_path.resolve().as_posix(),
                "summary_sha256": _sha256(summary_path),
            }
        )
        if reasons:
            failures.append(audit)
        else:
            rows.append(audit)

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["molecule_id"])].append(row)
    all_parent_ids = sorted({str(task["molecule_id"]) for task in tasks})
    parent_rows = []
    for molecule_id in all_parent_ids:
        successful = by_parent.get(molecule_id, [])
        parent_rows.append(
            {
                "molecule_id": molecule_id,
                "successful_geometry_count": len(successful),
                "all_four_geometries_successful": len(successful) == 4,
                "mean_energy_abs_error_hartree": (
                    float(np.mean([row["energy_abs_error_hartree"] for row in successful]))
                    if successful
                    else None
                ),
                "mean_force_mae_hartree_per_bohr": (
                    float(
                        np.mean(
                            [row["force_mae_hartree_per_bohr"] for row in successful]
                        )
                    )
                    if successful
                    else None
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_csv = args.output_dir / "replay_baseline_success.csv"
    failure_csv = args.output_dir / "replay_baseline_failures.csv"
    parent_csv = args.output_dir / "replay_parent_coverage.csv"
    _write_csv(success_csv, rows)
    _write_csv(failure_csv, failures)
    _write_csv(parent_csv, parent_rows)
    summary = {
        "definition": (
            "Authoritative strict complete-total density-relaxed E/F replay baseline manifest"
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha,
        "task_csv": args.task_csv.resolve().as_posix(),
        "task_csv_sha256": _sha256(args.task_csv),
        "run": args.run,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "counts": {
            "expected_tasks": len(tasks),
            "successful_tasks": len(rows),
            "failed_or_missing_tasks": len(failures),
            "expected_parents": len(all_parent_ids),
            "parents_with_all_four_successful": sum(
                bool(row["all_four_geometries_successful"]) for row in parent_rows
            ),
            "parents_with_at_least_one_successful": sum(
                int(row["successful_geometry_count"]) > 0 for row in parent_rows
            ),
        },
        "failure_reason_counts": {
            reason: sum(
                reason in str(row["failure_reason"]).split(";") for row in failures
            )
            for reason in sorted(
                {
                    item
                    for row in failures
                    for item in str(row["failure_reason"]).split(";")
                    if item
                }
            )
        },
        "successful_task_metrics": {
            "wall_time_s": _quantiles([float(row["wall_time_s"]) for row in rows]),
            "cycles": _quantiles([float(row["cycles"]) for row in rows]),
            "energy_abs_error_hartree": _quantiles(
                [float(row["energy_abs_error_hartree"]) for row in rows]
            ),
            "force_mae_hartree_per_bohr": _quantiles(
                [float(row["force_mae_hartree_per_bohr"]) for row in rows]
            ),
        },
        "artifacts": {
            "success_csv": {
                "path": success_csv.resolve().as_posix(),
                "sha256": _sha256(success_csv),
            },
            "failure_csv": {
                "path": failure_csv.resolve().as_posix(),
                "sha256": _sha256(failure_csv),
            },
            "parent_csv": {
                "path": parent_csv.resolve().as_posix(),
                "sha256": _sha256(parent_csv),
            },
        },
    }
    output = args.output_dir / "replay_baseline_manifest.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and failures:
        raise RuntimeError(f"{len(failures)} replay tasks are failed or missing")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--require-complete", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())

