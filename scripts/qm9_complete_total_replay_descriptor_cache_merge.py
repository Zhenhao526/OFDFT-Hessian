#!/usr/bin/env python3
"""Merge and fail-closed audit Stage-3 train800 replay descriptor shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.qm9_complete_total_replay_descriptor_cache import (
        _rows,
        _sha256,
        _task_hash,
    )
except ModuleNotFoundError:
    from qm9_complete_total_replay_descriptor_cache import _rows, _sha256, _task_hash


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
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return {key: None for key in ("mean", "median", "p90", "max")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "max": float(np.max(finite)),
    }


def _load_shard_contract(
    cache_root: Path,
    shard_count: int,
    task_csv_sha: str,
    schema_sha: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    settings = None
    failures = []
    for shard_index in range(shard_count):
        path = cache_root / f"shard_{shard_index:04d}_summary.json"
        if not path.is_file():
            failures.append(
                {
                    "scope": "shard",
                    "shard_index": shard_index,
                    "failure_reason": "missing_shard_summary",
                }
            )
            continue
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            failures.append(
                {
                    "scope": "shard",
                    "shard_index": shard_index,
                    "failure_reason": f"invalid_shard_summary:{type(error).__name__}",
                }
            )
            continue
        reasons = []
        if row.get("test100_accessed") is not False:
            reasons.append("test100_not_frozen")
        if row.get("task_csv_sha256") != task_csv_sha:
            reasons.append("task_csv_hash_mismatch")
        if row.get("schema_manifest_sha256") != schema_sha:
            reasons.append("schema_hash_mismatch")
        if int(row.get("shard_index", -1)) != shard_index:
            reasons.append("shard_index_mismatch")
        if int(row.get("shard_count", -1)) != shard_count:
            reasons.append("shard_count_mismatch")
        row_settings = row.get("descriptor_settings")
        if not isinstance(row_settings, dict):
            reasons.append("missing_descriptor_settings")
        elif settings is None:
            settings = row_settings
        elif row_settings != settings:
            reasons.append("descriptor_settings_mismatch")
        if reasons:
            failures.append(
                {
                    "scope": "shard",
                    "shard_index": shard_index,
                    "failure_reason": ";".join(reasons),
                }
            )
    return settings, failures


def merge(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _rows(args.task_csv)
    task_csv_sha = _sha256(args.task_csv)
    schema_sha = _sha256(args.schema_manifest)
    schema = json.loads(args.schema_manifest.read_text())
    if schema.get("test100_accessed") is not False:
        raise ValueError("active feature schema does not freeze Test100")
    active_feature_count = int(schema["active_feature_count"])
    settings, failures = _load_shard_contract(
        args.cache_root, args.shard_count, task_csv_sha, schema_sha
    )
    successes = []
    seen_pairs = set()
    for task_index, task in enumerate(tasks):
        molecule_id = str(task["molecule_id"])
        sample_id = int(task["sample_id"])
        pair = (molecule_id, sample_id)
        reasons = []
        if pair in seen_pairs:
            reasons.append("duplicate_molecule_sample")
        seen_pairs.add(pair)
        task_dir = (
            args.cache_root
            / f"task_{task_index:04d}_{molecule_id}_{sample_id:07d}"
        )
        summary_path = task_dir / "summary.json"
        audit: dict[str, Any] = {
            "scope": "task",
            "task_index": task_index,
            "molecule_id": molecule_id,
            "sample_id": sample_id,
            "summary_path": summary_path.resolve().as_posix(),
        }
        if not summary_path.is_file():
            reasons.append("missing_task_summary")
            summary = {}
        else:
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                summary = {}
                reasons.append(f"invalid_task_summary:{type(error).__name__}")
        if summary:
            if summary.get("test100_accessed") is not False:
                reasons.append("test100_not_frozen")
            if summary.get("task_csv_sha256") != task_csv_sha:
                reasons.append("task_csv_hash_mismatch")
            if summary.get("schema_manifest_sha256") != schema_sha:
                reasons.append("schema_hash_mismatch")
            if summary.get("descriptor_settings") != settings:
                reasons.append("descriptor_settings_mismatch")
            if settings is None or summary.get("task_sha256") != _task_hash(
                task, schema_sha, settings or {}
            ):
                reasons.append("task_hash_mismatch")
            if summary.get("baseline_array_sha256") != task["baseline_array_sha256"]:
                reasons.append("baseline_hash_mismatch")
            if summary.get("success") is not True:
                reasons.append("task_not_successful")
            cache_value = summary.get("descriptor_cache")
            cache_path = Path(cache_value) if cache_value else None
            if cache_path is None or not cache_path.is_file():
                reasons.append("missing_descriptor_cache")
            elif _sha256(cache_path) != summary.get("descriptor_cache_sha256"):
                reasons.append("descriptor_cache_hash_mismatch")
            else:
                try:
                    with np.load(cache_path) as payload:
                        descriptor = np.asarray(payload["descriptor"])
                        jacobian = np.asarray(payload["descriptor_jacobian"])
                        force_target = np.asarray(payload["force_target"])
                        energy_target = np.asarray(payload["energy_target"])
                    coordinate_count = int(force_target.size)
                    if descriptor.shape != (active_feature_count,):
                        reasons.append("descriptor_shape_mismatch")
                    if jacobian.shape != (coordinate_count, active_feature_count):
                        reasons.append("jacobian_shape_mismatch")
                    if energy_target.size != 1:
                        reasons.append("energy_target_shape_mismatch")
                    if not all(
                        np.all(np.isfinite(array))
                        for array in (descriptor, jacobian, force_target, energy_target)
                    ):
                        reasons.append("non_finite_cache_payload")
                except (OSError, ValueError, KeyError) as error:
                    reasons.append(f"invalid_cache_payload:{type(error).__name__}")
            audit.update(
                {
                    "descriptor_cache": cache_value,
                    "descriptor_cache_sha256": summary.get(
                        "descriptor_cache_sha256"
                    ),
                    "task_sha256": summary.get("task_sha256"),
                    "natoms": summary.get("natoms"),
                    "coordinate_count": summary.get("coordinate_count"),
                    "active_feature_count": summary.get("active_feature_count"),
                    "energy_target_hartree": summary.get("energy_target_hartree"),
                    "force_target_mae_hartree_per_bohr": summary.get(
                        "force_target_mae_hartree_per_bohr"
                    ),
                    "wall_time_s": summary.get("wall_time_s"),
                    "max_rss_mb": summary.get("max_rss_mb"),
                    "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
                    "three_body_matched_active_feature_fraction": summary.get(
                        "three_body_matched_active_feature_fraction"
                    ),
                    "four_body_matched_active_feature_fraction": summary.get(
                        "four_body_matched_active_feature_fraction"
                    ),
                }
            )
        if reasons:
            audit["failure_reason"] = ";".join(reasons)
            failures.append(audit)
        else:
            audit["summary_sha256"] = _sha256(summary_path)
            successes.append(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_csv = args.output_dir / "replay_descriptor_cache_success.csv"
    failure_csv = args.output_dir / "replay_descriptor_cache_failures.csv"
    _write_csv(success_csv, successes)
    _write_csv(failure_csv, failures)
    manifest = {
        "definition": "Audited train800 scalar descriptor/Jacobian E/F replay cache",
        "task_csv": args.task_csv.resolve().as_posix(),
        "task_csv_sha256": task_csv_sha,
        "schema_manifest": args.schema_manifest.resolve().as_posix(),
        "schema_manifest_sha256": schema_sha,
        "schema_checkpoint_sha256": schema["checkpoint_sha256"],
        "active_feature_count": active_feature_count,
        "descriptor_settings": settings,
        "shard_count": args.shard_count,
        "counts": {
            "expected_tasks": len(tasks),
            "successful_tasks": len(successes),
            "failed_tasks_or_shards": len(failures),
        },
        "metrics": {
            "wall_time_s": _quantiles(
                [float(row["wall_time_s"]) for row in successes]
            ),
            "force_target_mae_hartree_per_bohr": _quantiles(
                [
                    float(row["force_target_mae_hartree_per_bohr"])
                    for row in successes
                ]
            ),
            "three_body_matched_active_feature_fraction": _quantiles(
                [
                    float(row["three_body_matched_active_feature_fraction"])
                    for row in successes
                ]
            ),
            "four_body_matched_active_feature_fraction": _quantiles(
                [
                    float(row["four_body_matched_active_feature_fraction"])
                    for row in successes
                ]
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
        },
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "complete": len(successes) == len(tasks) and not failures,
    }
    output = args.output_dir / "replay_descriptor_cache_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.require_complete and not manifest["complete"]:
        raise RuntimeError(
            f"replay descriptor cache audit failed with {len(failures)} failures"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--schema-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--require-complete", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
