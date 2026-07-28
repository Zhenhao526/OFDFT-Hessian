#!/usr/bin/env python3
"""Retry missing and transient-I/O failures from a completed replay array."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TRANSIENT_ERROR_TYPES = {"OSError", "BlockingIOError", "TimeoutError"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_path(root: Path, index: int, row: dict[str, str]) -> Path:
    return root / (
        f"task_{index:04d}_{row['molecule_id']}_{int(row['sample_id']):07d}"
    ) / "summary.json"


def _retry_reason(path: Path) -> str | None:
    if not path.is_file():
        return "missing_summary"
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable_summary"
    if summary.get("success") is True:
        return None
    error_type = summary.get("error_type")
    if error_type in TRANSIENT_ERROR_TYPES:
        return f"transient_{error_type}"
    return None


def rescue(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows = _rows(args.task_csv)
    selected = []
    for index, row in enumerate(rows):
        reason = _retry_reason(_summary_path(args.output_dir, index, row))
        if reason is not None:
            selected.append((index, row, reason))
    if len(selected) > args.max_tasks:
        raise RuntimeError(
            f"refusing to retry {len(selected)} tasks above max_tasks={args.max_tasks}"
        )

    attempts = []
    for index, row, reason in selected:
        command = [
            sys.executable,
            "scripts/qm9_complete_total_replay_baseline.py",
            "--protocol",
            str(args.protocol),
            "--task-csv",
            str(args.task_csv),
            "--output-dir",
            str(args.output_dir),
            "--run",
            args.run,
            "--task-index",
            str(index),
            "--device",
            args.device,
            "--transform-device",
            "cpu",
            "--no-fallback-always",
            "--integral-derivative-workers",
            str(args.integral_derivative_workers),
        ]
        task_started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        summary_path = _summary_path(args.output_dir, index, row)
        remaining_reason = _retry_reason(summary_path)
        attempts.append(
            {
                "task_index": index,
                "molecule_id": row["molecule_id"],
                "sample_id": int(row["sample_id"]),
                "initial_reason": reason,
                "returncode": completed.returncode,
                "success": remaining_reason is None,
                "remaining_retry_reason": remaining_reason,
                "wall_time_s": time.perf_counter() - task_started,
                "summary": summary_path.resolve().as_posix(),
                "summary_sha256": (
                    _sha256(summary_path) if summary_path.is_file() else None
                ),
            }
        )

    result = {
        "definition": "missing/transient-I/O replay task rescue after main array",
        "task_csv": args.task_csv.resolve().as_posix(),
        "task_csv_sha256": _sha256(args.task_csv),
        "output_dir": args.output_dir.resolve().as_posix(),
        "selected_count": len(selected),
        "successful_rescue_count": sum(row["success"] for row in attempts),
        "failed_rescue_count": sum(not row["success"] for row in attempts),
        "attempts": attempts,
        "wall_time_s": time.perf_counter() - started,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed_rescue_count"]:
        raise RuntimeError("one or more transient replay rescues still failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--integral-derivative-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    rescue(parse_args())
