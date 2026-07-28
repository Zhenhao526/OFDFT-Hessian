#!/usr/bin/env python3
"""Freeze a resumable HVP branch-audit pause snapshot without touching source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _task_dir(task_root: Path, row: dict[str, str]) -> Path:
    return task_root / (
        f"task_{int(row['task_index']):04d}_{row['molecule_id']}_"
        f"d{int(row['direction_index'])}_{row['branch']}"
    )


def _slurm_snapshot(job_ids: list[str]) -> list[dict[str, str]]:
    if not job_ids:
        return []
    command = [
        "sacct",
        "-n",
        "-X",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,Elapsed,ExitCode",
        "--parsable2",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 5 or not fields[0]:
            continue
        rows.append(
            dict(
                zip(("job_id", "job_name", "state", "elapsed", "exit_code"), fields[:5])
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-table", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--provenance-file", type=Path, action="append", default=[])
    args = parser.parse_args()

    with args.task_table.open() as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    if not tasks:
        raise RuntimeError("Task table is empty")
    fields = list(tasks[0])
    completed_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    partial_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    validation_errors: list[str] = []

    for task in tasks:
        task_dir = _task_dir(args.task_root, task)
        summary_path = task_dir / "summary.json"
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text())
                summaries = payload.get("summaries", [])
                if len(summaries) != 1:
                    raise ValueError(f"expected one summary, got {len(summaries)}")
                summary = summaries[0]
                observed = (
                    str(summary["molecule_id"]).zfill(7),
                    int(summary["direction_index"]),
                )
                expected = (task["molecule_id"], int(task["direction_index"]))
                if observed != expected:
                    raise ValueError(f"summary key {observed} != task key {expected}")
                arrays = list(task_dir.glob("*_hvp_arrays.npz"))
                if len(arrays) != 1:
                    raise ValueError(f"expected one arrays NPZ, got {len(arrays)}")
                completed_rows.append(task)
                artifacts.append(
                    {
                        "task_index": int(task["task_index"]),
                        "molecule_id": task["molecule_id"],
                        "direction_index": int(task["direction_index"]),
                        "branch": task["branch"],
                        "task_dir": task_dir.as_posix(),
                        "summary_sha256": _sha256(summary_path),
                        "arrays_path": arrays[0].as_posix(),
                        "arrays_size_bytes": arrays[0].stat().st_size,
                        "wall_time_s": float(summary["wall_time_s"]),
                    }
                )
                continue
            except Exception as exc:
                validation_errors.append(
                    f"{task_dir}: {type(exc).__name__}: {exc}"
                )
        missing_rows.append(task)
        if task_dir.exists():
            files = [path for path in task_dir.rglob("*") if path.is_file()]
            partial_rows.append(
                {
                    **task,
                    "task_dir": task_dir.as_posix(),
                    "file_count": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.output_dir / "completed_train100_tasks.tsv"
    missing_path = args.output_dir / "missing_train100_tasks.tsv"
    partial_path = args.output_dir / "partial_train100_tasks.tsv"
    _write_tsv(completed_path, completed_rows, fields)
    _write_tsv(missing_path, missing_rows, fields)
    _write_tsv(
        partial_path,
        partial_rows,
        fields + ["task_dir", "file_count", "bytes"],
    )
    artifact_fields = sorted({key for row in artifacts for key in row})
    _write_tsv(args.output_dir / "completed_artifacts.tsv", artifacts, artifact_fields)

    provenance = []
    for path in args.provenance_file:
        if not path.is_file():
            raise FileNotFoundError(path)
        provenance.append(
            {
                "path": path.resolve().as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    try:
        squeue = subprocess.run(
            ["squeue", "-h", "-u", subprocess.getoutput("id -un")],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as exc:
        squeue = f"unavailable: {type(exc).__name__}: {exc}"

    snapshot = {
        "definition": __doc__,
        "paused_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "paused",
        "reason": "user_requested_pause_and_node01_only_future_compute",
        "task_table": args.task_table.resolve().as_posix(),
        "task_table_sha256": _sha256(args.task_table),
        "task_root": args.task_root.resolve().as_posix(),
        "expected_task_count": len(tasks),
        "completed_task_count": len(completed_rows),
        "missing_task_count": len(missing_rows),
        "partial_task_count": len(partial_rows),
        "validation_errors": validation_errors,
        "completed_tasks_sha256": _sha256(completed_path),
        "missing_tasks_sha256": _sha256(missing_path),
        "partial_tasks_sha256": _sha256(partial_path),
        "completed_artifact_bytes": sum(row["arrays_size_bytes"] for row in artifacts),
        "slurm_jobs": _slurm_snapshot(args.job_id),
        "active_squeue_after_pause": squeue,
        "future_default_compute_node": "node01",
        "future_default_max_gpu_tasks": 8,
        "other_nodes_require_explicit_user_instruction": True,
        "downstream_training_started": False,
        "test100_accessed": False,
        "source_labels_checkpoints_references_modified": False,
        "provenance": provenance,
    }
    snapshot_path = args.output_dir / "pause_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
