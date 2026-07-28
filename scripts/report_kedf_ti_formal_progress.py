#!/usr/bin/env python3
"""Report per-window progress for a formal two-phase KEDF TI grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.analyze_two_phase_run import parse_md_log, series_stats
from scripts.report_kedf_feasibility_progress import electronic_iterations


FATAL_MARKERS = (
    "segmentation fault",
    "mpi_abort",
    "not converged",
    "fatal error",
)


def report(root: Path, requested_steps: int) -> dict:
    rows = []
    for phase in ("solid", "liquid"):
        for run in sorted((root / phase).glob("lambda_*")):
            if not run.is_dir():
                continue
            logs = sorted(run.glob("OUT.*/running_md.log"))
            if not logs:
                rows.append(
                    {
                        "phase": phase,
                        "lambda_label": run.name,
                        "status": "waiting",
                    }
                )
                continue
            samples, max_step = parse_md_log(logs[-1])
            recent = samples[-50:]
            latter_half = samples[len(samples) // 2 :]
            stdout_path = run / "run.stdout"
            stdout = (
                stdout_path.read_text(errors="replace")
                if stdout_path.exists()
                else ""
            )
            iterations = electronic_iterations(stdout)
            elapsed = max(0.0, time.time() - logs[-1].parent.stat().st_mtime)
            seconds_per_step = elapsed / max_step if max_step else None
            remaining = (
                max(0, requested_steps - max_step) * seconds_per_step
                if seconds_per_step is not None
                else None
            )
            rows.append(
                {
                    "phase": phase,
                    "lambda_label": run.name,
                    "status": (
                        "running" if max_step < requested_steps else "md_complete"
                    ),
                    "max_step": max_step,
                    "requested_steps": requested_steps,
                    "temperature_recent_K": series_stats(
                        [sample["temperature_K"] for sample in recent]
                    ),
                    "temperature_latter_half_K": series_stats(
                        [sample["temperature_K"] for sample in latter_half]
                    ),
                    "electronic_iterations_recent": series_stats(iterations[-20:]),
                    "contains_nan": "nan" in stdout.lower(),
                    "contains_fatal_marker": any(
                        marker in stdout.lower() for marker in FATAL_MARKERS
                    ),
                    "elapsed_seconds": elapsed,
                    "seconds_per_step": seconds_per_step,
                    "remaining_seconds": remaining,
                }
            )
    active = [row for row in rows if row["status"] == "running"]
    waiting = [row for row in rows if row["status"] == "waiting"]
    completed = [row for row in rows if row["status"] == "md_complete"]
    return {
        "schema": "kedf-ti-formal-progress-v1",
        "root": str(root.resolve()),
        "requested_steps": requested_steps,
        "summary": {
            "windows": len(rows),
            "active": len(active),
            "waiting": len(waiting),
            "completed": len(completed),
            "maximum_remaining_seconds": max(
                (
                    row["remaining_seconds"]
                    for row in active
                    if row["remaining_seconds"] is not None
                ),
                default=0.0,
            ),
        },
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.steps), indent=2))


if __name__ == "__main__":
    main()
