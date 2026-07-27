#!/usr/bin/env python3
"""Report progress for a two-phase KEDF feasibility run."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from scripts.analyze_two_phase_run import parse_md_log, series_stats

STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")
TN_RE = re.compile(r"^\s*TN\d+\s+", re.MULTILINE)


def electronic_iterations(text: str) -> list[int]:
    matches = list(STEP_RE.finditer(text))
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append(len(TN_RE.findall(text[match.end() : end])))
    return result


def report(root: Path, requested_steps: int) -> dict:
    results = []
    for phase in ("solid", "liquid"):
        run = root / phase
        logs = sorted(run.glob("OUT.*/running_md.log"))
        stdout = run / "run.stdout"
        stdout_text = stdout.read_text(errors="replace") if stdout.exists() else ""
        if not logs:
            results.append({"phase": phase, "status": "initializing"})
            continue
        rows, max_step = parse_md_log(logs[-1])
        recent = rows[-50:]
        out_dir = logs[-1].parent
        elapsed = max(0.0, time.time() - out_dir.stat().st_mtime)
        seconds_per_step = elapsed / max_step if max_step else None
        remaining = (
            max(0, requested_steps - max_step) * seconds_per_step
            if seconds_per_step is not None
            else None
        )
        iterations = electronic_iterations(stdout_text)
        recent_iterations = iterations[-20:]
        pressures = [
            row["pressure_kbar"] for row in recent if "pressure_kbar" in row
        ]
        results.append(
            {
                "phase": phase,
                "status": "running" if max_step < requested_steps else "md_complete",
                "max_step": max_step,
                "requested_steps": requested_steps,
                "temperature_recent_K": series_stats(
                    [row["temperature_K"] for row in recent]
                ),
                "pressure_recent_kbar": series_stats(pressures),
                "electronic_iterations_recent": series_stats(recent_iterations),
                "contains_nan": "nan" in stdout_text.lower(),
                "elapsed_seconds": elapsed,
                "seconds_per_step": seconds_per_step,
                "remaining_seconds": remaining,
            }
        )
    return {
        "schema": "kedf-feasibility-progress-v1",
        "root": str(root.resolve()),
        "requested_steps": requested_steps,
        "phases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve(), args.steps), indent=2))


if __name__ == "__main__":
    main()
