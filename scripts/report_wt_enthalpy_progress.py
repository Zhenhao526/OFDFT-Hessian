#!/usr/bin/env python3
"""Report progress for locally rebuilt WT fusion-enthalpy trajectories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.analyze_two_phase_run import parse_md_log, series_stats


def report(root: Path, requested_steps: int) -> dict:
    rows = []
    for run in sorted(root.glob("T*_steps*/*")):
        if not run.is_dir() or run.name not in {"solid", "liquid"}:
            continue
        logs = sorted(run.glob("OUT.*/running_md.log"))
        if not logs:
            rows.append(
                {
                    "temperature_label": run.parent.name,
                    "phase": run.name,
                    "status": "initializing",
                }
            )
            continue
        samples, max_step = parse_md_log(logs[-1])
        recent = samples[-50:]
        recent_pressures = [
            sample["pressure_kbar"]
            for sample in recent
            if "pressure_kbar" in sample
        ]
        # The run directory mtime is set when ABACUS creates OUT.* and is not
        # touched by subsequent writes inside that directory.
        elapsed_seconds = max(0.0, time.time() - run.stat().st_mtime)
        seconds_per_step = (
            elapsed_seconds / max_step if max_step > 0 else None
        )
        remaining_seconds = (
            max(0, requested_steps - max_step) * seconds_per_step
            if seconds_per_step is not None
            else None
        )
        rows.append(
            {
                "temperature_label": run.parent.name,
                "phase": run.name,
                "status": "running" if max_step < requested_steps else "md_complete",
                "max_step": max_step,
                "requested_steps": requested_steps,
                "temperature_recent_K": series_stats(
                    [sample["temperature_K"] for sample in recent]
                ),
                "pressure_recent_kbar": series_stats(
                    recent_pressures
                ),
                "trajectory_pressure_available": len(recent_pressures)
                == len(recent)
                and bool(recent),
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_step": seconds_per_step,
                "remaining_seconds": remaining_seconds,
            }
        )
    return {
        "schema": "wt-enthalpy-progress-v1",
        "root": str(root.resolve()),
        "requested_steps": requested_steps,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve(), args.steps), indent=2))


if __name__ == "__main__":
    main()
