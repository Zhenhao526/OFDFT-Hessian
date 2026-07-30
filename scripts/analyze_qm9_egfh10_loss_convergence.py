#!/usr/bin/env python3
"""Evaluate EGFH training-loss convergence from adjacent 50-step windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPONENTS = ("total", "E", "G", "F", "H")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    windows: dict[int, dict] = {}
    sources: list[str] = []
    for path in args.summaries:
        payload = json.loads(path.read_text())
        losses = payload["losses"]
        total = losses["total"]
        count = int(total["count"])
        first_step = int(total["first"]["step"])
        last_step = int(total["last"]["step"])
        midpoint = count // 2
        if count != 100 or midpoint != 50 or last_step - first_step != 99:
            raise SystemExit(
                f"{path}: expected 100 consecutive steps, found "
                f"count={count}, range={first_step}-{last_step}"
            )
        sources.append(str(path))
        for offset, mean_key in (
            (0, "first_half_mean"),
            (midpoint, "second_half_mean"),
        ):
            start = first_step + offset
            windows[start] = {
                "start_step": start,
                "end_step": start + midpoint - 1,
                "means": {
                    component: float(losses[component][mean_key])
                    for component in COMPONENTS
                },
            }

    ordered = [windows[start] for start in sorted(windows)]
    for index, window in enumerate(ordered):
        if index == 0:
            window["total_relative_change_from_previous"] = None
            continue
        previous = ordered[index - 1]["means"]["total"]
        current = window["means"]["total"]
        window["total_relative_change_from_previous"] = (
            None if previous == 0.0 else (current - previous) / previous
        )

    recent_changes = [
        window["total_relative_change_from_previous"] for window in ordered[-2:]
    ]
    converged = len(recent_changes) == 2 and all(
        change is not None and abs(change) < args.relative_tolerance
        for change in recent_changes
    )
    payload = {
        "criterion": (
            "absolute relative change of mean total loss below tolerance "
            "for each of the two most recent adjacent 50-step transitions"
        ),
        "relative_tolerance": args.relative_tolerance,
        "converged": converged,
        "last_completed_step": ordered[-1]["end_step"] + 1,
        "recent_total_relative_changes": recent_changes,
        "windows": ordered,
        "source_summaries": sources,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
