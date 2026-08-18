#!/usr/bin/env python3
"""Summarize the latest successful EGFH10 TensorBoard event file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSS_TAGS = {
    "total": "train_loss/total",
    "E": "train_loss/energy_loss",
    "G": "train_loss/gradient_loss",
    "F": "train_loss/force_loss",
    "H": "train_loss/relaxed_force_secant_loss",
}


def scalar_summary(events) -> dict:
    steps = [int(event.step) for event in events]
    values = np.asarray([event.value for event in events], dtype=np.float64)
    midpoint = len(values) // 2
    result = {
        "count": int(len(values)),
        "first": {"step": steps[0], "value": float(values[0])},
        "last": {"step": steps[-1], "value": float(values[-1])},
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }
    if midpoint:
        first_half = float(values[:midpoint].mean())
        second_half = float(values[midpoint:].mean())
        result["first_half_mean"] = first_half
        result["second_half_mean"] = second_half
        result["half_mean_relative_change"] = (
            None if first_half == 0.0 else (second_half - first_half) / first_half
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    event_files = sorted(
        args.run_root.glob("events.out.tfevents.*"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not event_files:
        raise SystemExit(f"No TensorBoard event files under {args.run_root}")
    event_file = event_files[-1]
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    losses = {
        name: scalar_summary(accumulator.Scalars(tag))
        for name, tag in LOSS_TAGS.items()
        if tag in scalar_tags
    }
    gradient_norms = {
        tag.removeprefix("train_gradient_norm/"): scalar_summary(
            accumulator.Scalars(tag)
        )
        for tag in sorted(scalar_tags)
        if tag.startswith("train_gradient_norm/")
    }
    gradient_cosines = {
        tag.removeprefix("train_gradient_cosine/"): scalar_summary(
            accumulator.Scalars(tag)
        )
        for tag in sorted(scalar_tags)
        if tag.startswith("train_gradient_cosine/")
    }
    total_gradient_norm = (
        scalar_summary(accumulator.Scalars("grad_2.0_norm_total"))
        if "grad_2.0_norm_total" in scalar_tags
        else None
    )
    payload = {
        "event_file": str(event_file),
        "losses": losses,
        "loss_component_gradient_norms": gradient_norms,
        "loss_component_gradient_cosines": gradient_cosines,
        "total_gradient_norm": total_gradient_norm,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
