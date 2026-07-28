#!/usr/bin/env python3
"""Export A-E checkpoint, throughput, loss-gradient, and conflict diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scalar_events(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    events: dict[str, list[tuple[int, float, float]]] = {}
    for path in sorted(run_dir.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                events.setdefault(tag, []).append(
                    (int(event.step), float(event.wall_time), float(event.value))
                )
    for tag, values in events.items():
        deduplicated = {(step, wall_time, value) for step, wall_time, value in values}
        events[tag] = sorted(deduplicated)
    return events


def _checkpoint(run_dir: Path) -> tuple[str | None, int | None]:
    checkpoint = run_dir / "checkpoints/last.ckpt"
    if not checkpoint.is_file():
        candidates = sorted((run_dir / "checkpoints").glob("*.ckpt"))
        checkpoint = candidates[-1] if candidates else None
    if checkpoint is None:
        return None, None
    return checkpoint.as_posix(), int(checkpoint.stat().st_size)


def _final_throughput_step(run_dir: Path) -> int:
    path = run_dir / "throughput/throughput.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty throughput CSV: {path}")
    return int(rows[-1]["global_step"])


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _read_tasks(args.tasks)
    run_rows, scalar_rows = [], []
    for task in tasks:
        run_dir = args.run_root / task["run_name"]
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        checkpoint, checkpoint_size = _checkpoint(run_dir)
        final_global_step = _final_throughput_step(run_dir)
        throughput_path = run_dir / "throughput/throughput_summary.json"
        if not throughput_path.is_file():
            raise FileNotFoundError(throughput_path)
        throughput = json.loads(throughput_path.read_text())
        scalars = _scalar_events(run_dir)
        selected_tags = sorted(
            tag
            for tag in scalars
            if tag.startswith("train_gradient_norm/")
            or tag.startswith("train_gradient_cosine/")
            or tag in {
                "train_hvp/gradnorm_multiplier",
                "train_hvp/active_graph_fraction",
                "train_hvp/effective_weight",
            }
            or tag.startswith("train_loss/")
        )
        for tag in selected_tags:
            values = np.asarray([value for _, _, value in scalars[tag]], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            scalar_rows.append(
                {
                    "run_name": task["run_name"],
                    "variant": task["variant"],
                    "seed": int(task["seed"]),
                    "curvature_weight": float(task["curvature_weight"]),
                    "reference_floor": float(task["reference_floor"]),
                    "tag": tag,
                    "count": int(finite.size),
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "first": float(finite[0]),
                    "last": float(finite[-1]),
                }
            )
        run_rows.append(
            {
                **task,
                "checkpoint": checkpoint,
                "checkpoint_size_bytes": checkpoint_size,
                "final_global_step": final_global_step,
                "world_size": int(throughput["world_size"]),
                "total_batches": int(throughput["total_batches"]),
                "total_samples_estimated_global": float(
                    throughput["total_samples_estimated_global"]
                ),
                "training_wall_time_s": float(throughput["wall_time_s"]),
                "mean_samples_per_sec": float(throughput["mean_samples_per_sec"]),
                "peak_gpu_memory_mb": float(throughput["peak_gpu_memory_mb"]),
                "gradient_norm_tag_count": sum(
                    tag.startswith("train_gradient_norm/") for tag in selected_tags
                ),
                "gradient_cosine_tag_count": sum(
                    tag.startswith("train_gradient_cosine/") for tag in selected_tags
                ),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "training_runs.csv", run_rows)
    _write_csv(args.output_dir / "training_scalar_diagnostics.csv", scalar_rows)
    result = {
        "definition": __doc__,
        "tasks": str(args.tasks.resolve()),
        "run_count": len(run_rows),
        "all_checkpoints_at_requested_step": all(
            row["checkpoint"] is not None
            and int(row["final_global_step"]) == int(row["max_steps"])
            for row in run_rows
        ),
        "runs": run_rows,
        "scalar_diagnostics": scalar_rows,
        "test100_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "run_count": result["run_count"],
                "all_checkpoints_at_requested_step": result[
                    "all_checkpoints_at_requested_step"
                ],
                "scalar_rows": len(scalar_rows),
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
