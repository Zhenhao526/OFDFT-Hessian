#!/usr/bin/env python3
"""Aggregate wall time, throughput, timing, memory, and HVP activation for HVP100 runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PATTERN = re.compile(
    r"qm9_hvp100_(?P<name>.+)_seed(?P<seed>\d+)_s600_gated25v2_20260716$"
)


def _variant(name: str) -> tuple[str, float]:
    mapping = {
        "force": ("A", 0.0), "force_secant": ("B", 0.0),
        "force_hvp_w1e5": ("C", 1e-5), "force_hvp_w1e4": ("C", 1e-4),
        "force_hvp_w1e3": ("C", 1e-3),
        "force_secant_hvp_w1e5": ("D", 1e-5),
        "force_secant_hvp_w1e4": ("D", 1e-4),
        "force_secant_hvp_w1e3": ("D", 1e-3),
    }
    return mapping[name]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _event_mean(run_dir: Path, tag: str) -> float | None:
    files = list(run_dir.glob("events.out.tfevents.*"))
    if not files:
        return None
    accumulator = EventAccumulator(str(max(files, key=lambda p: p.stat().st_size)))
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None
    values = [float(event.value) for event in accumulator.Scalars(tag)]
    return float(np.mean(values)) if values else None


def analyze(args: argparse.Namespace) -> dict:
    rows = []
    timing_fields = (
        "data_loading_s_per_batch", "net_forward_s_per_batch",
        "density_gradient_autograd_s_per_batch", "force_autograd_s_per_batch",
        "hvp_autograd_s_per_batch", "backward_s_per_batch", "optimizer_s_per_batch",
    )
    for run_dir in sorted(args.runs_dir.glob("qm9_hvp100_*_s600_gated25v2_20260716")):
        match = PATTERN.match(run_dir.name)
        summary_path = run_dir / "throughput" / "throughput_summary.json"
        csv_path = run_dir / "throughput" / "throughput.csv"
        if match is None or not summary_path.exists() or not csv_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        with csv_path.open() as handle:
            intervals = list(csv.DictReader(handle))
        variant, weight = _variant(match.group("name"))
        row = {
            "run": run_dir.name,
            "variant": variant,
            "hvp_weight": weight,
            "seed": int(match.group("seed")),
            "wall_time_s": float(summary["wall_time_s"]),
            "mean_samples_per_sec": float(summary["mean_samples_per_sec"]),
            "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
            "hvp_active_batch_fraction": _event_mean(run_dir, "train_hvp/active_batch"),
            "hvp_active_graph_fraction": _event_mean(
                run_dir, "train_hvp/active_graph_fraction"
            ),
        }
        for field in timing_fields:
            values = [float(item.get(field) or 0.0) for item in intervals]
            row[f"mean_{field}"] = float(np.mean(values))
        rows.append(row)
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 complete runs, found {len(rows)}")
    baseline = {row["seed"]: row for row in rows if row["variant"] == "A"}
    for row in rows:
        reference = baseline[row["seed"]]
        row["wall_time_ratio_vs_A"] = row["wall_time_s"] / reference["wall_time_s"]
        row["memory_ratio_vs_A"] = row["peak_gpu_memory_mb"] / reference["peak_gpu_memory_mb"]
    groups = []
    for key in sorted({(row["variant"], row["hvp_weight"]) for row in rows}):
        members = [row for row in rows if (row["variant"], row["hvp_weight"]) == key]
        group = {"variant": key[0], "hvp_weight": key[1], "seeds": len(members)}
        for metric in (
            "wall_time_s", "mean_samples_per_sec", "peak_gpu_memory_mb",
            "wall_time_ratio_vs_A", "memory_ratio_vs_A",
            "hvp_active_batch_fraction", "hvp_active_graph_fraction",
            *[f"mean_{field}" for field in timing_fields],
        ):
            values = [row[metric] for row in members if row[metric] is not None]
            group[f"mean_{metric}"] = float(np.mean(values)) if values else None
        groups.append(group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "training_cost_runs.csv", rows)
    _write_csv(args.output_dir / "training_cost_groups.csv", groups)
    result = {
        "definition": "One-GPU equal-optimizer-step HVP100 training cost; Test100 not accessed.",
        "test_accessed": False,
        "runs": rows,
        "groups": groups,
    }
    (args.output_dir / "training_cost_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"groups": groups}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
