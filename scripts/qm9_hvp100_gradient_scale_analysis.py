#!/usr/bin/env python3
"""Collect per-loss parameter-gradient scales from the frozen HVP100 runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PATTERN = re.compile(
    r"qm9_hvp100_(?P<name>.+)_seed(?P<seed>\d+)_s600_(?:gated25v2_|gated25_)?20260716$"
)


def _variant(name: str) -> tuple[str, float]:
    mapping = {
        "force": ("A", 0.0),
        "force_secant": ("B", 0.0),
        "force_hvp_w1e5": ("C", 1e-5),
        "force_hvp_w1e4": ("C", 1e-4),
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
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict:
    rows: list[dict] = []
    for run_dir in sorted(args.runs_dir.glob(args.run_glob)):
        match = PATTERN.match(run_dir.name)
        if match is None:
            continue
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        accumulator = EventAccumulator(str(max(event_files, key=lambda p: p.stat().st_size)))
        accumulator.Reload()
        variant, weight = _variant(match.group("name"))
        for tag in accumulator.Tags().get("scalars", []):
            if not tag.startswith("train_gradient_norm/"):
                continue
            loss = tag.split("/", 1)[1]
            # Lightning may emit the same sparse snapshot on adjacent logging steps.
            unique: dict[float, float] = {}
            for event in accumulator.Scalars(tag):
                unique.setdefault(round(float(event.value), 14), float(event.step))
            for value, step in unique.items():
                rows.append(
                    {
                        "run": run_dir.name,
                        "variant": variant,
                        "hvp_weight": weight,
                        "seed": int(match.group("seed")),
                        "loss": loss,
                        "step": int(step),
                        "weighted_parameter_gradient_norm": value,
                    }
                )
    if not rows:
        raise RuntimeError("No gradient-norm TensorBoard scalars found")

    groups = []
    keys = sorted({(r["variant"], r["hvp_weight"], r["loss"]) for r in rows})
    for variant, weight, loss in keys:
        values = [
            r["weighted_parameter_gradient_norm"]
            for r in rows
            if (r["variant"], r["hvp_weight"], r["loss"]) == (variant, weight, loss)
            and r["weighted_parameter_gradient_norm"] > 0.0
        ]
        if not values:
            continue
        groups.append(
            {
                "variant": variant,
                "hvp_weight": weight,
                "loss": loss,
                "observations": len(values),
                "mean_gradient_norm": float(np.mean(values)),
                "median_gradient_norm": float(np.median(values)),
                "min_gradient_norm": float(np.min(values)),
                "max_gradient_norm": float(np.max(values)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "gradient_norm_observations.csv", rows)
    _write_csv(args.output_dir / "gradient_norm_groups.csv", groups)
    result = {
        "definition": "Sparse norms of each weighted loss contribution with respect to trainable parameters.",
        "test_accessed": False,
        "runs": len({r["run"] for r in rows}),
        "observations": rows,
        "groups": groups,
    }
    (args.output_dir / "gradient_norm_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"runs": result["runs"], "groups": groups}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-glob", default="qm9_hvp100_*_s600_gated25v2_20260716")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
