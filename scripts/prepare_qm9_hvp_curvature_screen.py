#!/usr/bin/env python3
"""Freeze the single-seed A-E curvature screen task table without Test100 access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Training protocol must forbid Test100")
    training = protocol["training"]
    seed = int(training["screen_seed"])
    max_steps = int(training["max_steps"])
    rows = []

    def append(variant: str, weight: float, floor: float) -> None:
        weight_name = f"{weight:.0e}".replace("-", "m").replace("+", "p")
        floor_name = f"{floor:.0e}".replace("-", "m").replace("+", "p")
        rows.append(
            {
                "task_index": len(rows),
                "variant": variant,
                "curvature_weight": f"{weight:.12g}",
                "reference_floor": f"{floor:.12g}",
                "seed": seed,
                "max_steps": max_steps,
                "run_name": (
                    f"qm9_hvp_curvature_v1_{variant}_w{weight_name}_f{floor_name}_"
                    f"seed{seed}_s{max_steps}"
                ),
            }
        )

    default_floor = float(training["reference_scale_floors"][0])
    append("A", 0.0, default_floor)
    append("B", float(protocol["variants"]["B"]["secant_weight"]), default_floor)
    for variant in ("C", "D", "E"):
        for weight in training["curvature_weights"]:
            for floor in training["reference_scale_floors"]:
                append(variant, float(weight), float(floor))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "task_count": len(rows),
        "variant_counts": {
            variant: sum(row["variant"] == variant for row in rows)
            for variant in ("A", "B", "C", "D", "E")
        },
        "screen_seed": seed,
        "test100_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
