#!/usr/bin/env python3
"""Prepare and summarize finite-difference pressures for MD snapshots."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.check_kedf_pressure_finite_difference import (
    analyze as analyze_snapshot,
    prepare as prepare_snapshot,
)


def sample_label(frame_index: int) -> str:
    return f"frame_{frame_index:04d}"


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    samples = []
    for frame_index in args.frame_indices:
        label = sample_label(frame_index)
        sample_root = out / label
        prepare_snapshot(
            argparse.Namespace(
                out=sample_root,
                source=args.source,
                source_frame=str(frame_index),
                config=args.config,
                phase=args.phase,
                temperature=args.temperature,
                linear_strain=args.linear_strain,
                volume_per_atom=None,
                ranks=args.ranks,
            )
        )
        sample_manifest = json.loads(
            (sample_root / "manifest.json").read_text()
        )
        samples.append(
            {
                "label": label,
                "frame_index": frame_index,
                "source_step": sample_manifest["source_step"],
                "run": str(sample_root),
            }
        )
    manifest = {
        "schema": "kedf-pressure-samples-v1",
        "phase": args.phase,
        "target_temperature_K": args.temperature,
        "source": str(Path(args.source).resolve()),
        "frame_indices": args.frame_indices,
        "linear_strain": args.linear_strain,
        "ranks_per_scf": args.ranks,
        "samples": samples,
        "status": "prepared",
    }
    (out / "samples_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def summarize_pressures(values: list[float]) -> dict:
    if not values:
        return {}
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": sd,
        "standard_error": sd / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads((root / "samples_manifest.json").read_text())
    samples = []
    for item in manifest["samples"]:
        sample_root = root / item["label"]
        analyze_snapshot(argparse.Namespace(root=sample_root))
        result = json.loads(
            (sample_root / "pressure_fd_result.json").read_text()
        )
        samples.append(
            {
                **item,
                "volume_per_atom_A3": result[
                    "base_volume_per_atom_A3"
                ],
                "static_pressure_kbar": result[
                    "finite_difference_static_pressure_kbar"
                ],
                "ideal_ionic_pressure_kbar": result[
                    "ideal_ionic_pressure_kbar"
                ],
                "estimated_total_pressure_kbar": result[
                    "estimated_total_pressure_kbar"
                ],
            }
        )
    pressure = summarize_pressures(
        [item["estimated_total_pressure_kbar"] for item in samples]
    )
    checks = {
        "at_least_five_snapshots": len(samples) >= 5,
        "mean_pressure_within_2_5_kbar": (
            bool(pressure) and abs(float(pressure["mean"])) <= 2.5
        ),
        "pressure_standard_error_le_2_5_kbar": (
            bool(pressure)
            and float(pressure["standard_error"]) <= 2.5
        ),
    }
    summary = {
        **manifest,
        "samples": samples,
        "pressure_kbar": pressure,
        "checks": checks,
        "status": (
            "snapshot_pressure_verified"
            if all(checks.values())
            else "snapshot_pressure_not_verified"
        ),
    }
    (root / "pressure_samples_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prep.add_argument("--temperature", type=float, required=True)
    prep.add_argument("--frame-indices", type=int, nargs="+", required=True)
    prep.add_argument("--linear-strain", type=float, default=0.001)
    prep.add_argument("--ranks", type=int, default=4)
    prep.set_defaults(func=prepare)
    gate = subparsers.add_parser("analyze")
    gate.add_argument("root", type=Path)
    gate.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
