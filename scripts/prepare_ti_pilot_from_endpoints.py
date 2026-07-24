#!/usr/bin/env python3
"""Prepare phase TI pilot grids after both endpoint validations pass."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from scripts.prepare_al108_ti_windows import prepare

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    endpoint_manifest = json.loads((args.endpoint_root / "endpoint_manifest.json").read_text())
    phase_inputs = {item["phase"]: item for item in endpoint_manifest["phases"]}
    prepared = []
    for phase_index, phase in enumerate(("solid", "liquid")):
        validation = json.loads((args.endpoint_root / phase / "endpoint_validation.json").read_text())
        if validation["status"] != "endpoint_validation_passed":
            raise RuntimeError(f"{phase} endpoint validation did not pass")
        item = phase_inputs[phase]
        prepare(
            Namespace(
                out=args.out / phase,
                source=item["source"],
                source_frame="last",
                phase=phase,
                temperature=endpoint_manifest["temperature_K"],
                volume_per_atom=item["zero_pressure_volume_per_atom_A3"],
                lambdas=LAMBDAS,
                steps=args.steps,
                dt=1.0,
                csvr_tau=20.0,
                dumpfreq=5,
                restartfreq=args.steps,
                seed=51000 + 1000 * phase_index,
                pair_model=args.pair_model,
                config=args.config,
            )
        )
        prepared.append(
            {
                "phase": phase,
                "target_kedf": "wt",
                "source": item["source"],
                "zero_pressure_volume_per_atom_A3": item["zero_pressure_volume_per_atom_A3"],
            }
        )
    manifest = {
        "schema": "wt-pair-ti-pilot-v2",
        "target_kedf": "wt",
        "temperature_K": endpoint_manifest["temperature_K"],
        "steps": args.steps,
        "lambdas": LAMBDAS,
        "phases": prepared,
    }
    (args.out / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
