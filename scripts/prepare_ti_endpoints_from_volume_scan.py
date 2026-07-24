#!/usr/bin/env python3
"""Gate zero-pressure scan results and prepare TI endpoint validation jobs."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from scripts.prepare_al108_ti_windows import prepare

ROOT = Path(__file__).resolve().parents[1]


def validated_source(scan_phase: Path) -> tuple[float, Path, dict]:
    result = json.loads((scan_phase / "nvt_volume_scan_result.json").read_text())
    rows = sorted(result["rows"], key=lambda row: row["volume_per_atom_A3"])
    for row in rows:
        metadata = json.loads((scan_phase / row["label"] / "metadata.json").read_text())
        if metadata.get("abacus", {}).get("of_kinetic", "").lower() != "wt":
            raise RuntimeError(f"non-WT volume point found at {scan_phase / row['label']}")
    if result["valid_phase_points"] != len(rows):
        raise RuntimeError(f"not all volume points passed the phase gate in {scan_phase}")
    pressures = [row["pressure_last_half_kbar"]["mean"] for row in rows]
    if not all(left > right for left, right in zip(pressures, pressures[1:])):
        raise RuntimeError(f"pressure is not strictly decreasing with volume in {scan_phase}")
    if min(pressures) > 0.0 or max(pressures) < 0.0:
        raise RuntimeError(f"NVT pressure points do not bracket zero in {scan_phase}")
    volume = result["linear_fit"]["zero_pressure_volume_per_atom_A3"]
    volumes = [row["volume_per_atom_A3"] for row in rows]
    if not min(volumes) <= volume <= max(volumes):
        raise RuntimeError(f"fitted zero-pressure volume lies outside scan range in {scan_phase}")
    source_row = min(rows, key=lambda row: abs(row["volume_per_atom_A3"] - volume))
    return volume, scan_phase / source_row["label"], result


def prepare_phase(
    phase: str,
    scan_phase: Path,
    output: Path,
    pair_model: Path,
    temperature: float,
    steps: int,
    seed: int,
    config: Path,
) -> dict:
    volume, source, scan = validated_source(scan_phase)
    common = {
        "source": str(source),
        "source_frame": "last",
        "phase": phase,
        "temperature": temperature,
        "volume_per_atom": volume,
        "steps": steps,
        "dt": 1.0,
        "csvr_tau": 20.0,
        "dumpfreq": 1,
        "restartfreq": steps,
        "pair_model": pair_model,
        "config": config,
    }
    prepare(Namespace(out=output / phase / "baseline", lambdas=[1.0], seed=seed + 1, **common))
    prepare(Namespace(out=output / phase / "ti", lambdas=[0.0, 1.0], seed=seed, **common))
    return {
        "phase": phase,
        "target_kedf": "wt",
        "zero_pressure_volume_per_atom_A3": volume,
        "source": str(source),
        "pressure_scan": scan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    phases = [
        prepare_phase(
            "solid", args.scan_root / "solid", args.out, args.pair_model,
            args.temperature, args.steps, 31000, args.config,
        ),
        prepare_phase(
            "liquid", args.scan_root / "liquid", args.out, args.pair_model,
            args.temperature, args.steps, 41000, args.config,
        ),
    ]
    manifest = {
        "schema": "wt-pair-ti-endpoints-v2",
        "target_kedf": "wt",
        "temperature_K": args.temperature,
        "steps": args.steps,
        "phases": phases,
    }
    (args.out / "endpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
