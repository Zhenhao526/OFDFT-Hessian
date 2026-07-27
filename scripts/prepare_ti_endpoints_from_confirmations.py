#!/usr/bin/env python3
"""Prepare KEDF-to-pair endpoint checks from gated zero-pressure confirmations."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from mpn_melting.abacus_input import load_json
from scripts.prepare_al108_ti_windows import SUPPORTED_KEDFS, prepare

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    final_summary = (
        args.confirmation_root
        / "zero_pressure_confirmation_summary.json"
    )
    summary_path = (
        final_summary
        if final_summary.exists()
        else args.confirmation_root / "confirmation_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "all_confirmations_passed":
        raise RuntimeError("zero-pressure confirmations did not pass")
    target_kedf = str(summary.get("target_kedf", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise RuntimeError(f"unsupported confirmation target {target_kedf!r}")
    config_kedf = str(load_json(args.config).get("of_kinetic", "")).lower()
    if config_kedf != target_kedf:
        raise RuntimeError(
            f"confirmation target {target_kedf!r} does not match config "
            f"target {config_kedf!r}"
        )

    phases = []
    for phase_index, item in enumerate(summary["phase_results"]):
        phase = item["phase"]
        if item["status"] != "confirmation_passed":
            raise RuntimeError(f"{phase} confirmation did not pass")
        common = {
            "source": item["run"],
            "source_frame": "last",
            "phase": phase,
            "temperature": summary["temperature_K"],
            "volume_per_atom": item["volume_per_atom_A3"],
            "steps": args.steps,
            "dt": 1.0,
            "csvr_tau": 20.0,
            "dumpfreq": 1,
            "restartfreq": args.steps,
            "pair_model": args.pair_model,
            "config": args.config,
            "ranks": None,
        }
        seed = 81000 + 1000 * phase_index
        prepare(Namespace(out=args.out / phase / "baseline", lambdas=[1.0], seed=seed + 1, **common))
        prepare(Namespace(out=args.out / phase / "ti", lambdas=[0.0, 1.0], seed=seed, **common))
        phases.append(
            {
                "phase": phase,
                "target_kedf": target_kedf,
                "source": item["run"],
                "zero_pressure_volume_per_atom_A3": item["volume_per_atom_A3"],
            }
        )
    manifest = {
        "schema": "kedf-pair-ti-endpoints-from-confirmations-v1",
        "target_kedf": target_kedf,
        "temperature_K": summary["temperature_K"],
        "steps": args.steps,
        "phases": phases,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "endpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
