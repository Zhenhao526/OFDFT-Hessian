#!/usr/bin/env python3
"""Prepare reference-potential two-phase NVT scans from a validated source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-frame", default="last")
    parser.add_argument("--volume-per-atom", type=float, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--dumpfreq", type=int, default=10)
    parser.add_argument("--restartfreq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202607500)
    parser.add_argument("--split", type=float, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(
        args.source,
        args.source_frame,
        "Al",
        include_velocities=True,
    )
    if source["atoms"].velocities is None:
        raise ValueError("two-phase source does not contain velocities")
    config = load_json(args.config)
    element = load_json(ROOT / "config" / "al.json")
    runs = []
    for index, temperature in enumerate(sorted(set(args.temperatures))):
        atoms = scaled_to_volume(
            source["atoms"], args.volume_per_atom * source["atoms"].natoms
        )
        point_config = dict(config)
        point_config.update(
            {
                "calculation": "md",
                "cal_force": 1,
                "cal_stress": 0,
                "md_type": "nvt",
                "md_nstep": args.steps,
                "md_dt": args.dt,
                "md_tfirst": temperature,
                "md_tlast": temperature,
                "md_thermostat": "csvr",
                "md_csvr_tau": args.csvr_tau,
                "md_dumpfreq": args.dumpfreq,
                "md_restartfreq": args.restartfreq,
                "md_seed": args.seed + index,
                "mpirun_np": 1,
            }
        )
        label = f"T{int(round(temperature)):04d}"
        run_dir = out / label
        write_job(
            run_dir,
            atoms,
            element,
            point_config,
            job_type="pair_reference_two_phase_coexistence",
            suffix=f"al{atoms.natoms}_reference_coexist_{label}",
            calculation="md",
            extra_metadata={
                "phase": "two_phase",
                "target_temperature_K": temperature,
                "volume_per_atom_A3": args.volume_per_atom,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": False,
                "pair_model": str(args.pair_model.resolve()),
                "reference_only": True,
                "split": args.split,
            },
        )
        runs.append(
            {
                "label": label,
                "temperature_K": temperature,
                "run_dir": str(run_dir.resolve()),
                "seed": args.seed + index,
            }
        )

    manifest = {
        "schema": "pair-reference-coexistence-scan-v1",
        "natoms": source["atoms"].natoms,
        "source": source["source"],
        "source_step": source["step"],
        "volume_per_atom_A3": args.volume_per_atom,
        "steps": args.steps,
        "dt_fs": args.dt,
        "split": args.split,
        "pair_model": str(args.pair_model.resolve()),
        "runs": runs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
