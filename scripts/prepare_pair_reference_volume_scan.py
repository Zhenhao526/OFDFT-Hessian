#!/usr/bin/env python3
"""Prepare small fixed-volume pair-reference MD scans for zero-pressure volumes."""

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
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-frame", default="last")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--volumes-per-atom", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--dumpfreq", type=int, default=10)
    parser.add_argument("--restartfreq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202607700)
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
        args.source, args.source_frame, "Al", include_velocities=False
    )
    config = load_json(args.config)
    element = load_json(ROOT / "config" / "al.json")
    points = []
    for index, volume_per_atom in enumerate(sorted(set(args.volumes_per_atom))):
        atoms = scaled_to_volume(
            source["atoms"], volume_per_atom * source["atoms"].natoms
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
                "md_tfirst": args.temperature,
                "md_tlast": args.temperature,
                "md_thermostat": "csvr",
                "md_csvr_tau": args.csvr_tau,
                "md_dumpfreq": args.dumpfreq,
                "md_restartfreq": args.restartfreq,
                "md_seed": args.seed + index,
                "mpirun_np": 1,
            }
        )
        label = f"vpa_{volume_per_atom:.3f}".replace(".", "p")
        run_dir = out / label
        write_job(
            run_dir,
            atoms,
            element,
            point_config,
            job_type="pair_reference_zero_pressure_volume_scan",
            suffix=f"al{atoms.natoms}_{args.phase}_reference_{label}",
            calculation="md",
            extra_metadata={
                "phase": args.phase,
                "target_temperature_K": args.temperature,
                "volume_per_atom_A3": volume_per_atom,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": True,
                "pair_model": str(args.pair_model.resolve()),
                "reference_only": True,
            },
        )
        points.append(
            {
                "label": label,
                "volume_per_atom_A3": volume_per_atom,
                "run_dir": str(run_dir.resolve()),
                "seed": args.seed + index,
            }
        )

    manifest = {
        "schema": "pair-reference-volume-scan-v1",
        "phase": args.phase,
        "natoms": source["atoms"].natoms,
        "target_temperature_K": args.temperature,
        "source": source["source"],
        "source_step": source["step"],
        "steps": args.steps,
        "pair_model": str(args.pair_model.resolve()),
        "points": points,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
