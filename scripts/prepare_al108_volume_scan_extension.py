#!/usr/bin/env python3
"""Continue every WT NVT volume-scan point from its thermalized final frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--dumpfreq", type=int, default=5)
    parser.add_argument("--restartfreq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf != "wt":
        raise RuntimeError(f"volume continuation requires WT, got {target_kedf!r}")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 1,
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": args.dt,
            "md_tfirst": args.temperature,
            "md_tlast": args.temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": args.csvr_tau,
            "md_dumpfreq": args.dumpfreq,
            "md_restartfreq": args.restartfreq,
            "init_vel": 1,
            "mpirun_np": 12,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    for phase_index, phase in enumerate(("solid", "liquid")):
        parent_phase = args.parent / phase
        parent_manifest = json.loads((parent_phase / "manifest.json").read_text())
        points = []
        for point_index, point in enumerate(parent_manifest["points"]):
            source_dir = parent_phase / point["label"]
            source = load_atom_source(source_dir.as_posix(), "last", "Al", include_velocities=True)
            atoms = source["atoms"]
            if atoms.velocities is None:
                raise RuntimeError(f"missing source velocities at {source_dir}")
            point_config = dict(config)
            point_config["md_seed"] = args.seed + 1000 * phase_index + point_index
            write_job(
                out / phase / point["label"],
                atoms,
                element,
                point_config,
                job_type="wt_nvt_zero_pressure_volume_scan_continuation",
                suffix=f"al108_{phase}_T{int(args.temperature):04d}_{point['label']}_continuation",
                calculation="md",
                extra_metadata={
                    "phase": phase,
                    "target_kedf": "wt",
                    "target_temperature_K": args.temperature,
                    "volume_A3": point["volume_A3"],
                    "volume_per_atom_A3": point["volume_per_atom_A3"],
                    "source": source["source"],
                    "source_step": source["step"],
                    "source_velocities_discarded": False,
                    "steps": args.steps,
                    "dt_fs": args.dt,
                    "thermostat": "csvr",
                    "csvr_tau": args.csvr_tau,
                },
            )
            points.append(point)
        manifest = {
            "phase": phase,
            "target_kedf": "wt",
            "target_temperature_K": args.temperature,
            "natoms": parent_manifest["natoms"],
            "parent_scan": str(parent_phase.resolve()),
            "thermalized_initial": True,
            "steps": args.steps,
            "points": points,
        }
        phase_out = out / phase
        phase_out.mkdir(parents=True, exist_ok=True)
        (phase_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
