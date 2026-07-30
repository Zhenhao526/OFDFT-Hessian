#!/usr/bin/env python3
"""Prepare a velocity-preserving KEDF temperature continuation or ramp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_KEDFS = {"xwm", "lkt"}


def validate_protocol(
    start_temperature_K: float,
    end_temperature_K: float,
    steps: int,
    csvr_tau_fs: float,
    ranks: int,
) -> None:
    if start_temperature_K <= 0.0 or end_temperature_K <= 0.0:
        raise ValueError("temperatures must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if csvr_tau_fs <= 0.0:
        raise ValueError("CSVR tau must be positive")
    if ranks <= 0:
        raise ValueError("MPI ranks must be positive")


def prepare(args: argparse.Namespace) -> dict:
    validate_protocol(
        args.start_temperature,
        args.end_temperature,
        args.steps,
        args.csvr_tau,
        args.ranks,
    )
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(
        str(args.source.resolve()),
        "last",
        "Al",
        include_velocities=True,
    )
    atoms = source["atoms"]
    if atoms.natoms != 108 or set(atoms.symbols) != {"Al"}:
        raise ValueError("source is not a 108-atom Al phase")
    if atoms.velocities is None:
        raise RuntimeError("source phase does not contain velocities")
    atoms = scaled_to_volume(atoms, args.volume * atoms.natoms)

    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise ValueError(f"unsupported KEDF {target_kedf!r}")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": int(bool(config.get("cal_stress", False))),
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": 1.0,
            "md_tfirst": args.start_temperature,
            "md_tlast": args.end_temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": args.csvr_tau,
            "md_dumpfreq": 5,
            "md_restartfreq": 100,
            "init_vel": 1,
            "md_seed": args.seed,
            "mpirun_np": args.ranks,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    write_job(
        out,
        atoms,
        element,
        config,
        job_type=f"{target_kedf}_temperature_continuation",
        suffix=(
            f"al108_{target_kedf}_{args.phase}_"
            f"T{int(args.start_temperature):04d}_to_"
            f"T{int(args.end_temperature):04d}"
        ),
        calculation="md",
        extra_metadata={
            "phase": args.phase,
            "target_kedf": target_kedf,
            "start_temperature_K": args.start_temperature,
            "end_temperature_K": args.end_temperature,
            "target_temperature_K": args.end_temperature,
            "volume_per_atom_A3": args.volume,
            "volume_A3": args.volume * atoms.natoms,
            "source": source["source"],
            "source_step": source["step"],
            "source_velocities_discarded": False,
            "steps": args.steps,
            "dt_fs": 1.0,
            "thermostat": "csvr",
            "csvr_tau": args.csvr_tau,
            "md_seed": args.seed,
            "mpi_ranks": args.ranks,
        },
    )
    manifest = {
        "schema": "kedf-temperature-continuation-v1",
        "target_kedf": target_kedf,
        "phase": args.phase,
        "start_temperature_K": args.start_temperature,
        "end_temperature_K": args.end_temperature,
        "volume_per_atom_A3": args.volume,
        "steps": args.steps,
        "csvr_tau_fs": args.csvr_tau,
        "md_seed": args.seed,
        "mpi_ranks": args.ranks,
        "run": str(out),
        "source": source["source"],
        "source_step": source["step"],
        "source_velocities_discarded": False,
    }
    (out / "temperature_continuation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    parser.add_argument("--volume", type=float, required=True)
    parser.add_argument("--start-temperature", type=float, required=True)
    parser.add_argument("--end-temperature", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--csvr-tau", type=float, default=2.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ranks", type=int, default=72)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
