#!/usr/bin/env python3
"""Prepare one WT phase continuation while preserving source velocities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]


def prepare(
    source_path: Path,
    out: Path,
    config_path: Path,
    phase: str,
    volume_per_atom: float,
    temperature: float,
    steps: int,
    csvr_tau: float,
    seed: int,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(
        str(source_path), "last", "Al", include_velocities=True
    )
    atoms = source["atoms"]
    if atoms.natoms != 108 or set(atoms.symbols) != {"Al"}:
        raise ValueError("source is not a 108-atom Al phase")
    if atoms.velocities is None:
        raise RuntimeError("source phase does not contain velocities")
    atoms = scaled_to_volume(atoms, volume_per_atom * atoms.natoms)

    config = load_json(config_path)
    if str(config.get("of_kinetic", "")).lower() != "wt":
        raise ValueError("phase continuation config must use WT")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 1,
            "md_type": "nvt",
            "md_nstep": steps,
            "md_dt": 1.0,
            "md_tfirst": temperature,
            "md_tlast": temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": csvr_tau,
            "md_dumpfreq": 5,
            "md_restartfreq": 100,
            "init_vel": 1,
            "md_seed": seed,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    write_job(
        out,
        atoms,
        element,
        config,
        job_type="wt_phase_continuation",
        suffix=f"al108_{phase}_T{int(temperature):04d}_continuation",
        calculation="md",
        extra_metadata={
            "phase": phase,
            "target_kedf": "wt",
            "target_temperature_K": temperature,
            "target_pressure_kbar": 0.0,
            "volume_per_atom_A3": volume_per_atom,
            "volume_A3": volume_per_atom * atoms.natoms,
            "source": source["source"],
            "source_step": source["step"],
            "source_velocities_discarded": False,
            "md_seed": seed,
            "steps": steps,
            "dt_fs": 1.0,
            "thermostat": "csvr",
            "csvr_tau": csvr_tau,
        },
    )
    manifest = {
        "schema": "wt-phase-continuation-v1",
        "target_kedf": "wt",
        "phase": phase,
        "temperature_K": temperature,
        "target_pressure_kbar": 0.0,
        "volume_per_atom_A3": volume_per_atom,
        "steps": steps,
        "csvr_tau_fs": csvr_tau,
        "md_seed": seed,
        "run": str(out.resolve()),
        "source": source["source"],
        "source_step": source["step"],
        "source_velocities_discarded": False,
    }
    (out / "phase_continuation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    parser.add_argument("--volume", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--csvr-tau", type=float, default=2.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    result = prepare(
        args.source.resolve(),
        args.out.resolve(),
        args.config.resolve(),
        args.phase,
        args.volume,
        args.temperature,
        args.steps,
        args.csvr_tau,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
