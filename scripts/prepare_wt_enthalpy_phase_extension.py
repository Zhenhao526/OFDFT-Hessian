#!/usr/bin/env python3
"""Continue one phase from a verified WT enthalpy confirmation pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume
from scripts.prepare_wt_enthalpy_extension import load_verified_source

ROOT = Path(__file__).resolve().parents[1]


def prepare_phase_extension(
    source_root: Path,
    out: Path,
    config_path: Path,
    phase: str,
    steps: int,
    csvr_tau: float,
    seed: int,
) -> dict:
    if phase not in {"solid", "liquid"}:
        raise ValueError(f"unsupported phase: {phase}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    manifest, _ = load_verified_source(source_root)
    source_phases = {row["phase"]: row for row in manifest["phases"]}
    source_spec = source_phases[phase]
    temperature = float(manifest["temperature_K"])
    pressure = float(manifest.get("target_pressure_kbar", 0.0))

    source = load_atom_source(
        source_spec["run"], "last", "Al", include_velocities=True
    )
    if source["atoms"].velocities is None:
        raise RuntimeError(f"missing source velocities for {phase}")
    volume_per_atom = float(source_spec["volume_per_atom_A3"])
    atoms = scaled_to_volume(
        source["atoms"], volume_per_atom * source["atoms"].natoms
    )

    config = load_json(config_path)
    if str(config.get("of_kinetic", "")).lower() != "wt":
        raise ValueError("enthalpy extension config must use WT")
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
            "mpirun_np": 18,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    run = out / phase
    write_job(
        run,
        atoms,
        element,
        config,
        job_type="wt_zero_pressure_enthalpy_phase_extension",
        suffix=f"al{atoms.natoms}_{phase}_T{int(temperature):04d}_enthalpy_ext",
        calculation="md",
        extra_metadata={
            "phase": phase,
            "target_kedf": "wt",
            "target_temperature_K": temperature,
            "target_pressure_kbar": pressure,
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
    phase_spec = {
        "phase": phase,
        "run": str(run.resolve()),
        "source": source["source"],
        "source_step": source["step"],
        "volume_per_atom_A3": volume_per_atom,
        "md_seed": seed,
    }
    extension_manifest = {
        "schema": "wt-zero-pressure-confirmation-v1",
        "extension_schema": "wt-zero-pressure-enthalpy-phase-extension-v1",
        "target_kedf": "wt",
        "temperature_K": temperature,
        "target_pressure_kbar": pressure,
        "steps": steps,
        "csvr_tau_fs": csvr_tau,
        "parent_confirmation": str(source_root.resolve()),
        "source_velocities_discarded": False,
        "phases": [phase_spec],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmation_manifest.json").write_text(
        json.dumps(extension_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return extension_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026122050)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_node01_local_cpu18.json",
    )
    args = parser.parse_args()
    result = prepare_phase_extension(
        args.source.resolve(),
        args.out.resolve(),
        args.config.resolve(),
        args.phase,
        args.steps,
        args.csvr_tau,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
