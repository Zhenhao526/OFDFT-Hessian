#!/usr/bin/env python3
"""Continue one verified WT solid/liquid confirmation pair for more sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]


def load_verified_source(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "confirmation_manifest.json").read_text())
    summary = json.loads((root / "confirmation_summary.json").read_text())
    if manifest.get("schema") != "wt-zero-pressure-confirmation-v1":
        raise ValueError("source confirmation manifest has the wrong schema")
    if str(manifest.get("target_kedf", "")).lower() != "wt":
        raise ValueError("source confirmation does not use WT")
    if summary.get("status") != "all_confirmations_passed":
        raise ValueError("source confirmation is not verified")
    results = {row["phase"]: row for row in summary.get("phase_results", [])}
    if set(results) != {"solid", "liquid"} or any(
        row.get("status") != "confirmation_passed" for row in results.values()
    ):
        raise ValueError("source solid and liquid confirmations must both pass")
    return manifest, summary


def prepare_extension(
    source_root: Path,
    out: Path,
    config_path: Path,
    steps: int,
    csvr_tau: float,
    seed: int,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    manifest, _ = load_verified_source(source_root)
    temperature = float(manifest["temperature_K"])
    target_pressure = float(manifest.get("target_pressure_kbar", 0.0))
    source_phases = {row["phase"]: row for row in manifest["phases"]}
    if set(source_phases) != {"solid", "liquid"}:
        raise ValueError("source manifest must contain solid and liquid")

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
            "mpirun_np": 36,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    phases = []
    for index, phase in enumerate(("solid", "liquid")):
        source_spec = source_phases[phase]
        source = load_atom_source(
            source_spec["run"], "last", "Al", include_velocities=True
        )
        if source["atoms"].velocities is None:
            raise RuntimeError(f"missing source velocities for {phase}")
        volume_per_atom = float(source_spec["volume_per_atom_A3"])
        atoms = scaled_to_volume(
            source["atoms"], volume_per_atom * source["atoms"].natoms
        )
        point_config = dict(config)
        point_config["md_seed"] = seed + index
        run = out / phase
        write_job(
            run,
            atoms,
            element,
            point_config,
            job_type="wt_zero_pressure_enthalpy_extension",
            suffix=f"al{atoms.natoms}_{phase}_T{int(temperature):04d}_enthalpy_ext",
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": "wt",
                "target_temperature_K": temperature,
                "target_pressure_kbar": target_pressure,
                "volume_per_atom_A3": volume_per_atom,
                "volume_A3": volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": False,
                "md_seed": point_config["md_seed"],
                "steps": steps,
                "dt_fs": 1.0,
                "thermostat": "csvr",
                "csvr_tau": csvr_tau,
            },
        )
        phases.append(
            {
                "phase": phase,
                "run": str(run.resolve()),
                "source": source["source"],
                "source_step": source["step"],
                "volume_per_atom_A3": volume_per_atom,
                "md_seed": point_config["md_seed"],
            }
        )
    extension_manifest = {
        "schema": "wt-zero-pressure-confirmation-v1",
        "extension_schema": "wt-zero-pressure-enthalpy-extension-v1",
        "target_kedf": "wt",
        "temperature_K": temperature,
        "target_pressure_kbar": target_pressure,
        "steps": steps,
        "csvr_tau_fs": csvr_tau,
        "parent_confirmation": str(source_root.resolve()),
        "source_velocities_discarded": False,
        "phases": phases,
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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026072601)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    result = prepare_extension(
        args.source.resolve(),
        args.out.resolve(),
        args.config.resolve(),
        args.steps,
        args.csvr_tau,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
