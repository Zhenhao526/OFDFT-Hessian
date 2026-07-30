#!/usr/bin/env python3
"""Continue a verified XWM/LKT enthalpy pair while preserving velocities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_KEDFS = {"xwm", "lkt"}
MAX_ABACUS_INTEGER = 2_147_483_647


def load_verified_source(root: Path) -> tuple[dict, dict]:
    manifest = json.loads(
        (root / "confirmation_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (root / "confirmation_summary.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "kedf-volume-confirmation-v1":
        raise ValueError("source confirmation manifest has the wrong schema")
    method = str(manifest.get("target_kedf", "")).lower()
    if method not in SUPPORTED_KEDFS:
        raise ValueError(f"unsupported source KEDF {method!r}")
    if summary.get("status") != "volume_confirmation_verified":
        raise ValueError("source confirmation is not verified")
    if str(summary.get("target_kedf", "")).lower() != method:
        raise ValueError("source manifest and summary use different KEDFs")
    results = {row["phase"]: row for row in summary.get("results", [])}
    if set(results) != {"solid", "liquid"} or any(
        row.get("status") != "passed" for row in results.values()
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
    ranks: int,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if seed < 0 or seed + 1 > MAX_ABACUS_INTEGER:
        raise ValueError("phase seeds must fit a signed 32-bit ABACUS integer")

    manifest, _ = load_verified_source(source_root)
    method = str(manifest["target_kedf"]).lower()
    temperature = float(manifest["temperature_K"])
    target_pressure = float(manifest.get("target_pressure_kbar", 0.0))
    stress_available = bool(manifest.get("stress_available", False))
    source_phases = {row["phase"]: row for row in manifest["phases"]}
    if set(source_phases) != {"solid", "liquid"}:
        raise ValueError("source manifest must contain solid and liquid")

    config = load_json(config_path)
    if str(config.get("of_kinetic", "")).lower() != method:
        raise ValueError("extension config and source use different KEDFs")
    if bool(config.get("cal_stress", False)) != stress_available:
        raise ValueError("extension config changes the source stress mode")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": int(stress_available),
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
            "mpirun_np": ranks,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    phases = []
    out.mkdir(parents=True)
    for index, phase in enumerate(("solid", "liquid")):
        source_spec = source_phases[phase]
        source = load_atom_source(
            source_spec["run"], "last", "Al", include_velocities=True
        )
        if source["atoms"].velocities is None:
            raise RuntimeError(f"missing source velocities for {phase}")
        source_path = Path(str(source["source"])).resolve()
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
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
            job_type=f"{method}_zero_pressure_enthalpy_extension",
            suffix=(
                f"al{atoms.natoms}_{method}_{phase}_"
                f"T{int(temperature):04d}_enthalpy_ext"
            ),
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": method,
                "target_temperature_K": temperature,
                "target_pressure_kbar": target_pressure,
                "volume_per_atom_A3": volume_per_atom,
                "volume_A3": volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_structure_sha256": source_sha256,
                "source_velocities_discarded": False,
                "abacus_init_vel": 1,
                "stress_available": stress_available,
                "steps": steps,
                "csvr_tau_fs": csvr_tau,
                "md_seed": point_config["md_seed"],
                "mpi_ranks": ranks,
            },
        )
        phases.append(
            {
                "phase": phase,
                "run": str(run.resolve()),
                "source": source["source"],
                "source_step": source["step"],
                "source_structure_sha256": source_sha256,
                "source_velocities_discarded": False,
                "volume_per_atom_A3": volume_per_atom,
                "md_seed": point_config["md_seed"],
            }
        )

    extension_manifest = {
        "schema": "kedf-volume-confirmation-v1",
        "extension_schema": "kedf-enthalpy-extension-v1",
        "target_kedf": method,
        "temperature_K": temperature,
        "target_pressure_kbar": target_pressure,
        "stress_available": stress_available,
        "steps": steps,
        "csvr_tau_fs": csvr_tau,
        "parent_confirmation": str(source_root.resolve()),
        "source_velocities_discarded": False,
        "phases": phases,
        "status": "prepared",
    }
    (out / "confirmation_manifest.json").write_text(
        json.dumps(extension_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return extension_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026073100)
    parser.add_argument("--ranks", type=int, default=36)
    args = parser.parse_args()
    result = prepare_extension(
        args.source.resolve(),
        args.out.resolve(),
        args.config.resolve(),
        args.steps,
        args.csvr_tau,
        args.seed,
        args.ranks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
