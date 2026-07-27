#!/usr/bin/env python3
"""Prepare and gate solid/liquid finite-temperature volume confirmations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_KEDFS = {"xwm", "lkt"}


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise ValueError(f"unsupported KEDF: {target_kedf}")
    stress_available = bool(config.get("cal_stress", False))
    common = {
        **config,
        "calculation": "md",
        "cal_force": 1,
        "cal_stress": int(stress_available),
        "md_type": "nvt",
        "md_nstep": args.steps,
        "md_dt": 1.0,
        "md_tfirst": args.temperature,
        "md_tlast": args.temperature,
        "md_thermostat": "csvr",
        "md_csvr_tau": args.csvr_tau,
        "md_dumpfreq": 5,
        "md_restartfreq": 100,
        "init_vel": 0,
        "mpirun_np": args.ranks,
    }
    element = load_json(ROOT / "config" / "al.json")
    phases = []
    requested_phases = set(
        getattr(args, "phases", None) or ("solid", "liquid")
    )
    specifications = tuple(
        item
        for item in (
        ("solid", args.solid_source, args.solid_volume),
        ("liquid", args.liquid_source, args.liquid_volume),
        )
        if item[0] in requested_phases
    )
    if not specifications:
        raise ValueError("at least one phase must be requested")
    out.mkdir(parents=True)
    for phase_index, (phase, source_path, volume_per_atom) in enumerate(
        specifications
    ):
        source = load_atom_source(
            source_path, "last", "Al", include_velocities=False
        )
        atoms = scaled_to_volume(
            source["atoms"], float(volume_per_atom) * source["atoms"].natoms
        )
        point_config = {
            **common,
            "md_seed": args.seed + phase_index,
        }
        run = out / phase
        write_job(
            run,
            atoms,
            element,
            point_config,
            job_type=f"{target_kedf}_finite_temperature_volume_confirmation",
            suffix=f"al108_{target_kedf}_{phase}_T{int(args.temperature):04d}_volume_confirm",
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": target_kedf,
                "target_temperature_K": args.temperature,
                "target_pressure_kbar": 0.0,
                "volume_per_atom_A3": volume_per_atom,
                "volume_A3": volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": True,
                "abacus_init_vel": 0,
                "stress_available": stress_available,
                "steps": args.steps,
                "csvr_tau_fs": args.csvr_tau,
                "md_seed": point_config["md_seed"],
                "mpi_ranks": args.ranks,
            },
        )
        phases.append(
            {
                "phase": phase,
                "run": str(run),
                "source": source["source"],
                "source_step": source["step"],
                "volume_per_atom_A3": volume_per_atom,
                "md_seed": point_config["md_seed"],
            }
        )
    manifest = {
        "schema": "kedf-volume-confirmation-v1",
        "target_kedf": target_kedf,
        "temperature_K": args.temperature,
        "target_pressure_kbar": 0.0,
        "stress_available": stress_available,
        "steps": args.steps,
        "csvr_tau_fs": args.csvr_tau,
        "phases": phases,
        "status": "prepared",
    }
    (out / "confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads(
        (root / "confirmation_manifest.json").read_text()
    )
    results = []
    for item in manifest["phases"]:
        phase = item["phase"]
        run = root / phase
        phase_result = analyze_phase(run, phase, thermalized_initial=True)
        (run / "phase_analysis.json").write_text(
            json.dumps(phase_result, indent=2, sort_keys=True) + "\n"
        )
        logs = sorted(run.glob("OUT.*/running_md.log"))
        rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
        late = rows[len(rows) // 2 :]
        temperature = series_stats(
            [row["temperature_K"] for row in late]
        )
        pressure_values = [
            row["pressure_kbar"] for row in late if "pressure_kbar" in row
        ]
        pressure = series_stats(pressure_values)
        temperature_error = abs(
            float(temperature.get("mean", math.inf))
            - float(manifest["temperature_K"])
        )
        pressure_error = abs(float(pressure.get("mean", math.inf)))
        nearest = phase_result.get("trajectory", {}).get(
            "nearest_neighbor_A", 0.0
        )
        checks = {
            "reached_requested_step": max_step >= int(manifest["steps"]),
            "phase_verified": phase_result.get("status")
            == f"{phase}_verified",
            "temperature_mean_within_20_K": temperature_error <= 20.0,
            "nearest_neighbor_gt_2_A": float(nearest) > 2.0,
            "pressure_mean_within_2_5_kbar_if_available": (
                pressure_error <= 2.5
                if manifest["stress_available"]
                else True
            ),
        }
        result = {
            **item,
            "max_step": max_step,
            "phase_status": phase_result.get("status"),
            "nearest_neighbor_A": nearest,
            "temperature_last_half_K": temperature,
            "temperature_target_error_K": temperature_error,
            "pressure_last_half_kbar": pressure,
            "pressure_target_error_kbar": (
                pressure_error if pressure_values else None
            ),
            "requires_posthoc_pressure": not manifest["stress_available"],
            "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
        }
        (run / "confirmation_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        results.append(result)
    summary = {
        **manifest,
        "results": results,
        "status": (
            "volume_confirmation_verified"
            if all(item["status"] == "passed" for item in results)
            else "volume_confirmation_failed"
        ),
    }
    (root / "confirmation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--solid-source", required=True)
    prep.add_argument("--liquid-source", required=True)
    prep.add_argument("--solid-volume", type=float, required=True)
    prep.add_argument("--liquid-volume", type=float, required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--temperature", type=float, default=975.0)
    prep.add_argument("--steps", type=int, default=300)
    prep.add_argument("--csvr-tau", type=float, default=5.0)
    prep.add_argument("--seed", type=int, default=2026072710)
    prep.add_argument("--ranks", type=int, default=36)
    prep.add_argument(
        "--phases",
        nargs="+",
        choices=("solid", "liquid"),
        default=("solid", "liquid"),
    )
    prep.set_defaults(func=prepare)
    gate = subparsers.add_parser("analyze")
    gate.add_argument("root", type=Path)
    gate.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
