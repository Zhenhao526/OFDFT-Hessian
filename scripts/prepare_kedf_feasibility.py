#!/usr/bin/env python3
"""Prepare and gate short XWM/LKT solid and liquid feasibility trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_KEDFS = {"xwm", "lkt"}


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    if target_kedf not in SUPPORTED_KEDFS:
        raise ValueError(f"expected one of {sorted(SUPPORTED_KEDFS)}, got {target_kedf!r}")
    stress_available = target_kedf == "lkt"
    if bool(config.get("cal_stress")) != stress_available:
        raise ValueError(
            f"{target_kedf} cal_stress must be {int(stress_available)} for this ABACUS build"
        )
    out.mkdir(parents=True)

    element = load_json(ROOT / "config" / "al.json")
    common = {
        **config,
        "calculation": "md",
        "cal_force": 1,
        "cal_stress": int(stress_available),
        "md_type": "nvt",
        "md_nstep": args.steps,
        "md_dt": args.dt,
        "md_tfirst": args.temperature,
        "md_tlast": args.temperature,
        "md_thermostat": "csvr",
        "md_csvr_tau": args.csvr_tau,
        "md_dumpfreq": 5,
        "md_restartfreq": 100,
        "init_vel": 0,
        "mpirun_np": args.ranks,
    }
    phases = []
    for phase_index, (phase, source_path) in enumerate(
        (("solid", args.solid_source), ("liquid", args.liquid_source))
    ):
        source = load_atom_source(
            source_path,
            args.source_frame,
            "Al",
            include_velocities=False,
        )
        point_config = {**common, "md_seed": args.seed + phase_index}
        run = out / phase
        write_job(
            run,
            source["atoms"],
            element,
            point_config,
            job_type=f"{target_kedf}_phase_feasibility",
            suffix=f"al108_{target_kedf}_{phase}_T{int(args.temperature):04d}_feasibility",
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": target_kedf,
                "target_temperature_K": args.temperature,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": True,
                "abacus_init_vel": 0,
                "stress_available": stress_available,
                "steps": args.steps,
                "dt_fs": args.dt,
                "thermostat": "csvr",
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
                "md_seed": point_config["md_seed"],
            }
        )
    manifest = {
        "schema": "kedf-phase-feasibility-v1",
        "target_kedf": target_kedf,
        "temperature_K": args.temperature,
        "steps": args.steps,
        "dt_fs": args.dt,
        "csvr_tau_fs": args.csvr_tau,
        "stress_available": stress_available,
        "phases": phases,
        "status": "prepared",
    }
    (out / "feasibility_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads(
        (root / "feasibility_manifest.json").read_text(encoding="utf-8")
    )
    results = []
    for item in manifest["phases"]:
        phase = item["phase"]
        run = root / phase
        phase_result = analyze_phase(run, phase, thermalized_initial=True)
        (run / "phase_analysis.json").write_text(
            json.dumps(phase_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logs = sorted(run.glob("OUT.*/running_md.log"))
        rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
        late = rows[len(rows) // 2 :]
        temperature = series_stats([row["temperature_K"] for row in late])
        pressure_values = [
            row["pressure_kbar"] for row in late if "pressure_kbar" in row
        ]
        pressure = series_stats(pressure_values)
        nearest = phase_result.get("trajectory", {}).get("nearest_neighbor_A", 0.0)
        temperature_error = abs(
            float(temperature.get("mean", math.inf))
            - float(manifest["temperature_K"])
        )
        checks = {
            "reached_requested_step": max_step >= int(manifest["steps"]),
            "phase_verified": phase_result.get("status") == f"{phase}_verified",
            "temperature_mean_within_35_K": temperature_error <= 35.0,
            "nearest_neighbor_gt_2_A": float(nearest) > 2.0,
            "pressure_present_if_available": (
                bool(pressure_values) if manifest["stress_available"] else True
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
            "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
        }
        (run / "feasibility_result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    summary = {
        **manifest,
        "results": results,
        "status": (
            "feasibility_verified"
            if all(item["status"] == "passed" for item in results)
            else "feasibility_failed"
        ),
    }
    (root / "feasibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--solid-source", required=True)
    prep.add_argument("--liquid-source", required=True)
    prep.add_argument("--source-frame", default="last")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--temperature", type=float, default=975.0)
    prep.add_argument("--steps", type=int, default=200)
    prep.add_argument("--dt", type=float, default=1.0)
    prep.add_argument("--csvr-tau", type=float, default=2.0)
    prep.add_argument("--seed", type=int, default=2026072701)
    prep.add_argument("--ranks", type=int, default=36)
    prep.set_defaults(func=prepare)
    gate = subparsers.add_parser("analyze")
    gate.add_argument("root", type=Path)
    gate.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
