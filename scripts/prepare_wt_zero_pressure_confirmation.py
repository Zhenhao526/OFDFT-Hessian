#!/usr/bin/env python3
"""Prepare long WT NVT confirmation runs at fitted zero-pressure volumes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from scripts.prepare_al108_volume_scan import scaled_to_volume

ROOT = Path(__file__).resolve().parents[1]


def volume_scan_specifications(
    volume_scan_root: Path,
) -> tuple[float, tuple[tuple[str, str, float], ...]]:
    temperatures = set()
    specifications = []
    for phase in ("solid", "liquid"):
        phase_root = volume_scan_root / phase
        report_path = phase_root / "nvt_volume_scan_result.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checks = report.get("checks", {})
        if report.get("status") != "zero_pressure_volume_verified" or not checks:
            raise RuntimeError(f"{phase} zero-pressure volume scan is not verified")
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise RuntimeError(f"{phase} zero-pressure volume checks failed: {failed}")
        if str(report.get("target_kedf", "")).lower() != "wt":
            raise RuntimeError(f"{phase} volume scan does not use WT")

        temperatures.add(float(report["target_temperature_K"]))
        fit = report.get("linear_fit") or {}
        volume = fit.get("zero_pressure_volume_per_atom_A3")
        bracket = report.get("zero_pressure_bracket_A3_per_atom")
        if volume is None or bracket is None or not (
            float(bracket[0]) <= float(volume) <= float(bracket[1])
        ):
            raise RuntimeError(f"{phase} fitted zero-pressure volume is not bracketed")

        valid_rows = [
            row
            for row in report.get("rows", [])
            if row.get("phase_status") == f"{phase}_verified"
            and row.get("temperature_mean_within_25_K") is True
            and row.get("pressure_last_half_kbar")
        ]
        if not valid_rows:
            raise RuntimeError(f"{phase} volume scan has no valid source point")
        source_row = min(
            valid_rows,
            key=lambda row: abs(float(row["volume_per_atom_A3"]) - float(volume)),
        )
        dumps = sorted((phase_root / source_row["label"]).glob("OUT.*/MD_dump"))
        if len(dumps) != 1:
            raise RuntimeError(
                f"expected one MD_dump for {phase} source point, found {len(dumps)}"
            )
        specifications.append((phase, str(dumps[0]), float(volume)))

    if len(temperatures) != 1:
        raise RuntimeError("solid and liquid volume scans use different temperatures")
    return temperatures.pop(), tuple(specifications)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solid-source")
    parser.add_argument("--liquid-source")
    parser.add_argument("--volume-scan-root", type=Path)
    parser.add_argument("--solid-volume", type=float, default=18.051036222240153)
    parser.add_argument("--liquid-volume", type=float, default=18.736615407000613)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026072201)
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
        raise RuntimeError(f"confirmation requires WT, got {target_kedf!r}")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 1,
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": 1.0,
            "md_tfirst": args.temperature,
            "md_tlast": args.temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": args.csvr_tau,
            "md_dumpfreq": 5,
            "md_restartfreq": 100,
            "init_vel": 1,
            "mpirun_np": 36,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    phases = []
    if args.volume_scan_root is not None:
        if args.solid_source is not None or args.liquid_source is not None:
            raise ValueError("volume-scan mode cannot be combined with explicit sources")
        scan_temperature, specifications = volume_scan_specifications(
            args.volume_scan_root.resolve()
        )
        if not math.isclose(args.temperature, scan_temperature, abs_tol=1.0e-9):
            raise ValueError(
                f"requested temperature {args.temperature} does not match scan {scan_temperature}"
            )
    else:
        if args.solid_source is None or args.liquid_source is None:
            raise ValueError("explicit mode requires both solid and liquid sources")
        specifications = (
            ("solid", args.solid_source, args.solid_volume),
            ("liquid", args.liquid_source, args.liquid_volume),
        )
    for phase_index, (phase, source_path, volume_per_atom) in enumerate(specifications):
        source = load_atom_source(source_path, "last", "Al", include_velocities=True)
        atoms = source["atoms"]
        if atoms.velocities is None:
            raise RuntimeError(f"missing source velocities for {phase}: {source_path}")
        atoms = scaled_to_volume(atoms, volume_per_atom * atoms.natoms)
        point_config = dict(config)
        point_config["md_seed"] = args.seed + phase_index
        write_job(
            out / phase,
            atoms,
            element,
            point_config,
            job_type="wt_zero_pressure_confirmation",
            suffix=f"al108_{phase}_T{int(args.temperature):04d}_zeroP_confirmation",
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": "wt",
                "target_temperature_K": args.temperature,
                "target_pressure_kbar": 0.0,
                "volume_per_atom_A3": volume_per_atom,
                "volume_A3": volume_per_atom * atoms.natoms,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": False,
                "md_seed": point_config["md_seed"],
                "steps": args.steps,
                "dt_fs": 1.0,
                "thermostat": "csvr",
                "csvr_tau": args.csvr_tau,
            },
        )
        phases.append(
            {
                "phase": phase,
                "run": str((out / phase).resolve()),
                "source": source["source"],
                "source_step": source["step"],
                "volume_per_atom_A3": volume_per_atom,
                "md_seed": point_config["md_seed"],
            }
        )
    manifest = {
        "schema": "wt-zero-pressure-confirmation-v1",
        "target_kedf": "wt",
        "temperature_K": args.temperature,
        "target_pressure_kbar": 0.0,
        "steps": args.steps,
        "csvr_tau_fs": args.csvr_tau,
        "phases": phases,
    }
    (out / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
