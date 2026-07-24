#!/usr/bin/env python3
"""Prepare and analyze fixed-configuration pressure pre-scans for Al108."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import lattice_volume
from scripts.check_mpn_pressure_finite_difference import parse_static_pressure_kbar
from scripts.analyze_phase_run import analyze as analyze_phase
from scripts.analyze_two_phase_run import parse_md_log, series_stats

ROOT = Path(__file__).resolve().parents[1]
KB_EV_PER_K = 8.617333262145e-5
EV_PER_A3_TO_KBAR = 1602.1766208
MAX_TEMPERATURE_MEAN_ERROR_K = 25.0


def point_seed(base_seed: int, point_index: int) -> int:
    """Give each volume point an independent thermostat random stream."""
    return int(base_seed) + int(point_index)


def point_seed_provenance(base_seed: int, point_index: int) -> dict[str, int]:
    """Keep the executed seed visible in point and run metadata."""
    return {"md_seed": point_seed(base_seed, point_index)}


def scaled_to_volume(atoms: AtomSet, target_volume_A3: float) -> AtomSet:
    current = lattice_volume(tuple(atoms.lattice_vectors))
    factor = (target_volume_A3 / current) ** (1.0 / 3.0)
    lattice = [tuple(value * factor for value in vector) for vector in atoms.lattice_vectors]
    return AtomSet(
        list(atoms.symbols),
        list(atoms.scaled_positions),
        lattice,
        velocities=list(atoms.velocities) if atoms.velocities is not None else None,
        movements=list(atoms.movements) if atoms.movements is not None else None,
    )


def running_log(point: Path) -> Path:
    matches = sorted(point.glob("OUT.*/running_scf.log"))
    if not matches:
        raise FileNotFoundError(f"no running_scf.log below {point}")
    return matches[-1]


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(args.source, args.source_frame, "Al", include_velocities=False)
    atoms = source["atoms"]
    config = load_json(args.config)
    config.update({"calculation": "scf", "cal_force": 0, "cal_stress": 1, "mpirun_np": 12})
    element = load_json(ROOT / "config" / "al.json")
    points = []
    for volume_per_atom in args.volumes_per_atom:
        volume = volume_per_atom * atoms.natoms
        label = f"vpa_{volume_per_atom:.3f}".replace(".", "p")
        point_atoms = scaled_to_volume(atoms, volume)
        write_job(
            out / label,
            point_atoms,
            element,
            config,
            job_type="mpn_static_pressure_prescan",
            suffix=f"al108_{args.phase}_pressure_{label}",
            calculation="scf",
            extra_metadata={
                "phase": args.phase,
                "target_temperature_K": args.temperature,
                "volume_A3": volume,
                "volume_per_atom_A3": volume_per_atom,
                "source": source["source"],
                "source_step": source["step"],
            },
        )
        points.append({"label": label, "volume_A3": volume, "volume_per_atom_A3": volume_per_atom})
    manifest = {
        "phase": args.phase,
        "target_temperature_K": args.temperature,
        "natoms": atoms.natoms,
        "source": source["source"],
        "source_step": source["step"],
        "points": points,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def analyze(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    natoms = int(manifest["natoms"])
    temperature = float(manifest["target_temperature_K"])
    rows = []
    for point in manifest["points"]:
        pressure_static = parse_static_pressure_kbar(running_log(root / point["label"]).read_text(errors="replace"))
        volume = float(point["volume_A3"])
        kinetic = natoms * KB_EV_PER_K * temperature / volume * EV_PER_A3_TO_KBAR
        rows.append(
            {
                **point,
                "static_pressure_kbar": pressure_static,
                "ideal_ionic_pressure_kbar": kinetic,
                "estimated_md_pressure_kbar": pressure_static + kinetic,
            }
        )
    ordered = sorted(rows, key=lambda row: row["volume_per_atom_A3"])
    bracket = None
    for left, right in zip(ordered, ordered[1:]):
        if left["estimated_md_pressure_kbar"] * right["estimated_md_pressure_kbar"] <= 0.0:
            bracket = [left["volume_per_atom_A3"], right["volume_per_atom_A3"]]
            break
    estimate = None
    if bracket:
        left = next(row for row in ordered if row["volume_per_atom_A3"] == bracket[0])
        right = next(row for row in ordered if row["volume_per_atom_A3"] == bracket[1])
        p0, p1 = left["estimated_md_pressure_kbar"], right["estimated_md_pressure_kbar"]
        estimate = left["volume_per_atom_A3"] - p0 * (right["volume_per_atom_A3"] - left["volume_per_atom_A3"]) / (p1 - p0)
    result = {**manifest, "rows": ordered, "zero_pressure_bracket_A3_per_atom": bracket, "linear_zero_pressure_estimate_A3_per_atom": estimate}
    (root / "volume_prescan_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def prepare_md(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(args.source, args.source_frame, "Al", include_velocities=False)
    atoms = source["atoms"]
    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "unknown")).lower()
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
            "md_seed": args.seed,
            "mpirun_np": 12,
        }
    )
    element = load_json(ROOT / "config" / "al.json")
    points = []
    for point_index, volume_per_atom in enumerate(args.volumes_per_atom):
        volume = volume_per_atom * atoms.natoms
        label = f"vpa_{volume_per_atom:.3f}".replace(".", "p")
        point_atoms = scaled_to_volume(atoms, volume)
        seed_provenance = point_seed_provenance(args.seed, point_index)
        point_config = {**config, **seed_provenance}
        write_job(
            out / label,
            point_atoms,
            element,
            point_config,
            job_type=f"{target_kedf}_nvt_zero_pressure_volume_scan",
            suffix=f"al108_{args.phase}_T{int(args.temperature):04d}_{label}",
            calculation="md",
            extra_metadata={
                "phase": args.phase,
                "target_kedf": target_kedf,
                "target_temperature_K": args.temperature,
                "volume_A3": volume,
                "volume_per_atom_A3": volume_per_atom,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": True,
                "steps": args.steps,
                "dt_fs": args.dt,
                "thermostat": "csvr",
                "csvr_tau": args.csvr_tau,
                **seed_provenance,
            },
        )
        points.append(
            {
                "label": label,
                "volume_A3": volume,
                "volume_per_atom_A3": volume_per_atom,
                **seed_provenance,
            }
        )
    manifest = {
        "phase": args.phase,
        "target_kedf": target_kedf,
        "target_temperature_K": args.temperature,
        "natoms": atoms.natoms,
        "source": source["source"],
        "source_step": source["step"],
        "steps": args.steps,
        "points": points,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def linear_fit_zero(rows: list[dict]) -> dict | None:
    if len(rows) < 2:
        return None
    xs = [float(row["volume_per_atom_A3"]) for row in rows]
    ys = [float(row["pressure_last_half_kbar"]["mean"]) for row in rows]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return {
        "slope_kbar_per_A3_per_atom": slope,
        "intercept_kbar": intercept,
        "zero_pressure_volume_per_atom_A3": -intercept / slope if slope else None,
    }


def pressure_bracket(rows: list[dict]) -> list[float] | None:
    ordered = sorted(rows, key=lambda row: float(row["volume_per_atom_A3"]))
    for left, right in zip(ordered, ordered[1:]):
        left_pressure = float(left["pressure_last_half_kbar"]["mean"])
        right_pressure = float(right["pressure_last_half_kbar"]["mean"])
        if left_pressure * right_pressure <= 0.0:
            return [
                float(left["volume_per_atom_A3"]),
                float(right["volume_per_atom_A3"]),
            ]
    return None


def bracket_linear_fit(rows: list[dict], bracket: list[float] | None) -> dict | None:
    if bracket is None:
        return None
    endpoints = [
        row
        for row in rows
        if float(row["volume_per_atom_A3"]) in (float(bracket[0]), float(bracket[1]))
    ]
    return linear_fit_zero(endpoints)


def pressure_monotonically_decreases(rows: list[dict]) -> bool:
    return all(row["consistent_with_decrease"] for row in pressure_monotonicity_diagnostic(rows))


def pressure_monotonicity_diagnostic(
    rows: list[dict], sigma_tolerance: float = 2.0
) -> list[dict]:
    ordered = sorted(rows, key=lambda row: float(row["volume_per_atom_A3"]))
    if len(ordered) < 2:
        return [{"consistent_with_decrease": False, "reason": "fewer_than_two_points"}]
    diagnostic = []
    for left, right in zip(ordered, ordered[1:]):
        left_stats = left["pressure_last_half_kbar"]
        right_stats = right["pressure_last_half_kbar"]
        pressure_drop = float(left_stats["mean"]) - float(right_stats["mean"])
        standard_errors = []
        for stats in (left_stats, right_stats):
            count = int(stats.get("n", 0))
            sd = stats.get("sd")
            if count > 0 and sd is not None:
                standard_errors.append(float(sd) / math.sqrt(count))
        combined_standard_error = (
            math.sqrt(sum(value**2 for value in standard_errors))
            if len(standard_errors) == 2
            else None
        )
        consistent = pressure_drop >= 0.0 or (
            combined_standard_error is not None
            and pressure_drop >= -sigma_tolerance * combined_standard_error
        )
        diagnostic.append(
            {
                "left_volume_per_atom_A3": float(left["volume_per_atom_A3"]),
                "right_volume_per_atom_A3": float(right["volume_per_atom_A3"]),
                "pressure_drop_kbar": pressure_drop,
                "combined_standard_error_kbar": combined_standard_error,
                "sigma_tolerance": sigma_tolerance,
                "consistent_with_decrease": consistent,
            }
        )
    return diagnostic


def zero_pressure_checks(
    rows: list[dict],
    valid: list[dict],
    requested_steps: int,
    fit: dict | None,
    bracket: list[float] | None,
) -> dict[str, bool]:
    estimate = None if fit is None else fit.get("zero_pressure_volume_per_atom_A3")
    return {
        "requested_steps_reached": bool(rows)
        and all(int(row["max_step"]) >= requested_steps for row in rows),
        "at_least_two_valid_phase_points": len(valid) >= 2,
        "valid_nearest_neighbors_gt_2_A": bool(valid)
        and all(
            row.get("nearest_neighbor_A") is not None
            and float(row["nearest_neighbor_A"]) > 2.0
            for row in valid
        ),
        "pressure_bracket_found": bracket is not None,
        "pressure_decreases_with_volume": pressure_monotonically_decreases(valid),
        "zero_pressure_fit_inside_bracket": bracket is not None
        and estimate is not None
        and bracket[0] <= float(estimate) <= bracket[1],
    }


def analyze_md(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    thermalized_initial = bool(manifest.get("thermalized_initial")) or int(
        manifest.get("source_step") or 0
    ) > 0
    rows = []
    for point in manifest["points"]:
        run = root / point["label"]
        phase_result = analyze_phase(
            run,
            manifest["phase"],
            thermalized_initial=thermalized_initial,
        )
        (run / "phase_analysis.json").write_text(json.dumps(phase_result, indent=2, sort_keys=True) + "\n")
        logs = sorted(run.glob("OUT.*/running_md.log"))
        md_rows, max_step = parse_md_log(logs[-1]) if logs else ([], 0)
        sampled = md_rows[len(md_rows) // 2 :]
        pressures = [row["pressure_kbar"] for row in sampled if "pressure_kbar" in row]
        temperatures = [row["temperature_K"] for row in sampled]
        temperature_stats = series_stats(temperatures)
        temperature_error = abs(
            float(temperature_stats.get("mean", math.inf))
            - float(manifest["target_temperature_K"])
        )
        rows.append(
            {
                **point,
                "max_step": max_step,
                "phase_status": phase_result["status"],
                "pressure_last_half_kbar": series_stats(pressures),
                "temperature_last_half_K": temperature_stats,
                "temperature_target_error_K": temperature_error,
                "temperature_mean_within_25_K": temperature_error <= MAX_TEMPERATURE_MEAN_ERROR_K,
                "nearest_neighbor_A": phase_result.get("trajectory", {}).get("nearest_neighbor_A"),
            }
        )
    valid = [
        row
        for row in rows
        if row["phase_status"] == f"{manifest['phase']}_verified"
        and row["pressure_last_half_kbar"]
        and row["temperature_mean_within_25_K"]
    ]
    bracket = pressure_bracket(valid)
    global_fit = linear_fit_zero(valid)
    fit = bracket_linear_fit(valid, bracket)
    checks = zero_pressure_checks(
        rows,
        valid,
        int(manifest["steps"]),
        fit,
        bracket,
    )
    result = {
        **manifest,
        "rows": rows,
        "valid_phase_points": len(valid),
        "zero_pressure_bracket_A3_per_atom": bracket,
        "linear_fit": fit,
        "global_linear_fit_diagnostic": global_fit,
        "pressure_monotonicity_diagnostic": pressure_monotonicity_diagnostic(valid),
        "checks": checks,
        "status": "zero_pressure_volume_verified"
        if all(checks.values())
        else "zero_pressure_volume_not_verified",
    }
    (root / "nvt_volume_scan_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--source-frame", default="last")
    prep.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prep.add_argument("--temperature", type=float, required=True)
    prep.add_argument("--volumes-per-atom", type=float, nargs="+", required=True)
    prep.add_argument("--config", type=Path, default=ROOT / "config" / "abacus_mpn_node04_cpu12_stress.json")
    prep.set_defaults(func=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("run_root", type=Path)
    analysis.set_defaults(func=analyze)
    prep_md = sub.add_parser("prepare-md")
    prep_md.add_argument("--out", type=Path, required=True)
    prep_md.add_argument("--source", required=True)
    prep_md.add_argument("--source-frame", default="last")
    prep_md.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prep_md.add_argument("--temperature", type=float, required=True)
    prep_md.add_argument("--volumes-per-atom", type=float, nargs="+", required=True)
    prep_md.add_argument("--steps", type=int, default=200)
    prep_md.add_argument("--dt", type=float, default=1.0)
    prep_md.add_argument("--csvr-tau", type=float, default=10.0)
    prep_md.add_argument("--dumpfreq", type=int, default=5)
    prep_md.add_argument("--restartfreq", type=int, default=100)
    prep_md.add_argument("--seed", type=int, default=850)
    prep_md.add_argument("--config", type=Path, default=ROOT / "config" / "abacus_mpn_node04_cpu12_stress.json")
    prep_md.set_defaults(func=prepare_md)
    analysis_md = sub.add_parser("analyze-md")
    analysis_md.add_argument("run_root", type=Path)
    analysis_md.set_defaults(func=analyze_md)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
