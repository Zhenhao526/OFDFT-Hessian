#!/usr/bin/env python3
"""Prepare Mg XWM-ensemble snapshots for finite-temperature KSDFT single points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import invert_3x3, matmul_row, nearest_neighbor_distance, parse_md_dump


KB_EV_PER_K = 8.617333262145e-5
RY_TO_EV = 13.605693122994
STEP_RE = re.compile(r"STEP OF MOLECULAR DYNAMICS:\s*(\d+)")
THERMO_RE = re.compile(
    r"Energy \(Ry\).*?Temperature \(K\)\s*\n\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.S,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"expected one {name} below {root}, found {len(paths)}")
    return paths[0]


def thermo_by_dump_step(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = list(STEP_RE.finditer(text))
    result: dict[int, dict[str, float]] = {}
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        row = THERMO_RE.search(text, marker.end(), stop)
        if row is None:
            continue
        # ABACUS reports the observables for MD_dump step n under MD block n+1.
        dump_step = int(marker.group(1)) - 1
        total_ry, potential_ry, kinetic_ry, temperature_k = map(float, row.groups())
        result[dump_step] = {
            "total_energy_Ry": total_ry,
            "potential_energy_Ry": potential_ry,
            "kinetic_energy_Ry": kinetic_ry,
            "temperature_K": temperature_k,
        }
    return result


def fractional_positions(frame) -> list[tuple[float, float, float]]:
    inverse = invert_3x3(frame.lattice)
    return [
        tuple(value - math.floor(value) for value in matmul_row(position, inverse))
        for position in frame.positions
    ]


def evenly_spaced_steps(start: int, stop: int, count: int, interval: int = 5) -> list[int]:
    available = list(range(start, stop + 1, interval))
    if count < 2 or count > len(available):
        raise ValueError("invalid frame count")
    indices = [round(i * (len(available) - 1) / (count - 1)) for i in range(count)]
    result = [available[index] for index in indices]
    if len(set(result)) != count:
        raise ValueError(f"duplicate selected steps: {result}")
    return result


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "PREPARATION_SHA256SUMS":
            rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / "PREPARATION_SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--element", type=Path, required=True)
    parser.add_argument("--frames-per-phase-temperature", type=int, default=10)
    parser.add_argument("--frame-start", type=int, default=500)
    parser.add_argument("--frame-stop", type=int, default=995)
    parser.add_argument("--pilot-step", type=int, default=775)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_root = workspace / "runs/mg_three_method_20260804/xwm/multitemp_enthalpy_v1/formal_steps1000_v2"
    temperatures = (950, 1000, 1050)
    phases = ("solid", "liquid")
    steps = evenly_spaced_steps(
        args.frame_start, args.frame_stop, args.frames_per_phase_temperature
    )
    if args.pilot_step not in steps:
        raise ValueError(f"pilot step {args.pilot_step} is not among {steps}")

    element = load_json(args.element)
    base_config = load_json(args.config)
    base_config.update(
        {
            "mpirun_extra_args": ["--bind-to", "none"],
            "kmesh": [2, 2, 2],
            "kshift": [0, 0, 0],
            "cal_force": 0,
            "cal_stress": 0,
            "out_chg": 0,
            "out_wfc_pw": 0,
            "out_band": 1,
            "nbands": 192,
            "smearing_method": "fd",
        }
    )

    assignments = {
        "pilot_node02": [],
        "pilot_node06": [],
        "production_node02": [],
        "production_node06": [],
    }
    jobs = []
    production_index = 0
    for temperature in temperatures:
        for phase in phases:
            source = source_root / f"T{temperature:04d}" / phase
            dump = one_file(source, "MD_dump")
            log = one_file(source, "running_md.log")
            frames = {frame.step: frame for frame in parse_md_dump(dump)}
            thermo = thermo_by_dump_step(log)
            missing = sorted(set(steps) - frames.keys())
            if missing:
                raise ValueError(f"{source}: missing frames {missing}")
            missing = sorted(set(steps) - thermo.keys())
            if missing:
                raise ValueError(f"{source}: missing thermodynamic rows {missing}")
            for step in steps:
                frame = frames[step]
                if len(frame.positions) != 128:
                    raise ValueError(f"{source} step {step}: expected 128 atoms")
                job_id = f"T{temperature:04d}_{phase}_step{step:04d}"
                relative = Path("jobs") / f"T{temperature:04d}" / phase / f"step{step:04d}"
                job = out / relative
                sigma_ry = KB_EV_PER_K * temperature / RY_TO_EV
                config = dict(base_config)
                config["smearing_sigma"] = f"{sigma_ry:.12f}"
                atoms = AtomSet(
                    symbols=["Mg"] * 128,
                    scaled_positions=fractional_positions(frame),
                    lattice_vectors=[tuple(row) for row in frame.lattice],
                )
                row = thermo[step]
                node = "node02" if production_index % 2 == 0 else "node06"
                production_index += 1
                write_job(
                    job,
                    atoms,
                    element,
                    config,
                    job_type="xwm_ensemble_ksdft_k222_temperature_consistent_fd_singlepoint",
                    suffix=f"mg128_ksdft_{job_id}",
                    calculation="scf",
                    extra_metadata={
                        "job_id": job_id,
                        "phase": phase,
                        "target_temperature_K": temperature,
                        "source_run": str(source.relative_to(workspace)),
                        "source_md_dump": str(dump.relative_to(workspace)),
                        "source_md_dump_sha256": sha256(dump),
                        "source_running_log": str(log.relative_to(workspace)),
                        "source_running_log_sha256": sha256(log),
                        "source_md_step": step,
                        "source_instantaneous_temperature_K": row["temperature_K"],
                        "xwm_potential_energy_Ry": row["potential_energy_Ry"],
                        "xwm_potential_energy_eV": row["potential_energy_Ry"] * RY_TO_EV,
                        "nearest_neighbor_A": nearest_neighbor_distance(frame.positions, frame.lattice),
                        "assigned_node": node,
                        "kmesh": [2, 2, 2],
                        "fd_smearing_sigma_Ry": sigma_ry,
                        "fd_smearing_sigma_eV": KB_EV_PER_K * temperature,
                        "electronic_surface": "Mermin_FINAL_ETOT_IS",
                        "internal_energy_formula": "U_KS=FINAL_ETOT_IS-E_entropy(-TS)",
                    },
                )
                input_rows = []
                for name in ("INPUT", "KPT", "STRU", "metadata.json"):
                    input_rows.append(f"{sha256(job / name)}  {name}")
                (job / "INPUT_SHA256SUMS").write_text(
                    "\n".join(input_rows) + "\n", encoding="utf-8"
                )
                relative_text = str(relative)
                assignments[f"production_{node}"].append(relative_text)
                if step == args.pilot_step:
                    pilot_node = "node02" if phase == "solid" else "node06"
                    assignments[f"pilot_{pilot_node}"].append(relative_text)
                jobs.append(
                    {
                        "job_id": job_id,
                        "relative_path": relative_text,
                        "temperature_K": temperature,
                        "phase": phase,
                        "source_md_step": step,
                        "assigned_node": node,
                        "is_pilot": step == args.pilot_step,
                    }
                )

    # Pilot jobs are retained in the full production manifests but skipped once complete.
    manifest_dir = out / "manifests"
    manifest_dir.mkdir()
    for name, entries in assignments.items():
        (manifest_dir / f"{name}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    manifest = {
        "schema": "mg-xwm-ensemble-ksdft-multitemperature-oneway-diagnostic-v1",
        "status": "prepared",
        "temperatures_K": list(temperatures),
        "phases": list(phases),
        "natoms": 128,
        "selected_steps": steps,
        "pilot_step": args.pilot_step,
        "kmesh": [2, 2, 2],
        "smearing_method": "fd",
        "sigma_formula": "k_B*T/Ry_to_eV",
        "target_kedf": "xwm",
        "reference": "finite-temperature KSDFT",
        "correction_scope": "one-way diagnostic; no reverse KS ensemble",
        "assignments": assignments,
        "jobs": jobs,
    }
    (out / "PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksums(out)
    print(
        json.dumps(
            {
                "root": str(out),
                "jobs": len(jobs),
                "selected_steps": steps,
                "assignments": {key: len(value) for key, value in assignments.items()},
                "preparation_sha256s_sha": sha256(out / "PREPARATION_SHA256SUMS"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
