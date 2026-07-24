#!/usr/bin/env python3
"""Prepare a WT solid/liquid continuation from complete ABACUS restart STRUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpn_melting.abacus_input import ANGSTROM_TO_BOHR, load_json, write_job
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import invert_3x3, matmul_row


ROOT = Path(__file__).resolve().parents[1]


def read_restart_stru(path: Path) -> AtomSet:
    """Read Direct or Cartesian ABACUS restart STRUs, including m/v fields."""
    lines = path.read_text(encoding="utf-8").splitlines()

    def section(name: str) -> int:
        try:
            return next(i for i, line in enumerate(lines) if line.strip() == name)
        except StopIteration as exc:
            raise ValueError(f"missing {name} section in {path}") from exc

    def next_nonempty(index: int) -> tuple[str, int]:
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            raise ValueError(f"unexpected end of STRU: {path}")
        return lines[index].strip(), index + 1

    lattice_constant_line, _ = next_nonempty(section("LATTICE_CONSTANT") + 1)
    scale = float(lattice_constant_line.split()[0]) / ANGSTROM_TO_BOHR
    cursor = section("LATTICE_VECTORS") + 1
    lattice = []
    for _ in range(3):
        line, cursor = next_nonempty(cursor)
        values = tuple(float(value) * scale for value in line.split()[:3])
        if len(values) != 3:
            raise ValueError(f"invalid lattice vector in {path}: {line}")
        lattice.append(values)

    coordinate_type, cursor = next_nonempty(section("ATOMIC_POSITIONS") + 1)
    coordinate_type = coordinate_type.lower()
    if coordinate_type not in {"direct", "cartesian"}:
        raise ValueError(f"unsupported coordinate type in {path}: {coordinate_type}")
    inverse_lattice = invert_3x3(lattice)

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float] | None] = []
    movements: list[tuple[int, int, int] | None] = []
    while True:
        try:
            species_line, cursor = next_nonempty(cursor)
        except ValueError:
            break
        symbol = species_line.split()[0]
        _, cursor = next_nonempty(cursor)  # species magnetization
        count_line, cursor = next_nonempty(cursor)
        count = int(count_line.split()[0])
        for _ in range(count):
            atom_line, cursor = next_nonempty(cursor)
            fields = atom_line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid atom line in {path}: {atom_line}")
            raw_position = tuple(float(value) for value in fields[:3])
            if coordinate_type == "cartesian":
                cartesian = tuple(value * scale for value in raw_position)
                position = matmul_row(cartesian, inverse_lattice)
            else:
                position = raw_position
            symbols.append(symbol)
            positions.append(tuple(value % 1.0 for value in position))

            if "m" in fields:
                movement_index = fields.index("m") + 1
                movements.append(
                    tuple(int(value) for value in fields[movement_index : movement_index + 3])
                )
            elif len(fields) >= 6:
                try:
                    movements.append(tuple(int(value) for value in fields[3:6]))
                except ValueError:
                    movements.append(None)
            else:
                movements.append(None)

            if "v" in fields:
                velocity_index = fields.index("v") + 1
                velocity = tuple(
                    float(value) for value in fields[velocity_index : velocity_index + 3]
                )
                if len(velocity) != 3:
                    raise ValueError(f"incomplete velocity in {path}: {atom_line}")
                velocities.append(velocity)
            else:
                velocities.append(None)

    if not symbols:
        raise ValueError(f"no atoms found in {path}")
    if not all(velocity is not None for velocity in velocities):
        raise ValueError(f"restart STRU does not contain complete velocities: {path}")
    complete_movements = None
    if all(movement is not None for movement in movements):
        complete_movements = [movement for movement in movements if movement is not None]
    return AtomSet(
        symbols=symbols,
        scaled_positions=positions,
        lattice_vectors=lattice,
        velocities=[velocity for velocity in velocities if velocity is not None],
        movements=complete_movements,
    )


def prepare_resume(
    solid_source: Path,
    liquid_source: Path,
    out: Path,
    config_path: Path,
    completed_steps: int,
    target_total_steps: int,
    temperature: float,
    csvr_tau: float,
    source_step: int,
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    remaining_steps = target_total_steps - completed_steps
    if remaining_steps <= 0:
        raise ValueError("target total steps must exceed completed steps")
    if source_step != completed_steps:
        raise ValueError("source step and completed steps must match")

    config = load_json(config_path)
    if str(config.get("of_kinetic", "")).lower() != "wt":
        raise ValueError("resume config must use WT")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 1,
            "md_type": "nvt",
            "md_nstep": remaining_steps,
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
    for phase, source in (("solid", solid_source), ("liquid", liquid_source)):
        atoms = read_restart_stru(source)
        if atoms.natoms != 108 or set(atoms.symbols) != {"Al"}:
            raise ValueError(f"{phase} source is not a 108-atom Al system")
        run = out / phase
        write_job(
            run,
            atoms,
            element,
            config,
            job_type="wt_zero_pressure_enthalpy_resume",
            suffix=f"al108_{phase}_T{int(temperature):04d}_resume_s{source_step}",
            calculation="md",
            extra_metadata={
                "phase": phase,
                "target_kedf": "wt",
                "target_temperature_K": temperature,
                "source": str(source.resolve()),
                "source_step": source_step,
                "source_velocities_discarded": False,
                "completed_steps_before_resume": completed_steps,
                "remaining_steps": remaining_steps,
                "combined_target_steps": target_total_steps,
                "thermostat": "csvr",
                "csvr_tau": csvr_tau,
            },
        )
        phases.append(
            {
                "phase": phase,
                "run": str(run.resolve()),
                "source": str(source.resolve()),
                "source_step": source_step,
            }
        )

    manifest = {
        "schema": "wt-zero-pressure-enthalpy-resume-v1",
        "target_kedf": "wt",
        "temperature_K": temperature,
        "source_step": source_step,
        "completed_steps_before_resume": completed_steps,
        "remaining_steps": remaining_steps,
        "combined_target_steps": target_total_steps,
        "source_velocities_discarded": False,
        "phases": phases,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solid-source", type=Path, required=True)
    parser.add_argument("--liquid-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--completed-steps", type=int, required=True)
    parser.add_argument("--target-total-steps", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1100.0)
    parser.add_argument("--csvr-tau", type=float, default=5.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()
    manifest = prepare_resume(
        args.solid_source.resolve(),
        args.liquid_source.resolve(),
        args.out.resolve(),
        args.config.resolve(),
        args.completed_steps,
        args.target_total_steps,
        args.temperature,
        args.csvr_tau,
        args.source_step,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
