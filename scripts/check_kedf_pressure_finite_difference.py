#!/usr/bin/env python3
"""Estimate a KEDF snapshot pressure from a central volume derivative."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import lattice_volume
from scripts.check_mpn_pressure_finite_difference import (
    EV_PER_A3_TO_KBAR,
    finite_difference_pressure_kbar,
    parse_static_pressure_kbar,
)

ROOT = Path(__file__).resolve().parents[1]
KB_EV_PER_K = 8.617333262145e-5
ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
ITERATION_ENERGY_RE = re.compile(r"E_Total\s+[-+0-9.eE]+\s+([-+0-9.eE]+)")


def scaled_atoms(atoms: AtomSet, factor: float) -> AtomSet:
    lattice = [
        tuple(component * factor for component in vector)
        for vector in atoms.lattice_vectors
    ]
    return AtomSet(list(atoms.symbols), list(atoms.scaled_positions), lattice)


def scaled_to_volume(atoms: AtomSet, volume_A3: float) -> AtomSet:
    current = lattice_volume(tuple(atoms.lattice_vectors))
    return scaled_atoms(atoms, (volume_A3 / current) ** (1.0 / 3.0))


def output_text(point: Path) -> str:
    logs = list(point.glob("OUT.*/running_scf.log"))
    if not logs:
        raise FileNotFoundError(f"no running_scf.log in {point}")
    return logs[-1].read_text(errors="replace")


def ionic_kinetic_pressure_kbar(
    natoms: int, temperature_K: float, volume_A3: float
) -> float:
    return natoms * KB_EV_PER_K * temperature_K / volume_A3 * EV_PER_A3_TO_KBAR


def prepare(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    config = load_json(args.config)
    target_kedf = str(config.get("of_kinetic", "")).lower()
    analytic_stress = bool(config.get("cal_stress", False))
    config.update(
        {
            "calculation": "scf",
            "cal_force": 0,
            "cal_stress": int(analytic_stress),
            "mpirun_np": args.ranks,
        }
    )
    source = load_atom_source(
        args.source, args.source_frame, "Al", include_velocities=False
    )
    atoms = source["atoms"]
    if args.volume_per_atom is not None:
        atoms = scaled_to_volume(
            atoms, float(args.volume_per_atom) * atoms.natoms
        )
    factors = {
        "minus": 1.0 - args.linear_strain,
        "center": 1.0,
        "plus": 1.0 + args.linear_strain,
    }
    element = load_json(ROOT / "config" / "al.json")
    volumes = {}
    for label, factor in factors.items():
        point_atoms = scaled_atoms(atoms, factor)
        volume = lattice_volume(tuple(point_atoms.lattice_vectors))
        volumes[label] = volume
        write_job(
            out / label,
            point_atoms,
            element,
            config,
            job_type=f"{target_kedf}_pressure_finite_difference",
            suffix=f"al{atoms.natoms}_{target_kedf}_{args.phase}_pressure_fd_{label}",
            calculation="scf",
            extra_metadata={
                "phase": args.phase,
                "target_kedf": target_kedf,
                "target_temperature_K": args.temperature,
                "source": source["source"],
                "source_step": source["step"],
                "linear_scale_factor": factor,
                "volume_A3": volume,
                "analytic_stress_available": analytic_stress,
            },
        )
    manifest = {
        "schema": "kedf-snapshot-pressure-fd-v1",
        "phase": args.phase,
        "target_kedf": target_kedf,
        "target_temperature_K": args.temperature,
        "source": source["source"],
        "source_step": source["step"],
        "natoms": atoms.natoms,
        "base_volume_A3": volumes["center"],
        "base_volume_per_atom_A3": volumes["center"] / atoms.natoms,
        "linear_strain": args.linear_strain,
        "analytic_stress_available": analytic_stress,
        "factors": factors,
        "volumes_A3": volumes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    energies = {}
    analytic_pressure = None
    for label in ("minus", "center", "plus"):
        text = output_text(root / label)
        matches = ENERGY_RE.findall(text) or ITERATION_ENERGY_RE.findall(text)
        if not matches:
            raise RuntimeError(f"no final energy for {label}")
        energies[label] = float(matches[-1])
        if label == "center" and manifest["analytic_stress_available"]:
            analytic_pressure = parse_static_pressure_kbar(text)
    volumes = manifest["volumes_A3"]
    static_pressure = finite_difference_pressure_kbar(
        energies["minus"],
        energies["plus"],
        volumes["minus"],
        volumes["plus"],
    )
    kinetic_pressure = ionic_kinetic_pressure_kbar(
        int(manifest["natoms"]),
        float(manifest["target_temperature_K"]),
        float(volumes["center"]),
    )
    result = {
        **manifest,
        "energies_eV": energies,
        "finite_difference_static_pressure_kbar": static_pressure,
        "ideal_ionic_pressure_kbar": kinetic_pressure,
        "estimated_total_pressure_kbar": static_pressure + kinetic_pressure,
        "analytic_static_pressure_kbar": analytic_pressure,
    }
    if analytic_pressure is not None:
        absolute_error = abs(analytic_pressure - static_pressure)
        result["analytic_finite_difference_absolute_error_kbar"] = absolute_error
        result["analytic_finite_difference_relative_error"] = absolute_error / max(
            abs(static_pressure), 1.0e-15
        )
    (root / "pressure_fd_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--source-frame", default="last")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--phase", choices=("solid", "liquid"), required=True)
    prep.add_argument("--temperature", type=float, required=True)
    prep.add_argument("--linear-strain", type=float, default=0.001)
    prep.add_argument("--volume-per-atom", type=float)
    prep.add_argument("--ranks", type=int, default=12)
    prep.set_defaults(func=prepare)
    gate = subparsers.add_parser("analyze")
    gate.add_argument("root", type=Path)
    gate.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
