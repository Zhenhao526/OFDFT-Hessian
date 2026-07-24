#!/usr/bin/env python3
"""Check isotropic ABACUS MPN pressure against a central volume derivative."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import lattice_volume

ROOT = Path(__file__).resolve().parents[1]
EV_PER_A3_TO_KBAR = 1602.1766208
ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
ITERATION_ENERGY_RE = re.compile(r"E_Total\s+[-+0-9.eE]+\s+([-+0-9.eE]+)")
PRESSURE_PATTERNS = (
    re.compile(r"ELECTRONIC\s+PART OF STRESS:\s+([-+0-9.eE]+)\s+kbar"),
    re.compile(r"#TOTAL-PRESSURE#.*?:\s*([-+0-9.eE]+)\s+kbar"),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--source-frame", default="last")
    prepare.add_argument(
        "--config",
        default=str(ROOT / "config" / "abacus_mpn_node04_cpu12_stress.json"),
    )
    prepare.add_argument(
        "--linear-strain",
        type=float,
        default=0.001,
        help="Isotropic linear strain half width; volume changes by approximately three times this value.",
    )
    prepare.set_defaults(func=prepare_jobs)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("run_root")
    analyze.set_defaults(func=analyze_jobs)
    return result


def scaled_atoms(atoms: AtomSet, factor: float) -> AtomSet:
    lattice = [tuple(component * factor for component in vector) for vector in atoms.lattice_vectors]
    return AtomSet(list(atoms.symbols), list(atoms.scaled_positions), lattice)


def finite_difference_pressure_kbar(
    energy_minus_eV: float,
    energy_plus_eV: float,
    volume_minus_A3: float,
    volume_plus_A3: float,
) -> float:
    derivative = (energy_plus_eV - energy_minus_eV) / (volume_plus_A3 - volume_minus_A3)
    return -derivative * EV_PER_A3_TO_KBAR


def prepare_jobs(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    config = load_json(Path(args.config))
    config.update({"calculation": "scf", "cal_force": 0, "cal_stress": 1, "mpirun_np": 12})
    element = load_json(ROOT / "config" / "al.json")
    source = load_atom_source(args.source, args.source_frame, "Al", include_velocities=False)
    atoms = source["atoms"]
    factors = {
        "minus": 1.0 - args.linear_strain,
        "center": 1.0,
        "plus": 1.0 + args.linear_strain,
    }
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
            job_type="mpn_pressure_finite_difference",
            suffix=f"al108_mpn_pressure_fd_{label}",
            calculation="scf",
            extra_metadata={
                "initial_structure_source": source["source"],
                "initial_structure_source_step": source["step"],
                "linear_scale_factor": factor,
                "volume_A3": volume,
            },
        )
    manifest = {
        "source": source["source"],
        "source_step": source["step"],
        "linear_strain": args.linear_strain,
        "natoms": atoms.natoms,
        "factors": factors,
        "volumes_A3": volumes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared pressure finite-difference jobs in {out}")


def output_text(point: Path) -> str:
    logs = list(point.glob("OUT.*/running_scf.log"))
    if not logs:
        raise FileNotFoundError(f"no running_scf.log in {point}")
    return logs[-1].read_text(errors="replace")


def parse_static_pressure_kbar(text: str) -> float:
    """Return the electronic/static pressure printed by supported ABACUS versions."""
    for pattern in PRESSURE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1])
    raise RuntimeError("no electronic or total static pressure in center output")


def analyze_jobs(args: argparse.Namespace) -> None:
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    energies = {}
    analytic_pressure = None
    for label in ("minus", "center", "plus"):
        text = output_text(root / label)
        matches = ENERGY_RE.findall(text) or ITERATION_ENERGY_RE.findall(text)
        if not matches:
            raise RuntimeError(f"no final energy for {label}")
        energies[label] = float(matches[-1])
        if label == "center":
            analytic_pressure = parse_static_pressure_kbar(text)
    volumes = manifest["volumes_A3"]
    finite_difference = finite_difference_pressure_kbar(
        energies["minus"], energies["plus"], volumes["minus"], volumes["plus"]
    )
    absolute_error = abs(analytic_pressure - finite_difference)
    relative_error = absolute_error / max(abs(finite_difference), 1.0e-15)
    result = {
        "energies_eV": energies,
        "volumes_A3": volumes,
        "analytic_electronic_pressure_kbar": analytic_pressure,
        "finite_difference_pressure_kbar": finite_difference,
        "absolute_error_kbar": absolute_error,
        "relative_error": relative_error,
        "linear_strain": manifest["linear_strain"],
        "pass_abs_0_1_kbar": absolute_error < 0.1,
        "pass_relative_1_percent": relative_error < 0.01,
    }
    (root / "pressure_fd_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
