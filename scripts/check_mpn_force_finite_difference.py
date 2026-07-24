#!/usr/bin/env python3
"""Check an ABACUS MPN ionic force against a central energy difference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.structures import AtomSet, build_fcc

ROOT = Path(__file__).resolve().parents[1]
ENERGY_RE = re.compile(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV")
ITERATION_ENERGY_RE = re.compile(r"E_Total\s+[-+0-9.eE]+\s+([-+0-9.eE]+)")
FORCE_RE = re.compile(r"^\s*Al1\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$", re.M)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", required=True)
    prepare.add_argument(
        "--config", default=str(ROOT / "config" / "abacus_mpn_node01_cpu16_paper_cutoff.json")
    )
    prepare.add_argument("--offset", type=float, default=0.05, help="Central Al1 x displacement in Angstrom.")
    prepare.add_argument("--delta", type=float, default=0.005, help="Finite-difference half width in Angstrom.")
    prepare.set_defaults(func=prepare_jobs)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("run_root")
    analyze.set_defaults(func=analyze_jobs)
    return result


def displaced_fcc(displacement_A: float) -> AtomSet:
    atoms = build_fcc("Al", 4.05, (2, 2, 2))
    box_x = atoms.lattice_vectors[0][0]
    positions = list(atoms.scaled_positions)
    first = positions[0]
    positions[0] = ((first[0] + displacement_A / box_x) % 1.0, first[1], first[2])
    return AtomSet(atoms.symbols, positions, atoms.lattice_vectors)


def prepare_jobs(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    config = load_json(Path(args.config))
    config["mpirun_np"] = 8
    element = load_json(ROOT / "config" / "al.json")
    points = {"minus": args.offset - args.delta, "center": args.offset, "plus": args.offset + args.delta}
    for label, displacement in points.items():
        write_job(
            out / label,
            displaced_fcc(displacement),
            element,
            config,
            job_type="mpn_force_finite_difference",
            suffix=f"al32_mpn_force_fd_{label}",
            calculation="scf",
            extra_metadata={
                "displaced_atom": 1,
                "displacement_A": displacement,
                "finite_difference_delta_A": args.delta,
            },
        )
    manifest = {"offset_A": args.offset, "delta_A": args.delta, "points": points}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared force finite-difference jobs in {out}")


def output_text(point: Path) -> str:
    logs = list(point.glob("OUT.*/running_scf.log"))
    if not logs:
        raise FileNotFoundError(f"no running_scf.log in {point}")
    return logs[-1].read_text(errors="replace")


def analyze_jobs(args: argparse.Namespace) -> None:
    root = Path(args.run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    energies = {}
    force = None
    for label in ("minus", "center", "plus"):
        text = output_text(root / label)
        matches = ENERGY_RE.findall(text) or ITERATION_ENERGY_RE.findall(text)
        if not matches:
            raise RuntimeError(f"no final energy for {label}")
        energies[label] = float(matches[-1])
        if label == "center":
            forces = FORCE_RE.findall(text)
            if not forces:
                raise RuntimeError("no analytic Al1 force in center output")
            force = tuple(float(value) for value in forces[-1])
    delta = float(manifest["delta_A"])
    finite_difference_x = -(energies["plus"] - energies["minus"]) / (2.0 * delta)
    absolute_error = abs(force[0] - finite_difference_x)
    result = {
        "energies_eV": energies,
        "analytic_force_Al1_eV_per_A": force,
        "finite_difference_force_x_eV_per_A": finite_difference_x,
        "absolute_error_eV_per_A": absolute_error,
        "relative_error": absolute_error / max(abs(finite_difference_x), 1e-15),
        "delta_A": delta,
        "pass_abs_1e-3_eV_per_A": absolute_error < 1e-3,
        "pass_relative_1_percent": absolute_error / max(abs(finite_difference_x), 1e-15) < 0.01,
    }
    (root / "force_fd_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
