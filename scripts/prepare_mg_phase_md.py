#!/usr/bin/env python3
"""Prepare a provenance-tracked Mg phase MD job for KSDFT, WT, or XWM."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.structures import AtomSet, build_hcp, lattice_volume, select_structure


METHOD_CONFIGS = {
    "ksdft": "abacus_mg_ksdft_cpu36.json",
    "wt": "abacus_mg_wt_cpu36.json",
    "xwm": "abacus_mg_xwm_cpu36.json",
}
KB_RY_PER_K = 8.617333262145e-5 / 13.605693122994


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_to_volume(atoms: AtomSet, volume_per_atom: float) -> AtomSet:
    if volume_per_atom <= 0.0:
        raise ValueError("volume per atom must be positive")
    scale = (
        volume_per_atom * atoms.natoms / lattice_volume(atoms.lattice_vectors)
    ) ** (1.0 / 3.0)
    return AtomSet(
        symbols=list(atoms.symbols),
        scaled_positions=list(atoms.scaled_positions),
        lattice_vectors=[
            tuple(component * scale for component in vector)
            for vector in atoms.lattice_vectors
        ],
        velocities=(list(atoms.velocities) if atoms.velocities is not None else None),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), required=True)
    parser.add_argument("--phase-role", choices=("solid", "liquid", "melt"), required=True)
    parser.add_argument("--volume-per-atom", type=float, required=True)
    parser.add_argument("--temperature-k", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--csvr-tau-fs", type=float, default=2.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ranks", type=int, default=36)
    parser.add_argument("--size", type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument("--c-over-a", type=float)
    parser.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--kshift", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--source")
    parser.add_argument("--source-frame", default="last")
    parser.add_argument("--preserve-velocities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if args.steps <= 0 or args.temperature_k <= 0.0:
        raise ValueError("steps and temperature must be positive")
    if args.preserve_velocities and not args.source:
        raise ValueError("--preserve-velocities requires --source")
    if args.source and args.c_over_a is not None:
        raise ValueError("--c-over-a is only valid for a generated hcp source")

    element = load_json(repo / "config" / "mg.json")
    source_record: dict[str, object]
    if args.source:
        loaded = load_atom_source(
            args.source,
            args.source_frame,
            "Mg",
            include_velocities=args.preserve_velocities,
        )
        atoms = loaded["atoms"]
        source_path = Path(str(loaded["source"])).resolve()
        source_record = {
            "kind": "trajectory_or_structure",
            "path": str(source_path),
            "sha256": sha256(source_path),
            "step": loaded["step"],
            "velocities_preserved": args.preserve_velocities,
        }
    else:
        if args.c_over_a is not None:
            if args.c_over_a <= 0.0:
                raise ValueError("--c-over-a must be positive")
            lattice_a = (
                4.0 * args.volume_per_atom / (sqrt(3.0) * args.c_over_a)
            ) ** (1.0 / 3.0)
            atoms = build_hcp(
                element["element"],
                lattice_a,
                args.c_over_a * lattice_a,
                args.size,
            )
        else:
            atoms = select_structure(element, args.size)
        source_record = {
            "kind": "generated_hcp",
            "size": list(args.size),
            "c_over_a": args.c_over_a,
            "velocities_preserved": False,
        }
    atoms = scaled_to_volume(atoms, args.volume_per_atom)

    config_path = repo / "config" / METHOD_CONFIGS[args.method]
    config = load_json(config_path)
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "md_type": "nvt",
            "md_nstep": args.steps,
            "md_dt": 1.0,
            "md_tfirst": args.temperature_k,
            "md_tlast": args.temperature_k,
            "md_thermostat": "csvr",
            "md_csvr_tau": args.csvr_tau_fs,
            "md_dumpfreq": 5,
            "md_restartfreq": 100,
            "md_seed": args.seed,
            # ABACUS uses 0 to generate a fresh Maxwell distribution and 1
            # to consume velocities explicitly written in STRU.
            "init_vel": 1 if args.preserve_velocities else 0,
            "mpirun_np": args.ranks,
            "kmesh": list(args.kmesh),
            "kshift": list(args.kshift),
        }
    )
    if args.method == "ksdft":
        config["smearing_sigma"] = args.temperature_k * KB_RY_PER_K

    write_job(
        out,
        atoms,
        element,
        config,
        job_type="mg_phase_md",
        suffix=(
            f"mg{atoms.natoms}_{args.method}_{args.phase_role}_"
            f"T{int(round(args.temperature_k)):04d}"
        ),
        calculation="md",
        extra_metadata={
            "element": "Mg",
            "method": args.method,
            "phase_role": args.phase_role,
            "temperature_K": args.temperature_k,
            "volume_per_atom_A3": args.volume_per_atom,
            "steps": args.steps,
            "csvr_tau_fs": args.csvr_tau_fs,
            "seed": args.seed,
            "hcp_c_over_a": args.c_over_a,
            "kmesh": list(args.kmesh),
            "kshift": list(args.kshift),
            "source": source_record,
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
        },
    )
    manifest = {
        "schema": "mg-phase-md-v1",
        "status": "prepared",
        "method": args.method,
        "phase_role": args.phase_role,
        "temperature_K": args.temperature_k,
        "volume_per_atom_A3": args.volume_per_atom,
        "natoms": atoms.natoms,
        "steps": args.steps,
        "csvr_tau_fs": args.csvr_tau_fs,
        "seed": args.seed,
        "hcp_c_over_a": args.c_over_a,
        "kmesh": list(args.kmesh),
        "kshift": list(args.kshift),
        "source": source_record,
        "abacus_init_vel": 1 if args.preserve_velocities else 0,
        "input_sha256": sha256(out / "INPUT"),
        "stru_sha256": sha256(out / "STRU"),
        "kpt_sha256": sha256(out / "KPT"),
    }
    manifest_path = out / "mg_phase_md_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "SHA256SUMS").write_text(
        f"{sha256(manifest_path)}  mg_phase_md_manifest.json\n"
        f"{sha256(out / 'INPUT')}  INPUT\n"
        f"{sha256(out / 'STRU')}  STRU\n"
        f"{sha256(out / 'KPT')}  KPT\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "prepared", "out": str(out)}))


if __name__ == "__main__":
    main()
