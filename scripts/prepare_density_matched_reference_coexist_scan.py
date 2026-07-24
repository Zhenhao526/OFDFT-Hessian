#!/usr/bin/env python3
"""Build a continuous two-phase cell at separate solid/liquid reference volumes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.cli import load_atom_source
from mpn_melting.coexistence import load_region_labels
from mpn_melting.structures import AtomSet
from mpn_melting.trajectory import matmul_row, nearest_neighbor_distance


ROOT = Path(__file__).resolve().parents[1]


def cross_region_nearest(
    positions: list[tuple[float, float, float]],
    lattice: tuple[tuple[float, float, float], ...],
    labels: list[str],
) -> float:
    solid = [position for position, label in zip(positions, labels) if label == "solid_seed"]
    liquid = [position for position, label in zip(positions, labels) if label == "liquid_seed"]
    minimum = math.inf
    lx, ly, lz = lattice[0][0], lattice[1][1], lattice[2][2]
    for left in solid:
        for right in liquid:
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            dz = right[2] - left[2]
            dx -= round(dx / lx) * lx
            dy -= round(dy / ly) * ly
            dz -= round(dz / lz) * lz
            minimum = min(minimum, math.sqrt(dx * dx + dy * dy + dz * dz))
    return minimum


def density_matched_atoms(
    source: AtomSet,
    labels: list[str],
    old_split: float,
    solid_volume_per_atom: float,
    liquid_volume_per_atom: float,
) -> tuple[AtomSet, dict]:
    if len(labels) != source.natoms:
        raise ValueError("region labels do not match source atom count")
    solid_count = labels.count("solid_seed")
    liquid_count = labels.count("liquid_seed")
    if solid_count == 0 or liquid_count == 0:
        raise ValueError("both solid_seed and liquid_seed atoms are required")
    cross_section = (solid_volume_per_atom * solid_count) ** (1.0 / 3.0)
    solid_length = cross_section
    liquid_length = liquid_volume_per_atom * liquid_count / cross_section**2
    total_length = solid_length + liquid_length
    new_split = solid_length / total_length
    positions = []
    for fx, fy, fz in source.scaled_positions:
        wrapped = fz % 1.0
        if wrapped < old_split:
            mapped_z = wrapped / old_split * new_split
        else:
            mapped_z = new_split + (wrapped - old_split) / (1.0 - old_split) * (1.0 - new_split)
        positions.append((fx % 1.0, fy % 1.0, mapped_z % 1.0))
    lattice = [
        (cross_section, 0.0, 0.0),
        (0.0, cross_section, 0.0),
        (0.0, 0.0, total_length),
    ]
    atoms = AtomSet(
        list(source.symbols),
        positions,
        lattice,
        velocities=list(source.velocities) if source.velocities is not None else None,
        movements=list(source.movements) if source.movements is not None else None,
    )
    cartesian = [matmul_row(position, tuple(lattice)) for position in positions]
    diagnostics = {
        "solid_atom_count": solid_count,
        "liquid_atom_count": liquid_count,
        "solid_volume_per_atom_A3": solid_volume_per_atom,
        "liquid_volume_per_atom_A3": liquid_volume_per_atom,
        "cross_section_A": cross_section,
        "solid_length_A": solid_length,
        "liquid_length_A": liquid_length,
        "total_length_A": total_length,
        "old_split": old_split,
        "new_split": new_split,
        "nearest_neighbor_A": nearest_neighbor_distance(cartesian, tuple(lattice)),
        "cross_region_nearest_neighbor_A": cross_region_nearest(
            cartesian, tuple(lattice), labels
        ),
    }
    return atoms, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-frame", default="last")
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--old-split", type=float, required=True)
    parser.add_argument("--solid-volume-per-atom", type=float, required=True)
    parser.add_argument("--liquid-volume-per-atom", type=float, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--dumpfreq", type=int, default=10)
    parser.add_argument("--restartfreq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202607800)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = load_atom_source(
        args.source, args.source_frame, "Al", include_velocities=True
    )
    if source["atoms"].velocities is None:
        raise ValueError("two-phase source must contain velocities")
    labels = load_region_labels(args.regions)
    atoms, diagnostics = density_matched_atoms(
        source["atoms"],
        labels,
        args.old_split,
        args.solid_volume_per_atom,
        args.liquid_volume_per_atom,
    )
    if diagnostics["nearest_neighbor_A"] <= 2.0:
        raise ValueError(f"density-matched cell overlaps: {diagnostics}")

    config = load_json(args.config)
    element = load_json(ROOT / "config" / "al.json")
    runs = []
    for index, temperature in enumerate(sorted(set(args.temperatures))):
        point_config = dict(config)
        point_config.update(
            {
                "calculation": "md",
                "cal_force": 1,
                "cal_stress": 0,
                "md_type": "nvt",
                "md_nstep": args.steps,
                "md_dt": args.dt,
                "md_tfirst": temperature,
                "md_tlast": temperature,
                "md_thermostat": "csvr",
                "md_csvr_tau": args.csvr_tau,
                "md_dumpfreq": args.dumpfreq,
                "md_restartfreq": args.restartfreq,
                "md_seed": args.seed + index,
                "mpirun_np": 1,
            }
        )
        label = f"T{int(round(temperature)):04d}"
        run_dir = out / label
        write_job(
            run_dir,
            atoms,
            element,
            point_config,
            job_type="density_matched_pair_reference_coexistence",
            suffix=f"al{atoms.natoms}_density_matched_reference_{label}",
            calculation="md",
            extra_metadata={
                "phase": "two_phase",
                "target_temperature_K": temperature,
                "source": source["source"],
                "source_step": source["step"],
                "source_velocities_discarded": False,
                "pair_model": str(args.pair_model.resolve()),
                **diagnostics,
            },
        )
        shutil.copy2(args.regions, run_dir / "regions.csv")
        runs.append(
            {
                "label": label,
                "temperature_K": temperature,
                "run_dir": str(run_dir.resolve()),
                "seed": args.seed + index,
            }
        )

    manifest = {
        "schema": "density-matched-pair-reference-coexistence-v1",
        "natoms": atoms.natoms,
        "source": source["source"],
        "source_step": source["step"],
        "pair_model": str(args.pair_model.resolve()),
        "steps": args.steps,
        "diagnostics": diagnostics,
        "runs": runs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
