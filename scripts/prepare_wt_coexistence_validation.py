#!/usr/bin/env python3
"""Prepare the density-matched WT interface used to validate a melting result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

from mpn_melting.abacus_input import load_json, write_job
from mpn_melting.structures import (
    build_coexistence_seed,
    constrain_regions,
    reshape_coexistence_z_lengths,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SIZE = (2, 2, 4)
TEMPLATE_PHASE_ATOMS = 32
TILE_REPEAT = (3, 3, 3)


def verified_melting_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    if report.get("schema") != "wt-gibbs-helmholtz-melting-v2":
        raise ValueError("melting result has the wrong schema")
    if report.get("status") != "verified" or not checks or not all(checks.values()):
        raise ValueError("melting result is not fully verified")
    melting_temperature = report.get("melting_temperature_k")
    if melting_temperature is None or not math.isfinite(float(melting_temperature)):
        raise ValueError("melting result has no finite melting temperature")
    return report


def interpolate(points: Sequence[Dict[str, Any]], temperature: float, field: str) -> float:
    rows = sorted((float(row["temperature_k"]), float(row[field])) for row in points)
    if len(rows) < 2:
        raise ValueError("at least two enthalpy temperatures are required")
    if temperature < rows[0][0] or temperature > rows[-1][0]:
        raise ValueError("melting temperature lies outside the sampled volume range")
    for sampled_temperature, value in rows:
        if math.isclose(temperature, sampled_temperature, abs_tol=1.0e-10):
            return value
    for (left_temperature, left_value), (right_temperature, right_value) in zip(
        rows, rows[1:]
    ):
        if left_temperature < temperature < right_temperature:
            fraction = (temperature - left_temperature) / (
                right_temperature - left_temperature
            )
            return left_value + fraction * (right_value - left_value)
    raise ValueError("could not bracket melting temperature")


def coexistence_geometry(report: Dict[str, Any]) -> Dict[str, Any]:
    temperature = float(report["melting_temperature_k"])
    points = report.get("enthalpy_points", [])
    if any(point.get("status") != "verified" for point in points):
        raise ValueError("all enthalpy points must be verified")
    solid_volume = interpolate(points, temperature, "solid_volume_per_atom_A3")
    liquid_volume = interpolate(points, temperature, "liquid_volume_per_atom_A3")
    if solid_volume <= 0.0 or liquid_volume <= 0.0:
        raise ValueError("phase volumes must be positive")

    lattice_a = (4.0 * solid_volume) ** (1.0 / 3.0)
    cross_section_length = 2.0 * lattice_a
    cross_section_area = cross_section_length**2
    solid_length = TEMPLATE_PHASE_ATOMS * solid_volume / cross_section_area
    liquid_length = TEMPLATE_PHASE_ATOMS * liquid_volume / cross_section_area
    total_length = solid_length + liquid_length
    split = solid_length / total_length
    repeat_factor = math.prod(TILE_REPEAT)
    tiled_solid_atoms = TEMPLATE_PHASE_ATOMS * repeat_factor
    tiled_liquid_atoms = TEMPLATE_PHASE_ATOMS * repeat_factor

    geometry = {
        "melting_temperature_k": temperature,
        "solid_volume_per_atom_A3": solid_volume,
        "liquid_volume_per_atom_A3": liquid_volume,
        "fcc_lattice_a_A": lattice_a,
        "template_size": list(TEMPLATE_SIZE),
        "template_phase_atoms": TEMPLATE_PHASE_ATOMS,
        "template_cell_lengths_A": [
            cross_section_length,
            cross_section_length,
            total_length,
        ],
        "solid_slab_length_A": solid_length,
        "liquid_slab_length_A": liquid_length,
        "split_fraction": split,
        "tile_repeat": list(TILE_REPEAT),
        "tiled_cell_lengths_A": [
            TILE_REPEAT[0] * cross_section_length,
            TILE_REPEAT[1] * cross_section_length,
            TILE_REPEAT[2] * total_length,
        ],
        "tiled_solid_atoms": tiled_solid_atoms,
        "tiled_liquid_atoms": tiled_liquid_atoms,
        "tiled_atoms": tiled_solid_atoms + tiled_liquid_atoms,
    }
    solid_volume_check = (
        cross_section_area * solid_length / TEMPLATE_PHASE_ATOMS
    )
    liquid_volume_check = (
        cross_section_area * liquid_length / TEMPLATE_PHASE_ATOMS
    )
    if not math.isclose(solid_volume_check, solid_volume, rel_tol=1.0e-12):
        raise AssertionError("solid slab geometry does not preserve its volume")
    if not math.isclose(liquid_volume_check, liquid_volume, rel_tol=1.0e-12):
        raise AssertionError("liquid slab geometry does not preserve its volume")
    if geometry["tiled_atoms"] != 1728:
        raise AssertionError("validation tiling must produce 1728 atoms")
    return geometry


def prepare_hot_template(
    out: Path,
    melting_path: Path,
    report: Dict[str, Any],
    geometry: Dict[str, Any],
    config_path: Path,
    hot_temperature: float,
    steps: int,
    csvr_tau: float,
    seed: int,
    mpi_ranks: int,
) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    element = load_json(ROOT / "config" / "al.json")
    element["lattice_a_angstrom"] = geometry["fcc_lattice_a_A"]
    atoms, labels = build_coexistence_seed(
        element,
        TEMPLATE_SIZE,
        split=0.5,
        liquid_displacement_angstrom=0.0,
        seed=seed,
    )
    atoms = reshape_coexistence_z_lengths(
        atoms,
        labels,
        geometry["solid_slab_length_A"],
        geometry["liquid_slab_length_A"],
        split=0.5,
    )
    atoms = constrain_regions(atoms, labels, ["solid_seed"])

    config = load_json(config_path)
    if str(config.get("of_kinetic", "")).lower() != "wt":
        raise ValueError("coexistence validation config must use WT")
    config.update(
        {
            "calculation": "md",
            "cal_force": 1,
            "cal_stress": 0,
            "md_type": "nvt",
            "md_nstep": steps,
            "md_dt": 1.0,
            "md_tfirst": hot_temperature,
            "md_tlast": hot_temperature,
            "md_thermostat": "csvr",
            "md_csvr_tau": csvr_tau,
            "md_dumpfreq": 5,
            "md_restartfreq": 100,
            "md_seed": seed,
            "mpirun_np": mpi_ranks,
        }
    )
    write_job(
        out,
        atoms,
        element,
        config,
        job_type="wt_coexistence_template_hot_melt",
        suffix=f"al64_wt_coexist_hot_T{int(hot_temperature)}",
        calculation="md",
        extra_metadata={
            "schema": "wt-coexistence-validation-hot-template-v1",
            "method": "continuous_fcc_interface_fixed_solid_hot_melt",
            "target_temperature_k": hot_temperature,
            "steps": steps,
            "csvr_tau_fs": csvr_tau,
            "seed": seed,
            "fixed_regions": ["solid_seed"],
            "melting_result": str(melting_path.resolve()),
            "predicted_melting_temperature_k": report["melting_temperature_k"],
            "geometry": geometry,
        },
        region_labels=labels,
    )
    return {
        "schema": "wt-coexistence-validation-plan-v1",
        "status": "hot_template_prepared",
        "target_kedf": "wt",
        "melting_result": str(melting_path.resolve()),
        "hot_template_run": str(out.resolve()),
        "geometry": geometry,
        "required_next_stages": [
            "verify_hot_template_two_phase",
            "cool_fixed_solid_to_predicted_melting_temperature",
            "verify_cooled_template_two_phase",
            "tile_3x3x3_to_1728_and_relax_fixed_solid",
            "release_and_thermalize_all_atoms",
            "run_nominal_and_bracketing_direct_coexistence",
            "verify_interface_migration_against_free_energy_prediction",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--melting", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "abacus_wt_ti_node04_cpu12.json",
    )
    parser.add_argument("--hot-temperature", type=float, default=1600.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--csvr-tau", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026072501)
    parser.add_argument("--mpi-ranks", type=int, default=16)
    args = parser.parse_args()

    melting_path = args.melting.resolve()
    report = verified_melting_report(melting_path)
    geometry = coexistence_geometry(report)
    out = args.out.resolve()
    plan = prepare_hot_template(
        out / "template_hot_T1600_fixsolid",
        melting_path,
        report,
        geometry,
        args.config.resolve(),
        args.hot_temperature,
        args.steps,
        args.csvr_tau,
        args.seed,
        args.mpi_ranks,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
