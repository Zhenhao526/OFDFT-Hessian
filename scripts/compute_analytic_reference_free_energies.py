#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path


KB_EV_PER_K = 8.617333262145e-5
KB_J_PER_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def suf_reduced_excess_free_energy(splines_path: Path, p: int, x: float) -> float:
    data = runpy.run_path(str(splines_path))
    splines = data["splines"]
    sums = {
        1: data["sum_spline1"],
        25: data["sum_spline25"],
        50: data["sum_spline50"],
        75: data["sum_spline75"],
        100: data["sum_spline100"],
    }
    if p not in splines:
        raise ValueError("supported sUF p values are 1, 25, 50, 75, and 100")
    if not 0.0 < x <= 4.0:
        raise ValueError("sUF reduced density x must be in (0, 4]")
    if x < 0.1:
        index = int(x * 400)
    elif x < 1.0:
        index = 40 + int(x * 40 - 4)
    elif x < 4.0:
        index = 76 + int(x * 10 - 10)
    else:
        index = 105
    coefficient = splines[p][index]
    if x < 0.0025:
        return coefficient[0] * x**2 / 2.0 + coefficient[1] * x
    if x < 0.1:
        x0 = 0.0025 * int(x * 400)
        on_knot = x * 10000 % 25 == 0
    elif x < 1.0:
        x0 = 0.025 * int(x * 40)
        on_knot = x * 1000 % 25 == 0
    else:
        x0 = 0.1 * int(x * 10)
        on_knot = x * 100 % 10 == 0
    if on_knot:
        return sums[p][index - 1]
    return (
        sums[p][index - 1]
        + coefficient[0] * (x**2 - x0**2) / 2.0
        + coefficient[1] * (x - x0)
        + (coefficient[2] - 1.0) * math.log(x / x0)
        - coefficient[3] * (1.0 / x - 1.0 / x0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate analytic Einstein-crystal and sUF free energies"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--natoms", type=int, required=True)
    parser.add_argument("--mass-amu", type=float, required=True)
    parser.add_argument("--solid-volume-per-atom", type=float, required=True)
    parser.add_argument("--spring-constant", type=float, required=True)
    parser.add_argument("--liquid-volume-per-atom", type=float, required=True)
    parser.add_argument("--suf-p", type=int, required=True)
    parser.add_argument("--suf-sigma", type=float, required=True)
    parser.add_argument("--calphy-splines", type=Path, required=True)
    args = parser.parse_args()

    temperature = args.temperature
    thermal_energy = KB_EV_PER_K * temperature
    beta_si = 1.0 / (KB_J_PER_K * temperature)
    mass_kg = args.mass_amu * AMU_KG
    spring_si = args.spring_constant * EV_J / 1.0e-20
    einstein_partition = (
        beta_si**2 * spring_si * PLANCK_J_S**2
        / (4.0 * math.pi**2 * mass_kg)
    ) ** 1.5
    einstein_uncorrected = thermal_energy * math.log(einstein_partition)
    solid_volume_m3 = args.solid_volume_per_atom * 1.0e-30
    sum_mu2_over_k = 1.0 / (args.natoms * spring_si)
    cm_log = math.log(
        solid_volume_m3
        * (beta_si / (2.0 * math.pi * sum_mu2_over_k)) ** 1.5
    )
    cm_correction = -thermal_energy * cm_log / args.natoms
    einstein = einstein_uncorrected + cm_correction

    density = 1.0 / args.liquid_volume_per_atom
    x = 0.5 * (math.pi * args.suf_sigma**2) ** 1.5 * density
    beta_f_excess = suf_reduced_excess_free_energy(
        args.calphy_splines, args.suf_p, x
    )
    thermal_wavelength = (
        PLANCK_J_S
        / math.sqrt(2.0 * math.pi * mass_kg * KB_J_PER_K * temperature)
        * 1.0e10
    )
    ideal = thermal_energy * (
        math.log(density * thermal_wavelength**3) - 1.0
    )
    excess = thermal_energy * beta_f_excess
    result = {
        "schema": "mpn-analytic-reference-free-energies-v1",
        "temperature_k": temperature,
        "natoms": args.natoms,
        "mass_amu": args.mass_amu,
        "einstein": {
            "solid_volume_per_atom_angstrom3": args.solid_volume_per_atom,
            "spring_constant_ev_per_angstrom2": args.spring_constant,
            "uncorrected_free_energy_ev_per_atom": einstein_uncorrected,
            "center_of_mass_correction_ev_per_atom": cm_correction,
            "corrected_free_energy_ev_per_atom": einstein,
        },
        "suf": {
            "liquid_volume_per_atom_angstrom3": args.liquid_volume_per_atom,
            "number_density_angstrom_minus3": density,
            "p": args.suf_p,
            "sigma_angstrom": args.suf_sigma,
            "reduced_density_x": x,
            "thermal_wavelength_angstrom": thermal_wavelength,
            "ideal_gas_free_energy_ev_per_atom": ideal,
            "beta_excess_free_energy": beta_f_excess,
            "excess_free_energy_ev_per_atom": excess,
            "total_free_energy_ev_per_atom": ideal + excess,
        },
        "suf_spline_source": {
            "path": str(args.calphy_splines.resolve()),
            "sha256": sha256(args.calphy_splines),
            "implementation": "Calphy spline tables derived from Leite et al. supplementary material",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
