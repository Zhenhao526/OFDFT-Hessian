#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


KB_EV_PER_K = 8.617333262145e-5
KB_J_PER_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27


def close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def finite_n_ideal_gas_correction_ev_per_atom(
    natoms: int, temperature_k: float
) -> float:
    """Exact N! ideal-gas term minus its thermodynamic-limit Stirling form."""
    if natoms < 1 or temperature_k <= 0.0:
        raise ValueError("natoms and temperature must be positive")
    return (
        KB_EV_PER_K
        * temperature_k
        * (math.lgamma(natoms + 1) - natoms * math.log(natoms) + natoms)
        / natoms
    )


def reconstruct_analytic(analytic: Dict[str, Any]) -> Dict[str, float]:
    temperature = float(analytic["temperature_k"])
    natoms = int(analytic["natoms"])
    mass_kg = float(analytic["mass_amu"]) * AMU_KG
    thermal_energy = KB_EV_PER_K * temperature
    beta_si = 1.0 / (KB_J_PER_K * temperature)
    spring = float(analytic["einstein"]["spring_constant_ev_per_angstrom2"])
    spring_si = spring * EV_J / 1.0e-20
    einstein_partition = (
        beta_si**2 * spring_si * PLANCK_J_S**2 / (4.0 * math.pi**2 * mass_kg)
    ) ** 1.5
    einstein_uncorrected = thermal_energy * math.log(einstein_partition)
    volume_per_atom_m3 = (
        float(analytic["einstein"]["solid_volume_per_atom_angstrom3"]) * 1.0e-30
    )
    sum_mu2_over_k = 1.0 / (natoms * spring_si)
    cm_log = math.log(
        volume_per_atom_m3
        * (beta_si / (2.0 * math.pi * sum_mu2_over_k)) ** 1.5
    )
    cm_correction = -thermal_energy * cm_log / natoms

    density = 1.0 / float(analytic["suf"]["liquid_volume_per_atom_angstrom3"])
    sigma = float(analytic["suf"]["sigma_angstrom"])
    reduced_density = 0.5 * (math.pi * sigma**2) ** 1.5 * density
    thermal_wavelength = (
        PLANCK_J_S
        / math.sqrt(2.0 * math.pi * mass_kg * KB_J_PER_K * temperature)
        * 1.0e10
    )
    ideal = thermal_energy * (math.log(density * thermal_wavelength**3) - 1.0)
    excess = thermal_energy * float(analytic["suf"]["beta_excess_free_energy"])
    return {
        "einstein_uncorrected_ev_per_atom": einstein_uncorrected,
        "einstein_cm_correction_ev_per_atom": cm_correction,
        "einstein_total_ev_per_atom": einstein_uncorrected + cm_correction,
        "suf_density_angstrom_minus3": density,
        "suf_reduced_density_x": reduced_density,
        "thermal_wavelength_angstrom": thermal_wavelength,
        "suf_ideal_ev_per_atom": ideal,
        "suf_excess_ev_per_atom": excess,
        "suf_total_ev_per_atom": ideal + excess,
    }


def inspect_anchor_window_provenance(anchor: Dict[str, Any]) -> Dict[str, Any]:
    windows = anchor.get("windows", [])
    if not windows:
        raise ValueError("reference anchor contains no TI windows")
    rows = []
    for window in windows:
        path = Path(window["path"]).resolve()
        with path.open(encoding="utf-8") as stream:
            first_line = next((line for line in stream if line.strip()), None)
        if first_line is None:
            raise ValueError(f"reference trajectory is empty: {path}")
        record = json.loads(first_line)
        positions = record.get("positions_angstrom")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"reference trajectory has no atom positions: {path}")
        rows.append(
            {
                "path": str(path),
                "natoms": len(positions),
                "window_lambda": float(window["lambda"]),
                "trajectory_lambda": float(record["lambda"]),
                "initial_temperature_k": float(record["temperature_k"]),
                "production_temperature_mean_k": float(
                    window["temperature_mean_k"]
                ),
            }
        )
    return {
        "windows": rows,
        "natoms": sorted({row["natoms"] for row in rows}),
        "lambdas": sorted(row["window_lambda"] for row in rows),
        "lambda_labels_match_trajectories": all(
            close(row["window_lambda"], row["trajectory_lambda"])
            for row in rows
        ),
    }


def audit(
    analytic: Dict[str, Any],
    solid_anchor: Dict[str, Any],
    liquid_anchor: Dict[str, Any],
) -> Dict[str, Any]:
    reconstructed = reconstruct_analytic(analytic)
    temperature = float(analytic["temperature_k"])
    natoms = int(analytic["natoms"])
    solid_provenance = inspect_anchor_window_provenance(solid_anchor)
    liquid_provenance = inspect_anchor_window_provenance(liquid_anchor)
    solid_has_strict_status = "status" in solid_anchor
    liquid_has_strict_status = "status" in liquid_anchor
    checks = {
        "analytic_schema": analytic.get("schema")
        == "mpn-analytic-reference-free-energies-v1",
        "solid_anchor_schema": solid_anchor.get("schema")
        == "mpn-classical-reference-pair-ti-analysis-v1",
        "liquid_anchor_schema": liquid_anchor.get("schema")
        == "mpn-classical-reference-pair-ti-analysis-v1",
        "solid_windows_stable": solid_anchor.get("all_windows_stable") is True,
        "liquid_windows_stable": liquid_anchor.get("all_windows_stable") is True,
        "solid_strict_status": not solid_has_strict_status
        or solid_anchor.get("status") == "verified",
        "liquid_strict_status": not liquid_has_strict_status
        or liquid_anchor.get("status") == "verified",
        "solid_target_windows_stable": solid_anchor.get(
            "target_windows_stable", True
        )
        is True,
        "liquid_target_windows_stable": liquid_anchor.get(
            "target_windows_stable", True
        )
        is True,
        "solid_window_natoms_match_analytic": solid_provenance["natoms"]
        == [natoms],
        "liquid_window_natoms_match_analytic": liquid_provenance["natoms"]
        == [natoms],
        "solid_window_initial_temperatures_match_analytic": all(
            close(row["initial_temperature_k"], temperature)
            for row in solid_provenance["windows"]
        ),
        "liquid_window_initial_temperatures_match_analytic": all(
            close(row["initial_temperature_k"], temperature)
            for row in liquid_provenance["windows"]
        ),
        "solid_window_mean_temperatures_within_25_K": all(
            abs(row["production_temperature_mean_k"] - temperature) <= 25.0
            for row in solid_provenance["windows"]
        ),
        "liquid_window_mean_temperatures_within_25_K": all(
            abs(row["production_temperature_mean_k"] - temperature) <= 25.0
            for row in liquid_provenance["windows"]
        ),
        "solid_window_lambda_labels_match_trajectories": solid_provenance[
            "lambda_labels_match_trajectories"
        ],
        "liquid_window_lambda_labels_match_trajectories": liquid_provenance[
            "lambda_labels_match_trajectories"
        ],
        "solid_lambda_grid_has_endpoints": close(
            solid_provenance["lambdas"][0], 0.0
        )
        and close(solid_provenance["lambdas"][-1], 1.0),
        "liquid_lambda_grid_has_endpoints": close(
            liquid_provenance["lambdas"][0], 0.0
        )
        and close(liquid_provenance["lambdas"][-1], 1.0),
        "solid_declared_natoms_match": "natoms" not in solid_anchor
        or int(solid_anchor["natoms"]) == natoms,
        "liquid_declared_natoms_match": "natoms" not in liquid_anchor
        or int(liquid_anchor["natoms"]) == natoms,
        "solid_declared_temperature_matches": "target_temperature_k"
        not in solid_anchor
        or close(float(solid_anchor["target_temperature_k"]), temperature),
        "liquid_declared_temperature_matches": "target_temperature_k"
        not in liquid_anchor
        or close(float(liquid_anchor["target_temperature_k"]), temperature),
        "solid_overlap_ess": float(
            solid_anchor.get("minimum_adjacent_effective_sample_fraction", 1.0)
        )
        >= 0.05,
        "liquid_overlap_ess": float(
            liquid_anchor.get("minimum_adjacent_effective_sample_fraction", 1.0)
        )
        >= 0.05,
        "solid_overlap_closure": float(
            solid_anchor.get("maximum_adjacent_closure_mev_per_atom", 0.0)
        )
        <= 2.0,
        "liquid_overlap_closure": float(
            liquid_anchor.get("maximum_adjacent_closure_mev_per_atom", 0.0)
        )
        <= 2.0,
        "einstein_uncorrected_reproduced": close(
            reconstructed["einstein_uncorrected_ev_per_atom"],
            float(analytic["einstein"]["uncorrected_free_energy_ev_per_atom"]),
        ),
        "einstein_cm_reproduced": close(
            reconstructed["einstein_cm_correction_ev_per_atom"],
            float(analytic["einstein"]["center_of_mass_correction_ev_per_atom"]),
        ),
        "einstein_total_reproduced": close(
            reconstructed["einstein_total_ev_per_atom"],
            float(analytic["einstein"]["corrected_free_energy_ev_per_atom"]),
        ),
        "suf_density_reproduced": close(
            reconstructed["suf_density_angstrom_minus3"],
            float(analytic["suf"]["number_density_angstrom_minus3"]),
        ),
        "suf_reduced_density_reproduced": close(
            reconstructed["suf_reduced_density_x"],
            float(analytic["suf"]["reduced_density_x"]),
        ),
        "suf_ideal_reproduced": close(
            reconstructed["suf_ideal_ev_per_atom"],
            float(analytic["suf"]["ideal_gas_free_energy_ev_per_atom"]),
        ),
        "suf_excess_reproduced": close(
            reconstructed["suf_excess_ev_per_atom"],
            float(analytic["suf"]["excess_free_energy_ev_per_atom"]),
        ),
        "suf_total_reproduced": close(
            reconstructed["suf_total_ev_per_atom"],
            float(analytic["suf"]["total_free_energy_ev_per_atom"]),
        ),
        "solid_anchor_block_se_below_1_mev": float(
            solid_anchor["block_standard_error_mev_per_atom"]
        )
        <= 1.0,
        "liquid_anchor_block_se_below_1_mev": float(
            liquid_anchor["block_standard_error_mev_per_atom"]
        )
        <= 1.0,
        "solid_anchor_quadrature_below_1_mev": float(
            solid_anchor["quadrature_difference_mev_per_atom"]
        )
        <= 1.0,
        "liquid_anchor_quadrature_below_1_mev": float(
            liquid_anchor["quadrature_difference_mev_per_atom"]
        )
        <= 1.0,
        "solid_anchor_half_drift_below_2_mev": float(
            solid_anchor["half_drift_mev_per_atom"]
        )
        <= 2.0,
        "liquid_anchor_half_drift_below_2_mev": float(
            liquid_anchor["half_drift_mev_per_atom"]
        )
        <= 2.0,
    }
    solid_pair = (
        reconstructed["einstein_total_ev_per_atom"]
        + float(solid_anchor["delta_f_pair_minus_reference_simpson_ev_per_atom"])
    )
    liquid_pair = (
        reconstructed["suf_total_ev_per_atom"]
        + float(liquid_anchor["delta_f_pair_minus_reference_simpson_ev_per_atom"])
    )
    ideal_finite_n = finite_n_ideal_gas_correction_ev_per_atom(natoms, temperature)
    polson_scale = KB_EV_PER_K * temperature * math.log(natoms) / natoms
    return {
        "schema": "mpn-reference-free-energy-anchor-audit-v1",
        "status": "verified_with_finite_size_sensitivity"
        if all(checks.values())
        else "gate_failed",
        "temperature_k": temperature,
        "natoms": natoms,
        "checks": checks,
        "anchor_window_provenance": {
            "solid": solid_provenance,
            "liquid": liquid_provenance,
        },
        "reconstructed_analytic": reconstructed,
        "absolute_pair_free_energy_ev_per_atom": {
            "solid": solid_pair,
            "liquid_thermodynamic_limit_ideal_convention": liquid_pair,
            "liquid_minus_solid_raw": liquid_pair - solid_pair,
        },
        "absolute_pair_free_energy_mev_per_atom": {
            "liquid_minus_solid_raw": 1000.0 * (liquid_pair - solid_pair),
        },
        "finite_size_sensitivity": {
            "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom": 1000.0
            * ideal_finite_n,
            "polson_kbt_ln_n_over_n_mev_per_atom": 1000.0 * polson_scale,
            "liquid_minus_solid_with_ideal_n_factor_only_mev_per_atom": 1000.0
            * (liquid_pair + ideal_finite_n - solid_pair),
            "interpretation": (
                "The exact ideal-gas N! correction is a sensitivity, not a complete "
                "finite-size correction; the solid and sUF excess 1/N terms require "
                "size extrapolation or an explicit uncertainty allowance."
            ),
        },
        "anchor_error_budget_mev_per_atom": {
            "solid_block_standard_error": float(
                solid_anchor["block_standard_error_mev_per_atom"]
            ),
            "solid_quadrature_difference": float(
                solid_anchor["quadrature_difference_mev_per_atom"]
            ),
            "solid_half_drift": float(solid_anchor["half_drift_mev_per_atom"]),
            "liquid_block_standard_error": float(
                liquid_anchor["block_standard_error_mev_per_atom"]
            ),
            "liquid_quadrature_difference": float(
                liquid_anchor["quadrature_difference_mev_per_atom"]
            ),
            "liquid_half_drift": float(liquid_anchor["half_drift_mev_per_atom"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the analytic and TI reference anchors")
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--solid-anchor", type=Path, required=True)
    parser.add_argument("--liquid-anchor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.analytic.read_text(encoding="utf-8")),
        json.loads(args.solid_anchor.read_text(encoding="utf-8")),
        json.loads(args.liquid_anchor.read_text(encoding="utf-8")),
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
