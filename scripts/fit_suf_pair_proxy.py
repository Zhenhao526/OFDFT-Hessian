#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch


KB_EV_PER_K = 8.617333262145e-5


def suf_value_and_derivative(
    distance: torch.Tensor,
    *,
    p: int,
    sigma_angstrom: float,
    temperature_k: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduced_squared = (distance / sigma_angstrom) ** 2
    exponential = torch.exp(-reduced_squared)
    denominator = -torch.expm1(-reduced_squared)
    thermal_energy = KB_EV_PER_K * temperature_k
    value = -p * thermal_energy * torch.log(denominator)
    derivative = (
        -2.0
        * p
        * thermal_energy
        * distance
        / sigma_angstrom**2
        * exponential
        / denominator
    )
    return value, derivative


def core_value_and_derivative(
    distance: torch.Tensor,
    *,
    amplitude_ev: float,
    cutoff_angstrom: float,
    power: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduced = torch.clamp(1.0 - distance / cutoff_angstrom, min=0.0)
    valid = distance < cutoff_angstrom
    value = amplitude_ev * reduced**power * valid
    derivative = (
        -amplitude_ev
        * power
        / cutoff_angstrom
        * reduced ** (power - 1)
        * valid
    )
    return value, derivative


def basis_and_derivative(
    distance: torch.Tensor,
    centers: torch.Tensor,
    *,
    width_angstrom: float,
    cutoff_angstrom: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = distance[:, None] - centers[None, :]
    gaussian = torch.exp(-0.5 * (delta / width_angstrom) ** 2)
    angle = math.pi * distance / cutoff_angstrom
    cutoff_value = 0.5 * (torch.cos(angle) + 1.0)
    cutoff_derivative = -0.5 * math.pi / cutoff_angstrom * torch.sin(angle)
    basis = gaussian * cutoff_value[:, None]
    derivative = (
        -delta / width_angstrom**2 * gaussian * cutoff_value[:, None]
        + gaussian * cutoff_derivative[:, None]
    )
    return basis, derivative


def fit_proxy(
    *,
    p: int,
    suf_sigma_angstrom: float,
    temperature_k: float,
    basis_count: int,
    basis_min_angstrom: float,
    basis_max_angstrom: float,
    basis_width_angstrom: float,
    cutoff_angstrom: float,
    core_amplitude_ev: float,
    core_cutoff_angstrom: float,
    core_power: int,
    samples: int,
    ridge: float,
    sample_min_angstrom: float | None = None,
) -> Dict[str, object]:
    if basis_count < 3 or samples < basis_count * 3:
        raise ValueError("insufficient basis functions or radial samples")
    if sample_min_angstrom is None:
        sample_min_angstrom = basis_min_angstrom
    if not (0.0 < sample_min_angstrom < cutoff_angstrom):
        raise ValueError("sample_min_angstrom must lie inside the cutoff")
    distance = torch.linspace(
        sample_min_angstrom,
        cutoff_angstrom - 1.0e-6,
        samples,
        dtype=torch.float64,
    )
    centers = torch.linspace(
        basis_min_angstrom,
        basis_max_angstrom,
        basis_count,
        dtype=torch.float64,
    )
    target_value, target_derivative = suf_value_and_derivative(
        distance,
        p=p,
        sigma_angstrom=suf_sigma_angstrom,
        temperature_k=temperature_k,
    )
    core_value, core_derivative = core_value_and_derivative(
        distance,
        amplitude_ev=core_amplitude_ev,
        cutoff_angstrom=core_cutoff_angstrom,
        power=core_power,
    )
    basis, derivative = basis_and_derivative(
        distance,
        centers,
        width_angstrom=basis_width_angstrom,
        cutoff_angstrom=cutoff_angstrom,
    )

    energy_scale = max(float(target_value.max()), 0.1)
    derivative_scale = max(float(target_derivative.abs().max()), 0.1)
    design = torch.cat(
        (basis / energy_scale, derivative / derivative_scale),
        dim=0,
    )
    target = torch.cat(
        (
            (target_value - core_value) / energy_scale,
            (target_derivative - core_derivative) / derivative_scale,
        )
    )
    normal = design.T @ design
    normal += torch.eye(basis_count, dtype=torch.float64) * ridge
    coefficients = torch.linalg.solve(normal, design.T @ target)

    predicted_value = basis @ coefficients + core_value
    predicted_derivative = derivative @ coefficients + core_derivative
    value_error = predicted_value - target_value
    derivative_error = predicted_derivative - target_derivative
    fit_region = distance >= 2.0
    fit_value_error = value_error[fit_region]
    fit_derivative_error = derivative_error[fit_region]

    def rms(values: torch.Tensor) -> float:
        return math.sqrt(float((values * values).mean()))

    diagnostics = {
        "radial_domain_angstrom": [
            float(distance[0]),
            float(distance[-1]),
        ],
        "energy_rmse_ev_fit_region": rms(fit_value_error),
        "energy_max_abs_ev_fit_region": float(fit_value_error.abs().max()),
        "derivative_rmse_ev_per_angstrom_fit_region": rms(
            fit_derivative_error
        ),
        "derivative_max_abs_ev_per_angstrom_fit_region": float(
            fit_derivative_error.abs().max()
        ),
    }

    def proxy_value_and_derivative(distance_value: float) -> Tuple[float, float]:
        value_tensor = torch.tensor(
            [distance_value], dtype=torch.float64
        )
        item_basis, item_derivative = basis_and_derivative(
            value_tensor,
            centers,
            width_angstrom=basis_width_angstrom,
            cutoff_angstrom=cutoff_angstrom,
        )
        item_core, item_core_derivative = core_value_and_derivative(
            value_tensor,
            amplitude_ev=core_amplitude_ev,
            cutoff_angstrom=core_cutoff_angstrom,
            power=core_power,
        )
        return (
            float(item_basis @ coefficients + item_core),
            float(item_derivative @ coefficients + item_core_derivative),
        )

    short_value, short_derivative = proxy_value_and_derivative(1.5)
    sampled_value, _ = proxy_value_and_derivative(2.0)
    # The exact p=50 sUF potential rises by about 0.78 eV between 2.0 and
    # 1.5 A. Preserve that physical guard instead of applying the 1 eV
    # threshold used for unconstrained fitted target-potential references.
    short_range_guard = (
        short_value > sampled_value + 0.5 and short_derivative < 0.0
    )
    approximation_gate = (
        diagnostics["energy_rmse_ev_fit_region"] <= 2.0e-4
        and diagnostics["energy_max_abs_ev_fit_region"] <= 2.0e-3
        and diagnostics["derivative_rmse_ev_per_angstrom_fit_region"]
        <= 2.0e-3
        and diagnostics["derivative_max_abs_ev_per_angstrom_fit_region"]
        <= 2.0e-2
    )
    return {
        "schema": "mpn-radial-pair-reference-v1",
        "target_kedf": "lkt",
        "reference_kind": "suf_radial_proxy",
        "reference_phase": "liquid",
        "purpose": (
            "A fluid phase-specific bridge from analytic sUF to LKT; "
            "the exact sUF-to-proxy correction must be sampled independently"
        ),
        "suf": {
            "p": p,
            "sigma_angstrom": suf_sigma_angstrom,
            "temperature_k": temperature_k,
        },
        "model": {
            "centers_angstrom": centers.tolist(),
            "sigma_angstrom": basis_width_angstrom,
            "cutoff_angstrom": cutoff_angstrom,
            "coefficients_ev": [0.0, *coefficients.tolist()],
            "repulsive_core": {
                "amplitude_ev": core_amplitude_ev,
                "cutoff_angstrom": core_cutoff_angstrom,
                "power": core_power,
            },
        },
        "fit_parameters": {
            "sample_min_angstrom": sample_min_angstrom,
            "basis_min_angstrom": basis_min_angstrom,
            "basis_max_angstrom": basis_max_angstrom,
            "basis_count": basis_count,
            "basis_width_angstrom": basis_width_angstrom,
            "samples": samples,
            "ridge": ridge,
        },
        "validation": {
            "frames": 0,
            "energy_rmse_ev_per_atom": diagnostics[
                "energy_rmse_ev_fit_region"
            ],
            "force_rmse_ev_per_angstrom": diagnostics[
                "derivative_rmse_ev_per_angstrom_fit_region"
            ],
            "phase_energy_bias_ev_per_atom": {"liquid": 0.0},
        },
        "radial_fit_diagnostics": diagnostics,
        "short_range_diagnostic": {
            "u_1p5_ev": short_value,
            "du_dr_1p5_ev_per_angstrom": short_derivative,
            "u_2p0_ev": sampled_value,
            "minimum_u_1p5_minus_u_2p0_ev": 0.5,
        },
        "overlap_gate_passed": approximation_gate,
        "short_range_guard_passed": short_range_guard,
        "reference_gate_passed": approximation_gate and short_range_guard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the exact sUF radial potential to ABACUS's pair basis"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--suf-sigma", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--basis-count", type=int, default=81)
    parser.add_argument("--basis-min", type=float, default=1.4)
    parser.add_argument(
        "--sample-min",
        type=float,
        help=(
            "minimum fitted radius; defaults to --basis-min. "
            "Set this above --basis-min to retain basis support at the "
            "fit boundary without fitting away the short-range core"
        ),
    )
    parser.add_argument("--basis-max", type=float, default=6.3)
    parser.add_argument("--basis-width", type=float, default=0.10)
    parser.add_argument("--cutoff", type=float, default=6.5)
    parser.add_argument("--core-amplitude", type=float, default=100.0)
    parser.add_argument("--core-cutoff", type=float, default=2.0)
    parser.add_argument("--core-power", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    args = parser.parse_args()

    result = fit_proxy(
        p=args.p,
        suf_sigma_angstrom=args.suf_sigma,
        temperature_k=args.temperature,
        basis_count=args.basis_count,
        basis_min_angstrom=args.basis_min,
        basis_max_angstrom=args.basis_max,
        basis_width_angstrom=args.basis_width,
        cutoff_angstrom=args.cutoff,
        core_amplitude_ev=args.core_amplitude,
        core_cutoff_angstrom=args.core_cutoff,
        core_power=args.core_power,
        samples=args.samples,
        ridge=args.ridge,
        sample_min_angstrom=args.sample_min,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["reference_gate_passed"]:
        raise SystemExit("sUF pair proxy failed its radial approximation gate")


if __name__ == "__main__":
    main()
