#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from mpn_melting.gibbs_helmholtz import (
    gibbs_over_temperature,
    solve_melting_temperature,
)


KB_EV_PER_K = 8.617333262145e-5


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def enthalpy_points(
    convergence: dict[str, Any], label: str
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    rows = convergence["windows"][label]["points"]
    means = {float(row["temperature_k"]): float(row["delta_h_ev_per_atom"]) for row in rows}
    errors = {
        float(row["temperature_k"]): float(row["block_standard_error_mev_per_atom"]) / 1000.0
        for row in rows
    }
    means[900.0] = 0.5 * (means[850.0] + means[950.0])
    errors[900.0] = 0.5 * math.sqrt(errors[850.0] ** 2 + errors[950.0] ** 2)
    return sorted(means.items()), sorted(errors.items())


def solve_or_none(anchor_g: float, points: list[tuple[float, float]]) -> float | None:
    try:
        return solve_melting_temperature(900.0, anchor_g, points)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a preliminary waived XWM anchor and melting root"
    )
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--classical-reports", type=Path, required=True)
    parser.add_argument("--target-diagnostics", type=Path, required=True)
    parser.add_argument("--enthalpy-convergence", type=Path, required=True)
    parser.add_argument("--waiver", type=Path, required=True)
    parser.add_argument("--zero-pressure-solid", type=Path, required=True)
    parser.add_argument("--zero-pressure-eps005", type=Path, required=True)
    parser.add_argument("--zero-pressure-eps003", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    analytic = load(args.analytic)
    target = load(args.target_diagnostics)
    convergence = load(args.enthalpy_convergence)
    waiver = load(args.waiver)
    zero_solid = load(args.zero_pressure_solid)
    zero_eps005 = load(args.zero_pressure_eps005)
    zero_eps003 = load(args.zero_pressure_eps003)
    classical = {
        phase: {
            label: load(args.classical_reports / f"{phase}_{label}.json")
            for label in ("d25", "d50", "d75")
        }
        for phase in ("solid", "liquid")
    }
    bridges = {
        label: load(args.classical_reports / f"solid_bridge2_{label}.json")
        for label in ("d25", "d50", "d75")
    }

    waived_ids = {row["id"] for row in waiver["waived_items"]}
    required_waivers = {
        "classical_reference_solid_d75_terminal_bridge_closure",
        "multitemp_1000K_primary_d50_half_drift",
        "multitemp_950K_d75_half_drift_and_temperature",
    }
    checks = {
        "waiver_status": waiver.get("status")
        == "user_authorized_specific_gate_waivers_recorded",
        "required_waivers_present": required_waivers <= waived_ids,
        "target_ti_verified": target.get("overall_status") == "verified",
        "liquid_classical_reports_verified": all(
            classical["liquid"][label].get("status") == "verified"
            for label in ("d25", "d50", "d75")
        ),
        "classical_quadrature_within_2_mev": all(
            report["quadrature_difference_mev_per_atom"] <= 2.0
            for phase in classical.values()
            for report in phase.values()
        ),
        "solid_d50_bridge_verified": bridges["d50"].get("status") == "verified",
        "solid_d75_bridge_failure_is_waived": bridges["d75"].get("status")
        == "overlap_gate_failed",
        "solid_zero_pressure_confirmed": zero_solid.get("status")
        == "zero_pressure_confirmed",
        "liquid_zero_pressure_root_consistent": abs(
            float(zero_eps005["roots"]["liquid"]["volume_per_atom_a3"])
            - float(zero_eps003["roots"]["liquid"]["volume_per_atom_a3"])
        )
        < 0.00030,
        "solid_zero_pressure_root_consistent": abs(
            float(zero_eps005["roots"]["solid"]["volume_per_atom_a3"])
            - float(zero_eps003["roots"]["solid"]["volume_per_atom_a3"])
        )
        < 0.00030,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"input verification failed: {failed}")

    target_correction = {
        phase: float(target[phase]["simpson_meV_per_atom"]) / 1000.0
        for phase in ("solid", "liquid")
    }
    analytic_energy = {
        "solid": float(analytic["einstein"]["corrected_free_energy_ev_per_atom"]),
        "liquid": float(analytic["suf"]["total_free_energy_ev_per_atom"]),
    }
    anchors: dict[str, dict[str, float]] = {}
    for label in ("d25", "d50", "d75"):
        phase_free_energy = {
            phase: analytic_energy[phase]
            + float(
                classical[phase][label][
                    "delta_f_pair_minus_reference_simpson_ev_per_atom"
                ]
            )
            + target_correction[phase]
            for phase in ("solid", "liquid")
        }
        anchors[label] = {
            **phase_free_energy,
            "liquid_minus_solid": phase_free_energy["liquid"]
            - phase_free_energy["solid"],
        }

    d50_legs = [classical[phase]["d50"] for phase in ("solid", "liquid")]
    statistical = rss(
        [float(row["block_standard_error_mev_per_atom"]) for row in d50_legs]
        + [
            float(target[phase]["block_standard_error_meV_per_atom"])
            for phase in ("solid", "liquid")
        ]
    )
    quadrature = rss(
        [float(row["quadrature_difference_mev_per_atom"]) for row in d50_legs]
        + [
            float(target[phase]["quadrature_difference_meV_per_atom"])
            for phase in ("solid", "liquid")
        ]
    )
    temporal = rss(
        [float(row["half_drift_mev_per_atom"]) for row in d50_legs]
        + [float(target[phase]["half_drift_meV_per_atom"]) for phase in ("solid", "liquid")]
    )
    discard_spreads = {
        phase: max(
            classical[phase][label][
                "delta_f_pair_minus_reference_simpson_mev_per_atom"
            ]
            for label in ("d25", "d50", "d75")
        )
        - min(
            classical[phase][label][
                "delta_f_pair_minus_reference_simpson_mev_per_atom"
            ]
            for label in ("d25", "d50", "d75")
        )
        for phase in ("solid", "liquid")
    }
    discard_rss = rss(list(discard_spreads.values()))
    ti_conservative = rss([statistical, quadrature, temporal, discard_rss])
    natoms = int(analytic["natoms"])
    temperature = float(analytic["temperature_k"])
    exact_finite_n = (
        KB_EV_PER_K
        * temperature
        * (math.lgamma(natoms + 1) - natoms * math.log(natoms) + natoms)
        / natoms
        * 1000.0
    )
    polson = KB_EV_PER_K * temperature * math.log(natoms) / natoms * 1000.0
    finite_size = max(abs(exact_finite_n), abs(polson))
    combined_conservative = ti_conservative + finite_size

    melting: dict[str, Any] = {}
    for label in ("d25", "d50", "d75"):
        points, _ = enthalpy_points(convergence, label)
        grid = [
            {"temperature_k": t, "delta_g_ev_per_atom": t * g_over_t}
            for t, g_over_t in gibbs_over_temperature(
                900.0, anchors[label]["liquid_minus_solid"], points
            )
        ]
        melting[label] = {
            "enthalpy_points_ev_per_atom": [
                {"temperature_k": t, "liquid_minus_solid": h} for t, h in points
            ],
            "delta_g_grid": grid,
            "melting_temperature_k": solve_or_none(
                anchors[label]["liquid_minus_solid"], points
            ),
        }

    d50_points, d50_errors = enthalpy_points(convergence, "d50")
    rng = random.Random(args.seed)
    bootstrap_roots: list[float] = []
    for _ in range(args.bootstrap_samples):
        sampled_anchor = rng.gauss(
            anchors["d50"]["liquid_minus_solid"], statistical / 1000.0
        )
        sampled_native = {
            t: rng.gauss(h, dict(d50_errors)[t])
            for t, h in d50_points
            if t != 900.0
        }
        sampled_native[900.0] = 0.5 * (
            sampled_native[850.0] + sampled_native[950.0]
        )
        sampled_points = sorted(sampled_native.items())
        root = solve_or_none(sampled_anchor, sampled_points)
        if root is not None:
            bootstrap_roots.append(root)
    if len(bootstrap_roots) < 0.9 * args.bootstrap_samples:
        raise RuntimeError("fewer than 90% of bootstrap samples bracket a melting root")

    conservative_roots = {
        side: solve_or_none(
            anchors["d50"]["liquid_minus_solid"] + shift * combined_conservative / 1000.0,
            d50_points,
        )
        for side, shift in (("anchor_minus", -1.0), ("anchor_plus", 1.0))
    }
    source_paths = {
        "analytic": args.analytic,
        "target_diagnostics": args.target_diagnostics,
        "enthalpy_convergence": args.enthalpy_convergence,
        "waiver": args.waiver,
        "zero_pressure_solid": args.zero_pressure_solid,
        "zero_pressure_eps005": args.zero_pressure_eps005,
        "zero_pressure_eps003": args.zero_pressure_eps003,
        **{
            f"classical_{phase}_{label}": args.classical_reports
            / f"{phase}_{label}.json"
            for phase in ("solid", "liquid")
            for label in ("d25", "d50", "d75")
        },
        **{
            f"solid_bridge2_{label}": args.classical_reports
            / f"solid_bridge2_{label}.json"
            for label in ("d25", "d50", "d75")
        },
    }
    result = {
        "schema": "xwm-preliminary-waived-anchor-and-melting-v1",
        "status": "preliminary_computed_with_user_authorized_gate_waivers",
        "reporting_label": "preliminary XWM analysis with user-authorized gate waivers",
        "checks": checks,
        "thermodynamic_convention": "DeltaG = G_liquid - G_solid; root at DeltaG=0",
        "temperature_anchor_k": 900.0,
        "free_energy_path_ev_per_atom": {
            "analytic_reference": analytic_energy,
            "pair_minus_reference_d50": {
                phase: classical[phase]["d50"][
                    "delta_f_pair_minus_reference_simpson_ev_per_atom"
                ]
                for phase in ("solid", "liquid")
            },
            "xwm_minus_pair": target_correction,
        },
        "anchor_variants_ev_per_atom": anchors,
        "primary_anchor": {
            "discard_window": "d50",
            "delta_g_liquid_minus_solid_ev_per_atom": anchors["d50"][
                "liquid_minus_solid"
            ],
            "delta_g_liquid_minus_solid_mev_per_atom": anchors["d50"][
                "liquid_minus_solid"
            ]
            * 1000.0,
        },
        "uncertainty_budget_mev_per_atom": {
            "statistical_rss": statistical,
            "quadrature_rss": quadrature,
            "temporal_drift_rss": temporal,
            "classical_discard_spreads": discard_spreads,
            "discard_spread_rss": discard_rss,
            "ti_combined_conservative": ti_conservative,
            "exact_finite_n_ideal_sensitivity": exact_finite_n,
            "polson_kbt_ln_n_over_n": polson,
            "finite_size_systematic_allowance": finite_size,
            "combined_conservative": combined_conservative,
        },
        "melting_results": melting,
        "primary_d50_melting_temperature_k": melting["d50"][
            "melting_temperature_k"
        ],
        "discard_window_tm_spread_k": max(
            row["melting_temperature_k"] for row in melting.values()
        )
        - min(row["melting_temperature_k"] for row in melting.values()),
        "parametric_bootstrap": {
            "seed": args.seed,
            "requested_samples": args.bootstrap_samples,
            "bracketed_samples": len(bootstrap_roots),
            "sampling_model": "independent Gaussian block-SE errors for d50 anchor and enthalpy points",
            "mean_k": sum(bootstrap_roots) / len(bootstrap_roots),
            "median_k": percentile(bootstrap_roots, 0.5),
            "ci95_k": [
                percentile(bootstrap_roots, 0.025),
                percentile(bootstrap_roots, 0.975),
            ],
        },
        "conservative_anchor_shift_roots_k": conservative_roots,
        "gate_waiver": waiver,
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "anchor_mev_per_atom": result["primary_anchor"][
                    "delta_g_liquid_minus_solid_mev_per_atom"
                ],
                "tm_d25_d50_d75_k": {
                    label: melting[label]["melting_temperature_k"]
                    for label in ("d25", "d50", "d75")
                },
                "bootstrap": result["parametric_bootstrap"],
                "combined_conservative_mev_per_atom": combined_conservative,
                "conservative_anchor_shift_roots_k": conservative_roots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
