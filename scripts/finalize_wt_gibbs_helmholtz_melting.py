#!/usr/bin/env python3
"""Finalize verified WT Gibbs-Helmholtz melting results and diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


DISCARDS = {"d25": 0.25, "d50": 0.5, "d75": 0.75}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_roots(interval: Dict[str, Any]) -> list[float]:
    roots = [scenario.get("root_k") for scenario in interval["scenarios"]]
    if any(root is None for root in roots):
        raise ValueError("an uncertainty scenario is not temperature-bracketed")
    return [float(root) for root in roots]


def aggregate_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if set(results) != set(DISCARDS):
        raise ValueError("results must contain d25, d50, and d75")

    reference_grid: tuple[float, ...] | None = None
    anchor_temperature: float | None = None
    anchor_delta_g: float | None = None
    nominal_roots: Dict[str, float] = {}
    statistical_roots: list[float] = []
    conservative_roots: list[float] = []
    profiles: Dict[str, list[Dict[str, float]]] = {}
    checks: Dict[str, bool] = {}

    for label, expected_discard in DISCARDS.items():
        result = results[label]
        prefix = f"{label}_"
        checks[prefix + "schema_verified"] = (
            result.get("schema") == "wt-gibbs-helmholtz-melting-v2"
        )
        checks[prefix + "status_verified"] = result.get("status") == "verified"
        checks[prefix + "all_checks_verified"] = bool(result.get("checks")) and all(
            result["checks"].values()
        )
        convention = result.get("thermodynamic_convention", {})
        checks[prefix + "sign_convention_verified"] = (
            convention.get("delta_g") == "G_liquid_minus_G_solid"
            and convention.get("delta_h") == "H_liquid_minus_H_solid"
            and convention.get("root_definition") == "delta_g(T_m)=0"
        )

        points = sorted(
            result["enthalpy_points"], key=lambda point: float(point["temperature_k"])
        )
        grid = tuple(float(point["temperature_k"]) for point in points)
        discard = float(
            load_discard_fraction(result, expected_discard=expected_discard)
        )
        checks[prefix + "discard_verified"] = abs(discard - expected_discard) < 1.0e-12
        if reference_grid is None:
            reference_grid = grid
        checks[prefix + "temperature_grid_verified"] = grid == reference_grid

        current_anchor_temperature = float(result["anchor_temperature_k"])
        current_anchor_delta_g = float(result["anchor_delta_g_ev_per_atom"])
        if anchor_temperature is None:
            anchor_temperature = current_anchor_temperature
            anchor_delta_g = current_anchor_delta_g
        checks[prefix + "anchor_verified"] = (
            current_anchor_temperature == anchor_temperature
            and current_anchor_delta_g == anchor_delta_g
            and current_anchor_delta_g > 0.0
        )

        profile = sorted(
            result["gibbs_profile"],
            key=lambda row: float(row["temperature_k"]),
        )
        profiles[label] = profile
        profile_temperatures = tuple(float(row["temperature_k"]) for row in profile)
        profile_values = [float(row["delta_g_ev_per_atom"]) for row in profile]
        checks[prefix + "profile_grid_verified"] = profile_temperatures == grid
        checks[prefix + "profile_monotonic_verified"] = all(
            right < left for left, right in zip(profile_values, profile_values[1:])
        )
        checks[prefix + "profile_sign_change_verified"] = (
            profile_values[0] > 0.0 and profile_values[-1] < 0.0
        )

        root = result.get("melting_temperature_k")
        if root is None:
            raise ValueError(f"{label} nominal root is not bracketed")
        nominal_roots[label] = float(root)
        statistical_roots.extend(_interval_roots(result["statistical_interval"]))
        conservative_roots.extend(_interval_roots(result["conservative_interval"]))

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"final result checks failed: {', '.join(failed)}")

    assert reference_grid is not None
    assert anchor_temperature is not None
    assert anchor_delta_g is not None
    central = nominal_roots["d50"]
    nominal_envelope = (min(nominal_roots.values()), max(nominal_roots.values()))
    statistical_envelope = (min(statistical_roots), max(statistical_roots))
    conservative_envelope = (min(conservative_roots), max(conservative_roots))
    grid_bounds = (min(reference_grid), max(reference_grid))
    all_roots = [
        *nominal_roots.values(),
        *statistical_roots,
        *conservative_roots,
    ]
    checks["all_roots_inside_temperature_grid"] = all(
        grid_bounds[0] <= root <= grid_bounds[1] for root in all_roots
    )
    if not checks["all_roots_inside_temperature_grid"]:
        raise ValueError("one or more roots lie outside the temperature grid")

    return {
        "schema": "wt-gibbs-helmholtz-melting-final-v1",
        "status": "verified",
        "checks": checks,
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "delta_h": "H_liquid_minus_H_solid",
            "gibbs_helmholtz_equation": "d(delta_g/T)/dT=-delta_h/T^2",
            "root_definition": "delta_g(T_m)=0",
        },
        "temperature_grid_k": list(reference_grid),
        "anchor_temperature_k": anchor_temperature,
        "anchor_delta_g_mev_per_atom": 1000.0 * anchor_delta_g,
        "central_discard": "d50",
        "melting_temperature_k": central,
        "nominal_roots_k": nominal_roots,
        "nominal_discard_envelope_k": {
            "lower": nominal_envelope[0],
            "upper": nominal_envelope[1],
            "minus_from_central": central - nominal_envelope[0],
            "plus_from_central": nominal_envelope[1] - central,
        },
        "d50_statistical_sensitivity_k": {
            "lower": float(results["d50"]["statistical_interval"]["lower_k"]),
            "upper": float(results["d50"]["statistical_interval"]["upper_k"]),
            "minus_from_central": central
            - float(results["d50"]["statistical_interval"]["lower_k"]),
            "plus_from_central": float(
                results["d50"]["statistical_interval"]["upper_k"]
            )
            - central,
        },
        "all_discard_statistical_envelope_k": {
            "lower": statistical_envelope[0],
            "upper": statistical_envelope[1],
        },
        "d50_conservative_sensitivity_k": {
            "lower": float(results["d50"]["conservative_interval"]["lower_k"]),
            "upper": float(results["d50"]["conservative_interval"]["upper_k"]),
        },
        "all_discard_conservative_envelope_k": {
            "lower": conservative_envelope[0],
            "upper": conservative_envelope[1],
            "minus_from_central": central - conservative_envelope[0],
            "plus_from_central": conservative_envelope[1] - central,
        },
        "uncertainty_interpretation": {
            "statistical": "coherent statistical sensitivity envelope",
            "discard": "nominal sensitivity to production discard fraction",
            "conservative": (
                "full envelope across discard fractions, anchor systematic terms, "
                "block error, half drift, and discard spread"
            ),
            "probabilistic_confidence_interval": False,
        },
        "profiles": profiles,
    }


def load_discard_fraction(
    result: Dict[str, Any], *, expected_discard: float
) -> float:
    points = result.get("enthalpy_points", [])
    provenance_path = result.get("provenance", {}).get("enthalpy_series")
    if provenance_path:
        path = Path(provenance_path)
        if path.exists():
            return float(load_json(path)["discard_fraction"])
    if points and "discard_fraction" in points[0]:
        return float(points[0]["discard_fraction"])
    return expected_discard


def write_curve_csv(
    output: Path, results: Dict[str, Dict[str, Any]]
) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["discard_label", "temperature_k", "delta_g_mev_per_atom"]
        )
        for label in DISCARDS:
            for row in results[label]["gibbs_profile"]:
                writer.writerow(
                    [
                        label,
                        float(row["temperature_k"]),
                        1000.0 * float(row["delta_g_ev_per_atom"]),
                    ]
                )


def write_plot(
    output: Path,
    results: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    colors = {"d25": "#267365", "d50": "#b33a3a", "d75": "#315a9b"}
    width, height = 900.0, 600.0
    left, right, top, bottom = 90.0, 35.0, 55.0, 75.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    temperatures = [
        float(row["temperature_k"])
        for result in results.values()
        for row in result["gibbs_profile"]
    ]
    values = [
        1000.0 * float(row["delta_g_ev_per_atom"])
        for result in results.values()
        for row in result["gibbs_profile"]
    ]
    x_min, x_max = min(temperatures), max(temperatures)
    y_pad = max(1.0, 0.12 * (max(values) - min(values)))
    y_min, y_max = min(values) - y_pad, max(values) + y_pad

    def x_map(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_map(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<text x="450" y="30" text-anchor="middle" '
            'font-family="sans-serif" font-size="20" fill="#202020">'
            "WT Al melting point: Gibbs-Helmholtz integration</text>"
        ),
    ]
    conservative = summary["all_discard_conservative_envelope_k"]
    statistical = summary["all_discard_statistical_envelope_k"]
    svg.append(
        (
            f'<rect x="{x_map(conservative["lower"]):.3f}" y="{top:.3f}" '
            f'width="{x_map(conservative["upper"]) - x_map(conservative["lower"]):.3f}" '
            f'height="{plot_height:.3f}" fill="#d9d9d9" fill-opacity="0.55"/>'
        )
    )
    svg.append(
        (
            f'<rect x="{x_map(statistical["lower"]):.3f}" y="{top:.3f}" '
            f'width="{x_map(statistical["upper"]) - x_map(statistical["lower"]):.3f}" '
            f'height="{plot_height:.3f}" fill="#a6a6a6" fill-opacity="0.60"/>'
        )
    )

    x_ticks = sorted(set(temperatures))
    for tick in x_ticks:
        x = x_map(tick)
        svg.extend(
            [
                (
                    f'<line x1="{x:.3f}" y1="{top:.3f}" x2="{x:.3f}" '
                    f'y2="{top + plot_height:.3f}" stroke="#e6e6e6" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x:.3f}" y="{top + plot_height + 25:.3f}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="13" '
                    f'fill="#303030">{tick:.0f}</text>'
                ),
            ]
        )
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5.0
        y = y_map(value)
        svg.extend(
            [
                (
                    f'<line x1="{left:.3f}" y1="{y:.3f}" '
                    f'x2="{left + plot_width:.3f}" y2="{y:.3f}" '
                    'stroke="#e6e6e6" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 12:.3f}" y="{y + 4:.3f}" '
                    'text-anchor="end" font-family="sans-serif" font-size="13" '
                    f'fill="#303030">{value:.1f}</text>'
                ),
            ]
        )

    zero_y = y_map(0.0)
    central_x = x_map(float(summary["melting_temperature_k"]))
    svg.extend(
        [
            (
                f'<line x1="{left:.3f}" y1="{zero_y:.3f}" '
                f'x2="{left + plot_width:.3f}" y2="{zero_y:.3f}" '
                'stroke="#202020" stroke-width="1.4"/>'
            ),
            (
                f'<line x1="{central_x:.3f}" y1="{top:.3f}" '
                f'x2="{central_x:.3f}" y2="{top + plot_height:.3f}" '
                f'stroke="{colors["d50"]}" stroke-width="1.4" '
                'stroke-dasharray="7 5"/>'
            ),
        ]
    )
    for label in DISCARDS:
        profile = results[label]["gibbs_profile"]
        coordinates = [
            (
                x_map(float(row["temperature_k"])),
                y_map(1000.0 * float(row["delta_g_ev_per_atom"])),
            )
            for row in profile
        ]
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates)
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[label]}" '
            'stroke-width="2.4"/>'
        )
        svg.extend(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.5" '
            f'fill="{colors[label]}" stroke="#ffffff" stroke-width="1"/>'
            for x, y in coordinates
        )

    svg.extend(
        [
            (
                f'<rect x="{left:.3f}" y="{top:.3f}" width="{plot_width:.3f}" '
                f'height="{plot_height:.3f}" fill="none" stroke="#303030" '
                'stroke-width="1.2"/>'
            ),
            (
                '<text x="450" y="580" text-anchor="middle" '
                'font-family="sans-serif" font-size="16" fill="#202020">'
                "Temperature (K)</text>"
            ),
            (
                '<text x="24" y="290" text-anchor="middle" '
                'font-family="sans-serif" font-size="16" fill="#202020" '
                'transform="rotate(-90 24 290)">Delta G liq-sol (meV/atom)</text>'
            ),
        ]
    )
    legend_x, legend_y = left + 18.0, top + 22.0
    for index, label in enumerate(DISCARDS):
        y = legend_y + 22.0 * index
        svg.extend(
            [
                (
                    f'<line x1="{legend_x:.3f}" y1="{y:.3f}" '
                    f'x2="{legend_x + 24.0:.3f}" y2="{y:.3f}" '
                    f'stroke="{colors[label]}" stroke-width="2.4"/>'
                ),
                (
                    f'<text x="{legend_x + 31.0:.3f}" y="{y + 4.0:.3f}" '
                    'font-family="sans-serif" font-size="13" fill="#202020">'
                    f"{label}</text>"
                ),
            ]
        )
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_report(output: Path, summary: Dict[str, Any]) -> None:
    central = summary["melting_temperature_k"]
    stat = summary["d50_statistical_sensitivity_k"]
    conservative = summary["all_discard_conservative_envelope_k"]
    nominal = summary["nominal_roots_k"]
    text = f"""# WT Al melting point handoff

Status: verified

## Result

- Central estimate (d50): {central:.6f} K
- d50 statistical sensitivity: {stat['lower']:.6f} to {stat['upper']:.6f} K
- Full conservative sensitivity envelope: {conservative['lower']:.6f} to {conservative['upper']:.6f} K
- Nominal discard roots: d25={nominal['d25']:.6f} K, d50={nominal['d50']:.6f} K, d75={nominal['d75']:.6f} K

The intervals are sensitivity envelopes, not probabilistic confidence intervals.

## Method

The free-energy sign is DeltaG = G_liquid - G_solid and the enthalpy sign is
DeltaH = H_liquid - H_solid. The calculation integrates
d(DeltaG/T)/dT = -DeltaH/T^2 from the verified 900 K absolute free-energy
anchor over the verified 900, 975, 1050, and 1100 K enthalpy grid. The melting
temperature is the root DeltaG = 0.

## Verification

All d25, d50, and d75 nominal, statistical, and conservative roots are inside
the 900-1100 K grid. Every input phase trajectory and enthalpy point passed its
physical and convergence gates before inclusion.
"""
    output.write_text(text, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Finalize three verified WT Gibbs-Helmholtz melting results"
    )
    for label in DISCARDS:
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--append-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result_paths = {
        label: getattr(args, label).resolve() for label in DISCARDS
    }
    results = {label: load_json(path) for label, path in result_paths.items()}
    append_manifest_path = args.append_manifest.resolve()
    append_manifest = load_json(append_manifest_path)
    if (
        append_manifest.get("schema") != "wt-melting-enthalpy-grid-append-v1"
        or append_manifest.get("status") != "verified"
        or not append_manifest.get("checks")
        or not all(append_manifest["checks"].values())
    ):
        raise ValueError("appended temperature-grid manifest is not verified")
    for output in append_manifest["outputs"].values():
        path = Path(output["path"])
        if sha256_file(path) != output["sha256"]:
            raise ValueError(f"appended-grid output SHA mismatch: {path}")

    summary = aggregate_results(results)
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["provenance"] = {
        "append_manifest": {
            "path": str(append_manifest_path),
            "sha256": sha256_file(append_manifest_path),
        },
        "solver_results": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in result_paths.items()
        },
    }

    summary_path = output_dir / "wt_al_melting_final.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_curve_csv(output_dir / "gibbs_helmholtz_curve.csv", results)
    write_plot(output_dir / "gibbs_helmholtz_curve.svg", results, summary)
    write_report(output_dir / "WT_AL_MELTING_HANDOFF.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
