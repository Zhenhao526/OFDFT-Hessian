#!/usr/bin/env python3
"""Extrapolate KEDF phase volumes after correcting residual pressures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("phase_results", report.get("results", []))
    return {str(row["phase"]): row for row in rows}


def verified_reference(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text())
    if report.get("status") != "all_confirmations_passed":
        raise ValueError(f"reference report is not verified: {path}")
    rows = phase_rows(report)
    if set(rows) != {"solid", "liquid"}:
        raise ValueError("reference report must contain solid and liquid rows")
    if any(row.get("status") != "confirmation_passed" for row in rows.values()):
        raise ValueError("reference report phases are not verified")
    return report, rows


def pressure_mean(row: dict[str, Any]) -> float:
    pressure = row.get("pressure_kbar", row.get("pressure_last_half_kbar", {}))
    return float(pressure["mean"])


def slope_point(path: Path, phase: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    rows = phase_rows(report)
    if phase not in rows:
        raise ValueError(f"{path} has no {phase} row")
    row = rows[phase]
    checks = row.get("checks", {})
    phase_status = row.get("phase_status")
    if phase_status != f"{phase}_verified":
        raise ValueError(f"{path} {phase} structure is not verified")
    if checks and not checks.get("phase_verified", checks.get("phase_confirmation_passed", True)):
        raise ValueError(f"{path} {phase} phase check failed")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "temperature_K": float(report["temperature_K"]),
        "phase": phase,
        "volume_per_atom_A3": float(row["volume_per_atom_A3"]),
        "pressure_kbar": pressure_mean(row),
        "phase_status": phase_status,
    }


def pressure_volume_slope(
    first_path: Path,
    second_path: Path,
    phase: str,
) -> dict[str, Any]:
    points = [
        slope_point(first_path.resolve(), phase),
        slope_point(second_path.resolve(), phase),
    ]
    if abs(points[0]["temperature_K"] - points[1]["temperature_K"]) > 1.0e-9:
        raise ValueError(f"{phase} pressure-volume points have different temperatures")
    delta_volume = points[1]["volume_per_atom_A3"] - points[0]["volume_per_atom_A3"]
    if abs(delta_volume) < 1.0e-8:
        raise ValueError(f"{phase} pressure-volume points do not span volume")
    slope = (points[1]["pressure_kbar"] - points[0]["pressure_kbar"]) / delta_volume
    if slope >= 0.0:
        raise ValueError(f"{phase} dP/dV must be negative")
    return {
        "phase": phase,
        "temperature_K": points[0]["temperature_K"],
        "points": points,
        "dP_dV_kbar_per_A3_per_atom": slope,
    }


def extrapolate(
    lower_path: Path,
    upper_path: Path,
    solid_slope_paths: tuple[Path, Path],
    liquid_slope_paths: tuple[Path, Path],
    targets: list[float],
) -> dict[str, Any]:
    lower_path = lower_path.resolve()
    upper_path = upper_path.resolve()
    lower_report, lower_rows = verified_reference(lower_path)
    upper_report, upper_rows = verified_reference(upper_path)
    target_kedf = str(lower_report["target_kedf"]).lower()
    if str(upper_report["target_kedf"]).lower() != target_kedf:
        raise ValueError("reference reports use different KEDFs")
    lower_t = float(lower_report["temperature_K"])
    upper_t = float(upper_report["temperature_K"])
    if lower_t >= upper_t:
        raise ValueError("reference temperatures must increase")
    slopes = {
        "solid": pressure_volume_slope(*solid_slope_paths, "solid"),
        "liquid": pressure_volume_slope(*liquid_slope_paths, "liquid"),
    }
    corrected: dict[str, list[dict[str, float]]] = {}
    thermal_slopes: dict[str, float] = {}
    for phase in ("solid", "liquid"):
        dp_dv = slopes[phase]["dP_dV_kbar_per_A3_per_atom"]
        rows = []
        for temperature, source in (
            (lower_t, lower_rows[phase]),
            (upper_t, upper_rows[phase]),
        ):
            volume = float(source["volume_per_atom_A3"])
            pressure = pressure_mean(source)
            zero_volume = volume - pressure / dp_dv
            rows.append(
                {
                    "temperature_K": temperature,
                    "reported_volume_per_atom_A3": volume,
                    "residual_pressure_kbar": pressure,
                    "zero_pressure_volume_per_atom_A3": zero_volume,
                }
            )
        corrected[phase] = rows
        thermal_slopes[phase] = (
            rows[1]["zero_pressure_volume_per_atom_A3"]
            - rows[0]["zero_pressure_volume_per_atom_A3"]
        ) / (upper_t - lower_t)
    predictions = []
    for temperature in targets:
        if temperature <= upper_t:
            raise ValueError("target temperatures must exceed the upper reference")
        predictions.append(
            {
                "temperature_K": temperature,
                "volumes_per_atom_A3": {
                    phase: corrected[phase][1]["zero_pressure_volume_per_atom_A3"]
                    + thermal_slopes[phase] * (temperature - upper_t)
                    for phase in ("solid", "liquid")
                },
            }
        )
    return {
        "schema": "kedf-zero-pressure-volume-residual-corrected-extrapolation-v1",
        "status": "candidate_volumes_require_md_confirmation",
        "target_kedf": target_kedf,
        "reference_reports": [
            {"path": str(lower_path), "sha256": sha256(lower_path)},
            {"path": str(upper_path), "sha256": sha256(upper_path)},
        ],
        "pressure_volume_slopes": slopes,
        "residual_pressure_corrections": corrected,
        "thermal_slopes_A3_per_atom_K": thermal_slopes,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", type=Path, required=True)
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--solid-slope-report", type=Path, action="append", required=True)
    parser.add_argument("--liquid-slope-report", type=Path, action="append", required=True)
    parser.add_argument("--target", type=float, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.solid_slope_report) != 2 or len(args.liquid_slope_report) != 2:
        parser.error("each phase requires exactly two slope reports")
    result = extrapolate(
        args.lower,
        args.upper,
        tuple(args.solid_slope_report),
        tuple(args.liquid_slope_report),
        args.target,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
