#!/usr/bin/env python3
"""Merge an archived verified anchor-temperature enthalpy into a current grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

from scripts.summarize_wt_enthalpy_convergence import summarize


DISCARD_FRACTIONS = (0.25, 0.5, 0.75)
ENERGY_DEFINITION = "sampled_total_energy_plus_external_pv"
SERIES_SCHEMA = "wt-zero-pressure-fusion-enthalpy-series-v2"
CONVERGENCE_SCHEMA = "wt-enthalpy-discard-convergence-summary-v2"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-11, abs_tol=1.0e-11)


def _point_at_temperature(
    convergence: Dict[str, Any], temperature_k: float
) -> Dict[str, Any]:
    matches = [
        point
        for point in convergence.get("points", [])
        if _same(point["temperature_k"], temperature_k)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one convergence point at {temperature_k:g} K, found {len(matches)}"
        )
    point = matches[0]
    if point.get("status") != "verified":
        raise ValueError(f"archived {temperature_k:g} K point is not verified")
    return point


def _report_at_discard(
    point: Dict[str, Any], discard_fraction: float
) -> Dict[str, Any]:
    matches = [
        report
        for report in point.get("reports", [])
        if _same(report["discard_fraction"], discard_fraction)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one report at discard={discard_fraction:g}, found {len(matches)}"
        )
    report = matches[0]
    if report.get("status") != "verified":
        raise ValueError(
            f"archived discard={discard_fraction:g} report is not verified"
        )
    return report


def validate_verified_grid(
    series_by_discard: Dict[float, Dict[str, Any]],
    convergence: Dict[str, Any],
    excluded_temperature_k: float | None = None,
) -> None:
    if convergence.get("schema") != CONVERGENCE_SCHEMA:
        raise ValueError("current convergence uses an unsupported schema")
    if convergence.get("status") != "verified":
        raise ValueError("current convergence is not verified")
    if convergence.get("enthalpy_energy_definition") != ENERGY_DEFINITION:
        raise ValueError("current convergence uses an unsupported energy definition")

    reference_temperatures: tuple[float, ...] | None = None
    for discard_fraction, series in sorted(series_by_discard.items()):
        if series.get("schema") != SERIES_SCHEMA:
            raise ValueError("current enthalpy series uses an unsupported schema")
        if series.get("status") != "verified":
            raise ValueError(f"current discard={discard_fraction:g} series is not verified")
        if series.get("enthalpy_energy_definition") != ENERGY_DEFINITION:
            raise ValueError("current series uses an unsupported energy definition")
        if not _same(series["discard_fraction"], discard_fraction):
            raise ValueError("discard label does not match series content")
        points = sorted(series.get("points", []), key=lambda row: row["temperature_k"])
        temperatures = tuple(float(point["temperature_k"]) for point in points)
        if (
            excluded_temperature_k is not None
            and any(
                _same(temperature, excluded_temperature_k)
                for temperature in temperatures
            )
        ):
            raise ValueError("current series already contains the excluded temperature")
        if any(point.get("status") != "verified" for point in points):
            raise ValueError("current enthalpy series contains an unverified point")
        if reference_temperatures is None:
            reference_temperatures = temperatures
        elif temperatures != reference_temperatures:
            raise ValueError("current enthalpy series use different temperature grids")

        convergence_points = {
            float(point["temperature_k"]): point
            for point in convergence.get("points", [])
        }
        if set(temperatures) != set(convergence_points):
            raise ValueError("current series and convergence temperature grids differ")
        for point in points:
            temperature = float(point["temperature_k"])
            convergence_point = convergence_points[temperature]
            if convergence_point.get("status") != "verified":
                raise ValueError("current convergence contains an unverified point")
            report = _report_at_discard(convergence_point, discard_fraction)
            if not _same(
                report["delta_h_mev_per_atom"],
                1000.0 * float(point["delta_h_ev_per_atom"]),
            ):
                raise ValueError("current series and convergence enthalpies differ")
            if not _same(
                report["block_standard_error_mev_per_atom"],
                point["block_standard_error_mev_per_atom"],
            ):
                raise ValueError("current series and convergence block errors differ")
            if not _same(
                report["half_drift_mev_per_atom"],
                point["half_drift_mev_per_atom"],
            ):
                raise ValueError("current series and convergence half drifts differ")


def merge_temperature_grid(
    *,
    anchor_temperature_k: float,
    archived_convergence: Dict[str, Any],
    current_series_by_discard: Dict[float, Dict[str, Any]],
    current_convergence: Dict[str, Any],
) -> tuple[Dict[float, Dict[str, Any]], Dict[str, Any]]:
    if set(current_series_by_discard) != set(DISCARD_FRACTIONS):
        raise ValueError("current series must contain d25, d50, and d75")
    if archived_convergence.get("schema") != CONVERGENCE_SCHEMA:
        raise ValueError("archived convergence uses an unsupported schema")
    if archived_convergence.get("enthalpy_energy_definition") != ENERGY_DEFINITION:
        raise ValueError("archived convergence uses an unsupported energy definition")

    validate_verified_grid(
        current_series_by_discard,
        current_convergence,
        excluded_temperature_k=anchor_temperature_k,
    )
    archived_point = _point_at_temperature(
        archived_convergence, anchor_temperature_k
    )

    merged: Dict[float, Dict[str, Any]] = {}
    for discard_fraction in DISCARD_FRACTIONS:
        report = _report_at_discard(archived_point, discard_fraction)
        anchor_point = {
            "temperature_k": float(anchor_temperature_k),
            "delta_h_ev_per_atom": float(report["delta_h_mev_per_atom"]) / 1000.0,
            "block_standard_error_mev_per_atom": float(
                report["block_standard_error_mev_per_atom"]
            ),
            "half_drift_mev_per_atom": float(report["half_drift_mev_per_atom"]),
            "liquid_minus_solid_temperature_mean_difference_k": float(
                report["liquid_minus_solid_temperature_mean_difference_k"]
            ),
            "checks": {
                "archived_point_verified": True,
                "archived_report_verified": True,
                "positive_fusion_enthalpy": float(
                    report["delta_h_mev_per_atom"]
                )
                > 0.0,
            },
            "status": "verified",
            "source": "archived_verified_convergence_point",
        }
        current = current_series_by_discard[discard_fraction]
        payload = {
            **current,
            "points": sorted(
                [anchor_point, *current["points"]],
                key=lambda point: float(point["temperature_k"]),
            ),
            "merged_anchor_temperature_k": float(anchor_temperature_k),
        }
        merged[discard_fraction] = payload

    convergence = summarize(
        [merged[fraction] for fraction in DISCARD_FRACTIONS],
        maximum_discard_spread_mev_per_atom=float(
            current_convergence["maximum_discard_spread_mev_per_atom"]
        ),
        maximum_block_standard_error_mev_per_atom=float(
            current_convergence["maximum_block_standard_error_mev_per_atom"]
        ),
    )
    if convergence["status"] != "verified":
        raise RuntimeError("merged three-temperature convergence is not verified")
    return merged, convergence


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge a verified archived enthalpy point into current d25/d50/d75"
    )
    parser.add_argument("--anchor-temperature", type=float, default=900.0)
    parser.add_argument("--anchor-combination", type=Path, required=True)
    parser.add_argument("--archived-convergence", type=Path, required=True)
    parser.add_argument("--current-d25", type=Path, required=True)
    parser.add_argument("--current-d50", type=Path, required=True)
    parser.add_argument("--current-d75", type=Path, required=True)
    parser.add_argument("--current-convergence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_paths = {
        "anchor_combination": args.anchor_combination.resolve(),
        "archived_convergence": args.archived_convergence.resolve(),
        "current_d25": args.current_d25.resolve(),
        "current_d50": args.current_d50.resolve(),
        "current_d75": args.current_d75.resolve(),
        "current_convergence": args.current_convergence.resolve(),
    }
    anchor = load_json(source_paths["anchor_combination"])
    if (
        anchor.get("schema") != "wt-melting-free-energy-combination-v2"
        or anchor.get("status") != "anchor_temperature_free_energy_verified"
        or not anchor.get("checks")
        or not all(anchor["checks"].values())
        or not _same(anchor["temperature_k"], args.anchor_temperature)
        or float(anchor["free_energy_ev_per_atom"]["liquid_minus_solid"]) <= 0.0
    ):
        raise ValueError("free-energy anchor is not verified with positive liquid-solid DeltaG")

    current_series = {
        0.25: load_json(source_paths["current_d25"]),
        0.5: load_json(source_paths["current_d50"]),
        0.75: load_json(source_paths["current_d75"]),
    }
    merged, convergence = merge_temperature_grid(
        anchor_temperature_k=args.anchor_temperature,
        archived_convergence=load_json(source_paths["archived_convergence"]),
        current_series_by_discard=current_series,
        current_convergence=load_json(source_paths["current_convergence"]),
    )

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for discard_fraction, label in zip(DISCARD_FRACTIONS, ("d25", "d50", "d75")):
        output = output_dir / f"enthalpy_{label}.json"
        output.write_text(
            json.dumps(merged[discard_fraction], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_paths[label] = output
    convergence["provenance"] = {
        "archived_convergence": str(source_paths["archived_convergence"]),
        "current_convergence": str(source_paths["current_convergence"]),
        "merged_reports": [str(output_paths[label]) for label in ("d25", "d50", "d75")],
    }
    convergence_path = output_dir / "discard_convergence_summary.json"
    convergence_path.write_text(
        json.dumps(convergence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths["convergence"] = convergence_path

    manifest = {
        "schema": "wt-melting-enthalpy-grid-merge-v1",
        "status": "verified",
        "checks": {
            "anchor_combination_verified": True,
            "archived_anchor_enthalpy_verified": True,
            "current_enthalpy_convergence_verified": True,
            "merged_enthalpy_convergence_verified": convergence["status"]
            == "verified",
        },
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "delta_h": "H_liquid_minus_H_solid",
        },
        "anchor_temperature_k": float(args.anchor_temperature),
        "temperature_grid_k": [
            float(point["temperature_k"]) for point in merged[0.5]["points"]
        ],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
        },
    }
    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
