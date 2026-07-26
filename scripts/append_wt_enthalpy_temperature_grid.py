#!/usr/bin/env python3
"""Append a disjoint verified WT enthalpy grid to another verified grid."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

from scripts.merge_wt_enthalpy_temperature_grid import (
    DISCARD_FRACTIONS,
    _same,
    load_json,
    sha256_file,
    validate_verified_grid,
)
from scripts.summarize_wt_enthalpy_convergence import summarize


LABELS = ("d25", "d50", "d75")


def append_temperature_grid(
    *,
    base_series_by_discard: Dict[float, Dict[str, Any]],
    base_convergence: Dict[str, Any],
    appended_series_by_discard: Dict[float, Dict[str, Any]],
    appended_convergence: Dict[str, Any],
) -> tuple[Dict[float, Dict[str, Any]], Dict[str, Any]]:
    if set(base_series_by_discard) != set(DISCARD_FRACTIONS):
        raise ValueError("base series must contain d25, d50, and d75")
    if set(appended_series_by_discard) != set(DISCARD_FRACTIONS):
        raise ValueError("appended series must contain d25, d50, and d75")

    validate_verified_grid(base_series_by_discard, base_convergence)
    validate_verified_grid(appended_series_by_discard, appended_convergence)

    for field in (
        "maximum_discard_spread_mev_per_atom",
        "maximum_block_standard_error_mev_per_atom",
    ):
        if not _same(base_convergence[field], appended_convergence[field]):
            raise ValueError(f"base and appended convergence use different {field}")

    base_temperatures = {
        float(point["temperature_k"])
        for point in base_series_by_discard[0.5]["points"]
    }
    appended_temperatures = {
        float(point["temperature_k"])
        for point in appended_series_by_discard[0.5]["points"]
    }
    duplicates = sorted(base_temperatures & appended_temperatures)
    if duplicates:
        raise ValueError(f"temperature grids overlap: {duplicates}")

    merged: Dict[float, Dict[str, Any]] = {}
    for discard_fraction in DISCARD_FRACTIONS:
        base = base_series_by_discard[discard_fraction]
        appended = appended_series_by_discard[discard_fraction]
        payload = deepcopy(base)
        payload["points"] = sorted(
            [
                *deepcopy(base["points"]),
                *deepcopy(appended["points"]),
            ],
            key=lambda point: float(point["temperature_k"]),
        )
        payload["appended_verified_temperature_grid_k"] = sorted(
            appended_temperatures
        )
        merged[discard_fraction] = payload

    convergence = summarize(
        [merged[fraction] for fraction in DISCARD_FRACTIONS],
        maximum_discard_spread_mev_per_atom=float(
            base_convergence["maximum_discard_spread_mev_per_atom"]
        ),
        maximum_block_standard_error_mev_per_atom=float(
            base_convergence["maximum_block_standard_error_mev_per_atom"]
        ),
    )
    if convergence["status"] != "verified":
        raise RuntimeError("appended enthalpy grid is not verified")
    return merged, convergence


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append a verified disjoint d25/d50/d75 WT enthalpy grid"
    )
    for prefix in ("base", "append"):
        for label in LABELS:
            parser.add_argument(
                f"--{prefix}-{label}".replace("_", "-"),
                type=Path,
                required=True,
            )
        parser.add_argument(
            f"--{prefix}-convergence".replace("_", "-"),
            type=Path,
            required=True,
        )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_paths = {
        f"{prefix}_{label}": getattr(args, f"{prefix}_{label}").resolve()
        for prefix in ("base", "append")
        for label in (*LABELS, "convergence")
    }
    base_series = {
        fraction: load_json(source_paths[f"base_{label}"])
        for fraction, label in zip(DISCARD_FRACTIONS, LABELS)
    }
    appended_series = {
        fraction: load_json(source_paths[f"append_{label}"])
        for fraction, label in zip(DISCARD_FRACTIONS, LABELS)
    }
    merged, convergence = append_temperature_grid(
        base_series_by_discard=base_series,
        base_convergence=load_json(source_paths["base_convergence"]),
        appended_series_by_discard=appended_series,
        appended_convergence=load_json(source_paths["append_convergence"]),
    )

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    for discard_fraction, label in zip(DISCARD_FRACTIONS, LABELS):
        output = output_dir / f"enthalpy_{label}.json"
        output.write_text(
            json.dumps(merged[discard_fraction], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_paths[label] = output

    convergence["provenance"] = {
        "base_convergence": str(source_paths["base_convergence"]),
        "appended_convergence": str(source_paths["append_convergence"]),
        "merged_reports": [str(output_paths[label]) for label in LABELS],
    }
    convergence_path = output_dir / "discard_convergence_summary.json"
    convergence_path.write_text(
        json.dumps(convergence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths["convergence"] = convergence_path

    manifest = {
        "schema": "wt-melting-enthalpy-grid-append-v1",
        "status": "verified",
        "checks": {
            "base_grid_verified": True,
            "appended_grid_verified": True,
            "temperature_grids_disjoint": True,
            "merged_convergence_verified": convergence["status"] == "verified",
        },
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "delta_h": "H_liquid_minus_H_solid",
        },
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
    manifest_path = output_dir / "append_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
