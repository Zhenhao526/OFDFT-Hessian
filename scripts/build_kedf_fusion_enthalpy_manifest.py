#!/usr/bin/env python3
"""Build a KEDF fusion-enthalpy manifest with zero-pressure provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["phase"]): row for row in rows}


def build_point(
    enthalpy_root: Path,
    zero_pressure_path: Path,
) -> dict[str, Any]:
    enthalpy_root = enthalpy_root.resolve()
    zero_pressure_path = zero_pressure_path.resolve()
    manifest_path = enthalpy_root / "confirmation_manifest.json"
    summary_path = enthalpy_root / "confirmation_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    zero_pressure = json.loads(zero_pressure_path.read_text(encoding="utf-8"))

    if summary.get("status") != "volume_confirmation_verified":
        raise ValueError("enthalpy trajectory physical gate is not verified")
    if zero_pressure.get("status") != "all_confirmations_passed":
        raise ValueError("zero-pressure confirmation is not verified")
    method = str(manifest.get("target_kedf", "")).lower()
    if method not in {"xwm", "lkt"}:
        raise ValueError(f"unsupported target KEDF {method!r}")
    if str(summary.get("target_kedf", "")).lower() != method:
        raise ValueError("enthalpy manifest and summary use different KEDFs")
    if str(zero_pressure.get("target_kedf", "")).lower() != method:
        raise ValueError("enthalpy and zero-pressure reports use different KEDFs")
    temperature = float(manifest["temperature_K"])
    if not math.isclose(
        float(summary["temperature_K"]), temperature, abs_tol=1.0e-9
    ) or not math.isclose(
        float(zero_pressure["temperature_K"]), temperature, abs_tol=1.0e-9
    ):
        raise ValueError("enthalpy and zero-pressure temperatures differ")

    phases = phase_map(manifest["phases"])
    results = phase_map(summary["results"])
    zero_phases = phase_map(zero_pressure["phase_results"])
    if any(
        set(rows) != {"solid", "liquid"}
        for rows in (phases, results, zero_phases)
    ):
        raise ValueError("solid and liquid evidence is required")
    if any(results[phase].get("status") != "passed" for phase in phases):
        raise ValueError("enthalpy trajectory phase gates did not all pass")
    if any(
        zero_phases[phase].get("status") != "confirmation_passed"
        for phase in phases
    ):
        raise ValueError("zero-pressure phase gates did not all pass")
    for phase in phases:
        if not math.isclose(
            float(phases[phase]["volume_per_atom_A3"]),
            float(zero_phases[phase]["volume_per_atom_A3"]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            raise ValueError(f"{phase} enthalpy and zero-pressure volumes differ")

    trajectory_pressure_required = bool(manifest.get("stress_available", False))
    if method == "lkt" and not trajectory_pressure_required:
        raise ValueError("LKT enthalpy trajectories must include analytic stress")
    if method == "xwm" and trajectory_pressure_required:
        raise ValueError("XWM enthalpy trajectories unexpectedly claim stress")

    steps = int(manifest["steps"])
    return {
        "temperature_k": temperature,
        "target_pressure_kbar": float(
            manifest.get("target_pressure_kbar", 0.0)
        ),
        "steps": steps,
        "solid_run": phases["solid"]["run"],
        "liquid_run": phases["liquid"]["run"],
        "solid_runs": [phases["solid"]["run"]],
        "liquid_runs": [phases["liquid"]["run"]],
        "segment_steps": [steps],
        "solid_volume_per_atom_A3": float(
            phases["solid"]["volume_per_atom_A3"]
        ),
        "liquid_volume_per_atom_A3": float(
            phases["liquid"]["volume_per_atom_A3"]
        ),
        "trajectory_pressure_required": trajectory_pressure_required,
        "zero_pressure_verified": True,
        "zero_pressure_provenance": {
            "path": str(zero_pressure_path),
            "sha256": sha256(zero_pressure_path),
            "pressure_sources": {
                phase: zero_phases[phase].get("pressure_source")
                for phase in ("solid", "liquid")
            },
        },
        "enthalpy_trajectory_provenance": {
            "confirmation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
            },
            "confirmation_summary": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
        },
    }


def build_manifest(
    pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one enthalpy/zero-pressure pair is required")
    points = [build_point(root, pressure) for root, pressure in pairs]
    methods = set()
    for point in points:
        summary_path = Path(
            point["enthalpy_trajectory_provenance"]["confirmation_summary"][
                "path"
            ]
        )
        methods.add(
            str(json.loads(summary_path.read_text())["target_kedf"]).lower()
        )
    if len(methods) != 1:
        raise ValueError("enthalpy points use different KEDFs")
    temperatures = [float(point["temperature_k"]) for point in points]
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("enthalpy points contain duplicate temperatures")
    return {
        "schema": "kedf-zero-pressure-fusion-enthalpy-manifest-v1",
        "target_kedf": methods.pop(),
        "points": sorted(points, key=lambda point: point["temperature_k"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point",
        action="append",
        nargs=2,
        metavar=("ENTHALPY_ROOT", "ZERO_PRESSURE_SUMMARY"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(
        [(Path(root), Path(pressure)) for root, pressure in args.point]
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
