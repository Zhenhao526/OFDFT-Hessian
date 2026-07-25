#!/usr/bin/env python3
"""Append a verified single-phase extension to an enthalpy manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def build_targeted_manifest(
    parent: Dict[str, Any],
    extension_manifest: Dict[str, Any],
    extension_summary: Dict[str, Any],
    *,
    parent_path: Path,
    extension_root: Path,
) -> Dict[str, Any]:
    if extension_manifest.get("extension_schema") != (
        "wt-zero-pressure-enthalpy-phase-extension-v1"
    ):
        raise ValueError("extension manifest has the wrong schema")
    if extension_summary.get("status") != "all_confirmations_passed":
        raise ValueError("targeted phase extension is not verified")
    phases = extension_manifest.get("phases", [])
    results = extension_summary.get("phase_results", [])
    if len(phases) != 1 or len(results) != 1:
        raise ValueError("targeted extension must contain exactly one phase")
    phase = phases[0]["phase"]
    if phase not in {"solid", "liquid"} or results[0].get("phase") != phase:
        raise ValueError("targeted extension phase metadata is inconsistent")
    if results[0].get("status") != "confirmation_passed" or not all(
        results[0].get("checks", {}).values()
    ):
        raise ValueError("targeted extension phase gates did not all pass")
    if str(parent.get("target_kedf", "")).lower() != str(
        extension_manifest.get("target_kedf", "")
    ).lower():
        raise ValueError("targeted extension and parent use different KEDFs")

    temperature = float(extension_manifest["temperature_K"])
    matches = [
        point
        for point in parent["points"]
        if math.isclose(
            float(point["temperature_k"]), temperature, abs_tol=1.0e-9
        )
    ]
    if len(matches) != 1:
        raise ValueError("targeted extension temperature is absent or duplicated")
    extension_steps = int(extension_manifest["steps"])
    updated_points = []
    for original in parent["points"]:
        point = dict(original)
        if original is matches[0]:
            pressure = float(extension_manifest.get("target_pressure_kbar", 0.0))
            if not math.isclose(
                float(point["target_pressure_kbar"]),
                pressure,
                abs_tol=1.0e-9,
            ):
                raise ValueError("targeted extension pressure differs")
            volume_key = f"{phase}_volume_per_atom_A3"
            if not math.isclose(
                float(point[volume_key]),
                float(phases[0]["volume_per_atom_A3"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("targeted extension phase volume differs")

            common_steps = list(point.get("segment_steps", [point["steps"]]))
            solid_steps = list(point.get("solid_segment_steps", common_steps))
            liquid_steps = list(point.get("liquid_segment_steps", common_steps))
            run_key = f"{phase}_run"
            runs_key = f"{phase}_runs"
            prior_run = point[run_key]
            prior_runs = list(point.get(runs_key, [prior_run]))
            point[run_key] = phases[0]["run"]
            point[runs_key] = [*prior_runs, phases[0]["run"]]
            if phase == "solid":
                solid_steps.append(extension_steps)
            else:
                liquid_steps.append(extension_steps)
            point["solid_segment_steps"] = solid_steps
            point["liquid_segment_steps"] = liquid_steps
            point["solid_steps"] = sum(solid_steps)
            point["liquid_steps"] = sum(liquid_steps)
            point["steps"] = min(point["solid_steps"], point["liquid_steps"])
            point["targeted_extension"] = {
                "phase": phase,
                "steps": extension_steps,
                "root": str(extension_root.resolve()),
                "confirmation_manifest": str(
                    (extension_root / "confirmation_manifest.json").resolve()
                ),
                "confirmation_summary": str(
                    (extension_root / "confirmation_summary.json").resolve()
                ),
            }
        updated_points.append(point)
    return {
        **parent,
        "schema": "wt-zero-pressure-fusion-enthalpy-manifest-v2",
        "points": updated_points,
        "targeted_extension_provenance": {
            "parent_manifest": str(parent_path.resolve()),
            "extension_root": str(extension_root.resolve()),
            "phase": phase,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_manifest", type=Path)
    parser.add_argument("extension_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent_manifest.resolve()
    extension_root = args.extension_root.resolve()
    result = build_targeted_manifest(
        json.loads(parent_path.read_text(encoding="utf-8")),
        json.loads(
            (extension_root / "confirmation_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (extension_root / "confirmation_summary.json").read_text(
                encoding="utf-8"
            )
        ),
        parent_path=parent_path,
        extension_root=extension_root,
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
