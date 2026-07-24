#!/usr/bin/env python3
"""Build a fusion-enthalpy manifest from a verified phase confirmation pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence


def build_manifest(
    confirmation_manifest: Dict[str, Any],
    confirmation_summary: Dict[str, Any],
    *,
    _seen_parents: set[Path] | None = None,
) -> Dict[str, Any]:
    if confirmation_summary.get("status") != "all_confirmations_passed":
        raise ValueError("zero-pressure confirmation is not verified")
    phases = {row["phase"]: row for row in confirmation_manifest["phases"]}
    if set(phases) != {"solid", "liquid"}:
        raise ValueError("confirmation manifest must contain one solid and one liquid")
    summary_phases = {
        row["phase"]: row for row in confirmation_summary["phase_results"]
    }
    if set(summary_phases) != {"solid", "liquid"} or any(
        row.get("status") != "confirmation_passed"
        for row in summary_phases.values()
    ):
        raise ValueError("both confirmation phase results must pass")
    temperature = float(confirmation_manifest["temperature_K"])
    pressure = float(confirmation_manifest.get("target_pressure_kbar", 0.0))
    steps = int(confirmation_manifest["steps"])
    point = {
        "temperature_k": temperature,
        "target_pressure_kbar": pressure,
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
    }
    parent = confirmation_manifest.get("parent_confirmation")
    if parent is not None:
        parent_root = Path(parent).resolve()
        seen = set() if _seen_parents is None else set(_seen_parents)
        if parent_root in seen:
            raise ValueError("enthalpy extension parent chain contains a cycle")
        seen.add(parent_root)
        parent_manifest = json.loads(
            (parent_root / "confirmation_manifest.json").read_text(encoding="utf-8")
        )
        parent_summary = json.loads(
            (parent_root / "confirmation_summary.json").read_text(encoding="utf-8")
        )
        parent_result = build_manifest(
            parent_manifest, parent_summary, _seen_parents=seen
        )
        parent_point = parent_result["points"][0]
        if str(parent_result["target_kedf"]).lower() != str(
            confirmation_manifest["target_kedf"]
        ).lower():
            raise ValueError("enthalpy extension and parent use different KEDFs")
        if not math.isclose(
            float(parent_point["temperature_k"]), temperature, abs_tol=1.0e-9
        ):
            raise ValueError("enthalpy extension and parent temperatures differ")
        if not math.isclose(
            float(parent_point["target_pressure_kbar"]), pressure, abs_tol=1.0e-9
        ):
            raise ValueError("enthalpy extension and parent pressures differ")
        for phase in ("solid", "liquid"):
            volume_key = f"{phase}_volume_per_atom_A3"
            if not math.isclose(
                float(parent_point[volume_key]),
                float(point[volume_key]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"enthalpy extension and parent {phase} volumes differ"
                )
            point[f"{phase}_runs"] = [
                *parent_point[f"{phase}_runs"],
                point[f"{phase}_run"],
            ]
        point["segment_steps"] = [*parent_point["segment_steps"], steps]
        point["steps"] = sum(point["segment_steps"])
        point["parent_confirmation"] = str(parent_root)

    return {
        "schema": "wt-zero-pressure-fusion-enthalpy-manifest-v1",
        "target_kedf": confirmation_manifest["target_kedf"],
        "points": [point],
    }


def merge_manifests(manifests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not manifests:
        raise ValueError("at least one verified confirmation manifest is required")
    target_kedfs = {str(manifest["target_kedf"]).lower() for manifest in manifests}
    if len(target_kedfs) != 1:
        raise ValueError("confirmation manifests use different target KEDFs")
    points = [point for manifest in manifests for point in manifest["points"]]
    temperatures = [float(point["temperature_k"]) for point in points]
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("confirmation manifests contain duplicate temperatures")
    return {
        "schema": "wt-zero-pressure-fusion-enthalpy-manifest-v1",
        "target_kedf": target_kedfs.pop(),
        "points": sorted(points, key=lambda point: float(point["temperature_k"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.confirmation_roots]
    result = merge_manifests(
        [
            build_manifest(
                json.loads((root / "confirmation_manifest.json").read_text()),
                json.loads((root / "confirmation_summary.json").read_text()),
            )
            for root in roots
        ]
    )
    result["provenance"] = {
        "confirmations": [
            {
                "confirmation_manifest": str(root / "confirmation_manifest.json"),
                "confirmation_summary": str(root / "confirmation_summary.json"),
            }
            for root in roots
        ]
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
