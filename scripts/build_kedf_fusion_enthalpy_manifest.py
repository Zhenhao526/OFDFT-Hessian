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


def load_document_chain(
    root: Path,
    *,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    root = root.resolve()
    visited = set() if seen is None else seen
    if root in visited:
        raise ValueError("enthalpy extension parent chain contains a cycle")
    visited.add(root)
    manifest_path = root / "confirmation_manifest.json"
    summary_path = root / "confirmation_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = []
    parent = manifest.get("parent_confirmation")
    if parent is not None:
        parent_root = Path(parent).resolve()
        parent_manifest = parent_root / "confirmation_manifest.json"
        if parent_manifest.is_file():
            documents.extend(load_document_chain(parent_root, seen=visited))
        else:
            source = manifest.get("source_zero_pressure_confirmation")
            if not isinstance(source, dict) or not source.get("path"):
                raise ValueError(
                    "enthalpy parent has no confirmation manifest or "
                    "zero-pressure provenance"
                )
            source_path = Path(source["path"]).resolve()
            expected_path = (
                parent_root / "zero_pressure_confirmation_summary.json"
            )
            if source_path != expected_path or not source_path.is_file():
                raise ValueError(
                    "enthalpy zero-pressure parent path is inconsistent"
                )
            if source.get("sha256") != sha256(source_path):
                raise ValueError(
                    "enthalpy zero-pressure parent SHA256 does not match"
                )
            source_report = json.loads(source_path.read_text(encoding="utf-8"))
            if source_report.get("status") != "all_confirmations_passed":
                raise ValueError(
                    "enthalpy zero-pressure parent is not verified"
                )
    documents.append(
        {
            "root": root,
            "manifest_path": manifest_path,
            "summary_path": summary_path,
            "manifest": manifest,
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }
    )
    return documents


def build_point_from_roots(
    enthalpy_roots: list[Path],
    zero_pressure_path: Path,
) -> dict[str, Any]:
    if not enthalpy_roots:
        raise ValueError("at least one enthalpy trajectory root is required")
    enthalpy_roots = [root.resolve() for root in enthalpy_roots]
    zero_pressure_path = zero_pressure_path.resolve()
    documents = []
    document_roots: set[Path] = set()
    for root in enthalpy_roots:
        for document in load_document_chain(root):
            if document["root"] not in document_roots:
                documents.append(document)
                document_roots.add(document["root"])
    zero_pressure = json.loads(zero_pressure_path.read_text(encoding="utf-8"))

    if zero_pressure.get("status") != "all_confirmations_passed":
        raise ValueError("zero-pressure confirmation is not verified")
    methods = {
        str(document["manifest"].get("target_kedf", "")).lower()
        for document in documents
    }
    methods.update(
        str(document["summary"].get("target_kedf", "")).lower()
        for document in documents
    )
    if len(methods) != 1:
        raise ValueError("enthalpy manifests and summaries use different KEDFs")
    method = methods.pop()
    if method not in {"xwm", "lkt"}:
        raise ValueError(f"unsupported target KEDF {method!r}")
    if str(zero_pressure.get("target_kedf", "")).lower() != method:
        raise ValueError("enthalpy and zero-pressure reports use different KEDFs")
    temperatures = {
        float(document[key]["temperature_K"])
        for document in documents
        for key in ("manifest", "summary")
    }
    if len(temperatures) != 1:
        raise ValueError("enthalpy manifests and summaries use different temperatures")
    temperature = temperatures.pop()
    if not math.isclose(
        float(zero_pressure["temperature_K"]), temperature, abs_tol=1.0e-9
    ):
        raise ValueError("enthalpy and zero-pressure temperatures differ")

    zero_phases = phase_map(zero_pressure["phase_results"])
    if set(zero_phases) != {"solid", "liquid"}:
        raise ValueError("solid and liquid zero-pressure evidence is required")
    if any(
        zero_phases[phase].get("status") != "confirmation_passed"
        for phase in zero_phases
    ):
        raise ValueError("zero-pressure phase gates did not all pass")

    selected: dict[str, list[dict[str, Any]]] = {
        "solid": [],
        "liquid": [],
    }
    for document in documents:
        manifest_phases = phase_map(document["manifest"]["phases"])
        summary_results = phase_map(document["summary"]["results"])
        for phase in set(manifest_phases) & set(summary_results):
            if summary_results[phase].get("status") != "passed":
                continue
            if selected[phase]:
                parent = document["manifest"].get("parent_confirmation")
                previous_root = selected[phase][-1]["root"]
                if parent is None or Path(parent).resolve() != previous_root:
                    raise ValueError(
                        f"duplicate passed {phase} enthalpy trajectory"
                    )
            selected[phase].append(
                {
                    "root": document["root"],
                    "manifest_phase": manifest_phases[phase],
                    "summary_result": summary_results[phase],
                    "steps": int(document["manifest"]["steps"]),
                    "stress_available": bool(
                        document["manifest"].get("stress_available", False)
                    ),
                    "manifest_path": document["manifest_path"],
                    "summary_path": document["summary_path"],
                }
            )
    if any(not selected[phase] for phase in ("solid", "liquid")):
        raise ValueError("verified solid and liquid enthalpy trajectories are required")
    for phase in ("solid", "liquid"):
        for segment in selected[phase]:
            if not math.isclose(
                float(segment["manifest_phase"]["volume_per_atom_A3"]),
                float(zero_phases[phase]["volume_per_atom_A3"]),
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            ):
                raise ValueError(
                    f"{phase} enthalpy and zero-pressure volumes differ"
                )

    stress_flags = {
        bool(segment["stress_available"])
        for phase in ("solid", "liquid")
        for segment in selected[phase]
    }
    if len(stress_flags) != 1:
        raise ValueError("solid and liquid enthalpy pressure modes differ")
    trajectory_pressure_required = stress_flags.pop()
    if method == "lkt" and not trajectory_pressure_required:
        raise ValueError("LKT enthalpy trajectories must include analytic stress")
    if method == "xwm" and trajectory_pressure_required:
        raise ValueError("XWM enthalpy trajectories unexpectedly claim stress")

    segment_steps = {
        phase: [int(segment["steps"]) for segment in selected[phase]]
        for phase in ("solid", "liquid")
    }
    phase_runs = {
        phase: [
            segment["manifest_phase"]["run"] for segment in selected[phase]
        ]
        for phase in ("solid", "liquid")
    }
    solid_steps = sum(segment_steps["solid"])
    liquid_steps = sum(segment_steps["liquid"])
    return {
        "temperature_k": temperature,
        "target_pressure_kbar": 0.0,
        "steps": max(solid_steps, liquid_steps),
        "solid_steps": solid_steps,
        "liquid_steps": liquid_steps,
        "solid_run": phase_runs["solid"][-1],
        "liquid_run": phase_runs["liquid"][-1],
        "solid_runs": phase_runs["solid"],
        "liquid_runs": phase_runs["liquid"],
        "solid_segment_steps": segment_steps["solid"],
        "liquid_segment_steps": segment_steps["liquid"],
        "solid_volume_per_atom_A3": float(
            selected["solid"][-1]["manifest_phase"]["volume_per_atom_A3"]
        ),
        "liquid_volume_per_atom_A3": float(
            selected["liquid"][-1]["manifest_phase"]["volume_per_atom_A3"]
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
            phase: {
                "confirmation_manifest": {
                    "path": str(selected[phase][-1]["manifest_path"]),
                    "sha256": sha256(selected[phase][-1]["manifest_path"]),
                },
                "confirmation_summary": {
                    "path": str(selected[phase][-1]["summary_path"]),
                    "sha256": sha256(selected[phase][-1]["summary_path"]),
                },
                "segments": [
                    {
                        "confirmation_manifest": {
                            "path": str(segment["manifest_path"]),
                            "sha256": sha256(segment["manifest_path"]),
                        },
                        "confirmation_summary": {
                            "path": str(segment["summary_path"]),
                            "sha256": sha256(segment["summary_path"]),
                        },
                    }
                    for segment in selected[phase]
                ],
            }
            for phase in ("solid", "liquid")
        },
    }


def build_point(
    enthalpy_root: Path,
    zero_pressure_path: Path,
) -> dict[str, Any]:
    return build_point_from_roots([enthalpy_root], zero_pressure_path)


def build_manifest(
    pairs: list[tuple[list[Path], Path]],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one enthalpy/zero-pressure pair is required")
    points = [
        build_point_from_roots(roots, pressure) for roots, pressure in pairs
    ]
    methods = set()
    for point in points:
        summary_path = Path(
            point["enthalpy_trajectory_provenance"]["solid"][
                "confirmation_summary"
            ]["path"]
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
    )
    parser.add_argument(
        "--split-point",
        action="append",
        nargs=3,
        metavar=("SOLID_ROOT", "LIQUID_ROOT", "ZERO_PRESSURE_SUMMARY"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = [
        ([Path(root)], Path(pressure)) for root, pressure in (args.point or [])
    ]
    pairs.extend(
        (
            [Path(solid_root), Path(liquid_root)],
            Path(pressure),
        )
        for solid_root, liquid_root, pressure in (args.split_point or [])
    )
    result = build_manifest(pairs)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
