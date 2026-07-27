#!/usr/bin/env python3
"""Combine phase and pressure gates into a normalized zero-pressure summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(
    confirmation_root: Path,
    pressure_samples_root: Path | None = None,
) -> dict:
    confirmation_path = confirmation_root / "confirmation_summary.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    target_kedf = str(confirmation.get("target_kedf", "")).lower()
    if target_kedf not in {"xwm", "lkt", "wt", "ext-wt"}:
        raise ValueError(f"unsupported KEDF {target_kedf!r}")

    phase_results = []
    for item in confirmation["results"]:
        phase = str(item["phase"])
        checks = {
            "phase_confirmation_passed": item.get("status") == "passed",
        }
        pressure_provenance = None
        if item.get("requires_posthoc_pressure"):
            if pressure_samples_root is None:
                checks["posthoc_pressure_verified"] = False
                pressure = {}
            else:
                pressure_path = (
                    pressure_samples_root
                    / phase
                    / "pressure_samples_summary.json"
                )
                pressure_document = json.loads(
                    pressure_path.read_text(encoding="utf-8")
                )
                pressure = pressure_document.get("pressure_kbar", {})
                checks["posthoc_pressure_verified"] = (
                    pressure_document.get("status")
                    == "snapshot_pressure_verified"
                )
                pressure_provenance = {
                    "path": str(pressure_path.resolve()),
                    "sha256": sha256(pressure_path),
                    "samples": pressure_document.get("samples", []),
                }
        else:
            pressure = item.get("pressure_last_half_kbar", {})
            checks["analytic_pressure_verified"] = bool(pressure) and abs(
                float(pressure.get("mean", float("inf")))
            ) <= 2.5

        passed = all(checks.values())
        phase_results.append(
            {
                "phase": phase,
                "run": item["run"],
                "volume_per_atom_A3": item["volume_per_atom_A3"],
                "max_step": item["max_step"],
                "phase_status": item["phase_status"],
                "nearest_neighbor_A": item["nearest_neighbor_A"],
                "temperature_last_half_K": item[
                    "temperature_last_half_K"
                ],
                "pressure_kbar": pressure,
                "pressure_source": (
                    "posthoc_finite_difference"
                    if item.get("requires_posthoc_pressure")
                    else "analytic_stress"
                ),
                "pressure_provenance": pressure_provenance,
                "checks": checks,
                "status": (
                    "confirmation_passed"
                    if passed
                    else "confirmation_failed"
                ),
            }
        )

    all_passed = all(
        item["status"] == "confirmation_passed"
        for item in phase_results
    )
    return {
        "schema": "kedf-zero-pressure-confirmation-final-v1",
        "target_kedf": target_kedf,
        "temperature_K": confirmation["temperature_K"],
        "target_pressure_kbar": 0.0,
        "source_confirmation": {
            "path": str(confirmation_path.resolve()),
            "sha256": sha256(confirmation_path),
        },
        "phase_results": phase_results,
        "status": (
            "all_confirmations_passed"
            if all_passed
            else "confirmation_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--pressure-samples-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = finalize(
        args.confirmation_root.resolve(),
        args.pressure_samples_root.resolve()
        if args.pressure_samples_root
        else None,
    )
    output = (
        args.out.resolve()
        if args.out
        else args.confirmation_root.resolve()
        / "zero_pressure_confirmation_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
