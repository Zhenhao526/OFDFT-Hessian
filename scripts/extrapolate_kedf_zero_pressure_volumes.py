#!/usr/bin/env python3
"""Linearly extrapolate phase volumes from two verified KEDF zero-pressure points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_volumes(report: dict[str, Any]) -> dict[str, float]:
    if report.get("status") != "all_confirmations_passed":
        raise ValueError("zero-pressure report is not verified")
    rows = {
        row["phase"]: float(row["volume_per_atom_A3"])
        for row in report["phase_results"]
        if row.get("status") == "confirmation_passed"
    }
    if set(rows) != {"solid", "liquid"}:
        raise ValueError("report must contain verified solid and liquid phases")
    return rows


def extrapolate(
    lower_path: Path,
    upper_path: Path,
    targets: list[float],
) -> dict[str, Any]:
    lower_path = lower_path.resolve()
    upper_path = upper_path.resolve()
    lower = json.loads(lower_path.read_text())
    upper = json.loads(upper_path.read_text())
    if lower.get("target_kedf") != upper.get("target_kedf"):
        raise ValueError("zero-pressure reports use different KEDFs")
    lower_t = float(lower["temperature_K"])
    upper_t = float(upper["temperature_K"])
    if not lower_t < upper_t:
        raise ValueError("reference temperatures must increase")
    lower_v = phase_volumes(lower)
    upper_v = phase_volumes(upper)
    predictions = []
    slopes = {}
    for phase in ("solid", "liquid"):
        slope = (upper_v[phase] - lower_v[phase]) / (upper_t - lower_t)
        slopes[phase] = slope
    for temperature in targets:
        if temperature <= upper_t:
            raise ValueError("target temperatures must exceed the upper reference")
        predictions.append(
            {
                "temperature_K": temperature,
                "volumes_per_atom_A3": {
                    phase: upper_v[phase]
                    + slopes[phase] * (temperature - upper_t)
                    for phase in ("solid", "liquid")
                },
            }
        )
    return {
        "schema": "kedf-zero-pressure-volume-extrapolation-v1",
        "status": "candidate_volumes_require_confirmation",
        "target_kedf": lower["target_kedf"],
        "reference_reports": [
            {
                "path": str(lower_path),
                "sha256": sha256(lower_path),
                "temperature_K": lower_t,
                "volumes_per_atom_A3": lower_v,
            },
            {
                "path": str(upper_path),
                "sha256": sha256(upper_path),
                "temperature_K": upper_t,
                "volumes_per_atom_A3": upper_v,
            },
        ],
        "linear_slopes_A3_per_atom_K": slopes,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", type=Path, required=True)
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--target", type=float, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = extrapolate(args.lower, args.upper, args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
