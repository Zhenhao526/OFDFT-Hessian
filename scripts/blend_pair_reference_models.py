#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compatible_sequence(left: Any, right: Any) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
            for a, b in zip(left, right)
        )
    )


def blend_models(
    base_path: Path,
    correction_path: Path,
    output_path: Path,
    *,
    alpha: float,
    target_kedf: str,
    phase: str,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if output_path.exists():
        raise FileExistsError(f"refusing existing output: {output_path}")
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    correction_document = json.loads(
        correction_path.read_text(encoding="utf-8")
    )
    if not base_document.get("reference_gate_passed"):
        raise ValueError("base model has not passed its reference gate")
    if not correction_document.get("reference_gate_passed"):
        raise ValueError("correction model has not passed its reference gate")
    if str(base_document.get("target_kedf", "")).lower() != target_kedf:
        raise ValueError("base model target_kedf mismatch")
    if str(correction_document.get("target_kedf", "")).lower() != target_kedf:
        raise ValueError("correction model target_kedf mismatch")

    base_model = base_document["model"]
    correction_model = correction_document["model"]
    if not compatible_sequence(
        base_model["centers_angstrom"],
        correction_model["centers_angstrom"],
    ):
        raise ValueError("pair model field differs: centers_angstrom")
    for field in ("sigma_angstrom", "cutoff_angstrom"):
        if not math.isclose(
            float(base_model[field]),
            float(correction_model[field]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"pair model field differs: {field}")
    base_core = base_model["repulsive_core"]
    correction_core = correction_model["repulsive_core"]
    for field in ("amplitude_ev", "cutoff_angstrom"):
        if not math.isclose(
            float(base_core[field]),
            float(correction_core[field]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"pair model repulsive core differs: {field}")
    if int(base_core["power"]) != int(correction_core["power"]):
        raise ValueError("pair model repulsive core differs: power")
    base_coefficients = [float(value) for value in base_model["coefficients_ev"]]
    correction_coefficients = [
        float(value) for value in correction_model["coefficients_ev"]
    ]
    if len(base_coefficients) != len(correction_coefficients):
        raise ValueError("pair model coefficient lengths differ")
    coefficients = [
        (1.0 - alpha) * base_value + alpha * correction_value
        for base_value, correction_value in zip(
            base_coefficients, correction_coefficients
        )
    ]
    result = {
        "schema": "kedf-blended-pair-reference-v1",
        "status": "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_kedf": target_kedf,
        "phase": phase,
        "reference_kind": "phase_specific_variance_reduction_bridge",
        "purpose": (
            "retain the verified hard-wall phase dynamics while reducing "
            "the target-minus-reference variance"
        ),
        "alpha_correction": alpha,
        "base": {
            "path": str(base_path.resolve()),
            "sha256": sha256(base_path),
            "reference_kind": base_document.get("reference_kind"),
        },
        "correction": {
            "path": str(correction_path.resolve()),
            "sha256": sha256(correction_path),
            "dataset": correction_document.get("dataset"),
        },
        "model": {
            "centers_angstrom": base_model["centers_angstrom"],
            "sigma_angstrom": base_model["sigma_angstrom"],
            "cutoff_angstrom": base_model["cutoff_angstrom"],
            "repulsive_core": base_model["repulsive_core"],
            "coefficients_ev": coefficients,
        },
        "reference_gate_passed": True,
        "short_range_guard_passed": (
            base_document.get("short_range_guard_passed") is True
            and correction_document.get("short_range_guard_passed") is True
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blend compatible phase-specific pair references"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--target-kedf", choices=("xwm", "lkt"), required=True)
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    args = parser.parse_args()
    result = blend_models(
        args.base,
        args.correction,
        args.out,
        alpha=args.alpha,
        target_kedf=args.target_kedf,
        phase=args.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
