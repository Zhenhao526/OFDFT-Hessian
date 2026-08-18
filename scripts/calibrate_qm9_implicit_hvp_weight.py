#!/usr/bin/env python3
"""Freeze the formal implicit-HVP weight from a zero-LR gradient smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(protocol: dict, row: dict[str, str]) -> dict[str, float | str]:
    physical_definition_id = protocol.get("definitions", {}).get(
        "physical_definition_id", "legacy_hybrid_egfh_v0"
    )
    if physical_definition_id == "qm9_complete_total_relaxed_egfh_v1":
        if row.get("physical_definition_id") != physical_definition_id:
            raise ValueError(
                "Canonical calibration curve has the wrong physical definition"
            )
        if set(row.get("computed_components", "").split(";")) != {
            "E",
            "G",
            "F",
            "H",
        }:
            raise ValueError(
                "Canonical calibration must compute exactly E/G/F/H"
            )
    calibration = protocol["gradient_calibration"]
    current_lambda = float(calibration["smoke"]["lambda_H"])
    target_ratio = float(calibration["target_hvp_to_egf_gradient_ratio"])
    lower = float(calibration["lambda_h_min"])
    upper = float(calibration["lambda_h_max"])
    norms = {
        name: float(row[f"gradient_norm/{name}"])
        for name in ("energy", "density", "force", "hvp")
    }
    if not all(math.isfinite(value) and value >= 0 for value in norms.values()):
        raise ValueError("Gradient norms must be finite and nonnegative")
    if norms["hvp"] <= 0:
        raise ValueError("The implicit-HVP gradient norm must be positive")
    quadrature_egf_norm = math.sqrt(
        norms["energy"] ** 2
        + norms["density"] ** 2
        + norms["force"] ** 2
    )
    if physical_definition_id == "qm9_complete_total_relaxed_egfh_v1":
        egf_norm = float(row.get("gradient_norm/aggregate_egf", math.nan))
        egf_definition = "norm_of_gradient_of_weighted_E_plus_G_plus_F"
    else:
        egf_norm = quadrature_egf_norm
        egf_definition = "legacy_quadrature_of_component_gradient_norms"
    if not math.isfinite(egf_norm):
        raise ValueError("The aggregate E/G/F gradient norm must be finite")
    if egf_norm <= 0:
        raise ValueError("The aggregate E/G/F gradient norm must be positive")
    unclipped = current_lambda * target_ratio * egf_norm / norms["hvp"]
    formal_lambda = min(upper, max(lower, unclipped))
    return {
        "weighted_energy_gradient_norm": norms["energy"],
        "weighted_density_gradient_norm": norms["density"],
        "weighted_force_gradient_norm": norms["force"],
        "weighted_hvp_gradient_norm_at_smoke_lambda": norms["hvp"],
        "aggregate_weighted_egf_gradient_norm": egf_norm,
        "component_gradient_norm_quadrature": quadrature_egf_norm,
        "aggregate_egf_gradient_definition": egf_definition,
        "smoke_lambda_h": current_lambda,
        "target_hvp_to_egf_gradient_ratio": target_ratio,
        "unclipped_formal_lambda_h": unclipped,
        "formal_lambda_h": formal_lambda,
        "lambda_h_lower_bound": lower,
        "lambda_h_upper_bound": upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())
    protocol_sha256 = _sha256(args.protocol)
    with args.training_curve.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one zero-LR smoke row, found {len(rows)}"
        )
    physical_definition_id = protocol.get("definitions", {}).get(
        "physical_definition_id", "legacy_hybrid_egfh_v0"
    )
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "physical_definition_id": physical_definition_id,
        "protocol_sha256": protocol_sha256,
        "training_curve": args.training_curve.resolve().as_posix(),
        "training_curve_sha256": _sha256(args.training_curve),
        **calibrate(protocol, rows[0]),
    }
    if physical_definition_id == "qm9_complete_total_relaxed_egfh_v1":
        training_summary = args.training_curve.parent / "summary.json"
        if not training_summary.is_file():
            raise ValueError(
                f"Canonical calibration summary is missing: {training_summary}"
            )
        summary = json.loads(training_summary.read_text())
        if (
            summary.get("physical_definition_id") != physical_definition_id
            or summary.get("semantic_version") != physical_definition_id
            or summary.get("protocol_id") != protocol["protocol_id"]
            or summary.get("protocol_sha256") != protocol_sha256
            or not isinstance(summary.get("code_provenance"), dict)
        ):
            raise ValueError(
                "Canonical calibration summary protocol/code provenance mismatch"
            )
        result.update(
            {
                "training_summary": training_summary.resolve().as_posix(),
                "training_summary_sha256": _sha256(training_summary),
                "code_provenance": summary["code_provenance"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
