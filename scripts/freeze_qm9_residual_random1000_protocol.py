#!/usr/bin/env python3
"""Freeze validation decisions and checkpoint hashes before Test100 access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCOPES = ("all_scf", "ground_state_sample0")
ARMS = ("apbek", "direct", "residual")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reduction(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError(f"Non-positive baseline metric {baseline}")
    return 1.0 - candidate / baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = {}
    decisions = {}
    for scope in SCOPES:
        scope_metrics = {}
        for arm in ARMS:
            path = args.validation_dir / f"{arm}_{scope}.json"
            payload = json.loads(path.read_text())
            if payload.get("split") != "val":
                raise ValueError(f"{path} is not a validation result")
            if payload.get("failures"):
                raise ValueError(f"{path} contains numerical failures")
            scope_metrics[arm] = payload
        metrics[scope] = scope_metrics

        residual_energy = float(scope_metrics["residual"]["energy_mae_hartree"])
        direct_energy = float(scope_metrics["direct"]["energy_mae_hartree"])
        apbek_energy = float(scope_metrics["apbek"]["energy_mae_hartree"])
        residual_gradient = float(
            scope_metrics["residual"]["projected_gradient_l2_per_sample"]
        )
        direct_gradient = float(
            scope_metrics["direct"]["projected_gradient_l2_per_sample"]
        )
        apbek_gradient = float(
            scope_metrics["apbek"]["projected_gradient_l2_per_sample"]
        )
        reductions = {
            "residual_vs_apbek_energy": _reduction(residual_energy, apbek_energy),
            "residual_vs_apbek_gradient_l2": _reduction(
                residual_gradient, apbek_gradient
            ),
            "residual_vs_direct_energy": _reduction(residual_energy, direct_energy),
            "residual_vs_direct_gradient_l2": _reduction(
                residual_gradient, direct_gradient
            ),
        }
        learned_residual_pass = (
            reductions["residual_vs_apbek_energy"] >= 0.30
            and reductions["residual_vs_apbek_gradient_l2"] >= 0.30
        )
        direct_comparison_pass = (
            reductions["residual_vs_direct_energy"] >= 0.10
            and reductions["residual_vs_direct_gradient_l2"] >= -0.05
        ) or (
            reductions["residual_vs_direct_gradient_l2"] >= 0.10
            and reductions["residual_vs_direct_energy"] >= -0.05
        )
        decisions[scope] = {
            "reductions": reductions,
            "learned_residual_vs_apbek_pass": learned_residual_pass,
            "residual_vs_direct_pass": direct_comparison_pass,
            "scope_pass": learned_residual_pass and direct_comparison_pass,
        }

    direct_checkpoint = args.direct_checkpoint.resolve()
    residual_checkpoint = args.residual_checkpoint.resolve()
    record = {
        "schema_version": 1,
        "protocol_id": "qm9_residual_random1000_graphformer_v1",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "validation_dir": str(args.validation_dir.resolve()),
        "validation_file_sha256": {
            f"{arm}_{scope}": _sha256(args.validation_dir / f"{arm}_{scope}.json")
            for scope in SCOPES
            for arm in ARMS
        },
        "checkpoints": {
            "direct": {
                "path": str(direct_checkpoint),
                "sha256": _sha256(direct_checkpoint),
            },
            "residual": {
                "path": str(residual_checkpoint),
                "sha256": _sha256(residual_checkpoint),
            },
        },
        "decisions": decisions,
        "residual_representation_gate_pass": all(
            item["scope_pass"] for item in decisions.values()
        ),
        "test_evaluation_arms_frozen": list(ARMS),
        "test_metrics_frozen": [
            "energy_mae_hartree",
            "energy_mae_hartree_per_electron",
            "projected_gradient_l1_per_sample",
            "projected_gradient_l2_per_sample",
            "projected_gradient_component_rmse",
        ],
        "test_labels_accessed": False,
        "model_selection_on_test": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
