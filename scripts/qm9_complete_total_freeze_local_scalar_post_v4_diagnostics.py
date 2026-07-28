#!/usr/bin/env python3
"""Freeze hash-checked stable5 diagnostics after a failed v4 formal decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    protocol_path: Path,
    formal_analysis_path: Path,
    run_root: Path,
    source_arm_id: str | None = None,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("source protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("source protocol Test100 count is nonzero")
    analysis = json.loads(formal_analysis_path.read_text())
    if analysis.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("formal analysis protocol hash drift")
    if analysis.get("test100_accessed") is not False:
        raise ValueError("formal analysis does not certify frozen Test100")
    if int(analysis.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("formal analysis Test100 count is nonzero")
    if bool(analysis.get("parent_cv_design_authorized")):
        raise ValueError("post-v4 failure diagnostics are not authorized after a pass")
    selected = [str(value) for value in analysis.get("selected_formal_arms", [])]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("post-v4 diagnostics require unique failed formal arms")
    arm_rows = [
        row for row in analysis.get("arms", []) if str(row.get("arm_id")) in selected
    ]
    if len(arm_rows) != len(selected) or any(
        bool(row.get("gate_passed")) for row in arm_rows
    ):
        raise ValueError("formal analysis does not contain only unique failed arms")
    if source_arm_id is not None:
        if source_arm_id not in selected:
            raise ValueError("requested diagnostic source arm was not formally selected")
        arm_id = source_arm_id
        source_selection_rule = "explicit_failed_formal_arm"
    else:
        def score(row: dict[str, Any]) -> tuple[float, str]:
            value = (
                float(row["median_relative_frobenius"])
                + float(row["max_relative_frobenius"])
                + max(0.0, float(row["energy_median_ratio_to_source"]) - 1.05)
                + max(0.0, float(row["force_median_ratio_to_source"]) - 1.05)
            )
            return value, str(row["arm_id"])

        arm_id = str(min(arm_rows, key=score)["arm_id"])
        source_selection_rule = "minimum_preregistered_joint_selection_score_over_failed_formal_arms"
    expected_parents = [str(value) for value in protocol["inputs"]["parents"]]
    if analysis.get("expected_parent_ids") != expected_parents:
        raise ValueError("formal analysis stable5 parent drift")

    run_dir = run_root / arm_id
    run_summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best.ckpt"
    run_summary = json.loads(run_summary_path.read_text())
    if run_summary.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("formal run protocol hash drift")
    if run_summary.get("arm_id") != arm_id or run_summary.get("run_mode") != "formal":
        raise ValueError("formal run identity mismatch")
    if run_summary.get("selected_parent_ids") != expected_parents:
        raise ValueError("formal run parent drift")
    if run_summary.get("validation_accessed") is not False:
        raise ValueError("formal run accessed validation")
    if run_summary.get("test100_accessed") is not False:
        raise ValueError("formal run accessed Test100")
    if int(run_summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("formal run Test100 count is nonzero")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("checkpoint protocol hash drift")
    if checkpoint.get("arm_id") != arm_id:
        raise ValueError("checkpoint arm mismatch")
    if checkpoint.get("selected_parent_ids") != expected_parents:
        raise ValueError("checkpoint parent drift")
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not certify frozen Test100")
    if int(checkpoint.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("checkpoint Test100 count is nonzero")

    return {
        "protocol_id": "qm9_complete_total_local_scalar_post_v4_diagnostics_v1",
        "frozen_at": str(protocol.get("frozen_at")),
        "scope": "stable5 fit-only optimizer/Jacobian diagnosis after the failed v4 decision",
        "source_protocol": protocol_path.resolve().as_posix(),
        "source_protocol_sha256": _sha256(protocol_path),
        "source_formal_analysis": formal_analysis_path.resolve().as_posix(),
        "source_formal_analysis_sha256": _sha256(formal_analysis_path),
        "source_run_summary": run_summary_path.resolve().as_posix(),
        "source_run_summary_sha256": _sha256(run_summary_path),
        "source_arm_id": arm_id,
        "source_selection_rule": source_selection_rule,
        "source_checkpoint": checkpoint_path.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_step": int(checkpoint["step"]),
        "parents": expected_parents,
        "jacobian": {
            "iterations": 30,
            "relative_tolerance": 1.0e-3,
            "fd_relative_step": 3.0e-6,
            "nonlinear_alphas": [0.01, 0.03, 0.1, 0.3, 1.0],
        },
        "lbfgs": {
            objective: {
                "iterations": 100,
                "learning_rate": 0.5,
                "history_size": 25,
                "max_evaluations_per_iteration": 5,
                "tolerance_grad": 1.0e-9,
                "tolerance_change": 1.0e-12,
                "log_interval": 5,
                "parent_cv_authorization_allowed": objective == "joint",
            }
            for objective in ("joint", "hessian_only")
        },
        "vibrational_gate": {
            "mean_frequency_mae_cm-1_max": 200.0,
            "absolute_total_imaginary_mode_count_error_max": 5,
            "mean_mode_overlap_min": 0.8,
            "required_molecule_count": len(expected_parents),
        },
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_manifest(
        protocol_path=args.protocol,
        formal_analysis_path=args.formal_analysis,
        run_root=args.run_root,
        source_arm_id=args.source_arm_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--formal-analysis", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-arm-id")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
