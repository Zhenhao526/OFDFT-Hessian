#!/usr/bin/env python3
"""Apply the frozen stable5 local-angular smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    if len(protocol["arms"]) != 1:
        raise ValueError("local-angular smoke protocol must contain exactly one arm")
    arm_id = str(protocol["arms"][0]["id"])
    summary_path = args.run_dir / "summary.json"
    metrics_path = args.run_dir / "training_metrics.jsonl"
    summary = json.loads(summary_path.read_text())
    metrics = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
    ]
    protocol_sha256 = _sha256(args.protocol)
    if summary.get("protocol_sha256") != protocol_sha256:
        raise ValueError("protocol hash drift")
    if summary.get("arm_id") != arm_id or summary.get("run_mode") != "smoke":
        raise ValueError("run identity mismatch")
    expected_parents = [str(value) for value in protocol["inputs"]["parents"]]
    if summary.get("selected_parent_ids") != expected_parents:
        raise ValueError("stable5 parent order/coverage mismatch")
    if int(summary.get("unselected_parent_artifacts_opened", -1)) != 0:
        raise ValueError("unselected parent artifacts were opened")
    if summary.get("validation_accessed") is not False:
        raise ValueError("validation was accessed")
    if summary.get("test100_accessed") is not False:
        raise ValueError("Test100 was accessed")
    if int(summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("Test100 count is not zero")
    if not metrics or int(metrics[0]["step"]) != 0:
        raise ValueError("missing untouched step-zero metrics")
    positive = [row for row in metrics if int(row["step"]) > 0]
    if not positive:
        raise ValueError("missing positive-step metrics")
    gate = protocol["smoke_gate"]
    final = positive[-1]
    checks = {
        "positive_steps_finite": _finite_tree(positive),
        "raw_hessian_gradient_positive": all(
            float(row.get("gradnorm/raw_hessian_gradient_norm", 0.0)) > 0.0
            for row in positive
        ),
        "energy_force_gradient_positive": all(
            float(row.get("gradnorm/energy_force_gradient_norm", 0.0)) > 0.0
            for row in positive
        ),
        "median_hessian_improved": float(final["median_relative_frobenius"])
        < float(metrics[0]["median_relative_frobenius"]),
        "final_median_hessian": float(final["median_relative_frobenius"])
        <= float(gate["final_median_relative_frobenius_max"]),
        "final_joint_score": float(final["selection_score"])
        <= float(gate["final_selection_score_max"]),
        "peak_gpu_memory": float(final["gpu_peak_memory_mb"])
        <= float(gate["peak_gpu_memory_mb_max"]),
    }
    authorized = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "definition": "Frozen stable5 local-angular representation smoke gate.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "arm_id": arm_id,
        "run_dir": args.run_dir.resolve().as_posix(),
        "summary_sha256": _sha256(summary_path),
        "metrics_sha256": _sha256(metrics_path),
        "checks": checks,
        "formal_run_authorized": authorized,
        "parent_cv_design_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "step_zero_median_relative_frobenius": float(
            metrics[0]["median_relative_frobenius"]
        ),
        "final_step": int(final["step"]),
        "final_median_relative_frobenius": float(
            final["median_relative_frobenius"]
        ),
        "final_max_relative_frobenius": float(final["max_relative_frobenius"]),
        "final_selection_score": float(final["selection_score"]),
        "peak_gpu_memory_mb": float(final["gpu_peak_memory_mb"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
