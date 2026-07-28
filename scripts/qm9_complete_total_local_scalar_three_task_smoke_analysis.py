#!/usr/bin/env python3
"""Select at most one preregistered three-task stable5 formal arm."""

from __future__ import annotations

import argparse
import csv
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
    protocol_sha256 = _sha256(args.protocol)
    expected_parents = [str(value) for value in protocol["inputs"]["parents"]]
    rows: list[dict[str, Any]] = []
    for arm in protocol["arms"]:
        arm_id = str(arm["id"])
        run_dir = args.run_root / arm_id
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "training_metrics.jsonl"
        summary = json.loads(summary_path.read_text())
        metrics = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if summary.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"protocol hash drift for {arm_id}")
        if summary.get("arm_id") != arm_id or summary.get("run_mode") != "smoke":
            raise ValueError(f"run identity mismatch for {arm_id}")
        if summary.get("selected_parent_ids") != expected_parents:
            raise ValueError(f"parent order/coverage mismatch for {arm_id}")
        if int(summary.get("unselected_parent_artifacts_opened", -1)) != 0:
            raise ValueError(f"unselected parents opened by {arm_id}")
        if summary.get("validation_accessed") is not False:
            raise ValueError(f"validation accessed by {arm_id}")
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"Test100 accessed by {arm_id}")
        if int(summary.get("test100_evaluations_used", -1)) != 0:
            raise ValueError(f"Test100 count is not zero for {arm_id}")
        if not metrics or int(metrics[0]["step"]) != 0:
            raise ValueError(f"missing step-zero metric for {arm_id}")
        positive = [row for row in metrics if int(row["step"]) > 0]
        if not positive:
            raise ValueError(f"missing positive-step metrics for {arm_id}")
        required_norms = (
            "pcgrad/energy_gradient_norm",
            "pcgrad/force_gradient_norm",
            "pcgrad/hessian_gradient_norm",
        )
        # Step zero is a forward-only baseline and intentionally logs NaN for losses and
        # gradients that have not been evaluated yet. Formal eligibility requires every
        # diagnostic produced by an actual update to be finite.
        finite = _finite_tree(positive) and _finite_tree(summary)
        positive_norms = all(
            all(float(row.get(key, 0.0)) > 0.0 for key in required_norms)
            for row in positive
        )
        final = positive[-1]
        paired_score = float(arm["paired_activation_smoke_final_selection_score"])
        improved_score = float(final["selection_score"]) < paired_score
        improved_hessian = float(final["median_relative_frobenius"]) < float(
            metrics[0]["median_relative_frobenius"]
        )
        eligible = finite and positive_norms and improved_score and improved_hessian
        rows.append(
            {
                "arm_id": arm_id,
                "eligible": eligible,
                "all_diagnostics_finite": finite,
                "all_named_task_gradient_norms_positive": positive_norms,
                "paired_activation_smoke_final_selection_score": paired_score,
                "final_selection_score": float(final["selection_score"]),
                "step_zero_median_relative_frobenius": float(
                    metrics[0]["median_relative_frobenius"]
                ),
                "final_median_relative_frobenius": float(
                    final["median_relative_frobenius"]
                ),
                "final_max_relative_frobenius": float(
                    final["max_relative_frobenius"]
                ),
                "summary_sha256": _sha256(summary_path),
                "metrics_sha256": _sha256(metrics_path),
            }
        )

    eligible_rows = [row for row in rows if row["eligible"]]
    eligible_rows.sort(
        key=lambda row: (
            float(row["final_selection_score"]),
            float(row["final_median_relative_frobenius"]),
            str(row["arm_id"]),
        )
    )
    selected = [
        str(row["arm_id"])
        for row in eligible_rows[
            : int(protocol["smoke_selection"]["maximum_formal_arms"])
        ]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "arm_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "definition": "Stable5 three-task E/F/H GradNorm-PCGrad smoke selection; no held parent, validation, or Test100 label is read.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "run_root": args.run_root.resolve().as_posix(),
        "arms": rows,
        "eligible_arms": [str(row["arm_id"]) for row in eligible_rows],
        "selected_formal_arms": selected,
        "formal_run_authorized": bool(selected),
        "parent_cv_design_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "arm_summary_csv": csv_path.resolve().as_posix(),
        "arm_summary_csv_sha256": _sha256(csv_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
