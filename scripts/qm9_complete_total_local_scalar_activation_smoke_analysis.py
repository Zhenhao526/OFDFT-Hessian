#!/usr/bin/env python3
"""Apply the frozen same-seed activation-smoke promotion rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _positive_step_diagnostics_are_finite(rows: list[dict[str, Any]]) -> bool:
    positive = [row for row in rows if int(row["step"]) > 0]
    if not positive:
        return False
    for row in positive:
        for value in row.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    protocol_sha256 = _sha256(args.protocol)
    arms = [str(row["id"]) for row in protocol["arms"]]
    selection = protocol["smoke_selection"]
    control_id = str(selection["control_arm"])
    maximum = int(selection["maximum_formal_non_control_arms"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    input_artifacts = []
    for arm_id in arms:
        run_dir = args.run_root / arm_id
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "training_metrics.jsonl"
        summary = json.loads(summary_path.read_text())
        if summary.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"protocol hash drift for {arm_id}")
        if summary.get("arm_id") != arm_id or summary.get("run_mode") != "smoke":
            raise ValueError(f"run identity mismatch for {arm_id}")
        if int(summary.get("unselected_parent_artifacts_opened", -1)) != 0:
            raise ValueError(f"unselected parent artifact opened by {arm_id}")
        if summary.get("validation_accessed") is not False:
            raise ValueError(f"validation accessed by {arm_id}")
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"Test100 accessed by {arm_id}")
        metrics = _load_jsonl(metrics_path)
        positive = [row for row in metrics if int(row["step"]) > 0]
        raw_ratios = [
            float(row["gradnorm/raw_hessian_gradient_norm"])
            / max(
                float(row["gradnorm/energy_force_gradient_norm"]),
                np.finfo(float).tiny,
            )
            for row in positive
        ]
        final = summary["final"]
        rows.append(
            {
                "arm_id": arm_id,
                "is_control": arm_id == control_id,
                "positive_step_diagnostics_finite": _positive_step_diagnostics_are_finite(
                    metrics
                ),
                "best_step": int(summary["best_step"]),
                "selection_score": float(final["selection_score"]),
                "median_raw_hessian_to_energy_force_gradient_ratio": float(
                    np.median(raw_ratios)
                ),
                "median_relative_frobenius": float(
                    final["hessian_relative_frobenius"]["median"]
                ),
                "max_relative_frobenius": float(
                    final["hessian_relative_frobenius"]["max"]
                ),
                "energy_median_ratio_to_source": float(
                    final["energy_median_ratio_to_source"]
                ),
                "force_median_ratio_to_source": float(
                    final["force_median_ratio_to_source"]
                ),
                "summary_sha256": _sha256(summary_path),
                "metrics_sha256": _sha256(metrics_path),
            }
        )
        input_artifacts.extend(
            [
                {"path": summary_path.resolve().as_posix(), "sha256": _sha256(summary_path)},
                {"path": metrics_path.resolve().as_posix(), "sha256": _sha256(metrics_path)},
            ]
        )

    control = next(row for row in rows if row["arm_id"] == control_id)
    eligible = []
    for row in rows:
        row["beats_control_selection_score"] = bool(
            float(row["selection_score"]) < float(control["selection_score"])
        )
        row["beats_control_raw_gradient_ratio"] = bool(
            float(row["median_raw_hessian_to_energy_force_gradient_ratio"])
            > float(control["median_raw_hessian_to_energy_force_gradient_ratio"])
        )
        row["formal_eligible"] = bool(
            not row["is_control"]
            and row["positive_step_diagnostics_finite"]
            and row["beats_control_selection_score"]
            and row["beats_control_raw_gradient_ratio"]
        )
        if row["formal_eligible"]:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            float(row["selection_score"]),
            -float(row["median_raw_hessian_to_energy_force_gradient_ratio"]),
            str(row["arm_id"]),
        )
    )
    selected = [str(row["arm_id"]) for row in eligible[:maximum]]
    csv_path = args.output_dir / "arm_summary.csv"
    _write_csv(csv_path, rows)
    result = {
        "definition": "Frozen same-seed stable5 activation-smoke promotion; step-0 NaN sentinels are excluded from positive-step finiteness checks.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "run_root": args.run_root.resolve().as_posix(),
        "control_arm": control_id,
        "arms": rows,
        "selected_formal_arms": selected,
        "formal_submission_authorized": bool(selected),
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "input_artifacts": input_artifacts,
        "arm_summary_csv": csv_path.resolve().as_posix(),
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
