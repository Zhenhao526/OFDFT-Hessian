#!/usr/bin/env python3
"""Summarize chemistry-aware train20 structured scalar subspaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("structured-subspace distribution is empty or non-finite")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _load_frozen(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("validation_accessed") is not False:
        raise ValueError(f"validation access drift: {path}")
    if result.get("test100_accessed") is not False:
        raise ValueError(f"Test100 access drift: {path}")
    if int(result.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"Test100 evaluation count drift: {path}")
    return result


def _run_record(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata = summary["structured_subspace"]
    solver = summary["solver"]
    return {
        "arm_id": str(metadata["arm_id"]),
        "dimension": int(metadata["subspace_dimension"]),
        "mapping_nonzero_count": int(metadata["mapping_nonzero_count"]),
        "mapping_sha256": str(metadata["mapping_sha256"]),
        "angular_radial_modes": int(metadata["angular_radial_modes"]),
        "four_body_radial_modes": int(metadata["four_body_radial_modes"]),
        "random_environment_radial_modes": int(
            metadata["random_environment_radial_modes"]
        ),
        "active_feature_count": int(solver["active_feature_count"]),
        "design_residual_relative": float(solver["design_residual_relative"]),
        "expanded_coefficient_norm": float(solver["expanded_coefficient_norm"]),
        "train_hvp_median": float(summary["train_hvp_relative_frobenius"]["median"]),
        "train_hvp_max": float(summary["train_hvp_relative_frobenius"]["max"]),
        "heldout_hvp_median": float(
            summary["heldout_hvp_relative_frobenius"]["median"]
        ),
        "heldout_hvp_p90": float(summary["heldout_hvp_relative_frobenius"]["p90"]),
        "heldout_hvp_max": float(summary["heldout_hvp_relative_frobenius"]["max"]),
        "full_hessian_median": float(summary["hessian_relative_frobenius"]["median"]),
        "full_hessian_p90": float(summary["hessian_relative_frobenius"]["p90"]),
        "full_hessian_max": float(summary["hessian_relative_frobenius"]["max"]),
        "fraction_full_below_0p15": float(summary["fraction_full_hessian_below_0p15"]),
        "energy_median_ratio": float(summary["energy_median_ratio_to_source"]),
        "force_median_ratio": float(summary["force_median_ratio_to_source"]),
        "numeric_gate_passed": bool(
            summary["stage2_direction_generalization_gate_passed"]
        ),
        "wall_time_s": float(summary["wall_time_s"]),
        "max_rss_mb": float(summary["max_rss_mb"]),
        "gpu_peak_memory_mb": float(summary["gpu_peak_memory_mb"]),
        "summary": path.resolve().as_posix(),
        "summary_sha256": _sha256(path),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("promotion_allowed") is not False:
        raise ValueError("structured path must remain diagnostic-only")
    source = protocol["source"]
    stage2_protocol = Path(source["stage2_protocol"])
    stage2_hash = str(source["stage2_protocol_sha256"])
    if _sha256(stage2_protocol) != stage2_hash:
        raise ValueError("Stage-2 protocol hash drift")
    baseline_path = Path(source["baseline_summary"])
    if _sha256(baseline_path) != str(source["baseline_summary_sha256"]):
        raise ValueError("baseline summary hash drift")
    baseline = _load_frozen(baseline_path)
    full20_path = Path(source["full20_capacity_summary"])
    if _sha256(full20_path) != str(source["full20_capacity_summary_sha256"]):
        raise ValueError("full20 summary hash drift")
    full20 = _load_frozen(full20_path)

    expected_arms = [str(value) for value in protocol["arms"]]
    records = []
    parent_rows = []
    direction_rows = []
    for arm_id in expected_arms:
        path = args.subspace_root / arm_id / "summary.json"
        summary = _load_frozen(path)
        if summary.get("diagnostic_structured_subspace") is not True:
            raise ValueError(f"missing structured diagnostic marker: {arm_id}")
        if summary.get("promotion_allowed") is not False:
            raise ValueError(f"structured arm unexpectedly allows promotion: {arm_id}")
        if str(summary["protocol_sha256"]) != stage2_hash:
            raise ValueError(f"Stage-2 protocol drift: {arm_id}")
        if str(summary["structured_subspace"]["arm_id"]) != arm_id:
            raise ValueError(f"structured arm id drift: {arm_id}")
        records.append(_run_record(summary, path))
        with (path.parent / "per_parent_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                parent_rows.append({"arm_id": arm_id, **row})
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (path.parent / "per_direction_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(str(row["role"]), str(row["kind"]))].append(
                    float(row["hvp_relative_l2"])
                )
        for (role, kind), values in sorted(grouped.items()):
            direction_rows.append(
                {"arm_id": arm_id, "role": role, "kind": kind, **_distribution(values)}
            )

    best = min(
        records,
        key=lambda row: (
            row["heldout_hvp_median"],
            row["heldout_hvp_p90"],
            row["full_hessian_median"],
        ),
    )
    baseline_record = {
        "train_hvp_median": float(baseline["train_hvp_relative_frobenius"]["median"]),
        "train_hvp_max": float(baseline["train_hvp_relative_frobenius"]["max"]),
        "heldout_hvp_median": float(
            baseline["heldout_hvp_relative_frobenius"]["median"]
        ),
        "heldout_hvp_p90": float(baseline["heldout_hvp_relative_frobenius"]["p90"]),
        "full_hessian_median": float(baseline["hessian_relative_frobenius"]["median"]),
        "full_hessian_p90": float(baseline["hessian_relative_frobenius"]["p90"]),
    }
    result = {
        "definition": (
            "One-shot chemistry-aware typed coefficient-subspace diagnostic on the already "
            "exposed train20 train17/held7 split."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "stage2_protocol_sha256": stage2_hash,
        "baseline_full_space": baseline_record,
        "full20_fit_only_hessian_median": float(
            full20["hessian_relative_frobenius"]["median"]
        ),
        "best_by_exposed_heldout_diagnostic": best,
        "any_numeric_gate_passed": any(row["numeric_gate_passed"] for row in records),
        "new_direction_confirmation_required_for_promotion": True,
        "promotion_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("structured_path.csv", records),
        ("per_parent_metrics.csv", parent_rows),
        ("direction_type_metrics.csv", direction_rows),
    ):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
