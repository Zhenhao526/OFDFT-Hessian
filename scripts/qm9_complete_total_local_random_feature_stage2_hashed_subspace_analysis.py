#!/usr/bin/env python3
"""Summarize frozen train20 label-independent hashed scalar subspaces."""

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
        raise ValueError("hashed-subspace distribution is empty or non-finite")
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


def _discover_dimension_summaries(
    root: Path, expected_dimensions: list[int]
) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    for path in sorted(root.glob("dim_*/summary.json")):
        summary = json.loads(path.read_text())
        metadata = summary.get("hashed_subspace")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing hashed-subspace metadata: {path}")
        dimension = int(metadata["subspace_dimension"])
        if dimension in discovered:
            raise ValueError(f"duplicate hashed subspace dimension {dimension}")
        discovered[dimension] = path
    if set(discovered) != set(expected_dimensions):
        raise ValueError(
            "hashed-subspace artifact set drift: "
            f"missing={sorted(set(expected_dimensions) - set(discovered))}, "
            f"extra={sorted(set(discovered) - set(expected_dimensions))}"
        )
    return discovered


def _run_record(
    dimension: int, path: Path, *, diagnostic: bool
) -> dict[str, Any]:
    summary = _load_frozen(path)
    solver = summary["solver"]
    metadata = summary.get("hashed_subspace") or {}
    return {
        "dimension": dimension,
        "diagnostic_subspace": diagnostic,
        "mapping_sha256": str(metadata.get("mapping_sha256", "baseline_full_space")),
        "bucket_occupancy_min": int(metadata.get("bucket_occupancy_min", 1)),
        "bucket_occupancy_median": float(metadata.get("bucket_occupancy_median", 1.0)),
        "bucket_occupancy_max": int(metadata.get("bucket_occupancy_max", 1)),
        "empty_bucket_count": int(metadata.get("empty_bucket_count", 0)),
        "active_feature_count": int(solver["active_feature_count"]),
        "coefficient_norm": float(solver["coefficient_norm"]),
        "expanded_coefficient_norm": float(
            solver.get("expanded_coefficient_norm", solver["coefficient_norm"])
        ),
        "design_residual_relative": float(solver["design_residual_relative"]),
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
        "stage2_numeric_gate_passed": bool(
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
        raise ValueError("hashed-subspace path must remain diagnostic-only")
    source = protocol["source"]
    stage2_protocol = Path(source["stage2_protocol"])
    stage2_hash = str(source["stage2_protocol_sha256"])
    if _sha256(stage2_protocol) != stage2_hash:
        raise ValueError("Stage-2 protocol hash drift")
    baseline_path = Path(source["baseline_summary"])
    if _sha256(baseline_path) != str(source["baseline_summary_sha256"]):
        raise ValueError("baseline summary hash drift")
    full20_path = Path(source["full20_capacity_summary"])
    if _sha256(full20_path) != str(source["full20_capacity_summary_sha256"]):
        raise ValueError("full20 capacity summary hash drift")
    full20 = _load_frozen(full20_path)

    dimensions = [int(value) for value in protocol["subspace"]["dimensions"]]
    paths = _discover_dimension_summaries(args.subspace_root, dimensions)
    records = [_run_record(38571, baseline_path, diagnostic=False)]
    direction_rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        path = paths[dimension]
        summary = _load_frozen(path)
        if str(summary["protocol_sha256"]) != stage2_hash:
            raise ValueError(f"Stage-2 protocol drift for dimension {dimension}")
        if summary.get("diagnostic_hashed_subspace") is not True:
            raise ValueError(f"missing diagnostic marker for dimension {dimension}")
        if summary.get("promotion_allowed") is not False:
            raise ValueError(f"dimension {dimension} unexpectedly allows promotion")
        metadata = summary["hashed_subspace"]
        if int(metadata["seed"]) != int(protocol["subspace"]["seed"]):
            raise ValueError(f"hash seed drift for dimension {dimension}")
        if int(metadata["repetitions"]) != int(protocol["subspace"]["repetitions"]):
            raise ValueError(f"hash repetition drift for dimension {dimension}")
        records.append(_run_record(dimension, path, diagnostic=True))
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (path.parent / "per_direction_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(str(row["role"]), str(row["kind"]))].append(
                    float(row["hvp_relative_l2"])
                )
        for (role, kind), values in sorted(grouped.items()):
            direction_rows.append(
                {
                    "dimension": dimension,
                    "role": role,
                    "kind": kind,
                    **_distribution(values),
                }
            )
    best = min(
        records,
        key=lambda row: (
            row["heldout_hvp_median"],
            row["heldout_hvp_p90"],
            row["full_hessian_median"],
        ),
    )
    full20_median = float(full20["hessian_relative_frobenius"]["median"])
    result = {
        "definition": (
            "One-shot train20 label-independent hashed scalar-subspace path on the existing "
            "train17/held7 split; no arm is directly promotable."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "stage2_protocol_sha256": stage2_hash,
        "best_by_preregistered_heldout_metric": best,
        "subspace_only_resolves_stage2_gate": any(
            row["stage2_numeric_gate_passed"] for row in records
        ),
        "full20_capacity_hessian_median": full20_median,
        "best_to_full20_capacity_median_ratio": (
            best["full_hessian_median"] / max(full20_median, 1e-30)
        ),
        "promotion_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": records,
        "direction_type_distributions": direction_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "subspace_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "direction_type_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_rows[0]))
        writer.writeheader()
        writer.writerows(direction_rows)
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

