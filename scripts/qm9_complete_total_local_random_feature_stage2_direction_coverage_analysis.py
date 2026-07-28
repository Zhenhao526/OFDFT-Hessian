#!/usr/bin/env python3
"""Summarize frozen train20 direction-coverage diagnostics."""

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
        raise ValueError("coverage distribution is empty or non-finite")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("validation_accessed") is not False:
        raise ValueError(f"validation access drift: {path}")
    if result.get("test100_accessed") is not False:
        raise ValueError(f"Test100 access drift: {path}")
    if int(result.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"Test100 evaluation count drift: {path}")
    return result


def _run_record(name: str, path: Path, diagnostic: bool) -> dict[str, Any]:
    summary = _load(path)
    extensions = summary.get("direction_extension", [])
    train_counts = (
        [int(row["final_train_count"]) for row in extensions]
        if extensions
        else [17]
    )
    added_counts = (
        [int(row["added_train_count"]) for row in extensions]
        if extensions
        else [0]
    )
    solver = summary["solver"]
    return {
        "arm": name,
        "diagnostic_extension": diagnostic,
        "train_count_min": min(train_counts),
        "train_count_median": float(np.median(train_counts)),
        "train_count_max": max(train_counts),
        "added_count_median": float(np.median(added_counts)),
        "design_row_count": int(solver["design_row_count"]),
        "active_feature_count": int(solver["active_feature_count"]),
        "coefficient_norm": float(solver["coefficient_norm"]),
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
        raise ValueError("direction coverage must remain diagnostic-only")
    source = protocol["source"]
    stage2_protocol = Path(source["stage2_protocol"])
    stage2_hash = str(source["stage2_protocol_sha256"])
    if _sha256(stage2_protocol) != stage2_hash:
        raise ValueError("Stage-2 protocol hash drift")
    baseline_path = Path(source["baseline_summary"])
    if _sha256(baseline_path) != str(source["baseline_summary_sha256"]):
        raise ValueError("baseline summary hash drift")
    run_paths = {
        "train17": baseline_path,
        "train24": args.coverage_root / "train24" / "summary.json",
        "train32": args.coverage_root / "train32" / "summary.json",
        "maximal": args.coverage_root / "maximal" / "summary.json",
    }
    records: list[dict[str, Any]] = []
    direction_type_rows: list[dict[str, Any]] = []
    for name, path in run_paths.items():
        summary = _load(path)
        if str(summary["protocol_sha256"]) != stage2_hash:
            raise ValueError(f"Stage-2 protocol drift: {name}")
        diagnostic = name != "train17"
        if diagnostic:
            if summary.get("diagnostic_direction_extension") is not True:
                raise ValueError(f"missing direction-extension marker: {name}")
            if summary.get("promotion_allowed") is not False:
                raise ValueError(f"direction arm unexpectedly promotable: {name}")
        records.append(_run_record(name, path, diagnostic))
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (path.parent / "per_direction_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(str(row["role"]), str(row["kind"]))].append(
                    float(row["hvp_relative_l2"])
                )
        for (role, kind), values in sorted(grouped.items()):
            direction_type_rows.append(
                {"arm": name, "role": role, "kind": kind, **_distribution(values)}
            )
    best = min(
        records,
        key=lambda row: (
            row["heldout_hvp_median"],
            row["heldout_hvp_p90"],
            row["full_hessian_median"],
        ),
    )
    result = {
        "definition": (
            "Frozen train20 coverage diagnosis with original seven held directions unchanged "
            "and geometry-only train-direction extensions."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "stage2_protocol_sha256": stage2_hash,
        "best_by_preregistered_heldout_metric": best,
        "coverage_only_resolves_stage2_gate": any(
            row["stage2_numeric_gate_passed"] for row in records
        ),
        "promotion_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": records,
        "direction_type_distributions": direction_type_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "coverage_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "direction_type_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_type_rows[0]))
        writer.writeheader()
        writer.writerows(direction_type_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

