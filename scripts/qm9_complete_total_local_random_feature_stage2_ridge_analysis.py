#!/usr/bin/env python3
"""Summarize the preregistered train20 ridge diagnostic without promotion."""

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
        raise ValueError("diagnostic distribution is empty or non-finite")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary.get("validation_accessed") is not False:
        raise ValueError(f"validation access is not frozen: {path}")
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"Test100 access is not frozen: {path}")
    if int(summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"Test100 evaluation count drift: {path}")
    return summary


def _discover_ridge_summaries(
    root: Path, expected_ridges: list[float]
) -> dict[float, Path]:
    discovered: dict[float, Path] = {}
    for path in sorted(root.glob("ridge_*/summary.json")):
        summary = json.loads(path.read_text())
        ridge = float(summary["effective_ridge"])
        if ridge in discovered:
            raise ValueError(f"duplicate effective ridge {ridge:g}")
        discovered[ridge] = path
    expected = set(expected_ridges)
    if set(discovered) != expected:
        missing = sorted(expected - set(discovered))
        extra = sorted(set(discovered) - expected)
        raise ValueError(f"ridge artifact set drift: missing={missing}, extra={extra}")
    return discovered


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic = yaml.safe_load(args.protocol.read_text())
    if diagnostic.get("promotion_allowed") is not False:
        raise ValueError("ridge path must remain diagnostic-only")
    if diagnostic.get("test100_access_allowed") is not False:
        raise ValueError("ridge path does not freeze Test100")
    source = diagnostic["source"]
    stage2_protocol = Path(source["stage2_protocol"])
    if _sha256(stage2_protocol) != str(source["stage2_protocol_sha256"]):
        raise ValueError("Stage-2 protocol hash drift")
    baseline_path = Path(source["baseline_summary"])
    if _sha256(baseline_path) != str(source["baseline_summary_sha256"]):
        raise ValueError("baseline summary hash drift")

    run_records: list[dict[str, Any]] = []
    direction_records: list[dict[str, Any]] = []
    ridge_grid = [float(value) for value in diagnostic["ridge_grid"]]
    ridge_paths = _discover_ridge_summaries(args.ridge_root, ridge_grid)
    run_specs = [(float(source["baseline_ridge"]), baseline_path, False)]
    run_specs.extend((ridge, ridge_paths[ridge], True) for ridge in ridge_grid)
    stage2_hash = str(source["stage2_protocol_sha256"])
    for ridge, path, is_override in run_specs:
        summary = _load_summary(path)
        if str(summary["protocol_sha256"]) != stage2_hash:
            raise ValueError(f"Stage-2 protocol drift: {path}")
        if is_override:
            if summary.get("diagnostic_ridge_override") is not True:
                raise ValueError(f"missing diagnostic override marker: {path}")
            if summary.get("promotion_allowed") is not False:
                raise ValueError(f"override unexpectedly allows promotion: {path}")
            if not np.isclose(float(summary["effective_ridge"]), ridge):
                raise ValueError(f"effective ridge drift: {path}")
        solver = summary["solver"]
        run_records.append(
            {
                "ridge": ridge,
                "diagnostic_override": is_override,
                "coefficient_norm": float(solver["coefficient_norm"]),
                "normalized_solution_norm": float(solver["normalized_solution_norm"]),
                "design_residual_relative": float(solver["design_residual_relative"]),
                "train_hvp_median": float(
                    summary["train_hvp_relative_frobenius"]["median"]
                ),
                "train_hvp_max": float(
                    summary["train_hvp_relative_frobenius"]["max"]
                ),
                "heldout_hvp_median": float(
                    summary["heldout_hvp_relative_frobenius"]["median"]
                ),
                "heldout_hvp_p90": float(
                    summary["heldout_hvp_relative_frobenius"]["p90"]
                ),
                "full_hessian_median": float(
                    summary["hessian_relative_frobenius"]["median"]
                ),
                "full_hessian_p90": float(
                    summary["hessian_relative_frobenius"]["p90"]
                ),
                "full_hessian_max": float(
                    summary["hessian_relative_frobenius"]["max"]
                ),
                "fraction_full_below_0p15": float(
                    summary["fraction_full_hessian_below_0p15"]
                ),
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
        )
        direction_path = path.parent / "per_direction_metrics.csv"
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        with direction_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(str(row["role"]), str(row["kind"]))].append(
                    float(row["hvp_relative_l2"])
                )
        for (role, kind), values in sorted(grouped.items()):
            direction_records.append(
                {"ridge": ridge, "role": role, "kind": kind, **_distribution(values)}
            )

    best = min(
        run_records,
        key=lambda row: (
            row["heldout_hvp_median"],
            row["heldout_hvp_p90"],
            row["full_hessian_median"],
        ),
    )
    baseline = run_records[0]
    result = {
        "definition": (
            "One-shot regularization-path diagnosis on the frozen train20 train17/held7 "
            "direction split. Override runs are not promotion candidates."
        ),
        "diagnostic_protocol": args.protocol.resolve().as_posix(),
        "diagnostic_protocol_sha256": _sha256(args.protocol),
        "stage2_protocol_sha256": stage2_hash,
        "run_count": len(run_records),
        "best_by_preregistered_heldout_metric": best,
        "baseline": baseline,
        "ridge_only_resolves_stage2_gate": any(
            row["stage2_numeric_gate_passed"] for row in run_records
        ),
        "baseline_heldout_to_train_median_ratio": (
            baseline["heldout_hvp_median"] / baseline["train_hvp_median"]
        ),
        "best_heldout_to_train_median_ratio": (
            best["heldout_hvp_median"] / max(best["train_hvp_median"], 1e-30)
        ),
        "promotion_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": run_records,
        "direction_type_distributions": direction_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ridge_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_records[0]))
        writer.writeheader()
        writer.writerows(run_records)
    with (args.output_dir / "direction_type_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_records[0]))
        writer.writeheader()
        writer.writerows(direction_records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
