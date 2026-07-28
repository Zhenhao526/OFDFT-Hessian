#!/usr/bin/env python3
"""Summarize the preregistered train20 random-feature width diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary.get("validation_accessed") is not False:
        raise ValueError(f"validation access is not frozen: {path}")
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"Test100 access is not frozen: {path}")
    if int(summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"Test100 evaluation count drift: {path}")
    return summary


def _discover_width_summaries(root: Path, expected_widths: list[int]) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    for path in sorted(root.glob("width_*/summary.json")):
        summary = json.loads(path.read_text())
        subset = summary.get("feature_subset")
        if not isinstance(subset, dict):
            raise ValueError(f"missing feature subset: {path}")
        width = int(subset["selected_width_per_scale"])
        if width in discovered:
            raise ValueError(f"duplicate random-feature width {width}")
        discovered[width] = path
    if set(discovered) != set(expected_widths):
        raise ValueError(
            "width artifact set drift: "
            f"missing={sorted(set(expected_widths) - set(discovered))}, "
            f"extra={sorted(set(discovered) - set(expected_widths))}"
        )
    return discovered


def _record(width: int, path: Path, *, diagnostic: bool) -> dict[str, Any]:
    summary = _load_frozen_summary(path)
    solver = summary["solver"]
    subset = summary.get("feature_subset")
    return {
        "width_per_scale": width,
        "diagnostic_subset": diagnostic,
        "selected_global_feature_count": (
            int(subset["selected_global_feature_count"])
            if isinstance(subset, dict)
            else int(summary["global_feature_count"])
        ),
        "active_feature_count": int(solver["active_feature_count"]),
        "coefficient_norm": float(solver["coefficient_norm"]),
        "normalized_solution_norm": float(solver["normalized_solution_norm"]),
        "design_residual_relative": float(solver["design_residual_relative"]),
        "train_hvp_median": float(summary["train_hvp_relative_frobenius"]["median"]),
        "train_hvp_max": float(summary["train_hvp_relative_frobenius"]["max"]),
        "heldout_hvp_median": float(
            summary["heldout_hvp_relative_frobenius"]["median"]
        ),
        "heldout_hvp_p90": float(summary["heldout_hvp_relative_frobenius"]["p90"]),
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
        raise ValueError("width path must remain diagnostic-only")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("width path does not freeze Test100")
    source = protocol["source"]
    stage2_protocol = Path(source["stage2_protocol"])
    stage2_hash = str(source["stage2_protocol_sha256"])
    if _sha256(stage2_protocol) != stage2_hash:
        raise ValueError("Stage-2 protocol hash drift")
    baseline_path = Path(source["baseline_summary"])
    if _sha256(baseline_path) != str(source["baseline_summary_sha256"]):
        raise ValueError("baseline summary hash drift")
    baseline_summary = _load_frozen_summary(baseline_path)
    if str(baseline_summary["protocol_sha256"]) != stage2_hash:
        raise ValueError("baseline Stage-2 protocol drift")

    widths = [int(value) for value in protocol["width_per_scale_grid"]]
    paths = _discover_width_summaries(args.width_root, widths)
    records = [
        _record(int(source["baseline_width_per_scale"]), baseline_path, diagnostic=False)
    ]
    for width in widths:
        summary = _load_frozen_summary(paths[width])
        if str(summary["protocol_sha256"]) != stage2_hash:
            raise ValueError(f"Stage-2 protocol drift for width {width}")
        if summary.get("diagnostic_feature_subset") is not True:
            raise ValueError(f"missing diagnostic subset marker for width {width}")
        if summary.get("promotion_allowed") is not False:
            raise ValueError(f"width {width} unexpectedly allows promotion")
        records.append(_record(width, paths[width], diagnostic=True))

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
            "One-shot frozen train20 random-feature width diagnosis; complete explicit angular "
            "and four-body basis retained; no subset is promotable."
        ),
        "diagnostic_protocol": args.protocol.resolve().as_posix(),
        "diagnostic_protocol_sha256": _sha256(args.protocol),
        "stage2_protocol_sha256": stage2_hash,
        "best_by_preregistered_heldout_metric": best,
        "width_only_resolves_stage2_gate": any(
            row["stage2_numeric_gate_passed"] for row in records
        ),
        "promotion_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "width_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--width-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

