#!/usr/bin/env python3
"""Compare cached and uncached density-relaxed Hessian evaluator outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _matrix_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference_norm = float(np.linalg.norm(reference))
    return {
        "hessian_mae": float(np.mean(np.abs(diff))),
        "hessian_rmse": float(np.sqrt(np.mean(diff * diff))),
        "hessian_relative_fro": (
            float(np.linalg.norm(diff) / reference_norm) if reference_norm else float("nan")
        ),
        "hessian_max_abs": float(np.max(np.abs(diff))),
    }


def _row(payload: dict[str, Any], run: str, molecule_id: str) -> dict[str, Any]:
    return next(
        row
        for row in payload["rows"]
        if row["run"] == run and row["molecule_id"] == molecule_id and row["success"]
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline_summary.read_text())
    cached = json.loads(args.cached_summary.read_text())
    runs = args.run or sorted({row["run"] for row in cached["rows"]})
    rows = []
    for run in runs:
        baseline_row = _row(baseline, run, args.molecule_id)
        cached_row = _row(cached, run, args.molecule_id)
        baseline_hessian = np.load(baseline_row["hessian_npz"])["density_relaxed_hessian"]
        cached_hessian = np.load(cached_row["hessian_npz"])["density_relaxed_hessian"]
        baseline_elapsed = float(baseline_row["elapsed_s"])
        cached_elapsed = float(cached_row["elapsed_s"])
        rows.append(
            {
                "run": run,
                "molecule_id": args.molecule_id,
                "baseline_elapsed_s": baseline_elapsed,
                "cached_elapsed_s": cached_elapsed,
                "speedup": baseline_elapsed / cached_elapsed,
                "baseline_mean_cycles": baseline_row.get("mean_opt_cycles"),
                "cached_mean_cycles": cached_row.get("mean_opt_cycles"),
                "baseline_pbe_mae": baseline_row.get("mae"),
                "cached_pbe_mae": cached_row.get("mae"),
                **_matrix_metrics(cached_hessian, baseline_hessian),
            }
        )

    baseline_total = sum(row["baseline_elapsed_s"] for row in rows)
    cached_total = sum(row["cached_elapsed_s"] for row in rows)
    result = {
        "definition": (
            "Correctness and model-time comparison for optional CPU prepared-geometry sharing "
            "across otherwise identical model runs."
        ),
        "baseline_summary": args.baseline_summary.as_posix(),
        "cached_summary": args.cached_summary.as_posix(),
        "molecule_id": args.molecule_id,
        "prepared_geometry_cache": cached.get("prepared_geometry_cache"),
        "baseline_total_model_elapsed_s": baseline_total,
        "cached_total_model_elapsed_s": cached_total,
        "combined_model_time_speedup": baseline_total / cached_total,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "cache_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--cached-summary", type=Path, required=True)
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
