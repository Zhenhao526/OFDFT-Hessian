#!/usr/bin/env python3
"""Summarize fail-closed one-parent stable5 LBFGS capacity diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    positions = np.arange(len(rows))
    values = [float(row["relative_frobenius"]) for row in rows]
    axis.bar(positions, values, color="#2f6f8f")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="5% gate")
    axis.set_xticks(positions, [str(row["molecule_id"]) for row in rows], rotation=25)
    axis.set_ylabel("Final relative Frobenius")
    axis.set_yscale("log")
    axis.grid(True, axis="y", which="both", linewidth=0.5, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(
    root: Path,
    parent_ids: list[str],
    expected_checkpoint_sha256: str,
    output_dir: Path,
    *,
    make_plot: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        path = root / parent_id / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing one-parent summary: {path}")
        summary = json.loads(path.read_text())
        if summary.get("diagnostic_parent_id") != parent_id:
            raise ValueError(f"diagnostic parent mismatch for {parent_id}")
        if summary.get("fit_parent_ids") != [parent_id]:
            raise ValueError(f"fit parent scope mismatch for {parent_id}")
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"Test100 was not frozen for {parent_id}")
        if int(summary.get("test100_evaluations_used", -1)) != 0:
            raise ValueError(f"nonzero Test100 count for {parent_id}")
        if summary.get("parent_cv_authorized") is not False:
            raise ValueError(f"one-parent run improperly authorized parent CV: {parent_id}")
        if summary.get("source_checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError(f"source checkpoint mismatch for {parent_id}")
        distribution = summary["final"]["hessian_relative_frobenius"]
        value = float(distribution["median"])
        rows.append(
            {
                "molecule_id": parent_id,
                "relative_frobenius": value,
                "passed_5pct": value <= 0.05,
                "best_iteration": int(summary["best_iteration"]),
                "closure_calls": int(summary["closure_calls"]),
                "wall_time_s": float(summary["wall_time_s"]),
                "gpu_peak_memory_mb": float(summary["gpu_peak_memory_mb"]),
                "max_rss_mb": float(summary["max_rss_mb"]),
                "summary_sha256": _sha256(path),
            }
        )
    values = np.asarray([row["relative_frobenius"] for row in rows], dtype=np.float64)
    passed_count = int(np.count_nonzero(values <= 0.05))
    result = {
        "scope": "stable5_one_parent_diagnostic_only",
        "root": root.resolve().as_posix(),
        "parent_ids": parent_ids,
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_cv_authorized": False,
        "rows": rows,
        "aggregate": {
            "median_relative_frobenius": float(np.median(values)),
            "p90_relative_frobenius": float(np.quantile(values, 0.9)),
            "max_relative_frobenius": float(np.max(values)),
            "passed_5pct_count": passed_count,
            "parent_count": len(rows),
        },
        "single_parent_capacity_passed": passed_count == len(rows),
        "decision": (
            "single_parent_span_supported_but_parent_cv_still_forbidden"
            if passed_count == len(rows)
            else "single_parent_span_not_supported_at_registered_budget"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (output_dir / "per_parent.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if make_plot:
        _write_plot(output_dir / "per_parent_relative_frobenius.png", rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent-id", required=True, action="append")
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(
        arguments.root,
        arguments.parent_id,
        arguments.source_checkpoint_sha256,
        arguments.output_dir,
    )
