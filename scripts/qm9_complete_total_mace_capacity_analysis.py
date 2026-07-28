#!/usr/bin/env python3
"""Merge stable5 MACE capacity runs without opening later-stage data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_summary(summary: dict[str, Any], label: str) -> None:
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"{label} does not prove Test100 remained frozen")
    if int(summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"{label} has a nonzero Test100 evaluation count")
    if summary.get("validation_accessed") is not False:
        raise ValueError(f"{label} accessed validation data")
    if int(summary.get("unselected_parent_artifacts_opened", -1)) != 0:
        raise ValueError(f"{label} opened unselected-parent artifacts")
    if summary.get("run_mode") != "formal":
        raise ValueError(f"{label} is not a completed formal run")


def _first_threshold_step(
    metrics: list[dict[str, Any]], threshold: float
) -> int | None:
    for row in metrics:
        if float(row["median_relative_frobenius"]) <= threshold:
            return int(row["step"])
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, curve_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(str(row["label"]) for row in curve_rows))
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for label in labels:
        rows = [row for row in curve_rows if row["label"] == label]
        steps = [int(row["step"]) for row in rows]
        axis.plot(
            steps,
            [float(row["median_relative_frobenius"]) for row in rows],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=f"{label} median",
        )
        axis.plot(
            steps,
            [float(row["max_relative_frobenius"]) for row in rows],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label=f"{label} max",
        )
    for threshold, label in ((0.15, "15%"), (0.05, "5% gate")):
        axis.axhline(threshold, color="black", linewidth=0.8, alpha=0.5)
        axis.text(0.99, threshold, label, transform=axis.get_yaxis_transform(), ha="right")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Hessian relative Frobenius")
    axis.set_yscale("log")
    axis.grid(True, which="both", linewidth=0.5, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(
    runs: list[tuple[str, Path]], output_dir: Path, *, make_plot: bool = True
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    per_parent_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    selected_parent_ids: list[str] | None = None

    for label, run_dir in runs:
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "training_metrics.jsonl"
        if not summary_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"incomplete run directory: {run_dir}")
        summary = json.loads(summary_path.read_text())
        metrics = _read_jsonl(metrics_path)
        _validate_summary(summary, label)
        parent_ids = [str(value) for value in summary["selected_parent_ids"]]
        if selected_parent_ids is None:
            selected_parent_ids = parent_ids
        elif parent_ids != selected_parent_ids:
            raise ValueError("MACE runs do not use identical ordered stable5 parents")
        final = summary["final"]
        distribution = final["hessian_relative_frobenius"]
        gate_passed = bool(final["gate"]["passed"])
        row = {
            "label": label,
            "arm_id": summary["arm_id"],
            "steps": int(summary["steps"]),
            "best_step": int(summary["best_step"]),
            "median_relative_frobenius": float(distribution["median"]),
            "p90_relative_frobenius": float(distribution["p90"]),
            "max_relative_frobenius": float(distribution["max"]),
            "energy_median_ratio_to_source": float(
                final["energy_median_ratio_to_source"]
            ),
            "force_median_ratio_to_source": float(
                final["force_median_ratio_to_source"]
            ),
            "max_antisymmetric_over_symmetric_frobenius": float(
                final["max_antisymmetric_over_symmetric_frobenius"]
            ),
            "wall_time_s": float(summary["wall_time_s"]),
            "gpu_peak_memory_mb": float(summary["gpu_peak_memory_mb"]),
            "max_rss_mb": float(summary["max_rss_mb"]),
            "gate_passed": gate_passed,
            "parent_cv_authorized": bool(summary["parent_cv_authorized"]),
        }
        final_rows.append(row)
        for metric in metrics:
            curve_rows.append(
                {
                    "label": label,
                    "step": int(metric["step"]),
                    "wall_time_s": float(metric["wall_time_s"]),
                    "median_relative_frobenius": float(
                        metric["median_relative_frobenius"]
                    ),
                    "p90_relative_frobenius": float(
                        metric["p90_relative_frobenius"]
                    ),
                    "max_relative_frobenius": float(
                        metric["max_relative_frobenius"]
                    ),
                    "selection_score": float(metric["selection_score"]),
                    "gpu_peak_memory_mb": float(metric["gpu_peak_memory_mb"]),
                }
            )
        for parent in summary["per_parent"]:
            per_parent_rows.append(
                {
                    "label": label,
                    "molecule_id": str(parent["molecule_id"]),
                    "natoms": int(parent["natoms"]),
                    "relative_frobenius": float(parent["relative_frobenius"]),
                    "mae": float(parent["mae"]),
                    "rmse": float(parent["rmse"]),
                    "source_relative_frobenius": float(
                        parent["source_relative_frobenius"]
                    ),
                }
            )
        run_records.append(
            {
                "label": label,
                "run_dir": run_dir.resolve().as_posix(),
                "summary_sha256": _sha256(summary_path),
                "training_metrics_sha256": _sha256(metrics_path),
                "protocol": summary["protocol"],
                "protocol_sha256": summary["protocol_sha256"],
                "source_split_sha256": summary["source_split_sha256"],
                "threshold_first_steps": {
                    str(threshold): _first_threshold_step(metrics, threshold)
                    for threshold in (1.0, 0.5, 0.15, 0.05)
                },
            }
        )

    passing = [row for row in final_rows if row["gate_passed"]]
    selected = min(
        passing,
        key=lambda row: (
            row["median_relative_frobenius"],
            row["max_relative_frobenius"],
        ),
        default=None,
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "stable5_capacity_only",
        "selected_parent_ids": selected_parent_ids,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "validation_accessed": False,
        "runs": run_records,
        "final_rows": final_rows,
        "selected_candidate": None if selected is None else selected["label"],
        "next_stage_authorized": selected is not None,
        "decision": (
            "freeze_new_train20_protocol"
            if selected is not None
            else "fail_closed_keep_train20_and_test100_frozen"
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_csv(output_dir / "final_comparison.csv", final_rows)
    _write_csv(output_dir / "learning_curves.csv", curve_rows)
    _write_csv(output_dir / "per_parent_comparison.csv", per_parent_rows)
    if make_plot:
        _write_plot(output_dir / "learning_curves.png", curve_rows)
    return result


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(arguments.run, arguments.output_dir)
