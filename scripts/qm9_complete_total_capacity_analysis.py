#!/usr/bin/env python3
"""Aggregate complete-total capacity runs without reading validation or Test100 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: str | None) -> float | int | str | None:
    if value in (None, ""):
        return None
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _normalized_rows(run_dir: Path, filename: str) -> list[dict[str, Any]]:
    return [
        {
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            **{key: _number(value) for key, value in row.items()},
        }
        for row in _read_csv(run_dir / filename)
    ]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _hessian_residual_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "hessian_arrays").glob("step_*_*.npz")):
        match = re.fullmatch(r"step_(\d+)_(.+)\.npz", path.name)
        if match is None:
            continue
        with np.load(path) as payload:
            prediction = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        if prediction.ndim != 2 or reference.ndim != 2:
            raise ValueError(f"Hessian arrays must be matrices: {path}")
        if prediction.shape[0] != reference.shape[0]:
            raise ValueError(f"Hessian row mismatch: {path}")
        reference = reference[:, : prediction.shape[1]]
        predicted_vector = prediction.reshape(-1)
        reference_vector = reference.reshape(-1)
        residual = predicted_vector - reference_vector
        reference_norm = max(np.linalg.norm(reference_vector), np.finfo(float).tiny)
        prediction_norm = max(np.linalg.norm(predicted_vector), np.finfo(float).tiny)
        best_scale = float(
            np.dot(predicted_vector, reference_vector) / reference_norm**2
        )
        orthogonal_residual = predicted_vector - best_scale * reference_vector
        natoms = prediction.shape[0] // 3
        predicted_rigid_sum = np.stack(
            [
                prediction[:, column].reshape(natoms, 3).sum(axis=0)
                for column in range(prediction.shape[1])
            ]
        )
        reference_rigid_sum = np.stack(
            [
                reference[:, column].reshape(natoms, 3).sum(axis=0)
                for column in range(reference.shape[1])
            ]
        )
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "array_path": str(path.resolve()),
                "step": int(match.group(1)),
                "molecule_id": match.group(2),
                "natoms": natoms,
                "column_count": prediction.shape[1],
                "relative_frobenius": float(
                    np.linalg.norm(residual) / reference_norm
                ),
                "prediction_reference_cosine": float(
                    np.dot(predicted_vector, reference_vector)
                    / (prediction_norm * reference_norm)
                ),
                "best_reference_scale": best_scale,
                "orthogonal_relative_frobenius": float(
                    np.linalg.norm(orthogonal_residual) / reference_norm
                ),
                "max_abs_error": float(np.max(np.abs(residual))),
                "predicted_rigid_sum_max_abs": float(
                    np.max(np.abs(predicted_rigid_sum))
                ),
                "reference_rigid_sum_max_abs": float(
                    np.max(np.abs(reference_rigid_sum))
                ),
            }
        )
    return rows


def _plot(output_dir: Path, hessian_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    if hessian_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
        metrics = (
            ("relative_frobenius", "Hessian relative Frobenius"),
            ("total_energy_abs_error_hartree", "Energy absolute error (Ha)"),
            ("complete_total_force_mae_hartree_per_bohr", "Force MAE (Ha/Bohr)"),
        )
        for run_name in sorted({str(row["run_name"]) for row in hessian_rows}):
            selected = [
                row
                for row in hessian_rows
                if row["run_name"] == run_name and _finite(row.get("step"))
            ]
            selected.sort(key=lambda row: float(row["step"]))
            for axis, (field, label) in zip(axes, metrics):
                points = [row for row in selected if _finite(row.get(field))]
                if points:
                    axis.plot(
                        [float(row["step"]) for row in points],
                        [float(row[field]) for row in points],
                        marker="o",
                        markersize=3,
                        linewidth=1.2,
                        label=run_name,
                    )
                axis.set_xlabel("capacity step")
                axis.set_ylabel(label)
                axis.grid(alpha=0.25)
        axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1.0)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        path = output_dir / "capacity_energy_force_hessian_trajectory.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def analyze(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hessian_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    summaries = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        hessian = _normalized_rows(run_dir, "full_hessian_metrics.csv")
        losses = _normalized_rows(run_dir, "training_curve.csv")
        hessian_rows.extend(hessian)
        loss_rows.extend(losses)
        residual_rows.extend(_hessian_residual_rows(run_dir))
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        if summary is not None and (
            summary.get("test100_accessed") is not False
            or int(summary.get("test100_evaluations_used", -1)) != 0
        ):
            raise ValueError(f"Test100 freeze certificate failed for {run_dir}")
        final = max(hessian, key=lambda row: int(row["step"])) if hessian else {}
        update_counts = Counter(str(row.get("update_kind")) for row in losses)
        summaries.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "completed": summary is not None,
                "hvp_loss_norm": summary.get("hvp_loss_norm") if summary else None,
                "source_capacity_step": (
                    summary.get("source_capacity_step") if summary else None
                ),
                "final_step": final.get("step"),
                "final_relative_frobenius": final.get("relative_frobenius"),
                "final_energy_abs_error_hartree": final.get(
                    "total_energy_abs_error_hartree"
                ),
                "final_force_mae_hartree_per_bohr": final.get(
                    "complete_total_force_mae_hartree_per_bohr"
                ),
                "joint_update_count": update_counts.get("joint", 0),
                "hvp_update_count": update_counts.get("hvp", 0),
                "replay_update_count": update_counts.get("replay", 0),
                "wall_time_s": summary.get("wall_time_s") if summary else None,
                "max_rss_mb": summary.get("max_rss_mb") if summary else None,
                "peak_gpu_memory_mb": (
                    summary.get("peak_gpu_memory_mb") if summary else None
                ),
                "stage1_gate_passed": (
                    summary.get("stage1_gate_passed") if summary else False
                ),
                "test100_accessed": (
                    summary.get("test100_accessed") if summary else False
                ),
            }
        )

    _write_csv(output_dir / "capacity_run_summary.csv", summaries)
    _write_csv(output_dir / "capacity_hessian_trajectory.csv", hessian_rows)
    _write_csv(output_dir / "capacity_loss_trajectory.csv", loss_rows)
    _write_csv(output_dir / "capacity_hessian_residuals.csv", residual_rows)
    result = {
        "definition": "Train-only complete-total relaxed capacity-run aggregation",
        "run_count": len(run_dirs),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": summaries,
        "hessian_residual_row_count": len(residual_rows),
        "plots": _plot(output_dir, hessian_rows),
    }
    (output_dir / "capacity_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.run_dir, args.output_dir)
