#!/usr/bin/env python3
"""Aggregate fixed-density Hessian precision and finite-difference scans."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _aggregate(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"cases": len(rows)}
    for field in fields:
        values = _finite([row.get(field) for row in rows])
        result[f"{field}_mean"] = float(np.mean(values)) if values else None
        result[f"{field}_median"] = float(np.median(values)) if values else None
        result[f"{field}_max"] = float(np.max(values)) if values else None
    return result


def _load_autograd_hessian(path: str) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["autograd_hessian"], dtype=np.float64)


def analyze(scan_root: Path, output_dir: Path) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    hvp_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    hessian_paths: dict[tuple[str, str, str, float], str] = {}

    for summary_path in sorted(scan_root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        dtype = str(payload["model_dtype"]).replace("torch.", "")
        displacement = float(payload["fd_displacement"])
        summaries.append(payload)
        for item in payload.get("fixed_density_full_hessian", []):
            row = {
                "condition": summary_path.parent.name,
                "dtype": dtype,
                "displacement": displacement,
                "run": item["run"],
                "molecule_id": item["molecule_id"],
                "natoms": item.get("natoms"),
                "success": item.get("success", False),
                "finite": item.get("autograd_stats", {}).get("finite", False),
                "autograd_symmetry_max_abs": item.get("autograd_stats", {}).get(
                    "symmetry_max_abs"
                ),
                "fd_symmetry_max_abs": item.get("fd_stats", {}).get("symmetry_max_abs"),
                "ag_fd_mae": item.get("autograd_vs_fd", {}).get("mae"),
                "ag_fd_rmse": item.get("autograd_vs_fd", {}).get("rmse"),
                "ag_fd_relative_fro": item.get("autograd_vs_fd", {}).get(
                    "relative_fro_error"
                ),
                "autograd_pbe_mae": item.get("autograd_vs_pbe", {}).get("mae"),
                "fd_pbe_mae": item.get("fd_vs_pbe", {}).get("mae"),
                "autograd_elapsed_s": item.get("autograd_elapsed_s"),
                "fd_elapsed_s": item.get("fd_elapsed_s"),
                "autograd_peak_cuda_mb": item.get("autograd_peak_cuda_mb"),
                "fd_peak_cuda_mb": item.get("fd_peak_cuda_mb"),
                "artifact_npz": item.get("artifact_npz"),
            }
            case_rows.append(row)
            if row["artifact_npz"]:
                hessian_paths[(dtype, row["run"], row["molecule_id"], displacement)] = str(
                    row["artifact_npz"]
                )
        for item in payload.get("hvp", []):
            hvp_rows.append(
                {
                    "condition": summary_path.parent.name,
                    "dtype": dtype,
                    "displacement": displacement,
                    "run": item["run"],
                    "molecule_id": item["molecule_id"],
                    "direction": item.get("direction"),
                    "finite": item.get("hvp_stats", {}).get("finite", False),
                    "hvp_fd_mae": item.get("hvp_vs_fd_directional", {}).get("mae"),
                    "hvp_fd_rmse": item.get("hvp_vs_fd_directional", {}).get("rmse"),
                    "hvp_fd_relative_fro": item.get("hvp_vs_fd_directional", {}).get(
                        "relative_fro_error"
                    ),
                    "hvp_elapsed_s": item.get("hvp_elapsed_s"),
                    "fd_directional_elapsed_s": item.get("fd_directional_elapsed_s"),
                    "hvp_peak_cuda_mb": item.get("hvp_peak_cuda_mb"),
                }
            )

    metric_fields = [
        "ag_fd_mae",
        "ag_fd_rmse",
        "ag_fd_relative_fro",
        "autograd_symmetry_max_abs",
        "fd_symmetry_max_abs",
        "autograd_elapsed_s",
        "fd_elapsed_s",
        "autograd_peak_cuda_mb",
        "fd_peak_cuda_mb",
    ]
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["dtype"], row["run"], row["displacement"])].append(row)
    condition_rows = []
    for (dtype, run, displacement), rows in sorted(grouped.items()):
        condition_rows.append(
            {
                "dtype": dtype,
                "run": run,
                "displacement": displacement,
                "finite_cases": sum(bool(row["finite"]) for row in rows),
                **_aggregate(rows, metric_fields),
            }
        )

    hvp_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in hvp_rows:
        hvp_grouped[(row["dtype"], row["run"], row["displacement"])].append(row)
    hvp_summary_rows = []
    for (dtype, run, displacement), rows in sorted(hvp_grouped.items()):
        hvp_summary_rows.append(
            {
                "dtype": dtype,
                "run": run,
                "displacement": displacement,
                "finite_cases": sum(bool(row["finite"]) for row in rows),
                **_aggregate(
                    rows,
                    [
                        "hvp_fd_mae",
                        "hvp_fd_rmse",
                        "hvp_fd_relative_fro",
                        "hvp_elapsed_s",
                        "fd_directional_elapsed_s",
                        "hvp_peak_cuda_mb",
                    ],
                ),
            }
        )

    dtype_rows: list[dict[str, Any]] = []
    for run in sorted({row["run"] for row in case_rows}):
        for molecule_id in sorted({row["molecule_id"] for row in case_rows if row["run"] == run}):
            for displacement in sorted({row["displacement"] for row in case_rows}):
                key32 = ("float32", run, molecule_id, displacement)
                key64 = ("float64", run, molecule_id, displacement)
                if key32 not in hessian_paths or key64 not in hessian_paths:
                    continue
                h32 = _load_autograd_hessian(hessian_paths[key32])
                h64 = _load_autograd_hessian(hessian_paths[key64])
                diff = h32 - h64
                dtype_rows.append(
                    {
                        "run": run,
                        "molecule_id": molecule_id,
                        "displacement": displacement,
                        "mae": float(np.mean(np.abs(diff))),
                        "rmse": float(np.sqrt(np.mean(diff * diff))),
                        "relative_fro": float(np.linalg.norm(diff) / np.linalg.norm(h64)),
                        "max_abs": float(np.max(np.abs(diff))),
                    }
                )
    dtype_summary_rows = []
    dtype_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dtype_rows:
        dtype_grouped[row["run"]].append(row)
    for run, rows in sorted(dtype_grouped.items()):
        dtype_summary_rows.append({"run": run, **_aggregate(rows, ["mae", "rmse", "relative_fro", "max_abs"])})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    _write_csv(output_dir / "condition_summary.csv", condition_rows)
    _write_csv(output_dir / "hvp_per_case_metrics.csv", hvp_rows)
    _write_csv(output_dir / "hvp_condition_summary.csv", hvp_summary_rows)
    _write_csv(output_dir / "float32_vs_float64_per_case.csv", dtype_rows)
    _write_csv(output_dir / "float32_vs_float64_summary.csv", dtype_summary_rows)

    plot_paths: list[str] = []
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
        colors = {"EG": "#c44e52", "EGF_lam1": "#4c72b0"}
        markers = {"float32": "o", "float64": "s"}
        for dtype in sorted({row["dtype"] for row in condition_rows}):
            for run in sorted({row["run"] for row in condition_rows}):
                selected = sorted(
                    (row for row in condition_rows if row["dtype"] == dtype and row["run"] == run),
                    key=lambda row: row["displacement"],
                )
                label = f"{run} {dtype}"
                axes[0].plot(
                    [row["displacement"] for row in selected],
                    [row["ag_fd_relative_fro_mean"] for row in selected],
                    marker=markers[dtype],
                    color=colors.get(run),
                    label=label,
                )
                axes[1].plot(
                    [row["displacement"] for row in selected],
                    [row["fd_symmetry_max_abs_mean"] for row in selected],
                    marker=markers[dtype],
                    color=colors.get(run),
                    label=label,
                )
        for ax, ylabel in zip(
            axes,
            ["autograd vs FD relative Frobenius", "FD symmetry max abs"],
            strict=True,
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("finite-difference displacement (Bohr)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        plot_path = output_dir / "precision_and_step_sensitivity.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(plot_path))
    except ImportError:
        pass

    result = {
        "scan_root": str(scan_root.resolve()),
        "conditions": len(summaries),
        "full_hessian_cases": len(case_rows),
        "full_hessian_finite_cases": sum(bool(row["finite"]) for row in case_rows),
        "hvp_cases": len(hvp_rows),
        "hvp_finite_cases": sum(bool(row["finite"]) for row in hvp_rows),
        "condition_summary": condition_rows,
        "float32_vs_float64_summary": dtype_summary_rows,
        "plots": plot_paths,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.scan_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
