#!/usr/bin/env python3
"""Quantify descriptor-scale extrapolation in frozen unseen-parent diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError("at least three unseen-parent records are required")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    molecule_ids = [str(row["molecule_id"]) for row in rows]
    relative = np.asarray([float(row["relative_frobenius"]) for row in rows])
    if np.any(relative <= 0.0) or not np.all(np.isfinite(relative)):
        raise ValueError("relative Frobenius errors must be positive and finite")

    variables = (
        ("descriptor_l2", True),
        ("descriptor_max_abs", True),
        ("activation_max_abs", True),
        ("hessian_correction_frobenius", True),
        ("three_body_matched_active_feature_fraction", False),
        ("four_body_matched_active_feature_fraction", False),
        ("natoms", False),
    )
    target = np.log10(relative)
    correlations = []
    for name, log_transform in variables:
        values = np.asarray([float(row[name]) for row in rows])
        if log_transform:
            if np.any(values <= 0.0):
                raise ValueError(f"{name} must be positive for log correlation")
            transformed = np.log10(values)
        else:
            transformed = values
        correlations.append(
            {
                "variable": name,
                "x_transform": "log10" if log_transform else "identity",
                **_correlations(transformed, target),
            }
        )

    with (output / "correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    plot_variables = (
        ("descriptor_l2", "Descriptor L2"),
        ("descriptor_max_abs", "Descriptor max abs"),
        ("hessian_correction_frobenius", "Correction Hessian Frobenius"),
    )
    by_name = {row["variable"]: row for row in correlations}
    for axis, (name, label) in zip(axes, plot_variables, strict=True):
        values = np.asarray([float(row[name]) for row in rows])
        axis.scatter(values, relative, color="#1f6f8b", s=35)
        for molecule_id, x_value, y_value in zip(
            molecule_ids, values, relative, strict=True
        ):
            axis.annotate(molecule_id, (x_value, y_value), fontsize=7, xytext=(3, 3),
                          textcoords="offset points")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Hessian relative Frobenius")
        stats = by_name[name]
        axis.set_title(f"Pearson {stats['pearson']:.3f}; Spearman {stats['spearman']:.3f}")
        axis.grid(True, which="both", alpha=0.25)
    figure_path = output / "descriptor_scaling_vs_hessian_error.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    result = {
        "definition": "frozen unseen-parent descriptor-scale extrapolation diagnostic",
        "input_csv": args.input_csv.resolve().as_posix(),
        "input_csv_sha256": _sha256(args.input_csv),
        "parent_count": len(rows),
        "correlations": correlations,
        "figure": figure_path.resolve().as_posix(),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
