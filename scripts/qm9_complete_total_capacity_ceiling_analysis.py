#!/usr/bin/env python3
"""Aggregate train-only complete-total capacity runs and create diagnostic plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _run_rows(name: str, run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"{name} does not certify frozen Test100")
    if "per_parent" in summary:
        rows = summary["per_parent"]
    elif "stages" in summary:
        rows = summary["stages"][-1]["per_parent"]
    else:
        raise ValueError(f"{name} has no per-parent capacity metrics")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "run": name,
                "molecule_id": str(row["molecule_id"]),
                "natoms": int(row["natoms"]),
                "relative_frobenius": float(row["relative_frobenius"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "asymmetry_ratio": float(
                    row["antisymmetric_over_symmetric_frobenius"]
                ),
                "energy_abs_error_hartree": float(row["energy_abs_error_hartree"]),
                "force_mae_hartree_per_bohr": float(
                    row["force_mae_hartree_per_bohr"]
                ),
            }
        )
    trajectory = []
    log_path = run_dir / "training_metrics.jsonl"
    if log_path.is_file():
        trajectory = [json.loads(line) for line in log_path.read_text().splitlines()]
    return summary, normalized, trajectory


def _baseline_rows(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("baseline manifest does not certify frozen Test100")
    rows = []
    for row in manifest["parents"]:
        with np.load(row["capacity_array"]) as payload:
            prediction = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        difference = prediction - reference
        antisymmetric = 0.5 * (prediction - prediction.T)
        symmetric = 0.5 * (prediction + prediction.T)
        rows.append(
            {
                "run": "strict_baseline",
                "molecule_id": str(row["molecule_id"]),
                "natoms": int(row["natoms"]),
                "relative_frobenius": float(
                    np.linalg.norm(difference) / np.linalg.norm(reference)
                ),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference * difference))),
                "asymmetry_ratio": float(
                    np.linalg.norm(antisymmetric)
                    / max(np.linalg.norm(symmetric), np.finfo(float).tiny)
                ),
                "asymmetry_over_pbe": float(
                    np.linalg.norm(antisymmetric)
                    / max(np.linalg.norm(reference), np.finfo(float).tiny)
                ),
                "energy_abs_error_hartree": float(
                    row["baseline_energy_abs_error_hartree"]
                ),
                "force_mae_hartree_per_bohr": float(
                    row["baseline_force_mae_hartree_per_bohr"]
                ),
            }
        )
    return manifest, rows


def _plots(output_dir: Path, rows, trajectories):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_names = list(dict.fromkeys(row["run"] for row in rows))
    molecule_ids = list(dict.fromkeys(row["molecule_id"] for row in rows))
    width = 0.8 / len(run_names)
    x = np.arange(len(molecule_ids))
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for index, run in enumerate(run_names):
        lookup = {
            row["molecule_id"]: row["relative_frobenius"]
            for row in rows
            if row["run"] == run
        }
        values = [lookup.get(molecule, np.nan) for molecule in molecule_ids]
        axis.bar(x + (index - (len(run_names) - 1) / 2) * width, values, width, label=run)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="Stage1 5% gate")
    axis.set_yscale("log")
    axis.set_ylabel("Hessian relative Frobenius")
    axis.set_xticks(x, molecule_ids, rotation=30)
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "per_parent_relative_frobenius.png", dpi=180)
    plt.close(figure)

    if trajectories:
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for name, trajectory in trajectories.items():
            axis.plot(
                [row["step"] for row in trajectory],
                [row["median_relative_frobenius"] for row in trajectory],
                label=f"{name} median",
            )
            axis.plot(
                [row["step"] for row in trajectory],
                [row["max_relative_frobenius"] for row in trajectory],
                linestyle="--",
                label=f"{name} max",
            )
        axis.axhline(0.05, color="black", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Hessian relative Frobenius")
        axis.legend(fontsize=8)
        figure.savefig(output_dir / "capacity_training_trajectory.png", dpi=180)
        plt.close(figure)


def run(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline, rows = _baseline_rows(args.baseline_manifest)
    summaries = {}
    trajectories = {}
    for specification in args.run:
        name, path = specification.split("=", maxsplit=1)
        summary, run_rows, trajectory = _run_rows(name, Path(path))
        summaries[name] = summary
        rows.extend(run_rows)
        if trajectory:
            trajectories[name] = trajectory
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "capacity_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _plots(args.output_dir, rows, trajectories)
    result = {
        "definition": "train-only complete-total capacity ceiling aggregation",
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "source_split_sha256": baseline["source_split_sha256"],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "runs": {
            name: {
                "median_relative_frobenius": float(
                    np.median(
                        [
                            row["relative_frobenius"]
                            for row in rows
                            if row["run"] == name
                        ]
                    )
                ),
                "max_relative_frobenius": float(
                    max(
                        row["relative_frobenius"]
                        for row in rows
                        if row["run"] == name
                    )
                ),
            }
            for name in ["strict_baseline", *summaries]
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
