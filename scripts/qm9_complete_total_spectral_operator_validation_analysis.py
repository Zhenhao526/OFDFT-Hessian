#!/usr/bin/env python3
"""Compare preregistered shared-operator arms on unseen validation parents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_run(value: str) -> tuple[str, Path]:
    name, path = value.split("=", maxsplit=1)
    return name, Path(path).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    expected_parents: list[str] | None = None
    expected_protocol_sha256: str | None = None
    for name, run_dir in map(_parse_run, args.run):
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"{name} does not certify frozen Test100")
        if int(summary.get("test100_evaluations_used", 0)) != 0:
            raise ValueError(f"{name} records Test100 access")
        parents = [str(row["molecule_id"]) for row in summary["per_parent"]]
        if expected_parents is None:
            expected_parents = parents
            expected_protocol_sha256 = summary["protocol_sha256"]
        elif parents != expected_parents:
            raise ValueError("run parent IDs or order differ")
        elif summary["protocol_sha256"] != expected_protocol_sha256:
            raise ValueError("run protocol hashes differ")
        runs.append((name, run_dir, summary_path, summary))
    if len(runs) < 2:
        raise ValueError("at least two preregistered runs are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    run_rows = []
    for name, run_dir, summary_path, summary in runs:
        for row in summary["per_parent"]:
            all_rows.append({"run": name, **row})
        distribution = summary["hessian_relative_frobenius"]
        run_rows.append(
            {
                "run": name,
                "parent_count": summary["parent_count"],
                "hessian_median_relative_frobenius": distribution["median"],
                "hessian_p90_relative_frobenius": distribution["p90"],
                "hessian_max_relative_frobenius": distribution["max"],
                "fraction_hessian_le_0p15": summary[
                    "fraction_relative_frobenius_at_or_below_0_15"
                ],
                "parent_win_fraction_vs_source": summary[
                    "parent_win_fraction_vs_source"
                ],
                "energy_median_abs_error_hartree": summary[
                    "energy_abs_error_hartree"
                ]["median"],
                "energy_p90_abs_error_hartree": summary[
                    "energy_abs_error_hartree"
                ]["p90"],
                "force_median_mae_hartree_per_bohr": summary[
                    "force_mae_hartree_per_bohr"
                ]["median"],
                "force_p90_mae_hartree_per_bohr": summary[
                    "force_mae_hartree_per_bohr"
                ]["p90"],
                "all_registered_gates_passed": summary[
                    "stage3_validation_gate"
                ]["passed"],
                "summary": summary_path.as_posix(),
                "summary_sha256": _sha256(summary_path),
                "run_dir": run_dir.as_posix(),
            }
        )
    _write_csv(args.output_dir / "per_parent_all_arms.csv", all_rows)
    _write_csv(args.output_dir / "run_summary.csv", run_rows)

    parents = expected_parents or []
    x = np.arange(len(parents), dtype=np.float64)
    width = 0.8 / len(runs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for run_index, (name, _, _, summary) in enumerate(runs):
        rows = summary["per_parent"]
        offset = (run_index - (len(runs) - 1) / 2) * width
        axes[0, 0].bar(
            x + offset,
            [float(row["relative_frobenius"]) for row in rows],
            width=width,
            label=name,
        )
        axes[0, 1].plot(
            np.linspace(1 / len(rows), 1.0, len(rows)),
            np.sort([float(row["relative_frobenius"]) for row in rows]),
            marker="o",
            label=name,
        )
        axes[1, 0].plot(
            x,
            [
                float(row["source_energy_abs_error_hartree"])
                / max(
                    float(row["baseline_energy_abs_error_hartree"]),
                    np.finfo(float).tiny,
                )
                for row in rows
            ],
            marker="o",
            label=name,
        )
        axes[1, 1].plot(
            x,
            [
                float(row["source_force_mae_hartree_per_bohr"])
                / max(
                    float(row["baseline_force_mae_hartree_per_bohr"]),
                    np.finfo(float).tiny,
                )
                for row in rows
            ],
            marker="o",
            label=name,
        )
    axes[0, 0].axhline(0.15, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Hessian relative Frobenius")
    axes[0, 0].set_xticks(x, parents, rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.20, color="gray", linestyle=":", linewidth=1)
    axes[0, 1].set_xlabel("Empirical CDF")
    axes[0, 1].set_ylabel("Hessian relative Frobenius")
    axes[0, 1].legend()
    for axis, label in ((axes[1, 0], "Energy error ratio"), (axes[1, 1], "Force MAE ratio")):
        axis.axhline(1.05, color="black", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_ylabel(f"{label}: source / original-A")
        axis.set_xticks(x, parents, rotation=45, ha="right")
        axis.legend()
    plot_path = args.output_dir / "unseen_parent_shared_operator_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    result = {
        "definition": (
            "Read-only paired comparison of preregistered shared spectral/block "
            "operator arms on identical unseen validation parents."
        ),
        "parent_ids": parents,
        "runs": run_rows,
        "protocol_sha256": expected_protocol_sha256,
        "plot": plot_path.resolve().as_posix(),
        "plot_sha256": _sha256(plot_path),
        "per_parent_csv": (args.output_dir / "per_parent_all_arms.csv").resolve().as_posix(),
        "run_summary_csv": (args.output_dir / "run_summary.csv").resolve().as_posix(),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
