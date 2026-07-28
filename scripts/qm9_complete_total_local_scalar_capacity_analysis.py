#!/usr/bin/env python3
"""Merge preregistered stable5 local-scalar full-Hessian capacity runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    protocol_sha256 = _sha256(args.protocol)
    expected_parents = [str(value) for value in protocol["inputs"]["parents"]]
    expected_arms = [str(row["id"]) for row in protocol["arms"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arm_rows = []
    parent_rows = []
    curves: dict[str, list[dict[str, Any]]] = {}
    passing = []
    input_artifacts = []
    for arm_id in expected_arms:
        run_dir = args.run_root / arm_id
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "training_metrics.jsonl"
        summary = json.loads(summary_path.read_text())
        if summary.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"protocol hash drift for {arm_id}")
        if summary.get("arm_id") != arm_id or summary.get("run_mode") != "formal":
            raise ValueError(f"run identity mismatch for {arm_id}")
        if summary.get("selected_parent_ids") != expected_parents:
            raise ValueError(f"parent order/coverage mismatch for {arm_id}")
        if int(summary.get("unselected_parent_artifacts_opened", -1)) != 0:
            raise ValueError(f"unselected parent artifacts were opened by {arm_id}")
        if summary.get("validation_accessed") is not False:
            raise ValueError(f"validation access is not false for {arm_id}")
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"Test100 access is not false for {arm_id}")
        if int(summary.get("test100_evaluations_used", -1)) != 0:
            raise ValueError(f"Test100 count is not zero for {arm_id}")
        final = summary["final"]
        gate_passed = bool(final["gate"]["passed"])
        if bool(summary.get("parent_cv_authorized")) != gate_passed:
            raise ValueError(f"authorization/gate mismatch for {arm_id}")
        if gate_passed:
            passing.append(arm_id)
        arm_parent_rows = []
        for row in summary["per_parent"]:
            molecule_id = str(row["molecule_id"])
            with np.load(run_dir / f"{molecule_id}_result.npz") as payload:
                predicted = np.asarray(payload["predicted_hessian"], dtype=np.float64)
                source = np.asarray(
                    payload["source_hessian_symmetric"], dtype=np.float64
                )
                reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
            correction = predicted - source
            target = reference - source
            correction_norm = np.linalg.norm(correction)
            target_norm = max(np.linalg.norm(target), np.finfo(float).tiny)
            cosine = float(
                np.sum(correction * target)
                / max(correction_norm * target_norm, np.finfo(float).tiny)
            )
            alpha = float(
                np.sum(correction * target)
                / max(np.sum(correction * correction), np.finfo(float).tiny)
            )
            oracle_error = source + alpha * correction - reference
            diagnostic_row = {
                "arm_id": arm_id,
                **row,
                "correction_target_cosine": cosine,
                "correction_to_target_frobenius_ratio": float(
                    correction_norm / target_norm
                ),
                "oracle_scalar_amplitude": alpha,
                "oracle_scalar_relative_frobenius": float(
                    np.linalg.norm(oracle_error)
                    / max(np.linalg.norm(reference), np.finfo(float).tiny)
                ),
            }
            arm_parent_rows.append(diagnostic_row)
            parent_rows.append(diagnostic_row)
        arm_rows.append(
            {
                "arm_id": arm_id,
                "architecture": summary["architecture"]["architecture"],
                "trainable_parameter_count": summary["architecture"][
                    "trainable_parameter_count"
                ],
                "best_step": summary["best_step"],
                "median_relative_frobenius": final["hessian_relative_frobenius"][
                    "median"
                ],
                "p90_relative_frobenius": final["hessian_relative_frobenius"]["p90"],
                "max_relative_frobenius": final["hessian_relative_frobenius"]["max"],
                "energy_median_ratio_to_source": final[
                    "energy_median_ratio_to_source"
                ],
                "force_median_ratio_to_source": final[
                    "force_median_ratio_to_source"
                ],
                "max_antisymmetric_over_symmetric_frobenius": final[
                    "max_antisymmetric_over_symmetric_frobenius"
                ],
                "median_correction_target_cosine": float(
                    np.median(
                        [row["correction_target_cosine"] for row in arm_parent_rows]
                    )
                ),
                "median_correction_to_target_frobenius_ratio": float(
                    np.median(
                        [
                            row["correction_to_target_frobenius_ratio"]
                            for row in arm_parent_rows
                        ]
                    )
                ),
                "median_oracle_scalar_relative_frobenius": float(
                    np.median(
                        [
                            row["oracle_scalar_relative_frobenius"]
                            for row in arm_parent_rows
                        ]
                    )
                ),
                "gate_passed": gate_passed,
                "wall_time_s": summary["wall_time_s"],
                "max_rss_mb": summary["max_rss_mb"],
                "gpu_peak_memory_mb": summary["gpu_peak_memory_mb"],
                "summary_sha256": _sha256(summary_path),
            }
        )
        curves[arm_id] = _load_metrics(metrics_path)
        input_artifacts.extend(
            [
                {
                    "arm_id": arm_id,
                    "path": summary_path.resolve().as_posix(),
                    "sha256": _sha256(summary_path),
                },
                {
                    "arm_id": arm_id,
                    "path": metrics_path.resolve().as_posix(),
                    "sha256": _sha256(metrics_path),
                },
                {
                    "arm_id": arm_id,
                    "path": (run_dir / "best.ckpt").resolve().as_posix(),
                    "sha256": _sha256(run_dir / "best.ckpt"),
                },
            ]
        )

    selected = None
    if passing:
        selected = min(
            (row for row in arm_rows if row["arm_id"] in passing),
            key=lambda row: (
                float(row["median_relative_frobenius"]),
                float(row["max_relative_frobenius"]),
                str(row["arm_id"]),
            ),
        )["arm_id"]
    arm_csv = args.output_dir / "arm_summary.csv"
    parent_csv = args.output_dir / "per_parent_metrics.csv"
    _write_csv(arm_csv, arm_rows)
    _write_csv(parent_csv, parent_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm_id, rows in curves.items():
        steps = np.asarray([int(row["step"]) for row in rows])
        axes[0].plot(
            steps,
            [float(row["median_relative_frobenius"]) for row in rows],
            label=arm_id,
        )
        axes[1].plot(
            steps,
            [float(row["max_relative_frobenius"]) for row in rows],
            label=arm_id,
        )
    for axis, title in zip(axes, ("Stable5 median", "Stable5 maximum"), strict=True):
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Training step")
        axis.set_ylabel("Hessian relative Frobenius")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    plot_path = args.output_dir / "local_scalar_capacity.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    result = {
        "definition": "Stable5 fit-only local-scalar full-Hessian capacity decision; no held parent, validation, or Test100 label is read.",
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": protocol_sha256,
        "run_root": args.run_root.resolve().as_posix(),
        "expected_parent_ids": expected_parents,
        "arms": arm_rows,
        "passing_arms": passing,
        "selected_capacity_arm": selected,
        "parent_cv_design_authorized": selected is not None,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "input_artifacts": input_artifacts,
        "arm_summary_csv": arm_csv.resolve().as_posix(),
        "per_parent_csv": parent_csv.resolve().as_posix(),
        "plot": plot_path.resolve().as_posix(),
        "plot_sha256": _sha256(plot_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
