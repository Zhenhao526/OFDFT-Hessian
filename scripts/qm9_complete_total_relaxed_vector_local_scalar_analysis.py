#!/usr/bin/env python3
"""Read-only paired analysis for the same-source vector-HVP diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty vector-HVP stratum")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"role:{row['role']}"].append(row)
        groups[
            f"role:{row['role']}/kind:{row['direction_kind']}"
        ].append(row)
    strata = {}
    for name, selected in sorted(groups.items()):
        candidate = [
            float(row["relative_l2_with_floor"]) for row in selected
        ]
        source = [
            float(row["source_relative_l2_with_floor"]) for row in selected
        ]
        strata[name] = {
            "candidate_relative_l2": _distribution(candidate),
            "source_relative_l2": _distribution(source),
            "candidate_improved_fraction": float(
                np.mean([bool(row["improved"]) for row in selected])
            ),
            "candidate_fraction_at_or_below_0_15": float(
                np.mean(np.asarray(candidate) <= 0.15)
            ),
            "candidate_component_mae": _distribution(
                [float(row["mae_hartree_per_bohr2"]) for row in selected]
            ),
        }
    hard = sorted(
        rows,
        key=lambda row: float(row["relative_l2_with_floor"]),
        reverse=True,
    )[:20]
    held = [row for row in rows if row["role"] == "heldout"]
    held_without_worst = sorted(
        [float(row["relative_l2_with_floor"]) for row in held]
    )[:-1]
    return {
        "strata": strata,
        "hard": hard,
        "heldout_sensitivity": {
            "all": _distribution(
                [float(row["relative_l2_with_floor"]) for row in held]
            ),
            "without_single_largest_posthoc_diagnostic_only": _distribution(
                held_without_worst
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(fit_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    fit = json.loads(fit_summary_path.read_text())
    if (
        fit.get("validation_accessed") is not False
        or fit.get("test100_accessed") is not False
    ):
        raise ValueError("fit opened validation/Test100")
    if fit.get("same_source_energy_force_hvp") is not True:
        raise ValueError("fit is not same-source E/F/HVP")
    if fit.get("formal_stage3_authorized") is not False:
        raise ValueError("diagnostic unexpectedly authorizes Stage 3")
    arm_id = str(fit["selected_arm_id"])
    arm_dir = fit_summary_path.parent / arm_id
    direction_path = arm_dir / "per_direction.json"
    parent_path = arm_dir / "per_parent.json"
    rows = json.loads(direction_path.read_text())
    analysis = summarize_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_direction_paired.csv"
    _write_csv(csv_path, rows)
    summary = {
        "definition": (
            "Read-only same-source vector-HVP paired analysis. The worst-row "
            "removal is sensitivity reporting only and cannot change selection."
        ),
        "fit_summary": fit_summary_path.resolve().as_posix(),
        "fit_summary_sha256": _sha256(fit_summary_path),
        "selected_arm_id": arm_id,
        "selected_checkpoint": fit["selected_checkpoint"],
        "selected_checkpoint_sha256": fit["selected_checkpoint_sha256"],
        "direction_metrics": direction_path.resolve().as_posix(),
        "direction_metrics_sha256": _sha256(direction_path),
        "parent_metrics": parent_path.resolve().as_posix(),
        "parent_metrics_sha256": _sha256(parent_path),
        **analysis,
        "diagnostic_gate": fit["diagnostic_gate"],
        "same_source_energy_force_hvp": True,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "artifacts": {
            "per_direction_csv": csv_path.resolve().as_posix(),
            "per_direction_csv_sha256": _sha256(csv_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    analyze(args.fit_summary, args.output_dir)
