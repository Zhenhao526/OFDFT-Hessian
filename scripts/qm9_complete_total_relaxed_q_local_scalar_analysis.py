#!/usr/bin/env python3
"""Summarize source-to-candidate relaxed-q changes without refitting a model."""

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
        raise ValueError("cannot summarize an empty relaxed-q stratum")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def summarize_rows(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    enriched = []
    for row in rows:
        baseline_q = float(row["baseline_q"])
        pbe_q = float(row["pbe_q"])
        predicted_q = float(row["predicted_q"])
        denominator = max(abs(pbe_q), floor)
        source_error = abs(baseline_q - pbe_q) / denominator
        candidate_error = abs(predicted_q - pbe_q) / denominator
        enriched.append(
            {
                **row,
                "source_relative_error_with_floor": source_error,
                "candidate_relative_error_with_floor": candidate_error,
                "candidate_over_source": candidate_error
                / max(source_error, np.finfo(float).tiny),
                "candidate_improved": candidate_error < source_error,
                "required_correction_abs": abs(pbe_q - baseline_q),
            }
        )

    strata: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[f"role:{row['role']}"] .append(row)
        groups[f"role:{row['role']}/kind:{row['direction_kind']}"] .append(row)
    for name, selected in sorted(groups.items()):
        source = [float(row["source_relative_error_with_floor"]) for row in selected]
        candidate = [
            float(row["candidate_relative_error_with_floor"]) for row in selected
        ]
        strata[name] = {
            "source_relative_error": _distribution(source),
            "candidate_relative_error": _distribution(candidate),
            "candidate_over_source_median": float(
                np.median(
                    [float(row["candidate_over_source"]) for row in selected]
                )
            ),
            "candidate_improved_fraction": float(
                np.mean([bool(row["candidate_improved"]) for row in selected])
            ),
            "fraction_at_or_below_0_15": float(np.mean(np.asarray(candidate) <= 0.15)),
        }
    hard = sorted(
        enriched,
        key=lambda row: float(row["candidate_relative_error_with_floor"]),
        reverse=True,
    )[:20]
    correction = _distribution(
        [float(row["required_correction_abs"]) for row in enriched]
    )
    return {"strata": strata, "required_correction_abs": correction, "hard": hard, "rows": enriched}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(fit_summary_path: Path, output_dir: Path, floor: float) -> dict[str, Any]:
    fit_summary = json.loads(fit_summary_path.read_text())
    if fit_summary.get("test100_accessed") is not False:
        raise ValueError("fit summary does not freeze Test100")
    if fit_summary.get("validation_accessed") is not False:
        raise ValueError("fit summary opened validation parents")
    if fit_summary.get("formal_stage3_authorized") is not False:
        raise ValueError("diagnostic fit unexpectedly authorizes Stage 3")
    arm_id = str(fit_summary["selected_arm_id"])
    arm_dir = fit_summary_path.parent / arm_id
    direction_path = arm_dir / "per_direction.json"
    parent_path = arm_dir / "per_parent.json"
    rows = json.loads(direction_path.read_text())
    analysis = summarize_rows(rows, floor)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "per_direction_source_candidate.csv"
    _write_csv(enriched_path, analysis.pop("rows"))
    summary = {
        "definition": (
            "Read-only paired source/candidate relaxed scalar-q analysis; no model "
            "selection, validation-parent access, or Test100 access"
        ),
        "fit_summary": fit_summary_path.resolve().as_posix(),
        "fit_summary_sha256": _sha256(fit_summary_path),
        "selected_arm_id": arm_id,
        "selected_direction_metrics": direction_path.resolve().as_posix(),
        "selected_direction_metrics_sha256": _sha256(direction_path),
        "selected_parent_metrics": parent_path.resolve().as_posix(),
        "selected_parent_metrics_sha256": _sha256(parent_path),
        "q_reference_floor_hartree_per_bohr2": floor,
        **analysis,
        "diagnostic_gate": fit_summary["diagnostic_gate"],
        "train800_full_replay_included": False,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "artifacts": {
            "per_direction_csv": enriched_path.resolve().as_posix(),
            "per_direction_csv_sha256": _sha256(enriched_path),
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
    parser.add_argument("--q-reference-floor", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    analyze(arguments.fit_summary, arguments.output_dir, arguments.q_reference_floor)
