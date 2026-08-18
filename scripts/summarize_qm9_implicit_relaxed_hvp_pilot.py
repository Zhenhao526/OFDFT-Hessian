#!/usr/bin/env python3
"""Create a fail-closed, machine-readable summary of one implicit-HVP pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={row[key]}")
    return value


def _comparison(initial: dict[str, str], final: dict[str, str], key: str) -> dict[str, float]:
    before = _float(initial, key)
    after = _float(final, key)
    return {
        "initial": before,
        "final": after,
        "absolute_change": after - before,
        "relative_change": (after - before) / before if before != 0.0 else math.nan,
    }


def summarize(
    run_dir: Path,
    *,
    protocol: Path,
    prefix_training_curve: Path | None = None,
    prefix_density_refresh_points: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "training_curve.csv"
    hessian_path = run_dir / "full_hessian_metrics.csv"
    density_path = run_dir / "density_refresh_points.csv"
    for path in (summary_path, curve_path, hessian_path, density_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = json.loads(summary_path.read_text())
    protocol_payload = yaml.safe_load(protocol.read_text())
    if protocol_payload["protocol_id"] != source["protocol_id"]:
        raise ValueError("protocol ID does not match the training summary")
    protocol_sha256 = _sha256(protocol)
    if source.get("validation_accessed") is not False:
        raise ValueError("run does not certify frozen validation")
    if source.get("test100_accessed") is not False:
        raise ValueError("run does not certify frozen Test100")
    if int(source.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("run records Test100 evaluations")

    curve = (
        _rows(prefix_training_curve) if prefix_training_curve is not None else []
    ) + _rows(curve_path)
    hessian = _rows(hessian_path)
    density = (
        _rows(prefix_density_refresh_points)
        if prefix_density_refresh_points is not None
        else []
    ) + _rows(density_path)
    if not curve:
        raise ValueError("training curve is empty")
    if len(hessian) == 2:
        hessian.sort(key=lambda row: int(row["step"]))
        initial, final = hessian
    elif len(hessian) == 1 and source.get("root_initial_full_hessian_metrics"):
        initial = source["root_initial_full_hessian_metrics"][0]
        final = hessian[0]
    else:
        raise ValueError(
            "expected initial/final Hessian rows or one final row plus frozen root metrics"
        )
    if int(initial["step"]) != 0 or int(final["step"]) != int(source["final_step"]):
        raise ValueError("Hessian rows do not bind step 0 to the final step")

    update_counts = Counter(row["update_kind"] for row in curve)
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.ckpt"))
    if not checkpoints:
        raise ValueError("no checkpoints found")
    final_checkpoint = run_dir / "checkpoints" / f"step_{int(source['final_step']):07d}.ckpt"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(final_checkpoint)

    density_gate = float(source["training_density_stationarity_threshold"])
    solver_target = float(source["density_solver_target"])
    max_graph_density = max(
        _float(row, "max_cached_density_gradient_norm") for row in curve
    )
    max_refresh_density = max(_float(row, "final_gradient_norm") for row in density)
    max_response_residual = max(
        _float(row, "analytic_response_max_residual") for row in curve
    )
    max_correction_fraction = max(
        _float(row, "analytic_response_correction_fraction_max") for row in curve
    )
    max_cancellation = max(
        _float(row, "analytic_cancellation_index_max") for row in curve
    )
    all_finite = all(
        math.isfinite(float(value))
        for row in curve
        for value in row.values()
        if value not in ("", "True", "False", "hvp", "replay")
        and ":" not in value
        and ";" not in value
    )

    return {
        "definition": (
            "Train-only audit of alternating E/G/F replay and model-self-consistent "
            "implicit relaxed-HVP updates on QM9 molecule 0016298."
        ),
        "run_directory": run_dir.as_posix(),
        "protocol_id": source["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "final_checkpoint": final_checkpoint.as_posix(),
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "steps": len(curve),
        "update_counts": dict(sorted(update_counts.items())),
        "hessian": {
            key: _comparison(initial, final, key)
            for key in (
                "relative_frobenius",
                "symmetric_relative_frobenius",
                "mae",
                "rmse",
                "train_hvp_relative_median",
                "train_hvp_relative_p90",
                "train_hvp_relative_max",
                "total_energy_abs_error_hartree",
                "complete_total_force_mae_hartree_per_bohr",
            )
        },
        "density": {
            "solver_target": solver_target,
            "training_graph_gate": density_gate,
            "maximum_training_graph_gradient_norm": max_graph_density,
            "maximum_refresh_gradient_norm": max_refresh_density,
            "maximum_refresh_cycles": max(int(row["cycles"]) for row in density),
            "passed": max_graph_density < density_gate
            and max_refresh_density < solver_target,
        },
        "implicit_response": {
            "maximum_residual": max_response_residual,
            "maximum_correction_fraction": max_correction_fraction,
            "maximum_cancellation_index": max_cancellation,
            "passed": max_response_residual <= 1.0e-8
            and max_correction_fraction <= 5.0
            and max_cancellation <= 10.0,
        },
        "all_training_values_finite": all_finite,
        "stage1_gate_passed": bool(source["stage1_gate_passed"]),
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "source_summary_sha256": _sha256(summary_path),
        "training_curve_sha256": _sha256(curve_path),
        "prefix_training_curve_sha256": (
            _sha256(prefix_training_curve)
            if prefix_training_curve is not None
            else None
        ),
        "full_hessian_metrics_sha256": _sha256(hessian_path),
        "density_refresh_points_sha256": _sha256(density_path),
        "prefix_density_refresh_points_sha256": (
            _sha256(prefix_density_refresh_points)
            if prefix_density_refresh_points is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-training-curve", type=Path, default=None)
    parser.add_argument("--prefix-density-refresh-points", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(
        args.run_dir,
        protocol=args.protocol,
        prefix_training_curve=args.prefix_training_curve,
        prefix_density_refresh_points=args.prefix_density_refresh_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
