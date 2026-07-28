"""Merge representative validation density-optimization task reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _stats(rows: list[dict], key: str) -> dict:
    values = np.asarray(
        [row[key] for row in rows if row.get(key) is not None and math.isfinite(row[key])],
        dtype=np.float64,
    )
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "median": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=6)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    errors = []
    for index in range(args.num_tasks):
        path = args.run_root / "rows" / f"task_{index}.json"
        if not path.is_file():
            errors.append(f"missing task report: {path}")
            continue
        rows.append(json.loads(path.read_text()))
    groups = {}
    for dataset in sorted({row["dataset_name"] for row in rows}):
        subset = [row for row in rows if row["dataset_name"] == dataset]
        valid = [row for row in subset if row["status"] == "passed"]
        groups[dataset] = {
            "tasks": len(subset),
            "runtime_success": len(valid),
            "strict_converged": sum(bool(row.get("converged")) for row in valid),
            "strict_convergence_rate": (
                sum(bool(row.get("converged")) for row in valid) / len(valid) if valid else None
            ),
            "failure_reasons": [row.get("error") for row in subset if row.get("error")],
            "metrics": {
                key: _stats(valid, key)
                for key in (
                    "final_projected_gradient_norm",
                    "optimized_density_l2",
                    "optimized_coefficient_mae",
                    "optimized_coefficient_rmse",
                    "optimized_electron_error",
                    "label_density_model_energy_error",
                    "optimized_model_energy_error",
                    "cycles",
                    "elapsed_s",
                )
            },
        }
    runtime_failures = [row for row in rows if row["status"] != "passed"]
    errors.extend(
        f"runtime failure {row['dataset_name']} index {row['selection_index']}: {row.get('error')}"
        for row in runtime_failures
    )
    report = {
        "definition": (
            "Validation-only representative model-driven density optimization. Density error is "
            "against the archived final ground-state coefficients; label-density model error and "
            "optimized-density model error are reported separately. This is not a force/Hessian result."
        ),
        "groups": groups,
        "rows": rows,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
