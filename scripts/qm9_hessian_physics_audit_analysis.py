#!/usr/bin/env python3
"""Aggregate force-loop and relaxed scalar-energy Hessian physics audits."""

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
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _stats(values: list[Any], prefix: str) -> dict[str, Any]:
    finite = _finite(values)
    if not finite:
        return {f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_max": None}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    pairs = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and math.isfinite(float(row[x_key]))
        and math.isfinite(float(row[y_key]))
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson_r": None, "spearman_r": None}
    x, y = np.asarray(pairs).T
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(x, y).statistic)
    except ImportError:
        spearman = None
    return {"n": len(pairs), "pearson_r": pearson, "spearman_r": spearman}


def _array_difference(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference_norm = float(np.linalg.norm(reference))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_fro": (
            float(np.linalg.norm(diff) / reference_norm) if reference_norm else None
        ),
    }


def _loop_tolerance_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["run"],
                row["method"],
                row["molecule_id"],
                float(row["half_width"]),
            )
        ].append(row)
    output = []
    for (run, method, molecule_id, half_width), group in sorted(grouped.items()):
        strictest = min(group, key=lambda row: float(row["target_tolerance"]))
        for row in sorted(group, key=lambda item: float(item["target_tolerance"])):
            reference_curl = float(strictest["curl_estimate_hartree_per_bohr2"])
            reference_work = float(strictest["loop_work_hartree"])
            output.append(
                {
                    "run": run,
                    "method": method,
                    "molecule_id": molecule_id,
                    "half_width": half_width,
                    "tolerance": float(row["target_tolerance"]),
                    "reference_tolerance": float(strictest["target_tolerance"]),
                    "curl_estimate": float(row["curl_estimate_hartree_per_bohr2"]),
                    "reference_curl_estimate": reference_curl,
                    "abs_curl_difference": abs(
                        float(row["curl_estimate_hartree_per_bohr2"]) - reference_curl
                    ),
                    "relative_curl_difference": (
                        abs(float(row["curl_estimate_hartree_per_bohr2"]) - reference_curl)
                        / abs(reference_curl)
                        if reference_curl
                        else None
                    ),
                    "abs_loop_work_difference": abs(
                        float(row["loop_work_hartree"]) - reference_work
                    ),
                    "strict_all_corners": bool(row["strict_all_corners"]),
                    "max_final_gradient_norm": row.get("max_final_gradient_norm"),
                }
            )
    return output


def _energy_block_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Every comparison row for one condition points to the same NPZ. Deduplicate first.
    cases: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in rows:
        key = (row["run"], row["molecule_id"], float(row["displacement"]))
        cases.setdefault(key, row)
    grouped: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for (run, molecule_id, displacement), row in cases.items():
        grouped[(run, molecule_id)].append((displacement, row))

    output = []
    array_keys = {
        "total_energy": "total_energy_hessian_block",
        "relaxed_model_energy": "relaxed_model_energy_hessian_block",
    }
    for (run, molecule_id), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        group = sorted(group)
        reference_displacement, reference_row = group[0]
        with np.load(reference_row["block_npz"]) as payload:
            reference_arrays = {
                name: np.asarray(payload[key], dtype=np.float64)
                for name, key in array_keys.items()
            }
        for displacement, row in group:
            with np.load(row["block_npz"]) as payload:
                for name, key in array_keys.items():
                    output.append(
                        {
                            "run": run,
                            "molecule_id": molecule_id,
                            "quantity": name,
                            "displacement": displacement,
                            "reference_displacement": reference_displacement,
                            "strict_all_points": bool(row["strict_all_points"]),
                            "reference_strict_all_points": bool(
                                reference_row["strict_all_points"]
                            ),
                            **_array_difference(
                                np.asarray(payload[key], dtype=np.float64),
                                reference_arrays[name],
                            ),
                        }
                    )
    return output


def analyze(root: Path, output_dir: Path) -> dict[str, Any]:
    loop_rows: list[dict[str, Any]] = []
    loop_point_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    energy_point_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("loop_*/summary.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            baseline_asymmetry = float(row["baseline_max_symmetry_abs"])
            curl = float(row["curl_estimate_hartree_per_bohr2"])
            loop_rows.append(
                {
                    "condition": path.parent.name,
                    "target_tolerance": float(payload["target_tolerance"]),
                    "abs_curl_estimate": abs(curl),
                    "strict_all_corners": (
                        row["method"] == "fixed_density"
                        or row.get("strict_converged_corners") == 4
                    ),
                    "curl_over_baseline_max_asymmetry": (
                        abs(curl) / abs(baseline_asymmetry) if baseline_asymmetry else None
                    ),
                    **row,
                }
            )
        for row in payload.get("optimization_points", []):
            loop_point_rows.append({"condition": path.parent.name, **row})
    for path in sorted(root.glob("energy_block_*/summary.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            energy_rows.append(
                {
                    "condition": path.parent.name,
                    "strict_all_points": row.get("strict_converged_points") == 9,
                    **row,
                }
            )
        for row in payload.get("optimization_points", []):
            energy_point_rows.append({"condition": path.parent.name, **row})

    grouped: dict[tuple[str, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in loop_rows:
        grouped[
            (
                row["run"],
                row["method"],
                float(row["half_width"]),
                float(row["target_tolerance"]),
            )
        ].append(row)
    loop_summary = []
    for (run, method, half_width, tolerance), rows in sorted(grouped.items()):
        strict_rows = [row for row in rows if row["strict_all_corners"]]
        loop_summary.append(
            {
                "run": run,
                "method": method,
                "half_width": half_width,
                "tolerance": tolerance,
                "cases": len(rows),
                "strict_cases": len(strict_rows),
                **_stats([abs(row["loop_work_hartree"]) for row in rows], "abs_loop_work"),
                **_stats([abs(row["curl_estimate_hartree_per_bohr2"]) for row in rows], "abs_curl"),
                **_stats(
                    [abs(row["loop_work_hartree"]) for row in strict_rows],
                    "strict_abs_loop_work",
                ),
                **_stats(
                    [abs(row["curl_estimate_hartree_per_bohr2"]) for row in strict_rows],
                    "strict_abs_curl",
                ),
                **_stats([row.get("max_final_gradient_norm") for row in rows], "max_final_gradient_norm"),
            }
        )

    energy_summary = []
    energy_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in energy_rows:
        energy_grouped[(row["run"], row["comparison"], float(row["displacement"]))].append(row)
    for (run, comparison, displacement), rows in sorted(energy_grouped.items()):
        strict_rows = [row for row in rows if row["strict_all_points"]]
        energy_summary.append(
            {
                "run": run,
                "comparison": comparison,
                "displacement": displacement,
                "cases": len(rows),
                "strict_cases": len(strict_rows),
                **_stats([row.get("mae") for row in rows], "mae"),
                **_stats([row.get("rmse") for row in rows], "rmse"),
                **_stats([row.get("relative_fro") for row in rows], "relative_fro"),
                **_stats([row.get("mae") for row in strict_rows], "strict_mae"),
                **_stats([row.get("rmse") for row in strict_rows], "strict_rmse"),
                **_stats(
                    [row.get("relative_fro") for row in strict_rows],
                    "strict_relative_fro",
                ),
                **_stats([row.get("max_final_gradient_norm") for row in rows], "max_final_gradient_norm"),
            }
        )

    relaxed_rows = [row for row in loop_rows if row["method"] == "relaxed_density"]
    correlations = []
    for run in sorted({row["run"] for row in relaxed_rows}):
        selected = [row for row in relaxed_rows if row["run"] == run]
        for target in ("abs_loop_work_hartree", "abs_curl_estimate"):
            correlations.append(
                {
                    "run": run,
                    "predictor": "max_final_gradient_norm",
                    "target": target,
                    **_correlation(selected, "max_final_gradient_norm", target),
                }
            )

    loop_stability = _loop_tolerance_stability(loop_rows)
    energy_stability = _energy_block_stability(energy_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "loop_per_case.csv", loop_rows)
    _write_csv(output_dir / "loop_optimization_points.csv", loop_point_rows)
    _write_csv(output_dir / "loop_condition_summary.csv", loop_summary)
    _write_csv(output_dir / "energy_block_per_case.csv", energy_rows)
    _write_csv(output_dir / "energy_optimization_points.csv", energy_point_rows)
    _write_csv(output_dir / "energy_block_summary.csv", energy_summary)
    _write_csv(output_dir / "loop_residual_correlations.csv", correlations)
    _write_csv(output_dir / "loop_tolerance_stability.csv", loop_stability)
    _write_csv(output_dir / "energy_block_stability.csv", energy_stability)

    plots: list[str] = []
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
        colors = {"EG": "#c44e52", "EGF_lam1": "#4c72b0"}
        markers = {"fixed_density": "s", "relaxed_density": "o"}
        for run in sorted({row["run"] for row in loop_rows}):
            for method in ("fixed_density", "relaxed_density"):
                selected = [
                    row for row in loop_rows if row["run"] == run and row["method"] == method
                ]
                axes[0].scatter(
                    [row["half_width"] for row in selected],
                    [abs(row["curl_estimate_hartree_per_bohr2"]) for row in selected],
                    alpha=0.65,
                    s=28,
                    marker=markers[method],
                    color=colors.get(run),
                    label=f"{run} {method}",
                )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("loop half-width (Bohr)")
        axes[0].set_ylabel("absolute loop curl estimate")
        axes[0].legend(fontsize=7)
        axes[1].scatter(
            [row["max_final_gradient_norm"] for row in relaxed_rows],
            [abs(row["loop_work_hartree"]) for row in relaxed_rows],
            c=[colors.get(row["run"], "black") for row in relaxed_rows],
            s=28,
            alpha=0.7,
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("max final projected density-gradient norm")
        axes[1].set_ylabel("absolute relaxed loop work (Hartree)")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        plot_path = output_dir / "loop_work_diagnostics.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plots.append(str(plot_path))
    except ImportError:
        pass

    result = {
        "root": str(root.resolve()),
        "loop_cases": len(loop_rows),
        "loop_optimization_points": len(loop_point_rows),
        "energy_block_comparisons": len(energy_rows),
        "energy_optimization_points": len(energy_point_rows),
        "loop_condition_summary": loop_summary,
        "energy_block_summary": energy_summary,
        "loop_residual_correlations": correlations,
        "loop_tolerance_stability_rows": len(loop_stability),
        "energy_block_stability_rows": len(energy_stability),
        "plots": plots,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output_dir)


if __name__ == "__main__":
    main()
