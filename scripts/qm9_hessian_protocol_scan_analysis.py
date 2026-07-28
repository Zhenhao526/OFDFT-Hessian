#!/usr/bin/env python3
"""Aggregate density-relaxed Hessian protocol scans and diagnose asymmetry sources."""

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


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _maximum(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.max(finite)) if finite else None


def _metrics(hessian: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    hessian = np.asarray(hessian, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    h_sym = 0.5 * (hessian + hessian.T)
    h_asym = 0.5 * (hessian - hessian.T)
    ref_sym = 0.5 * (reference + reference.T)
    raw_diff = hessian - reference
    sym_diff = h_sym - ref_sym
    raw_abs = np.abs(raw_diff).reshape(-1)
    sym_abs = np.abs(sym_diff).reshape(-1)
    asym_abs = np.abs(hessian - hessian.T).reshape(-1)
    ref_norm = float(np.linalg.norm(reference))
    ref_sym_norm = float(np.linalg.norm(ref_sym))
    sym_norm = float(np.linalg.norm(h_sym))
    result = {
        "finite": bool(np.isfinite(hessian).all()),
        "raw_mae": float(np.mean(raw_abs)),
        "raw_rmse": float(np.sqrt(np.mean(raw_diff * raw_diff))),
        "raw_relative_fro": float(np.linalg.norm(raw_diff) / ref_norm) if ref_norm else None,
        "sym_mae": float(np.mean(sym_abs)),
        "sym_rmse": float(np.sqrt(np.mean(sym_diff * sym_diff))),
        "sym_relative_fro": (
            float(np.linalg.norm(sym_diff) / ref_sym_norm) if ref_sym_norm else None
        ),
        "symmetric_fro_norm": sym_norm,
        "antisymmetric_fro_norm": float(np.linalg.norm(h_asym)),
        "antisymmetric_over_symmetric_fro": (
            float(np.linalg.norm(h_asym) / sym_norm) if sym_norm else None
        ),
        "symmetry_max_abs": float(np.max(asym_abs)),
        "antisymmetric_max_abs": float(np.max(np.abs(h_asym))),
    }
    for name, values in (
        ("raw_abs_error", raw_abs),
        ("sym_abs_error", sym_abs),
        ("symmetry_abs", asym_abs),
    ):
        for label, quantile in (("q50", 0.5), ("q90", 0.9), ("q95", 0.95), ("q99", 0.99)):
            result[f"{name}_{label}"] = float(np.quantile(values, quantile))
        result[f"{name}_q100"] = float(np.max(values))
    return result


def _load_hessians(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    model_key = next(
        key
        for key in ("density_relaxed_hessian", "model_hessian", "autograd_hessian")
        if key in payload
    )
    ref_key = next(key for key in ("pbe_hessian", "reference_hessian") if key in payload)
    return np.asarray(payload[model_key]), np.asarray(payload[ref_key])


def _classify_optimization_trace(path_value: Any, tolerance: float) -> dict[str, Any]:
    if not path_value:
        return {"trace_status": "not_recorded", "trace_points": 0}
    path = Path(str(path_value))
    if not path.exists():
        return {"trace_status": "missing", "trace_points": 0}
    with np.load(path) as payload:
        gradients = np.asarray(payload["projected_gradient_norm"], dtype=np.float64)
        energies = np.asarray(payload["total_energy"], dtype=np.float64)
    result: dict[str, Any] = {
        "trace_status": "unknown",
        "trace_points": int(gradients.size),
        "trace_initial_gradient_norm": float(gradients[0]) if gradients.size else None,
        "trace_final_gradient_norm": float(gradients[-1]) if gradients.size else None,
        "trace_min_gradient_norm": float(np.min(gradients)) if gradients.size else None,
        "trace_energy_span": (
            float(np.max(energies) - np.min(energies)) if energies.size else None
        ),
    }
    if not gradients.size:
        result["trace_status"] = "empty"
        return result
    if not np.isfinite(gradients).all() or not np.isfinite(energies).all():
        result["trace_status"] = "nonfinite"
        return result
    if gradients[-1] < tolerance:
        result["trace_status"] = "converged"
        return result

    positive = np.maximum(gradients, np.finfo(np.float64).tiny)
    log_grad = np.log10(positive)
    tail_count = min(len(log_grad), max(40, len(log_grad) // 5))
    tail = log_grad[-tail_count:]
    delta = np.diff(tail)
    nonzero = delta[np.abs(delta) > 1e-8]
    reversals = (
        float(np.mean(nonzero[1:] * nonzero[:-1] < 0)) if nonzero.size > 1 else 0.0
    )
    tail_span = float(np.quantile(tail, 0.95) - np.quantile(tail, 0.05))
    tail_slope = float(np.polyfit(np.arange(tail_count), tail, 1)[0]) if tail_count > 1 else 0.0
    result.update(
        {
            "trace_tail_points": tail_count,
            "trace_tail_log10_span": tail_span,
            "trace_tail_log10_slope_per_cycle": tail_slope,
            "trace_tail_reversal_fraction": reversals,
        }
    )
    if gradients[-1] > max(10.0 * gradients[0], 1.0):
        result["trace_status"] = "diverging"
    elif abs(tail_slope) < 2e-4 and tail_span < 0.35:
        result["trace_status"] = "plateau"
    elif reversals > 0.55 and tail_span >= 0.20:
        result["trace_status"] = "oscillatory"
    elif tail_slope < -2e-4:
        result["trace_status"] = "slow_improvement"
    else:
        result["trace_status"] = "stalled_mixed"
    return result


def _condition_from_result(result: dict[str, Any], source: str) -> tuple[float, float]:
    displacement = float(result.get("displacement", result.get("model_displacement", 1e-3)))
    fallback = result.get("fallback_optimizer") or {}
    tolerance = float(
        fallback.get(
            "convergence_tolerance",
            result.get("optimizer", {}).get("convergence_tolerance", 1e-4),
        )
    )
    if "20260714_150546" in source:
        displacement, tolerance = 1e-3, 1e-4
    return displacement, tolerance


def _collect_result_paths(args: argparse.Namespace) -> list[Path]:
    paths = [args.baseline_summary]
    paths.extend(sorted(args.scan_root.glob("conditions/*/summary.json")))
    for root in args.additional_scan_root:
        paths.extend(sorted(root.glob("conditions/*/summary.json")))
    return [path.resolve() for path in paths if path.exists()]


def _correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None
    outputs: list[dict[str, Any]] = []
    predictors = [
        "mean_final_gradient_norm",
        "max_final_gradient_norm",
        "mean_cycles",
        "max_cycles",
        "plus_minus_gradient_mismatch_mean",
    ]
    targets = ["antisymmetric_over_symmetric_fro", "symmetry_max_abs", "raw_mae"]
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["displacement"], row["tolerance"])].append(row)
    groups.update({(run, -1.0, -1.0): [row for row in rows if row["run"] == run] for run in {row["run"] for row in rows}})
    for (run, displacement, tolerance), group in groups.items():
        for predictor in predictors:
            for target in targets:
                pairs = [
                    (float(row[predictor]), float(row[target]))
                    for row in group
                    if row.get(predictor) is not None
                    and row.get(target) is not None
                    and math.isfinite(float(row[predictor]))
                    and math.isfinite(float(row[target]))
                ]
                if len(pairs) < 3:
                    continue
                x, y = np.asarray(pairs).T
                pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
                spearman = float(spearmanr(x, y).statistic) if spearmanr is not None else None
                outputs.append(
                    {
                        "run": run,
                        "displacement": None if displacement < 0 else displacement,
                        "tolerance": None if tolerance < 0 else tolerance,
                        "scope": "all_conditions" if displacement < 0 else "condition",
                        "predictor": predictor,
                        "target": target,
                        "n": len(pairs),
                        "pearson_r": pearson,
                        "spearman_r": spearman,
                    }
                )
    return outputs


def _stability_rows(
    rows: list[dict[str, Any]], hessians: dict[tuple[str, str, float, float], np.ndarray]
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    keys = {(row["run"], row["molecule_id"]) for row in rows}
    for run, molecule_id in sorted(keys):
        conditions = sorted(
            (h, tol)
            for model, mol, h, tol in hessians
            if model == run and mol == molecule_id
        )
        for tolerance in sorted({tol for _, tol in conditions}):
            reference_key = (run, molecule_id, 1e-3, tolerance)
            if reference_key not in hessians:
                continue
            ref = hessians[reference_key]
            ref_sym = 0.5 * (ref + ref.T)
            for displacement in sorted(h for h, tol in conditions if tol == tolerance):
                current = hessians[(run, molecule_id, displacement, tolerance)]
                current_sym = 0.5 * (current + current.T)
                diff = current - ref
                sym_diff = current_sym - ref_sym
                outputs.append(
                    {
                        "axis": "displacement",
                        "run": run,
                        "molecule_id": molecule_id,
                        "fixed_value": tolerance,
                        "value": displacement,
                        "reference_value": 1e-3,
                        "mae_vs_reference": float(np.mean(np.abs(diff))),
                        "rmse_vs_reference": float(np.sqrt(np.mean(diff * diff))),
                        "relative_fro_vs_reference": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
                        "sym_mae_vs_reference": float(np.mean(np.abs(sym_diff))),
                        "sym_relative_fro_vs_reference": (
                            float(np.linalg.norm(sym_diff) / np.linalg.norm(ref_sym))
                            if np.linalg.norm(ref_sym)
                            else None
                        ),
                    }
                )
        for displacement in sorted({h for h, _ in conditions}):
            available = sorted(tol for h, tol in conditions if h == displacement)
            reference_tolerance = min(available)
            reference = hessians[(run, molecule_id, displacement, reference_tolerance)]
            reference_sym = 0.5 * (reference + reference.T)
            for tolerance in available:
                current = hessians[(run, molecule_id, displacement, tolerance)]
                current_sym = 0.5 * (current + current.T)
                diff = current - reference
                sym_diff = current_sym - reference_sym
                outputs.append(
                    {
                        "axis": "tolerance",
                        "run": run,
                        "molecule_id": molecule_id,
                        "fixed_value": displacement,
                        "value": tolerance,
                        "reference_value": reference_tolerance,
                        "mae_vs_reference": float(np.mean(np.abs(diff))),
                        "rmse_vs_reference": float(np.sqrt(np.mean(diff * diff))),
                        "relative_fro_vs_reference": (
                            float(np.linalg.norm(diff) / np.linalg.norm(reference))
                            if np.linalg.norm(reference)
                            else None
                        ),
                        "sym_mae_vs_reference": float(np.mean(np.abs(sym_diff))),
                        "sym_relative_fro_vs_reference": (
                            float(np.linalg.norm(sym_diff) / np.linalg.norm(reference_sym))
                            if np.linalg.norm(reference_sym)
                            else None
                        ),
                    }
                )
    return outputs


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[str] = []
    colors = {"EG": "#c44e52", "EGF_lam1": "#4c72b0"}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for run in sorted({row["run"] for row in rows}):
        selected = [row for row in rows if row["run"] == run]
        x = [row["max_final_gradient_norm"] for row in selected]
        y = [row["antisymmetric_over_symmetric_fro"] for row in selected]
        ax.scatter(x, y, s=24, alpha=0.65, label=run, color=colors.get(run))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max final projected density-gradient norm")
    ax.set_ylabel(r"$\|H_{asym}\|_F / \|H_{sym}\|_F$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "asymmetry_vs_density_residual.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.as_posix())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    for run in sorted({row["run"] for row in rows}):
        for tolerance in sorted({row["tolerance"] for row in rows}):
            points = []
            for displacement in sorted({row["displacement"] for row in rows}):
                group = [
                    row
                    for row in rows
                    if row["run"] == run
                    and row["tolerance"] == tolerance
                    and row["displacement"] == displacement
                ]
                if group:
                    points.append(
                        (
                            displacement,
                            _mean([row["raw_mae"] for row in group]),
                            _mean([row["antisymmetric_over_symmetric_fro"] for row in group]),
                        )
                    )
            if points:
                x, raw_mae, asymmetry = zip(*points)
                label = f"{run}, tol={tolerance:.0e}"
                axes[0].plot(x, raw_mae, marker="o", label=label, color=colors.get(run), alpha=0.75)
                axes[1].plot(x, asymmetry, marker="o", label=label, color=colors.get(run), alpha=0.75)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("finite-difference displacement [Bohr]")
    axes[0].set_ylabel("mean raw Hessian MAE [Ha/Bohr^2]")
    axes[1].set_ylabel(r"mean $\|H_{asym}\|_F / \|H_{sym}\|_F$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "step_and_tolerance_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.as_posix())

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for run in sorted({row["run"] for row in rows}):
        selected = [row for row in rows if row["run"] == run]
        ax.scatter(
            [row["raw_mae"] for row in selected],
            [row["sym_mae"] for row in selected],
            s=24,
            alpha=0.65,
            label=run,
            color=colors.get(run),
        )
    limits = ax.get_xlim()
    low = max(min(limits), 1e-7)
    high = max(limits)
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("raw Hessian MAE")
    ax.set_ylabel("symmetrized Hessian MAE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "raw_vs_symmetrized_mae.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.as_posix())
    return paths


def _plot_trace_examples(point_rows: list[dict[str, Any]], output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    candidates = [
        row
        for row in point_rows
        if row.get("trace_status")
        not in (None, "converged", "not_recorded", "missing", "empty")
        and row.get("trace_file")
        and Path(str(row["trace_file"])).exists()
    ]
    candidates.sort(
        key=lambda row: float(row.get("final_gradient_norm") or 0.0), reverse=True
    )
    candidates = candidates[:12]
    if not candidates:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for row in candidates:
        with np.load(row["trace_file"]) as payload:
            gradients = np.asarray(payload["projected_gradient_norm"], dtype=np.float64)
        label = (
            f"{row['run']} {row['molecule_id']} c{row.get('coord_idx')} "
            f"{row.get('side')} {row.get('trace_status')}"
        )
        ax.plot(np.arange(len(gradients)), gradients, linewidth=1.0, alpha=0.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("density optimization cycle")
    ax.set_ylabel("projected density-gradient norm")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = output_dir / "nonconverged_optimization_trace_examples.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.as_posix()


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        row["molecule_id"] for row in json.loads(args.selected_manifest.read_text())
    }
    molecule_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    hessians: dict[tuple[str, str, float, float], np.ndarray] = {}

    for result_path in _collect_result_paths(args):
        result = json.loads(result_path.read_text())
        displacement, tolerance = _condition_from_result(result, result_path.as_posix())
        optimization_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for point in result.get("optimization_rows", []):
            if point.get("molecule_id") in selected:
                enriched = {
                    "source_summary": result_path.as_posix(),
                    "displacement": displacement,
                    "tolerance": tolerance,
                    **point,
                    **_classify_optimization_trace(point.get("trace_file"), tolerance),
                }
                point_rows.append(enriched)
                optimization_by_key[(point["run"], point["molecule_id"])].append(point)

        for row in result.get("rows", []):
            if row.get("molecule_id") not in selected or not row.get("success"):
                continue
            hessian_path = Path(row["hessian_npz"])
            hessian, reference = _load_hessians(hessian_path)
            key = (row["run"], row["molecule_id"], displacement, tolerance)
            hessians[key] = hessian
            opts = optimization_by_key[(row["run"], row["molecule_id"])]
            grads = [
                float(point["final_gradient_norm"])
                for point in opts
                if point.get("final_gradient_norm") is not None
            ]
            cycles = [int(point["cycles"]) for point in opts if point.get("cycles") is not None]
            by_coord: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
            for point in opts:
                if point.get("coord_idx") is not None:
                    by_coord[int(point["coord_idx"])][point["side"]] = point
            gradient_mismatches = []
            energy_odd_derivatives = []
            for sides in by_coord.values():
                if "plus" not in sides or "minus" not in sides:
                    continue
                plus, minus = sides["plus"], sides["minus"]
                if plus.get("final_gradient_norm") is not None and minus.get("final_gradient_norm") is not None:
                    gradient_mismatches.append(
                        abs(float(plus["final_gradient_norm"]) - float(minus["final_gradient_norm"]))
                    )
                if plus.get("final_total_energy") is not None and minus.get("final_total_energy") is not None:
                    energy_odd_derivatives.append(
                        (float(plus["final_total_energy"]) - float(minus["final_total_energy"]))
                        / (2.0 * displacement)
                    )
            molecule_rows.append(
                {
                    "source_summary": result_path.as_posix(),
                    "run": row["run"],
                    "molecule_id": row["molecule_id"],
                    "sample_id": int(row["sample_id"]),
                    "natoms": int(row["natoms"]),
                    "displacement": displacement,
                    "tolerance": tolerance,
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                    "n_optimization_points": len(opts),
                    "strict_converged_points": sum(
                        1 for value in grads if value < tolerance
                    ),
                    "mean_final_gradient_norm": _mean(grads),
                    "max_final_gradient_norm": _maximum(grads),
                    "mean_cycles": _mean(cycles),
                    "max_cycles": _maximum(cycles),
                    "plus_minus_gradient_mismatch_mean": _mean(gradient_mismatches),
                    "relaxed_total_energy_gradient_rms": (
                        float(np.sqrt(np.mean(np.square(energy_odd_derivatives))))
                        if energy_odd_derivatives
                        else None
                    ),
                    "hessian_npz": hessian_path.as_posix(),
                    **_metrics(hessian, reference),
                }
            )

    condition_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in molecule_rows:
        grouped[(row["run"], row["displacement"], row["tolerance"])].append(row)
    metric_names = [
        "raw_mae",
        "raw_rmse",
        "raw_relative_fro",
        "sym_mae",
        "sym_rmse",
        "sym_relative_fro",
        "antisymmetric_over_symmetric_fro",
        "symmetry_max_abs",
        "elapsed_s",
        "mean_final_gradient_norm",
        "max_final_gradient_norm",
        "mean_cycles",
    ]
    metric_names.extend(
        f"{prefix}_{quantile}"
        for prefix in ("raw_abs_error", "sym_abs_error", "symmetry_abs")
        for quantile in ("q50", "q90", "q95", "q99", "q100")
    )
    for (run, displacement, tolerance), group in sorted(grouped.items()):
        strict_group = [
            row
            for row in group
            if row["n_optimization_points"] > 0
            and row["strict_converged_points"] == row["n_optimization_points"]
        ]
        condition_points = [
            row
            for row in point_rows
            if row["run"] == run
            and row["displacement"] == displacement
            and row["tolerance"] == tolerance
        ]
        trace_counts = {
            status: sum(row.get("trace_status") == status for row in condition_points)
            for status in (
                "converged",
                "plateau",
                "oscillatory",
                "diverging",
                "slow_improvement",
                "stalled_mixed",
                "nonfinite",
                "not_recorded",
                "missing",
            )
        }
        condition_rows.append(
            {
                "run": run,
                "displacement": displacement,
                "tolerance": tolerance,
                "n_molecules": len(group),
                "strict_molecules": len(strict_group),
                "strict_converged_points": sum(row["strict_converged_points"] for row in group),
                "optimization_points": sum(row["n_optimization_points"] for row in group),
                **{f"trace_{key}_points": value for key, value in trace_counts.items()},
                **{f"mean_{name}": _mean([row[name] for row in group]) for name in metric_names},
                **{f"max_{name}": _maximum([row[name] for row in group]) for name in metric_names},
                **{
                    f"strict_mean_{name}": _mean([row[name] for row in strict_group])
                    for name in metric_names
                },
                **{
                    f"strict_max_{name}": _maximum([row[name] for row in strict_group])
                    for name in metric_names
                },
            }
        )

    stability_rows = _stability_rows(molecule_rows, hessians)
    correlation_rows = _correlation_rows(molecule_rows)
    _write_csv(args.output_dir / "per_molecule_metrics.csv", molecule_rows)
    _write_csv(args.output_dir / "per_displacement_optimization.csv", point_rows)
    _write_csv(args.output_dir / "condition_summary.csv", condition_rows)
    _write_csv(args.output_dir / "hessian_stability.csv", stability_rows)
    _write_csv(args.output_dir / "residual_asymmetry_correlations.csv", correlation_rows)
    plot_paths = _plot(molecule_rows, args.output_dir)
    trace_plot = _plot_trace_examples(point_rows, args.output_dir)
    if trace_plot is not None:
        plot_paths.append(trace_plot)
    result = {
        "baseline_summary": args.baseline_summary.as_posix(),
        "scan_root": args.scan_root.as_posix(),
        "additional_scan_roots": [path.as_posix() for path in args.additional_scan_root],
        "selected_manifest": args.selected_manifest.as_posix(),
        "n_molecule_condition_rows": len(molecule_rows),
        "n_optimization_point_rows": len(point_rows),
        "n_condition_rows": len(condition_rows),
        "n_stability_rows": len(stability_rows),
        "plots": plot_paths,
        "condition_summary": condition_rows,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument(
        "--additional-scan-root",
        type=Path,
        action="append",
        default=[],
        help="Optional additional root containing conditions/*/summary.json files.",
    )
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
