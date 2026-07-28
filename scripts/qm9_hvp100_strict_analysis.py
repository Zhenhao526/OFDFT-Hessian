#!/usr/bin/env python3
"""Aggregate strict complete-total HVP validation without accessing Test100."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"qm9_hvp100_(?P<name>.+)_seed(?P<seed>\d+)_s600_(?:gated25v2_|gated25_)?20260716$"
)


def _variant(name: str) -> tuple[str, float]:
    if name == "force":
        return "A", 0.0
    if name == "force_secant":
        return "B", 0.0
    if name.startswith("force_secant_hvp_w"):
        variant = "D"
    elif name.startswith("force_hvp_w"):
        variant = "C"
    else:
        raise ValueError(name)
    encoded = name.rsplit("_w", 1)[1]
    return variant, {"1e5": 1e-5, "1e4": 1e-4, "1e3": 1e-3}[encoded]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pbe_rms(strict_dir: Path, run: str, molecule: str) -> float:
    path = strict_dir / run / f"{run}_{molecule}_0000000_hvp_arrays.npz"
    pbe = np.asarray(np.load(path)["pbe_hvp"], dtype=np.float64)
    return float(np.sqrt(np.mean(pbe**2)))


def _plot_results(output_dir: Path, groups: list[dict], molecule_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compared = [row for row in groups if row["variant"] != "A"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"B": "#009E73", "C": "#0072B2", "D": "#D55E00"}
    for row in compared:
        ax.scatter(
            100 * row["mean_force_component_mae_change"],
            100 * row["mean_strict_hvp_mae_change"],
            s=75,
            color=colors[row["variant"]],
            label=f"{row['variant']} w={row['hvp_weight']:.0e}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation force MAE change vs A (%)")
    ax.set_ylabel("Strict complete-total HVP MAE change vs A (%)")
    ax.set_title("HVP100 strict validation tradeoff")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "strict_force_hvp_tradeoff.png", dpi=180)
    plt.close(fig)

    baseline = {
        (row["seed"], row["molecule_id"]): row
        for row in molecule_rows if row["variant"] == "A"
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for variant in ("B", "C", "D"):
        selected = [row for row in molecule_rows if row["variant"] == variant]
        if not selected:
            continue
        x = [baseline[(row["seed"], row["molecule_id"])]["strict_hvp_mae"] for row in selected]
        y = [row["strict_hvp_mae"] for row in selected]
        ax.scatter(x, y, s=18, alpha=0.65, color=colors[variant], label=variant)
    values = [
        value
        for collection in ax.collections
        for value in collection.get_offsets().ravel()
        if np.isfinite(value) and value > 0
    ]
    lower = min(values) * 0.8
    upper = max(values) * 1.25
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("A strict HVP MAE (Ha/Bohr^2)")
    ax.set_ylabel("Candidate strict HVP MAE (Ha/Bohr^2)")
    ax.set_title("Per-molecule strict HVP comparison")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "strict_per_molecule_scatter.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict:
    stage1 = json.loads(args.stage1_summary.read_text())
    stage1_runs = {row["run"]: row for row in stage1["runs"]}
    molecule_rows: list[dict] = []
    run_rows: list[dict] = []

    for run_dir in sorted(args.strict_dir.glob("qm9_hvp100_*")):
        match = PATTERN.match(run_dir.name)
        metrics_path = run_dir / "metrics.csv"
        summary_path = run_dir / "summary.json"
        if match is None or not metrics_path.exists() or not summary_path.exists():
            continue
        variant, weight = _variant(match.group("name"))
        seed = int(match.group("seed"))
        with metrics_path.open() as handle:
            metrics = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text())
        for row in metrics:
            molecule = row["molecule_id"]
            reference_rms = _pbe_rms(args.strict_dir, run_dir.name, molecule)
            molecule_rows.append(
                {
                    "run": run_dir.name,
                    "variant": variant,
                    "hvp_weight": weight,
                    "seed": seed,
                    "molecule_id": molecule,
                    "natoms": int(row["natoms"]),
                    "direction_kind": row["direction_kind"],
                    "reference_hvp_rms": reference_rms,
                    "small_reference": reference_rms < args.small_reference_rms,
                    "strict_hvp_mae": float(row["strict_relaxed_vs_pbe_mae"]),
                    "strict_hvp_rmse": float(row["strict_relaxed_vs_pbe_rmse"]),
                    "strict_hvp_relative_frobenius": float(
                        row["strict_relaxed_vs_pbe_relative_frobenius"]
                    ),
                    "fixed_hvp_mae": float(row["fixed_vs_pbe_mae"]),
                    "max_density_gradient": float(row["strict_relaxed_max_gradient_norm"]),
                    "wall_time_s": float(row["wall_time_s"]),
                    "response_relative_residual": float(row["response_relative_residual"]),
                }
            )
        run_molecules = [r for r in molecule_rows if r["run"] == run_dir.name]
        stage = stage1_runs[run_dir.name]
        run_rows.append(
            {
                "run": run_dir.name,
                "variant": variant,
                "hvp_weight": weight,
                "seed": seed,
                "molecules": len(run_molecules),
                "energy_mae": stage["energy_mae"],
                "force_component_mae": stage["force_component_mae"],
                "strict_hvp_mae": float(np.mean([r["strict_hvp_mae"] for r in run_molecules])),
                "strict_hvp_rmse": float(np.mean([r["strict_hvp_rmse"] for r in run_molecules])),
                "strict_hvp_relative_frobenius": float(
                    np.mean([r["strict_hvp_relative_frobenius"] for r in run_molecules])
                ),
                "small_reference_hvp_mae": float(
                    np.mean([r["strict_hvp_mae"] for r in run_molecules if r["small_reference"]])
                ) if any(r["small_reference"] for r in run_molecules) else None,
                "wall_time_s": float(summary["wall_time_s"]),
                "max_rss_mb": float(summary["max_rss_mb"]),
                "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
            }
        )

    expected = int(args.expected_runs)
    if len(run_rows) != expected:
        raise RuntimeError(f"Expected {expected} strict runs, found {len(run_rows)}")
    baseline = {r["seed"]: r for r in run_rows if r["variant"] == "A"}
    baseline_molecule = {
        (r["seed"], r["molecule_id"]): r for r in molecule_rows if r["variant"] == "A"
    }
    for row in run_rows:
        reference = baseline[row["seed"]]
        for metric in ("energy_mae", "force_component_mae", "strict_hvp_mae"):
            row[f"{metric}_change"] = row[metric] / reference[metric] - 1.0
        paired = [r for r in molecule_rows if r["run"] == row["run"]]
        row["strict_hvp_win_fraction"] = float(
            np.mean(
                [
                    r["strict_hvp_mae"]
                    < baseline_molecule[(row["seed"], r["molecule_id"])]["strict_hvp_mae"]
                    for r in paired
                ]
            )
        )

    groups: list[dict] = []
    for key in sorted({(r["variant"], r["hvp_weight"]) for r in run_rows}):
        members = [r for r in run_rows if (r["variant"], r["hvp_weight"]) == key]
        group = {
            "variant": key[0],
            "hvp_weight": key[1],
            "seeds": len(members),
            "energy_gate_passed_seeds": sum(
                r["energy_mae_change"] <= args.energy_tolerance for r in members
            ),
            "force_improved_seeds": sum(r["force_component_mae_change"] < 0 for r in members),
            "strict_hvp_improved_seeds": sum(r["strict_hvp_mae_change"] < 0 for r in members),
        }
        for metric in (
            "energy_mae", "force_component_mae", "strict_hvp_mae",
            "strict_hvp_relative_frobenius", "energy_mae_change",
            "force_component_mae_change", "strict_hvp_mae_change",
            "strict_hvp_win_fraction", "wall_time_s", "peak_gpu_memory_mb",
        ):
            group[f"mean_{metric}"] = float(np.mean([r[metric] for r in members]))
        group["strict_gate_passed"] = bool(
            group["energy_gate_passed_seeds"] == 3
            and group["force_improved_seeds"] >= 2
            and group["strict_hvp_improved_seeds"] >= 2
            and group["mean_strict_hvp_mae_change"] < 0
            and group["mean_strict_hvp_win_fraction"] > 0.5
        )
        groups.append(group)

    eligible = [g for g in groups if g["variant"] in {"C", "D"} and g["strict_gate_passed"]]
    frozen_candidates = []
    for group in sorted(eligible, key=lambda g: g["mean_strict_hvp_mae"]):
        members = [
            r for r in run_rows
            if (r["variant"], r["hvp_weight"]) == (group["variant"], group["hvp_weight"])
        ]
        selected = sorted(members, key=lambda r: r["strict_hvp_mae"])[len(members) // 2]
        frozen_candidates.append(
            {
                "variant": group["variant"],
                "hvp_weight": group["hvp_weight"],
                "run": selected["run"],
                "seed_selection": "median strict complete-total HVP MAE among three seeds",
            }
        )
    frozen_candidates = frozen_candidates[:2]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "strict_runs.csv", run_rows)
    _write_csv(args.output_dir / "strict_groups.csv", groups)
    _write_csv(args.output_dir / "strict_per_molecule.csv", molecule_rows)
    _plot_results(args.output_dir, groups, molecule_rows)
    result = {
        "definition": "Validation-only strict density-relaxed complete-total HVP selection.",
        "test_accessed": False,
        "energy_relative_tolerance": args.energy_tolerance,
        "small_reference_rms_threshold": args.small_reference_rms,
        "runs": run_rows,
        "groups": groups,
        "frozen_candidates_for_full_hessian": frozen_candidates,
        "test100_allowed_after_full_hessian_gate": False,
    }
    (args.output_dir / "strict_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"groups": groups, "candidates": frozen_candidates}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--energy-tolerance", type=float, default=0.05)
    parser.add_argument("--small-reference-rms", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
