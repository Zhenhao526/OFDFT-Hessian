#!/usr/bin/env python3
"""Merge the random1000 conservative total-OFDFT model-optimization funnel."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _validation_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="ignore").replace("\r", "\n")
    pattern = re.compile(
        r"Epoch (\d+): 100%.*?val_loss/energy_loss=([0-9.eE+-]+), "
        r"val_loss/force_loss=([0-9.eE+-]+), val_loss/total=([0-9.eE+-]+)"
    )
    rows: dict[int, dict[str, Any]] = {}
    for match in pattern.finditer(text):
        epoch, energy, force, total = match.groups()
        rows[int(epoch)] = {
            "epoch": int(epoch),
            "energy_loss": float(energy),
            "force_loss": float(force),
            "total_loss": float(total),
        }
    return [rows[key] for key in sorted(rows)]


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values))


def _relative_change(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_confirm_dir = args.full_confirm_dir or args.total_confirm_dir
    training_rows = _validation_rows(args.training_log)
    _write_csv(args.output_dir / "training_validation.csv", training_rows)

    fast = json.loads((args.fast_eval_dir / "summary.json").read_text())
    force_rows = []
    for split, split_payload in fast["force_proxy"].items():
        for run, metrics in split_payload.items():
            force_rows.append({"split": split, "run": run, **metrics})
    _write_csv(args.output_dir / "energy_force_metrics.csv", force_rows)

    fixed_payload = json.loads(
        (args.fast_eval_dir / "fixed_density_incomplete_proxy" / "summary.json").read_text()
    )
    fixed_rows = []
    for row in fixed_payload["fixed_density_full_hessian"]:
        fixed_rows.append(
            {
                "run": row["run"],
                "molecule_id": row["molecule_id"],
                "natoms": row["natoms"],
                "success": row["success"],
                **_flatten_metrics("autograd_vs_pbe", row["autograd_vs_pbe"]),
                **_flatten_metrics("autograd_vs_fd", row["autograd_vs_fd"]),
                "autograd_elapsed_s": row["autograd_elapsed_s"],
                "fd_elapsed_s": row["fd_elapsed_s"],
            }
        )
    _write_csv(args.output_dir / "fixed_incomplete_proxy_per_molecule.csv", fixed_rows)

    hvp_rows = []
    curvature_rows = []
    for run in (args.baseline_label, args.candidate_label):
        run_dir = args.total_confirm_dir / "hvp" / run
        payload = json.loads((run_dir / "summary.json").read_text())
        for row in payload["summaries"]:
            hvp_rows.append(
                {
                    "run": run,
                    "molecule_id": row["molecule_id"],
                    "natoms": row["natoms"],
                    "response_solver": row["response_solver"],
                    "response_relative_residual": row["response_relative_residual"],
                    "implicit_vs_strict_relative_frobenius": row[
                        "implicit_vs_relaxed"
                    ]["relative_frobenius"],
                    **_flatten_metrics("strict_vs_pbe", row["strict_relaxed_vs_pbe"]),
                    **_flatten_metrics("implicit_vs_pbe", row["implicit_vs_pbe"]),
                    "strict_relaxed_max_gradient_norm": row[
                        "strict_relaxed_max_gradient_norm"
                    ],
                    "wall_time_s": row["wall_time_s"],
                }
            )
            curvature_rows.extend(row["curvature_step_scan"])
    _write_csv(args.output_dir / "strict_total_hvp_per_molecule.csv", hvp_rows)
    _write_csv(args.output_dir / "directional_curvature_step_scan.csv", curvature_rows)

    full_rows = []
    for run in (args.baseline_label, args.candidate_label):
        path = full_confirm_dir / "full_hessian_0000777" / run / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for row in payload["metric_rows"]:
            full_rows.append(row)
    _write_csv(args.output_dir / "strict_total_full_hessian.csv", full_rows)

    full_step_rows = []
    if args.full_step_check_dir is not None:
        payload = json.loads((args.full_step_check_dir / "summary.json").read_text())
        full_step_rows = payload["metric_rows"]
        _write_csv(args.output_dir / "strict_total_full_hessian_step_check.csv", full_step_rows)

    weight3_rows = []
    if args.weight3_hvp_summary is not None:
        payload = json.loads(args.weight3_hvp_summary.read_text())
        for row in payload["summaries"]:
            weight3_rows.append(
                {
                    "run": "ForceWeight3",
                    "molecule_id": row["molecule_id"],
                    **_flatten_metrics("strict_vs_pbe", row["strict_relaxed_vs_pbe"]),
                    "implicit_vs_strict_relative_frobenius": row[
                        "implicit_vs_relaxed"
                    ]["relative_frobenius"],
                }
            )
        _write_csv(args.output_dir / "weight3_strict_total_hvp.csv", weight3_rows)

    test100_total_hvp = None
    test100_total_rows: list[dict[str, Any]] = []
    if args.test100_total_hvp_dir is not None:
        test100_total_hvp = json.loads(
            (args.test100_total_hvp_dir / "summary.json").read_text()
        )
        with (args.test100_total_hvp_dir / "per_molecule.csv").open() as handle:
            test100_total_rows = list(csv.DictReader(handle))
        _write_csv(
            args.output_dir / "test100_strict_total_hvp_per_molecule.csv",
            test100_total_rows,
        )

    baseline_force = next(
        row for row in force_rows if row["split"] == "test" and row["run"] == args.baseline_label
    )
    candidate_force = next(
        row for row in force_rows if row["split"] == "test" and row["run"] == args.candidate_label
    )
    baseline_fixed = [row for row in fixed_rows if row["run"] == args.baseline_label]
    candidate_fixed = [row for row in fixed_rows if row["run"] == args.candidate_label]
    candidate_fixed_by_molecule = {
        row["molecule_id"]: row for row in candidate_fixed
    }
    baseline_hvp = [row for row in hvp_rows if row["run"] == args.baseline_label]
    candidate_hvp = [row for row in hvp_rows if row["run"] == args.candidate_label]
    paired_hvp = {
        row["molecule_id"]: row for row in candidate_hvp
    }
    improved_hvp = sum(
        paired_hvp[row["molecule_id"]]["strict_vs_pbe_mae"]
        < row["strict_vs_pbe_mae"]
        for row in baseline_hvp
    )
    summary = {
        "definitions": {
            "fast_force_and_fixed": (
                "Learned kin_plus_xc scalar-derived force/fixed-density Hessian regression "
                "proxy; not a complete total-OFDFT derivative."
            ),
            "strict_total_hvp": (
                "Complete scalar total-OFDFT force finite difference after strict density "
                "relaxation, with KKT implicit-response cross-check."
            ),
        },
        "inputs": {
            "training_log": str(args.training_log),
            "fast_eval_dir": str(args.fast_eval_dir),
            "total_confirm_dir": str(args.total_confirm_dir),
            "full_confirm_dir": str(full_confirm_dir),
            "weight3_hvp_summary": (
                str(args.weight3_hvp_summary) if args.weight3_hvp_summary else None
            ),
            "full_step_check_dir": (
                str(args.full_step_check_dir) if args.full_step_check_dir else None
            ),
            "test100_total_hvp_dir": (
                str(args.test100_total_hvp_dir) if args.test100_total_hvp_dir else None
            ),
        },
        "training_epochs": len(training_rows),
        "test100": {
            "baseline_energy_mae": baseline_force["energy_mae"],
            "candidate_energy_mae": candidate_force["energy_mae"],
            "energy_relative_change": _relative_change(
                candidate_force["energy_mae"], baseline_force["energy_mae"]
            ),
            "baseline_force_component_mae": baseline_force["force_component_mae"],
            "candidate_force_component_mae": candidate_force["force_component_mae"],
            "force_relative_change": _relative_change(
                candidate_force["force_component_mae"],
                baseline_force["force_component_mae"],
            ),
        },
        "fixed_incomplete_proxy_10mol": {
            "baseline_mean_mae": _mean(baseline_fixed, "autograd_vs_pbe_mae"),
            "candidate_mean_mae": _mean(candidate_fixed, "autograd_vs_pbe_mae"),
            "baseline_mean_relative_fro": _mean(
                baseline_fixed, "autograd_vs_pbe_relative_fro_error"
            ),
            "candidate_mean_relative_fro": _mean(
                candidate_fixed, "autograd_vs_pbe_relative_fro_error"
            ),
            "candidate_wins": sum(
                candidate_fixed_by_molecule[row["molecule_id"]]["autograd_vs_pbe_mae"]
                < row["autograd_vs_pbe_mae"]
                for row in baseline_fixed
            ),
        },
        "strict_total_hvp": {
            "molecules": len(baseline_hvp),
            "baseline_mean_mae": _mean(baseline_hvp, "strict_vs_pbe_mae"),
            "candidate_mean_mae": _mean(candidate_hvp, "strict_vs_pbe_mae"),
            "baseline_mean_relative_fro": _mean(
                baseline_hvp, "strict_vs_pbe_relative_frobenius"
            ),
            "candidate_mean_relative_fro": _mean(
                candidate_hvp, "strict_vs_pbe_relative_frobenius"
            ),
            "candidate_wins": improved_hvp,
        },
        "full_hessian_rows": len(full_rows),
        "full_step_check_rows": len(full_step_rows),
        "test100_total_hvp": (
            {
                "summaries": test100_total_hvp["summaries"],
                "candidate_mae_wins": test100_total_hvp["candidate_mae_wins"],
                "candidate_mae_losses": test100_total_hvp["candidate_mae_losses"],
            }
            if test100_total_hvp is not None
            else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    epochs = [row["epoch"] for row in training_rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for axis, key, title in zip(
        axes,
        ("energy_loss", "force_loss", "total_loss"),
        ("Validation energy", "Validation force", "Validation total"),
    ):
        axis.plot(epochs, [row[key] for row in training_rows], marker="o")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_validation.png", dpi=180)
    plt.close(fig)

    molecules = [row["molecule_id"] for row in baseline_hvp]
    x = np.arange(len(molecules))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8.2, 3.8))
    axis.bar(
        x - width / 2,
        [row["strict_vs_pbe_mae"] for row in baseline_hvp],
        width,
        label=args.baseline_label,
    )
    axis.bar(
        x + width / 2,
        [paired_hvp[molecule]["strict_vs_pbe_mae"] for molecule in molecules],
        width,
        label=args.candidate_label,
    )
    axis.set(xticks=x, xticklabels=molecules, ylabel="Strict total Hv MAE", yscale="log")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "strict_total_hvp_mae.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(molecules), figsize=(3.2 * len(molecules), 3.2))
    if len(molecules) == 1:
        axes = [axes]
    for axis, molecule in zip(axes, molecules):
        for run in (args.baseline_label, args.candidate_label):
            rows = [
                row
                for row in curvature_rows
                if row["run"] == run and row["molecule_id"] == molecule
            ]
            rows.sort(key=lambda row: row["step_bohr"])
            axis.plot(
                [row["step_bohr"] for row in rows],
                [row["relative_difference"] for row in rows],
                marker="o",
                label=run,
            )
        axis.set(xscale="log", yscale="log", title=molecule, xlabel="h (Bohr)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Energy/force curvature relative difference")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "curvature_step_stability.png", dpi=180)
    plt.close(fig)

    if test100_total_rows:
        baseline_test100 = {
            row["molecule_id"]: float(row["strict_vs_pbe_mae"])
            for row in test100_total_rows
            if row["run"] == args.baseline_label
        }
        candidate_test100 = {
            row["molecule_id"]: float(row["strict_vs_pbe_mae"])
            for row in test100_total_rows
            if row["run"] == args.candidate_label
        }
        molecules_test100 = sorted(baseline_test100)
        baseline_values = np.asarray(
            [baseline_test100[molecule] for molecule in molecules_test100]
        )
        candidate_values = np.asarray(
            [candidate_test100[molecule] for molecule in molecules_test100]
        )
        lower = max(
            min(float(baseline_values.min()), float(candidate_values.min())) * 0.8,
            np.finfo(float).tiny,
        )
        upper = max(float(baseline_values.max()), float(candidate_values.max())) * 1.25
        fig, axis = plt.subplots(figsize=(5.2, 4.6))
        improved = candidate_values < baseline_values
        axis.scatter(
            baseline_values[improved],
            candidate_values[improved],
            s=24,
            alpha=0.8,
            label=f"Improved ({int(improved.sum())})",
        )
        axis.scatter(
            baseline_values[~improved],
            candidate_values[~improved],
            s=32,
            marker="x",
            label=f"Regressed ({int((~improved).sum())})",
        )
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
        axis.set(
            xscale="log",
            yscale="log",
            xlim=(lower, upper),
            ylim=(lower, upper),
            xlabel="Baseline strict total Hv MAE",
            ylabel="Candidate strict total Hv MAE",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "test100_strict_total_hvp_scatter.png", dpi=180)
        plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--fast-eval-dir", type=Path, required=True)
    parser.add_argument("--total-confirm-dir", type=Path, required=True)
    parser.add_argument("--full-confirm-dir", type=Path)
    parser.add_argument("--weight3-hvp-summary", type=Path)
    parser.add_argument("--full-step-check-dir", type=Path)
    parser.add_argument("--test100-total-hvp-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
