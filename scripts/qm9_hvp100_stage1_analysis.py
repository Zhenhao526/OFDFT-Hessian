#!/usr/bin/env python3
"""Aggregate three-seed HVP100 validation and shortlist C/D weights without Test100."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _plot_groups(output_dir: Path, groups: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hvp_groups = [row for row in groups if row["variant"] in {"C", "D"}]
    colors = {"C": "#0072B2", "D": "#D55E00"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in hvp_groups:
        label = f"{row['variant']}  w={row['hvp_weight']:.0e}"
        ax.scatter(
            100 * row["mean_force_component_mae_change"],
            100 * row["mean_fixed_hvp_mae_change"],
            s=70,
            color=colors[row["variant"]],
            label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation force MAE change vs A (%)")
    ax.set_ylabel("Fixed-density HVP MAE change vs A (%)")
    ax.set_title("HVP100 validation-only screening")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_force_hvp_tradeoff.png", dpi=180)
    plt.close(fig)


PATTERN = re.compile(
    r"qm9_hvp100_(?P<name>.+)_seed(?P<seed>\d+)_s600_(?:gated25v2_|gated25_)?20260716$"
)


def _variant(name: str) -> tuple[str, float]:
    mapping = {
        "force": ("A", 0.0),
        "force_secant": ("B", 0.0),
        "force_hvp_w1e5": ("C", 1e-5),
        "force_hvp_w1e4": ("C", 1e-4),
        "force_hvp_w1e3": ("C", 1e-3),
        "force_secant_hvp_w1e5": ("D", 1e-5),
        "force_secant_hvp_w1e4": ("D", 1e-4),
        "force_secant_hvp_w1e3": ("D", 1e-3),
    }
    return mapping[name]


def analyze(args: argparse.Namespace) -> dict:
    rows = []
    per_molecule = []
    for run_dir in sorted(args.posttrain_dir.glob("qm9_hvp100_*")):
        match = PATTERN.match(run_dir.name)
        if match is None:
            continue
        force_path = run_dir / "validation_energy_force.json"
        hvp_path = run_dir / "fixed_hvp" / "summary.json"
        if not force_path.exists() or not hvp_path.exists():
            continue
        force = json.loads(force_path.read_text())
        hvp = json.loads(hvp_path.read_text())
        variant, weight = _variant(match.group("name"))
        row = {
            "run": run_dir.name,
            "variant": variant,
            "hvp_weight": weight,
            "seed": int(match.group("seed")),
            "energy_mae": force["energy_mae"],
            "force_component_mae": force["force_component_mae"],
            "force_component_rmse": force["force_component_rmse"],
            "fixed_hvp_mae": hvp["mean_mae"],
            "fixed_hvp_rmse": hvp["mean_rmse"],
            "fixed_hvp_relative_frobenius": hvp["mean_relative_frobenius"],
        }
        rows.append(row)
        for item in hvp["rows"]:
            per_molecule.append(
                {
                    "run": run_dir.name,
                    "variant": variant,
                    "hvp_weight": weight,
                    "seed": int(match.group("seed")),
                    **item,
                }
            )
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 complete runs, found {len(rows)}")

    baseline = {row["seed"]: row for row in rows if row["variant"] == "A"}
    baseline_mol = {
        (row["seed"], row["molecule_id"]): row
        for row in per_molecule
        if row["variant"] == "A"
    }
    for row in rows:
        ref = baseline[row["seed"]]
        for metric in ("energy_mae", "force_component_mae", "fixed_hvp_mae"):
            row[f"{metric}_change"] = row[metric] / ref[metric] - 1.0
        molecule_rows = [
            item for item in per_molecule if item["run"] == row["run"]
        ]
        row["fixed_hvp_win_fraction"] = float(
            np.mean(
                [
                    item["corrected_vs_pbe_mae"]
                    < baseline_mol[(row["seed"], item["molecule_id"])]["corrected_vs_pbe_mae"]
                    for item in molecule_rows
                ]
            )
        )

    grouped = []
    for key in sorted({(row["variant"], row["hvp_weight"]) for row in rows}):
        group = [row for row in rows if (row["variant"], row["hvp_weight"]) == key]
        grouped.append(
            {
                "variant": key[0],
                "hvp_weight": key[1],
                "seeds": len(group),
                **{
                    f"mean_{metric}": float(np.mean([row[metric] for row in group]))
                    for metric in (
                        "energy_mae",
                        "force_component_mae",
                        "fixed_hvp_mae",
                        "energy_mae_change",
                        "force_component_mae_change",
                        "fixed_hvp_mae_change",
                        "fixed_hvp_win_fraction",
                    )
                },
                "force_improved_seeds": sum(row["force_component_mae_change"] < 0 for row in group),
                "hvp_improved_seeds": sum(row["fixed_hvp_mae_change"] < 0 for row in group),
                "energy_gate_passed_seeds": sum(row["energy_mae_change"] <= args.energy_tolerance for row in group),
            }
        )
    eligible = [
        row for row in grouped
        if row["variant"] in {"C", "D"}
        and row["energy_gate_passed_seeds"] == 3
        and row["force_improved_seeds"] >= 2
        and row["hvp_improved_seeds"] >= 2
        and row["mean_fixed_hvp_win_fraction"] > 0.5
    ]
    shortlist = []
    for variant in ("C", "D"):
        candidates = [row for row in eligible if row["variant"] == variant]
        gate_passed = bool(candidates)
        if not candidates:
            candidates = [row for row in grouped if row["variant"] == variant]
        selected = dict(min(candidates, key=lambda row: row["mean_fixed_hvp_mae"]))
        selected["stage1_gate_passed"] = gate_passed
        shortlist.append(selected)
    output = {
        "definition": "Validation-only stage-1 selection; Test100 was not read.",
        "energy_relative_tolerance": args.energy_tolerance,
        "runs": rows,
        "groups": grouped,
        "shortlist": shortlist,
        "test_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage1_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for path, records in ((args.output_dir / "stage1_runs.csv", rows), (args.output_dir / "stage1_groups.csv", grouped), (args.output_dir / "stage1_per_molecule.csv", per_molecule)):
        fields = sorted({key for row in records for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(records)
    _plot_groups(args.output_dir, grouped)
    print(json.dumps({"groups": grouped, "shortlist": shortlist}, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posttrain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-tolerance", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
