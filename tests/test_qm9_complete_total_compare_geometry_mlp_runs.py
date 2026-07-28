from __future__ import annotations

import argparse
import csv
import json

import pytest

from scripts.qm9_complete_total_compare_geometry_mlp_runs import compare


def _run(tmp_path, name, values, *, exposed=False):
    root = tmp_path / name
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "test100_accessed": exposed,
                "test100_evaluations_used": int(exposed),
            }
        )
    )
    rows = []
    for index, value in enumerate(values):
        rows.append(
            {
                "molecule_id": f"{index:07d}",
                "natoms": 2 + index,
                "relative_frobenius": value,
                "train_hvp_relative_frobenius": value / 2,
                "heldout_hvp_relative_frobenius": value * 2,
                "energy_abs_error_hartree": value / 100,
                "force_mae_hartree_per_bohr": value / 50,
                "antisymmetric_over_symmetric_frobenius": 1.0e-5,
            }
        )
    with (root / "per_parent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (root / "training_metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                "wall_time_s": 1.0,
                "median_relative_frobenius": values[0],
                "median_train_hvp_relative_frobenius": values[0] / 2,
                "median_heldout_hvp_relative_frobenius": values[0] * 2,
            }
        )
        + "\n"
    )
    return root


def test_compare_reports_paired_wins_and_thresholds(tmp_path):
    first = _run(tmp_path, "first", [0.4, 0.2])
    second = _run(tmp_path, "second", [0.1, 0.3])
    result = compare(
        argparse.Namespace(
            run=[("first", first), ("second", second)],
            output_dir=tmp_path / "comparison",
        )
    )
    full = next(
        row
        for row in result["pairwise_comparisons"]
        if row["metric"] == "relative_frobenius"
    )
    assert full["second_wins"] == 1
    assert full["first_wins"] == 1
    paired_rows = list(
        csv.DictReader(
            (tmp_path / "comparison" / "paired_parent_differences.csv").open()
        )
    )
    full_rows = [row for row in paired_rows if row["metric"] == "relative_frobenius"]
    assert len(full_rows) == 2
    assert {row["winner"] for row in full_rows} == {"first", "second"}
    distribution = next(
        row
        for row in result["distributions"]
        if row["run"] == "second" and row["metric"] == "relative_frobenius"
    )
    assert distribution["count_le_0p15"] == 1
    assert result["test100_accessed"] is False


def test_compare_rejects_test100_exposed_run(tmp_path):
    safe = _run(tmp_path, "safe", [0.2])
    exposed = _run(tmp_path, "exposed", [0.2], exposed=True)
    with pytest.raises(ValueError, match="does not certify frozen Test100"):
        compare(
            argparse.Namespace(
                run=[("safe", safe), ("exposed", exposed)],
                output_dir=tmp_path / "comparison",
            )
        )
