from __future__ import annotations

import argparse
import json

from scripts.qm9_complete_total_spectral_operator_validation_analysis import analyze


def _summary(scale: float) -> dict:
    rows = []
    for index, molecule_id in enumerate(("0000001", "0000002")):
        rows.append(
            {
                "molecule_id": molecule_id,
                "relative_frobenius": scale * (index + 1),
                "baseline_energy_abs_error_hartree": 0.1,
                "source_energy_abs_error_hartree": 0.09,
                "baseline_force_mae_hartree_per_bohr": 0.2,
                "source_force_mae_hartree_per_bohr": 0.18,
            }
        )
    return {
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "protocol_sha256": "protocol",
        "parent_count": 2,
        "per_parent": rows,
        "hessian_relative_frobenius": {
            "median": 1.5 * scale,
            "p90": 1.9 * scale,
            "max": 2.0 * scale,
        },
        "fraction_relative_frobenius_at_or_below_0_15": 1.0,
        "parent_win_fraction_vs_source": 1.0,
        "energy_abs_error_hartree": {"median": 0.09, "p90": 0.09},
        "force_mae_hartree_per_bohr": {"median": 0.18, "p90": 0.18},
        "stage3_validation_gate": {"passed": True},
    }


def test_analysis_writes_paired_tables_and_plot(tmp_path) -> None:
    runs = []
    for name, scale in (("a", 0.04), ("b", 0.05)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(_summary(scale)))
        runs.append(f"{name}={run_dir}")
    output = tmp_path / "analysis"

    result = analyze(argparse.Namespace(run=runs, output_dir=output))

    assert result["parent_ids"] == ["0000001", "0000002"]
    assert len(result["runs"]) == 2
    assert (output / "per_parent_all_arms.csv").is_file()
    assert (output / "run_summary.csv").is_file()
    assert (output / "unseen_parent_shared_operator_comparison.png").is_file()
