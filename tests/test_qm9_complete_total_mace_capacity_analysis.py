from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qm9_complete_total_mace_capacity_analysis import analyze


def _run(path: Path, *, gate_passed: bool, test100_accessed: bool = False) -> Path:
    path.mkdir()
    summary = {
        "test100_accessed": test100_accessed,
        "test100_evaluations_used": 0,
        "validation_accessed": False,
        "unselected_parent_artifacts_opened": 0,
        "run_mode": "formal",
        "selected_parent_ids": ["0000001"],
        "arm_id": path.name,
        "steps": 2,
        "best_step": 2,
        "final": {
            "hessian_relative_frobenius": {"median": 0.2, "p90": 0.2, "max": 0.2},
            "energy_median_ratio_to_source": 1.0,
            "force_median_ratio_to_source": 1.0,
            "max_antisymmetric_over_symmetric_frobenius": 1e-16,
            "gate": {"passed": gate_passed},
        },
        "wall_time_s": 3.0,
        "gpu_peak_memory_mb": 4.0,
        "max_rss_mb": 5.0,
        "parent_cv_authorized": gate_passed,
        "per_parent": [
            {
                "molecule_id": "0000001",
                "natoms": 3,
                "relative_frobenius": 0.2,
                "mae": 0.01,
                "rmse": 0.02,
                "source_relative_frobenius": 2.0,
            }
        ],
        "protocol": "/frozen/protocol.yaml",
        "protocol_sha256": "a" * 64,
        "source_split_sha256": "b" * 64,
    }
    (path / "summary.json").write_text(json.dumps(summary))
    metrics = [
        {
            "step": step,
            "wall_time_s": float(step),
            "median_relative_frobenius": value,
            "p90_relative_frobenius": value,
            "max_relative_frobenius": value,
            "selection_score": value * 2,
            "gpu_peak_memory_mb": 4.0,
        }
        for step, value in ((0, 2.0), (1, 0.8), (2, 0.2))
    ]
    (path / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )
    return path


def test_analysis_fails_closed_when_no_run_passes(tmp_path: Path) -> None:
    first = _run(tmp_path / "h8", gate_passed=False)
    second = _run(tmp_path / "h16", gate_passed=False)
    output = tmp_path / "analysis"

    result = analyze([("h8", first), ("h16", second)], output, make_plot=False)

    assert result["selected_candidate"] is None
    assert result["next_stage_authorized"] is False
    assert result["decision"] == "fail_closed_keep_train20_and_test100_frozen"
    assert result["runs"][0]["threshold_first_steps"]["1.0"] == 1
    assert (output / "final_comparison.csv").is_file()
    assert (output / "learning_curves.csv").is_file()
    assert (output / "per_parent_comparison.csv").is_file()


def test_analysis_rejects_test100_access(tmp_path: Path) -> None:
    run = _run(tmp_path / "unsafe", gate_passed=True, test100_accessed=True)

    with pytest.raises(ValueError, match="Test100 remained frozen"):
        analyze([("unsafe", run)], tmp_path / "analysis", make_plot=False)
