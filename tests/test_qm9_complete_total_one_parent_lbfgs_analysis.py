from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qm9_complete_total_one_parent_lbfgs_analysis import analyze


CHECKPOINT_HASH = "a" * 64


def _summary(root: Path, parent_id: str, value: float) -> None:
    path = root / parent_id
    path.mkdir()
    summary = {
        "diagnostic_parent_id": parent_id,
        "fit_parent_ids": [parent_id],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_cv_authorized": False,
        "source_checkpoint_sha256": CHECKPOINT_HASH,
        "final": {"hessian_relative_frobenius": {"median": value}},
        "best_iteration": 10,
        "closure_calls": 20,
        "wall_time_s": 30.0,
        "gpu_peak_memory_mb": 40.0,
        "max_rss_mb": 50.0,
    }
    (path / "summary.json").write_text(json.dumps(summary))


def test_one_parent_analysis_remains_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _summary(root, "a", 0.03)
    _summary(root, "b", 0.20)

    result = analyze(
        root, ["a", "b"], CHECKPOINT_HASH, tmp_path / "out", make_plot=False
    )

    assert result["single_parent_capacity_passed"] is False
    assert result["parent_cv_authorized"] is False
    assert result["aggregate"]["passed_5pct_count"] == 1
    assert (tmp_path / "out" / "per_parent.csv").is_file()


def test_one_parent_analysis_rejects_parent_cv_authorization(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _summary(root, "a", 0.03)
    path = root / "a" / "summary.json"
    summary = json.loads(path.read_text())
    summary["parent_cv_authorized"] = True
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="improperly authorized"):
        analyze(root, ["a"], CHECKPOINT_HASH, tmp_path / "out", make_plot=False)
