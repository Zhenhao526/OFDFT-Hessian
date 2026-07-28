from __future__ import annotations

import pytest

from scripts.qm9_complete_total_local_random_feature_stage2_ridge_analysis import (
    _discover_ridge_summaries,
    _distribution,
)


def test_distribution_reports_frozen_quantiles() -> None:
    result = _distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == {
        "min": 1.0,
        "median": 3.0,
        "mean": 3.0,
        "p90": pytest.approx(4.6),
        "max": 5.0,
    }


def test_distribution_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="empty or non-finite"):
        _distribution([1.0, float("nan")])


def test_ridge_discovery_uses_artifact_value_not_directory_format(tmp_path) -> None:
    first = tmp_path / "ridge_1e-7"
    second = tmp_path / "ridge_0.001"
    first.mkdir()
    second.mkdir()
    (first / "summary.json").write_text('{"effective_ridge": 1e-7}\n')
    (second / "summary.json").write_text('{"effective_ridge": 1e-3}\n')

    result = _discover_ridge_summaries(tmp_path, [1e-7, 1e-3])

    assert result == {
        1e-7: first / "summary.json",
        1e-3: second / "summary.json",
    }
