from __future__ import annotations

import pytest

from scripts.qm9_complete_total_relaxed_q_local_scalar_analysis import summarize_rows


def test_summarize_rows_reports_paired_improvement_and_floor() -> None:
    rows = [
        {
            "molecule_id": "0000001",
            "direction_index": 0,
            "direction_kind": "bond_stretch",
            "role": "train",
            "baseline_q": 0.3,
            "pbe_q": 0.0,
            "predicted_q": 0.01,
        },
        {
            "molecule_id": "0000001",
            "direction_index": 1,
            "direction_kind": "angle_bend",
            "role": "heldout",
            "baseline_q": 1.0,
            "pbe_q": 2.0,
            "predicted_q": 1.8,
        },
    ]
    result = summarize_rows(rows, floor=0.1)
    assert result["strata"]["role:train"]["source_relative_error"]["median"] == pytest.approx(3.0)
    assert result["strata"]["role:train"]["candidate_relative_error"]["median"] == pytest.approx(0.1)
    assert result["strata"]["role:heldout"]["candidate_improved_fraction"] == 1.0
    assert len(result["hard"]) == 2
