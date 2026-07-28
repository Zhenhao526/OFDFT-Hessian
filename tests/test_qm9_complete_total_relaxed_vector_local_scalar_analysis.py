from __future__ import annotations

import pytest

from scripts.qm9_complete_total_relaxed_vector_local_scalar_analysis import (
    summarize_rows,
)


def test_summarize_rows_keeps_worst_removal_diagnostic_only() -> None:
    rows = [
        {
            "role": "heldout",
            "direction_kind": "bond_stretch",
            "relative_l2_with_floor": 0.1,
            "source_relative_l2_with_floor": 0.2,
            "improved": True,
            "mae_hartree_per_bohr2": 0.01,
        },
        {
            "role": "heldout",
            "direction_kind": "bond_stretch",
            "relative_l2_with_floor": 10.0,
            "source_relative_l2_with_floor": 9.0,
            "improved": False,
            "mae_hartree_per_bohr2": 1.0,
        },
        {
            "role": "train",
            "direction_kind": "angle_bend",
            "relative_l2_with_floor": 0.05,
            "source_relative_l2_with_floor": 2.0,
            "improved": True,
            "mae_hartree_per_bohr2": 0.02,
        },
    ]
    result = summarize_rows(rows)
    assert result["strata"]["role:heldout"][
        "candidate_improved_fraction"
    ] == 0.5
    assert result["heldout_sensitivity"]["all"]["median"] == pytest.approx(5.05)
    assert result["heldout_sensitivity"][
        "without_single_largest_posthoc_diagnostic_only"
    ]["median"] == pytest.approx(0.1)
