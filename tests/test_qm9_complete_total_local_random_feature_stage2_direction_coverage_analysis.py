from __future__ import annotations

import pytest

from scripts.qm9_complete_total_local_random_feature_stage2_direction_coverage_analysis import (
    _distribution,
)


def test_coverage_distribution() -> None:
    assert _distribution([1.0, 2.0, 3.0, 4.0])["p90"] == pytest.approx(3.7)


def test_coverage_distribution_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty or non-finite"):
        _distribution([])

