from __future__ import annotations

import json

import pytest

from scripts.qm9_complete_total_local_random_feature_stage2_hashed_subspace_analysis import (
    _discover_dimension_summaries,
    _distribution,
)


def test_hashed_subspace_discovery_uses_internal_dimension(tmp_path) -> None:
    for directory, dimension in (("dim_small", 1024), ("dim_large", 4096)):
        path = tmp_path / directory
        path.mkdir()
        (path / "summary.json").write_text(
            json.dumps({"hashed_subspace": {"subspace_dimension": dimension}})
            + "\n"
        )
    assert set(_discover_dimension_summaries(tmp_path, [1024, 4096])) == {
        1024,
        4096,
    }


def test_hashed_subspace_distribution_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="empty or non-finite"):
        _distribution([float("inf")])

