from __future__ import annotations

import json

import pytest

from scripts.qm9_complete_total_local_random_feature_stage2_width_analysis import (
    _discover_width_summaries,
)


def test_width_discovery_uses_artifact_metadata(tmp_path) -> None:
    for directory, width in (("width_zero", 0), ("width_sixteen", 16)):
        path = tmp_path / directory
        path.mkdir()
        (path / "summary.json").write_text(
            json.dumps({"feature_subset": {"selected_width_per_scale": width}})
            + "\n"
        )

    result = _discover_width_summaries(tmp_path, [0, 16])

    assert set(result) == {0, 16}


def test_width_discovery_rejects_incomplete_grid(tmp_path) -> None:
    path = tmp_path / "width_zero"
    path.mkdir()
    (path / "summary.json").write_text(
        '{"feature_subset": {"selected_width_per_scale": 0}}\n'
    )
    with pytest.raises(ValueError, match="artifact set drift"):
        _discover_width_summaries(tmp_path, [0, 16])

