from __future__ import annotations

import json

import pytest

from scripts.qm9_complete_total_structured_subspace_analysis import _load_frozen


def test_structured_analysis_rejects_test_or_validation_access(tmp_path) -> None:
    valid = {
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(valid))
    assert _load_frozen(path) == valid

    for key, value in (
        ("validation_accessed", True),
        ("test100_accessed", True),
        ("test100_evaluations_used", 1),
    ):
        invalid = dict(valid)
        invalid[key] = value
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="access drift|evaluation count drift"):
            _load_frozen(path)
