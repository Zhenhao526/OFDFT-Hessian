from __future__ import annotations

import torch

from scripts.qm9_complete_total_relaxed_q_feature_jet_audit import (
    _feature_chunk_metadata,
    _hessian_symmetry_metrics,
    _hessian_symmetry_max,
)


def test_hessian_symmetry_max_is_chunk_invariant() -> None:
    hessian = torch.zeros((5, 3, 3), dtype=torch.float64)
    hessian[4, 0, 1] = 0.25
    assert _hessian_symmetry_max(hessian, chunk_size=2) == 0.25
    assert _hessian_symmetry_max(hessian, chunk_size=8) == 0.25
    metrics = _hessian_symmetry_metrics(hessian, chunk_size=2)
    assert metrics["transpose_difference_max_abs"] == 0.25
    assert metrics["asymmetry_max_over_symmetry_max"] == 1.0
    assert metrics["asymmetry_fro_over_symmetry_fro"] == 1.0


def test_feature_chunk_metadata_supports_only_nonoverride_legacy_payload() -> None:
    assert _feature_chunk_metadata(
        {}, {"feature_chunk_size": 32}, "0000001"
    ) == (32, False)
    assert _feature_chunk_metadata(
        {"feature_chunk_size": 8, "feature_chunk_size_override": True},
        {"feature_chunk_size": 8, "feature_chunk_size_override": True},
        "0000002",
    ) == (8, True)

    import pytest

    with pytest.raises(ValueError, match="legacy feature jet"):
        _feature_chunk_metadata(
            {},
            {"feature_chunk_size": 8, "feature_chunk_size_override": True},
            "0000003",
        )
    with pytest.raises(ValueError, match="chunk-size drift"):
        _feature_chunk_metadata(
            {"feature_chunk_size": 8},
            {"feature_chunk_size": 16},
            "0000004",
        )
