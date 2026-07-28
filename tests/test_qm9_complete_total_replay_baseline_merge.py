from __future__ import annotations

from scripts.qm9_complete_total_replay_baseline_merge import (
    _failure_reason,
    _quantiles,
)


def test_failure_reason_separates_tensor_stationarity_mismatch() -> None:
    assert (
        _failure_reason(
            {
                "finite": True,
                "strict_converged": False,
                "final_projected_density_gradient_norm": 1.0e-10,
                "tensor_projected_density_gradient_norm": 0.5,
            }
        )
        == "tensor_legacy_stationarity_mismatch"
    )
    assert (
        _failure_reason(
            {
                "finite": True,
                "strict_converged": False,
                "final_projected_density_gradient_norm": 1.0e-3,
                "tensor_projected_density_gradient_norm": 1.0e-3,
            }
        )
        == "density_not_strict"
    )


def test_quantiles_are_finite_and_ordered() -> None:
    result = _quantiles([1.0, 2.0, 3.0, float("nan")])
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["p90"] <= result["max"]

