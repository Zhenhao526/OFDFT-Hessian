from __future__ import annotations

import numpy as np

from scripts.qm9_complete_total_relaxed_q_sidecar_audit import (
    _relative_difference,
    assess_directional_q,
)


def _branch(*, q: float, implicit_q: float | None = None) -> dict[str, object]:
    direction = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    values = np.asarray([q, q * 1.001, q * 0.999, q], dtype=np.float64)
    scan = np.zeros((4, 1, 3), dtype=np.float64)
    scan[:, 0, 0] = values
    implicit = np.asarray([[implicit_q if implicit_q is not None else q, 0.0, 0.0]])
    return {
        "direction": direction,
        "curvature_steps_bohr": np.asarray([1e-5, 3e-5, 3e-6, 1e-3]),
        "relaxed_hvp_step_scan": scan,
        "implicit_hvp": implicit,
        "pbe_hvp": np.asarray([[0.75, 0.0, 0.0]]),
        "energy_curvature_closure": q * 1.001,
        "force_curvature_closure": q,
    }


def _gate() -> dict[str, object]:
    return {
        "curvature_floor_hartree_per_bohr2": 0.1,
        "selected_force_secant_step_bohr": 1e-5,
        "small_steps_bohr": [3e-6, 1e-5, 3e-5],
        "maximum_direction_mismatch": 1e-10,
        "maximum_small_step_relative_spread": 0.05,
        "maximum_branch_relative_spread": 0.05,
        "maximum_implicit_relaxed_relative_difference": 0.05,
        "maximum_energy_force_relative_difference": 0.05,
    }


def test_relative_difference_uses_symmetric_floor() -> None:
    assert _relative_difference(0.001, -0.001, 0.1) == 0.02
    assert _relative_difference(2.0, 1.0, 0.1) == 0.5


def test_directional_q_gate_accepts_stable_branches() -> None:
    result = assess_directional_q([_branch(q=0.5) for _ in range(3)], _gate())

    assert result["eligible"] is True
    assert result["checks"] == {
        "direction_match": True,
        "small_step_stability": True,
        "branch_stability": True,
        "implicit_relaxed_agreement": True,
        "energy_force_closure": True,
        "pbe_branch_match": True,
        "finite": True,
    }
    assert np.isclose(result["correction_q_hartree_per_bohr2"], 0.25)


def test_directional_q_gate_rejects_implicit_disagreement() -> None:
    branches = [_branch(q=0.5), _branch(q=0.5), _branch(q=0.5, implicit_q=0.2)]

    result = assess_directional_q(branches, _gate())

    assert result["eligible"] is False
    assert result["checks"]["implicit_relaxed_agreement"] is False
