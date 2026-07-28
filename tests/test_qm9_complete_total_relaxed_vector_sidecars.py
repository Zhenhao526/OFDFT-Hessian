from __future__ import annotations

import numpy as np
import pytest

from scripts.qm9_complete_total_relaxed_vector_sidecars import (
    _selected_step,
    _vector_spread,
)


def test_selected_step_requires_exact_requested_value() -> None:
    steps = np.asarray([3.0e-6, 1.0e-5, 3.0e-5])
    assert _selected_step(steps, 1.0e-5) == 1
    with pytest.raises(ValueError, match="missing"):
        _selected_step(steps, 2.0e-5)


def test_vector_spread_uses_vector_norm_and_floor() -> None:
    vectors = np.asarray([[1.0, 0.0], [1.2, 0.0], [0.8, 0.0]])
    assert _vector_spread(vectors, floor=0.1) == pytest.approx(0.2)
    tiny = np.asarray([[0.01, 0.0], [-0.01, 0.0]])
    assert _vector_spread(tiny, floor=0.1) == pytest.approx(0.1)
