from __future__ import annotations

import numpy as np

from scripts.qm9_complete_total_mace_checkpoint_hessian_geometry import (
    correction_geometry,
)


def test_correction_geometry_reports_amplitude_and_alignment() -> None:
    source = np.zeros((2, 2))
    reference = np.diag([2.0, 1.0])
    predicted = 0.25 * reference

    metrics = correction_geometry(predicted, source, reference)

    assert np.isclose(metrics["learned_over_target_norm"], 0.25)
    assert np.isclose(metrics["learned_target_cosine"], 1.0)
    assert np.isclose(
        metrics["residual_frobenius"], 0.75 * np.linalg.norm(reference)
    )
