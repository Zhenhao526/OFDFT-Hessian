import numpy as np
import pytest

from scripts.qm9_complete_total_capacity_analysis import _hessian_residual_rows


def test_hessian_residual_rows_decomposes_partial_hessian(tmp_path):
    run_dir = tmp_path / "run"
    array_dir = run_dir / "hessian_arrays"
    array_dir.mkdir(parents=True)
    reference = np.zeros((6, 6), dtype=np.float64)
    reference[:, 0] = np.asarray([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    prediction = 0.5 * reference[:, :1]
    np.savez(
        array_dir / "step_0000007_0000001.npz",
        predicted_hessian=prediction,
        pbe_hessian=reference,
    )

    rows = _hessian_residual_rows(run_dir)

    assert len(rows) == 1
    assert rows[0]["step"] == 7
    assert rows[0]["molecule_id"] == "0000001"
    assert rows[0]["relative_frobenius"] == pytest.approx(0.5)
    assert rows[0]["prediction_reference_cosine"] == pytest.approx(1.0)
    assert rows[0]["best_reference_scale"] == pytest.approx(0.5)
    assert rows[0]["orthogonal_relative_frobenius"] == pytest.approx(0.0)
    assert rows[0]["predicted_rigid_sum_max_abs"] == pytest.approx(0.0)
