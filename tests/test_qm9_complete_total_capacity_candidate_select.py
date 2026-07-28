import argparse
import json

from scripts.qm9_complete_total_capacity_candidate_select import select


def _summary(path, hessian, energy, force):
    payload = {
        "test100_accessed": False,
        "per_parent": [
            {
                "relative_frobenius": hessian_value,
                "energy_abs_error_hartree": energy_value,
                "force_mae_hartree_per_bohr": force_value,
            }
            for hessian_value, energy_value, force_value in zip(
                hessian, energy, force, strict=True
            )
        ],
    }
    path.write_text(json.dumps(payload))


def test_selects_largest_floor_that_passes_stable5_only(tmp_path):
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _summary(low, [0.02, 0.04], [1e-4, 2e-4], [2e-4, 3e-4])
    _summary(high, [0.02, 0.06], [1e-4, 2e-4], [2e-4, 3e-4])
    args = argparse.Namespace(
        candidate=[("floor_0p1", 0.1, low), ("floor_1p0", 1.0, high)],
        output_dir=tmp_path / "out",
        hessian_max=0.05,
        energy_median_max=1e-3,
        energy_all_max=2e-3,
        force_median_max=1e-3,
        force_all_max=3e-3,
    )

    result = select(args)

    assert result["selected_candidate"]["name"] == "floor_0p1"
    assert result["validation_parent_metrics_used"] is False
    assert result["test100_accessed"] is False
