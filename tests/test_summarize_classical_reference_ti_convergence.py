import json

from scripts.summarize_classical_reference_ti_convergence import summarize


def test_verified_summary(tmp_path):
    reports = []
    for index, value in enumerate((1.0, 1.5, 2.0)):
        path = tmp_path / f"d{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "verified",
                    "natoms": 128,
                    "target_temperature_k": 900.0,
                    "reference_label": "Einstein",
                    "delta_f_pair_minus_reference_simpson_mev_per_atom": value,
                    "block_standard_error_mev_per_atom": 0.4,
                    "half_drift_mev_per_atom": 0.7,
                    "quadrature_difference_mev_per_atom": 0.2,
                    "minimum_adjacent_effective_sample_fraction": 0.5,
                    "maximum_adjacent_closure_mev_per_atom": 0.3,
                }
            )
        )
        reports.append(path)
    phase = tmp_path / "phase.json"
    phase.write_text(json.dumps({"status": "solid_verified"}))

    result = summarize(reports, phase, "solid", "wt", 2.0)

    assert result["status"] == "verified"
    assert result["integral_discard_spread_mev_per_atom"] == 1.0


def test_spread_failure(tmp_path):
    reports = []
    for index, value in enumerate((1.0, 2.0, 4.0)):
        path = tmp_path / f"d{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "verified",
                    "natoms": 128,
                    "target_temperature_k": 900.0,
                    "reference_label": "sUF",
                    "delta_f_pair_minus_reference_simpson_mev_per_atom": value,
                    "block_standard_error_mev_per_atom": 0.4,
                    "half_drift_mev_per_atom": 0.7,
                    "quadrature_difference_mev_per_atom": 0.2,
                    "minimum_adjacent_effective_sample_fraction": 0.5,
                    "maximum_adjacent_closure_mev_per_atom": 0.3,
                }
            )
        )
        reports.append(path)
    phase = tmp_path / "phase.json"
    phase.write_text(json.dumps({"status": "liquid_verified"}))

    result = summarize(reports, phase, "liquid", "wt", 2.0)

    assert result["status"] == "needs_extension"
    assert not result["checks"]["discard_spread_within_gate"]
