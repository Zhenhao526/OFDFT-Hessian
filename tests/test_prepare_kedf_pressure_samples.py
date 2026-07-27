from scripts.prepare_kedf_pressure_samples import summarize_pressures


def test_pressure_summary_reports_standard_error() -> None:
    result = summarize_pressures([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert result["n"] == 5
    assert result["mean"] == 0.0
    assert 0.0 < result["standard_error"] < result["sd"]
