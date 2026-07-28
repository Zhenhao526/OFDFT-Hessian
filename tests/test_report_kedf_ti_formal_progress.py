from pathlib import Path

from scripts.report_kedf_ti_formal_progress import report


def test_reports_active_and_waiting_windows(tmp_path: Path):
    active = tmp_path / "solid" / "lambda_0p000"
    waiting = tmp_path / "liquid" / "lambda_0p000"
    output = active / "OUT.test"
    output.mkdir(parents=True)
    waiting.mkdir(parents=True)
    (active / "run.stdout").write_text(
        "STEP OF MOLECULAR DYNAMICS: 1\n TN0 x\n TN1 y\n"
    )
    (output / "running_md.log").write_text(
        " STEP OF MOLECULAR DYNAMICS: 1\n"
        " Energy (Ry) Potential (Ry) Kinetic (Ry) Temperature (K)\n"
        " -1.0 -2.0 1.0 900.0\n"
    )

    result = report(tmp_path, 3000)

    assert result["summary"]["windows"] == 2
    assert result["summary"]["active"] == 1
    assert result["summary"]["waiting"] == 1
    assert result["summary"]["completed"] == 0
    assert result["summary"]["maximum_remaining_seconds"] > 0
    active_row = next(row for row in result["windows"] if row["status"] == "running")
    assert active_row["max_step"] == 1
    assert active_row["temperature_recent_K"]["mean"] == 900.0
    assert active_row["electronic_iterations_recent"]["mean"] == 2
