import json
from pathlib import Path

from scripts.analyze_kedf_fd_volume_scan import phase_summary


def test_phase_summary_brackets_zero_pressure(tmp_path: Path) -> None:
    for index, (volume, pressure) in enumerate(
        ((18.0, 4.0), (18.2, -1.0), (18.4, -5.0))
    ):
        point = tmp_path / f"vpa_{index}"
        point.mkdir()
        (point / "pressure_fd_result.json").write_text(
            json.dumps(
                {
                    "base_volume_per_atom_A3": volume,
                    "finite_difference_static_pressure_kbar": pressure - 7.0,
                    "ideal_ionic_pressure_kbar": 7.0,
                    "estimated_total_pressure_kbar": pressure,
                }
            )
        )
    result = phase_summary(tmp_path)
    assert result["status"] == "zero_pressure_prescan_verified"
    assert result["zero_pressure_bracket_A3_per_atom"] == [18.0, 18.2]
    assert 18.0 < result["linear_zero_pressure_estimate_A3_per_atom"] < 18.2
