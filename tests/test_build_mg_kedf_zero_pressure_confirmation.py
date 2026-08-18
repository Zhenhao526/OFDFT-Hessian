import json

import pytest

from scripts.build_mg_kedf_zero_pressure_confirmation import build


def make_run(tmp_path, phase, *, status="physical_verified"):
    run = tmp_path / phase
    run.mkdir()
    (run / "mg_phase_md_manifest.json").write_text(
        json.dumps(
            {
                "method": "wt",
                "temperature_K": 900.0,
                "volume_per_atom_A3": 23.18 if phase == "solid" else 24.1,
            }
        )
    )
    (run / "physical_gate.json").write_text(
        json.dumps(
            {
                "status": status,
                "method": "wt",
                "expected_phase": phase,
                "hcp_phase_status": f"{phase}_verified",
                "checks": {"a": True, "b": True},
                "temperature_last_half_K": {"mean": 902.0},
                "pressure_last_half_kbar": {"mean": 0.5},
                "minimum_nearest_neighbor_A": 2.3,
            }
        )
    )
    return run


def test_build_verified_summary(tmp_path):
    result = build(make_run(tmp_path, "solid"), make_run(tmp_path, "liquid"), "wt")
    assert result["status"] == "all_confirmations_passed"
    assert result["element"] == "Mg"
    assert [item["phase"] for item in result["phase_results"]] == ["solid", "liquid"]


def test_rejects_failed_phase(tmp_path):
    solid = make_run(tmp_path, "solid", status="failed")
    liquid = make_run(tmp_path, "liquid")
    with pytest.raises(ValueError, match="physical gate"):
        build(solid, liquid, "wt")
