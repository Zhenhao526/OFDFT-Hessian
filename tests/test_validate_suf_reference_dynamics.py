from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_suf_reference_dynamics import validate


def write_inputs(
    root: Path,
    *,
    minimum_neighbor: float = 1.64,
    late_slope: float = 0.00043,
) -> None:
    root.mkdir()
    summary = {
        "minimum_distance_angstrom": minimum_neighbor,
        "samples": 2001,
        "stable": True,
        "steps": 20000,
        "target_temperature_k": 900.0,
        "temperature_mean_k": 901.3,
    }
    phase = {
        "CSP_final": {
            "median_A2": 19.5,
            "ordered_fraction_CSP_lt_2_5": 0.0,
        },
        "MSD_A2": {
            "last": 5.65,
            "late_slope_per_step": late_slope,
        },
        "expected_phase": "liquid",
        "minimum_nearest_neighbor_A": minimum_neighbor,
        "status": "liquid_not_verified",
        "steps": 20000,
        "temperature_K": {"late_mean": 904.6},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "phase_analysis.json").write_text(json.dumps(phase), encoding="utf-8")
    (root / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "checkpoint.json").write_text("{}\n", encoding="utf-8")


def test_auxiliary_suf_allows_stable_contacts_below_pair_gate(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_inputs(run_dir, minimum_neighbor=1.64)

    result = validate(
        run_dir,
        expected_steps=20000,
        target_temperature_k=900.0,
    )

    assert result["status"] == "verified"
    assert result["checks"]["auxiliary_nearest_neighbor_stable"] is True
    assert result["observed"]["generic_pair_phase_status"] == "liquid_not_verified"
    assert set(result["input_sha256"]) == {
        "summary",
        "phase_analysis",
        "trajectory",
        "checkpoint",
    }


def test_auxiliary_suf_rejects_numerical_close_contact(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_inputs(run_dir, minimum_neighbor=1.49)

    result = validate(
        run_dir,
        expected_steps=20000,
        target_temperature_k=900.0,
    )

    assert result["status"] == "validation_failed"
    assert result["checks"]["auxiliary_nearest_neighbor_stable"] is False


def test_auxiliary_suf_rejects_nondiffusive_liquid(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_inputs(run_dir, late_slope=0.00005)

    result = validate(
        run_dir,
        expected_steps=20000,
        target_temperature_k=900.0,
    )

    assert result["status"] == "validation_failed"
    assert result["checks"]["liquid_late_msd_slope"] is False
