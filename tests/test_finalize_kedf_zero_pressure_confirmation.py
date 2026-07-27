from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_kedf_zero_pressure_confirmation import finalize


def write_confirmation(root: Path, method: str, posthoc: bool) -> None:
    root.mkdir(parents=True)
    (root / "confirmation_summary.json").write_text(
        json.dumps(
            {
                "target_kedf": method,
                "temperature_K": 900.0,
                "results": [
                    {
                        "phase": phase,
                        "run": str(root / phase),
                        "volume_per_atom_A3": volume,
                        "max_step": 300,
                        "phase_status": f"{phase}_verified",
                        "nearest_neighbor_A": 2.2,
                        "temperature_last_half_K": {"mean": 900.0},
                        "pressure_last_half_kbar": (
                            {} if posthoc else {"mean": pressure}
                        ),
                        "requires_posthoc_pressure": posthoc,
                        "status": "passed",
                    }
                    for phase, volume, pressure in (
                        ("solid", 18.0, -1.0),
                        ("liquid", 19.0, 1.0),
                    )
                ],
            }
        )
    )


def test_lkt_analytic_pressures_finalize(tmp_path: Path) -> None:
    root = tmp_path / "lkt"
    write_confirmation(root, "lkt", posthoc=False)
    result = finalize(root)
    assert result["status"] == "all_confirmations_passed"
    assert all(
        item["pressure_source"] == "analytic_stress"
        for item in result["phase_results"]
    )


def test_xwm_requires_verified_posthoc_pressures(tmp_path: Path) -> None:
    root = tmp_path / "xwm"
    samples = tmp_path / "samples"
    write_confirmation(root, "xwm", posthoc=True)
    for phase, pressure in (("solid", -1.2), ("liquid", 0.8)):
        phase_root = samples / phase
        phase_root.mkdir(parents=True)
        (phase_root / "pressure_samples_summary.json").write_text(
            json.dumps(
                {
                    "status": "snapshot_pressure_verified",
                    "pressure_kbar": {
                        "n": 5,
                        "mean": pressure,
                        "standard_error": 0.5,
                    },
                    "samples": [],
                }
            )
        )
    result = finalize(root, samples)
    assert result["status"] == "all_confirmations_passed"
    assert all(
        item["pressure_source"] == "posthoc_finite_difference"
        for item in result["phase_results"]
    )


def test_xwm_without_posthoc_data_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "xwm"
    write_confirmation(root, "xwm", posthoc=True)
    result = finalize(root)
    assert result["status"] == "confirmation_failed"
