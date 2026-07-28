from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_kedf_fusion_enthalpy_manifest import build_manifest


def write_pair(
    root: Path,
    zero_pressure: Path,
    *,
    method: str,
    temperature: float = 975.0,
    stress_available: bool,
) -> None:
    root.mkdir()
    phases = [
        {
            "phase": phase,
            "run": str(root / phase),
            "volume_per_atom_A3": volume,
        }
        for phase, volume in (("solid", 18.4), ("liquid", 18.9))
    ]
    (root / "confirmation_manifest.json").write_text(
        json.dumps(
            {
                "target_kedf": method,
                "temperature_K": temperature,
                "target_pressure_kbar": 0.0,
                "steps": 3000,
                "stress_available": stress_available,
                "phases": phases,
            }
        )
    )
    (root / "confirmation_summary.json").write_text(
        json.dumps(
            {
                "status": "volume_confirmation_verified",
                "target_kedf": method,
                "temperature_K": temperature,
                "results": [
                    {"phase": phase["phase"], "status": "passed"}
                    for phase in phases
                ],
            }
        )
    )
    zero_pressure.write_text(
        json.dumps(
            {
                "status": "all_confirmations_passed",
                "target_kedf": method,
                "temperature_K": temperature,
                "phase_results": [
                    {
                        **phase,
                        "status": "confirmation_passed",
                        "pressure_source": (
                            "analytic_stress"
                            if stress_available
                            else "posthoc_finite_difference"
                        ),
                    }
                    for phase in phases
                ],
            }
        )
    )


@pytest.mark.parametrize(
    ("method", "stress_available"),
    [("lkt", True), ("xwm", False)],
)
def test_builds_method_specific_pressure_gate(
    tmp_path: Path, method: str, stress_available: bool
) -> None:
    root = tmp_path / "run"
    pressure = tmp_path / "pressure.json"
    write_pair(
        root,
        pressure,
        method=method,
        stress_available=stress_available,
    )

    result = build_manifest([(root, pressure)])
    point = result["points"][0]

    assert result["target_kedf"] == method
    assert point["trajectory_pressure_required"] is stress_available
    assert point["zero_pressure_verified"] is True
    assert point["zero_pressure_provenance"]["sha256"]


def test_rejects_volume_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    pressure = tmp_path / "pressure.json"
    write_pair(root, pressure, method="xwm", stress_available=False)
    report = json.loads(pressure.read_text())
    report["phase_results"][1]["volume_per_atom_A3"] = 19.0
    pressure.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="liquid.*volumes differ"):
        build_manifest([(root, pressure)])


def test_rejects_unverified_zero_pressure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    pressure = tmp_path / "pressure.json"
    write_pair(root, pressure, method="lkt", stress_available=True)
    report = json.loads(pressure.read_text())
    report["status"] = "confirmation_failed"
    pressure.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="zero-pressure.*not verified"):
        build_manifest([(root, pressure)])
