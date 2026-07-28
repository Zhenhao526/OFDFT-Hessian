import json
from pathlib import Path

import pytest

from scripts.extrapolate_kedf_zero_pressure_volumes import extrapolate


def write_report(
    path: Path,
    *,
    kedf: str,
    temperature: float,
    solid: float,
    liquid: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "all_confirmations_passed",
                "target_kedf": kedf,
                "temperature_K": temperature,
                "phase_results": [
                    {
                        "phase": "solid",
                        "status": "confirmation_passed",
                        "volume_per_atom_A3": solid,
                    },
                    {
                        "phase": "liquid",
                        "status": "confirmation_passed",
                        "volume_per_atom_A3": liquid,
                    },
                ],
            }
        )
    )


def test_extrapolates_each_phase_independently(tmp_path):
    lower = tmp_path / "lower.json"
    upper = tmp_path / "upper.json"
    write_report(
        lower, kedf="xwm", temperature=900, solid=18.0, liquid=19.0
    )
    write_report(
        upper, kedf="xwm", temperature=975, solid=18.3, liquid=19.15
    )

    result = extrapolate(lower, upper, [1050, 1100])

    assert result["target_kedf"] == "xwm"
    assert result["predictions"][0]["volumes_per_atom_A3"] == pytest.approx(
        {"solid": 18.6, "liquid": 19.3}
    )
    assert result["predictions"][1]["volumes_per_atom_A3"] == pytest.approx(
        {"solid": 18.8, "liquid": 19.4}
    )


def test_rejects_unverified_reference(tmp_path):
    lower = tmp_path / "lower.json"
    upper = tmp_path / "upper.json"
    write_report(
        lower, kedf="xwm", temperature=900, solid=18.0, liquid=19.0
    )
    write_report(
        upper, kedf="xwm", temperature=975, solid=18.3, liquid=19.15
    )
    report = json.loads(upper.read_text())
    report["status"] = "failed"
    upper.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="not verified"):
        extrapolate(lower, upper, [1050])
