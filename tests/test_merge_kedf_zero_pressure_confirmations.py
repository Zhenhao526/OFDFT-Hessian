from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_kedf_zero_pressure_confirmations import merge


def write_summary(
    path: Path,
    method: str,
    temperature: float,
    phase: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "target_kedf": method,
                "temperature_K": temperature,
                "phase_results": [
                    {
                        "phase": phase,
                        "status": "confirmation_passed",
                    }
                ],
                "status": "confirmation_failed",
            }
        )
    )


def test_merge_selects_independently_passed_phases(tmp_path: Path) -> None:
    solid = tmp_path / "solid.json"
    liquid = tmp_path / "liquid.json"
    write_summary(solid, "lkt", 900.0, "solid")
    write_summary(liquid, "lkt", 900.0, "liquid")
    result = merge([solid, liquid])
    assert result["status"] == "all_confirmations_passed"
    assert [item["phase"] for item in result["phase_results"]] == [
        "solid",
        "liquid",
    ]


def test_merge_rejects_mixed_methods(tmp_path: Path) -> None:
    solid = tmp_path / "solid.json"
    liquid = tmp_path / "liquid.json"
    write_summary(solid, "xwm", 900.0, "solid")
    write_summary(liquid, "lkt", 900.0, "liquid")
    with pytest.raises(ValueError, match="different KEDFs"):
        merge([solid, liquid])
