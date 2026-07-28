import json
from pathlib import Path

import pytest

from scripts.extrapolate_kedf_zero_pressure_volumes_residual_corrected import (
    extrapolate,
)


def write_report(
    path: Path,
    *,
    temperature: float,
    solid_volume: float,
    solid_pressure: float,
    liquid_volume: float,
    liquid_pressure: float,
    final: bool,
) -> None:
    rows = []
    for phase, volume, pressure in (
        ("solid", solid_volume, solid_pressure),
        ("liquid", liquid_volume, liquid_pressure),
    ):
        rows.append(
            {
                "phase": phase,
                "phase_status": f"{phase}_verified",
                "status": "confirmation_passed" if final else "passed",
                "volume_per_atom_A3": volume,
                "pressure_kbar": {"mean": pressure},
                "checks": {"phase_verified": True},
            }
        )
    path.write_text(
        json.dumps(
            {
                "target_kedf": "lkt",
                "temperature_K": temperature,
                "status": "all_confirmations_passed" if final else "diagnostic",
                "phase_results": rows,
            }
        )
    )


def test_corrects_residual_pressure_before_temperature_extrapolation(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "lower.json"
    upper = tmp_path / "upper.json"
    slope_a = tmp_path / "slope_a.json"
    slope_b = tmp_path / "slope_b.json"
    write_report(
        lower,
        temperature=900,
        solid_volume=18.4,
        solid_pressure=-2.0,
        liquid_volume=18.7,
        liquid_pressure=0.0,
        final=True,
    )
    write_report(
        upper,
        temperature=975,
        solid_volume=18.5,
        solid_pressure=-1.0,
        liquid_volume=18.9,
        liquid_pressure=-2.0,
        final=True,
    )
    write_report(
        slope_a,
        temperature=975,
        solid_volume=18.4,
        solid_pressure=0.0,
        liquid_volume=18.8,
        liquid_pressure=0.0,
        final=False,
    )
    write_report(
        slope_b,
        temperature=975,
        solid_volume=18.5,
        solid_pressure=-4.0,
        liquid_volume=18.9,
        liquid_pressure=-2.0,
        final=False,
    )

    result = extrapolate(
        lower,
        upper,
        (slope_a, slope_b),
        (slope_a, slope_b),
        [1050],
    )

    assert result["pressure_volume_slopes"]["solid"][
        "dP_dV_kbar_per_A3_per_atom"
    ] == pytest.approx(-40.0)
    assert result["pressure_volume_slopes"]["liquid"][
        "dP_dV_kbar_per_A3_per_atom"
    ] == pytest.approx(-20.0)
    assert result["predictions"][0]["volumes_per_atom_A3"] == pytest.approx(
        {"solid": 18.6, "liquid": 18.9}
    )


def test_rejects_positive_pressure_volume_slope(tmp_path: Path) -> None:
    lower = tmp_path / "lower.json"
    upper = tmp_path / "upper.json"
    slope_a = tmp_path / "slope_a.json"
    slope_b = tmp_path / "slope_b.json"
    for path, volume, pressure in (
        (lower, 18.0, 0.0),
        (upper, 18.1, 0.0),
        (slope_a, 18.0, 0.0),
        (slope_b, 18.1, 1.0),
    ):
        write_report(
            path,
            temperature=900 if path in {lower, slope_a, slope_b} else 975,
            solid_volume=volume,
            solid_pressure=pressure,
            liquid_volume=volume + 1.0,
            liquid_pressure=-pressure,
            final=path in {lower, upper},
        )
    with pytest.raises(ValueError, match="dP/dV must be negative"):
        extrapolate(
            lower,
            upper,
            (slope_a, slope_b),
            (slope_a, slope_b),
            [1050],
        )
