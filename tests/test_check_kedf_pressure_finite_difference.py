import math

from scripts.check_kedf_pressure_finite_difference import (
    EV_PER_A3_TO_KBAR,
    KB_EV_PER_K,
    ionic_kinetic_pressure_kbar,
)


def test_ionic_kinetic_pressure_matches_nkbt_over_volume() -> None:
    pressure = ionic_kinetic_pressure_kbar(108, 975.0, 108 * 18.0)
    expected = KB_EV_PER_K * 975.0 / 18.0 * EV_PER_A3_TO_KBAR
    assert math.isclose(pressure, expected, rel_tol=1.0e-14)
