import math

from scripts.design_kedf_ti_production import (
    choose_steps,
    projected_integral_standard_error,
    projected_window_standard_error,
    simpson_weights,
)


def test_simpson_weights_uniform_nine_window_grid():
    weights = simpson_weights([index / 8 for index in range(9)])
    assert math.isclose(sum(weights), 1.0)
    assert weights[0] == weights[-1]
    assert weights[1] == 4 * weights[0]


def test_projected_standard_errors():
    window_error, effective = projected_window_standard_error(
        2.0, 5.0, 2000, 0.25
    )
    assert math.isclose(effective, 50.0)
    assert math.isclose(window_error, 2.0 / math.sqrt(50.0))
    assert math.isclose(
        projected_integral_standard_error([0.5, 0.5], [1.0, 1.0]),
        math.sqrt(0.5),
    )


def test_choose_first_passing_candidate():
    selected = choose_steps(
        [1000, 2000],
        [0.5, 0.5],
        [2.0, 2.0],
        [5.0, 5.0],
        retained_fraction=0.25,
        minimum_effective_samples=30.0,
        maximum_window_standard_error=1.0,
        maximum_integral_standard_error=0.5,
    )
    assert selected["selected"]["steps"] == 2000
    assert not all(selected["candidates"][0]["checks"].values())
    assert all(selected["selected"]["checks"].values())
