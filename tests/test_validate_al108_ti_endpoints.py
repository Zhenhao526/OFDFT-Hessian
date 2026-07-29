from scripts.validate_al108_ti_endpoints import (
    RY_TO_EV,
    baseline_energy_by_component_step,
    centered_max_abs,
    component_run_reached_steps,
)


def test_baseline_energy_maps_one_based_md_steps_to_zero_based_components():
    rows = [
        {"step": 1, "potential_Ry": -2.0},
        {"step": 2, "potential_Ry": -1.5},
    ]

    assert baseline_energy_by_component_step(rows) == {
        0: -2.0 * RY_TO_EV,
        1: -1.5 * RY_TO_EV,
    }


def test_component_step_count_includes_zero_based_first_step():
    rows = [{"step": step} for step in range(10)]

    assert component_run_reached_steps(rows, 10)
    assert not component_run_reached_steps(rows[:-1], 10)
    assert not component_run_reached_steps([], 1)


def test_centered_error_accepts_a_stable_energy_bookkeeping_offset():
    assert centered_max_abs([0.00225, 0.00227, 0.00223]) < 3.0e-5


def test_centered_error_rejects_a_configuration_dependent_reference_mismatch():
    assert centered_max_abs([0.00225, 0.00227, 0.00280]) > 1.0e-4
