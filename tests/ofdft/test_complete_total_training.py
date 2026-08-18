import pytest
import torch

from mldft.ofdft.implicit_response import ConstrainedResponseSystem
from mldft.ofdft.complete_total_training import (
    assign_multi_task_pcgrad,
    assign_parameter_only_gradients,
    assign_two_task_pcgrad,
    alternating_update_kind,
    central_energy_directional_curvature,
    assemble_hessian_columns,
    central_force_secant_hvp,
    clear_implicit_response_warm_starts,
    consume_implicit_response_diagnostics,
    hutchinson_internal_frobenius_squared_loss,
    differentiable_constrained_density_unroll,
    electron_number_tangent_projection,
    hessian_error_metrics,
    implicit_stationary_density_parameter_response,
    internal_coordinate_projector,
    hutchinson_internal_projected_frobenius_squared_loss,
    low_mode_curvature_loss,
    mixed_absolute_relative_l1,
    mixed_absolute_relative_rmse,
    normalized_energy_l1,
    parameter_gradient_diagnostics,
    stationary_density_parameter_step_prediction,
    structures25_projected_gradient_mse,
)


def test_parameter_only_backward_leaves_coordinate_leaf_unrequested():
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    coordinate = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss = (parameter * coordinate.square()).square()

    gradients = assign_parameter_only_gradients(loss, [parameter])

    torch.testing.assert_close(
        gradients[0], torch.tensor(324.0, dtype=torch.float64)
    )
    torch.testing.assert_close(parameter.grad, gradients[0])
    assert coordinate.grad is None


def test_hutchinson_loss_is_unbiased_for_internal_frobenius_error():
    generator = torch.Generator().manual_seed(17)
    cartesian_dimension = 7
    internal_dimension = 4
    raw = torch.randn(
        cartesian_dimension,
        internal_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    error_hessian = torch.randn(
        cartesian_dimension,
        cartesian_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    expected = torch.sum((error_hessian @ basis) ** 2)

    all_signs = torch.cartesian_prod(
        *[
            torch.tensor([-1.0, 1.0], dtype=torch.float64)
            for _ in range(internal_dimension)
        ]
    )
    estimates = []
    for signs in all_signs:
        direction = basis @ signs
        prediction = error_hessian @ direction
        estimates.append(
            hutchinson_internal_frobenius_squared_loss(
                prediction,
                torch.zeros_like(prediction),
                internal_dimension=internal_dimension,
            )
        )
    torch.testing.assert_close(torch.stack(estimates).mean(), expected)


def test_internal_projected_hutchinson_loss_matches_two_sided_internal_frobenius():
    generator = torch.Generator().manual_seed(23)
    cartesian_dimension = 6
    internal_dimension = 3
    raw = torch.randn(
        cartesian_dimension,
        internal_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    # The canonical helper receives the internal basis as rows: B @ B.T = I.
    basis_columns, _ = torch.linalg.qr(raw, mode="reduced")
    basis = basis_columns.T
    error_hessian = torch.randn(
        cartesian_dimension,
        cartesian_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    expected = torch.linalg.matrix_norm(
        basis @ error_hessian @ basis.T
    ).square()

    all_signs = torch.cartesian_prod(
        *[
            torch.tensor([-1.0, 1.0], dtype=torch.float64)
            for _ in range(internal_dimension)
        ]
    )
    estimates = []
    mean_estimates = []
    for signs in all_signs:
        direction = basis.T @ signs
        prediction = error_hessian @ direction
        target = torch.zeros_like(prediction)
        estimates.append(
            hutchinson_internal_projected_frobenius_squared_loss(
                prediction,
                target,
                basis,
            )
        )
        mean_estimates.append(
            hutchinson_internal_projected_frobenius_squared_loss(
                prediction,
                target,
                basis,
                reduction="mean_internal_matrix",
            )
        )

    torch.testing.assert_close(torch.stack(estimates).mean(), expected)
    torch.testing.assert_close(
        torch.stack(mean_estimates).mean(),
        expected / float(internal_dimension**2),
    )


def test_internal_projected_hutchinson_loss_ignores_external_output_leakage():
    basis = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.zeros(4, dtype=torch.float64)
    internal_prediction = torch.tensor([1.0, -2.0, 0.0, 0.0], dtype=torch.float64)
    external_contamination = torch.tensor(
        [0.0, 0.0, 100.0, -200.0], dtype=torch.float64
    )

    clean = hutchinson_internal_projected_frobenius_squared_loss(
        internal_prediction,
        target,
        basis,
    )
    contaminated = hutchinson_internal_projected_frobenius_squared_loss(
        internal_prediction + external_contamination,
        target,
        basis,
    )

    torch.testing.assert_close(clean, torch.tensor(5.0, dtype=torch.float64))
    torch.testing.assert_close(contaminated, clean)


def test_internal_projected_hutchinson_loss_rejects_nonorthonormal_basis():
    nonorthonormal_basis = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="rows must be orthonormal"):
        hutchinson_internal_projected_frobenius_squared_loss(
            torch.ones(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            nonorthonormal_basis,
        )


def test_canonical_projection_helpers_reject_nonfinite_inputs():
    with pytest.raises(ValueError, match="must be finite"):
        electron_number_tangent_projection(
            torch.tensor([1.0, float("nan")], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="must be finite"):
        hutchinson_internal_projected_frobenius_squared_loss(
            torch.tensor([1.0, float("inf")], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )


def test_multi_task_pcgrad_projects_three_conflicting_tasks() -> None:
    parameter = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    diagnostics = assign_multi_task_pcgrad(
        {
            "energy": parameter[0],
            "force": -parameter[0] + parameter[1],
            "hessian": -parameter[1],
        },
        [parameter],
    )

    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.0, -0.5], dtype=torch.float64),
    )
    assert diagnostics["pcgrad/pairwise_conflict_count"] == 2.0
    assert diagnostics["pcgrad/energy_vs_force_conflict"] == 1.0
    assert diagnostics["pcgrad/force_vs_hessian_conflict"] == 1.0


def test_multi_task_pcgrad_matches_sum_for_aligned_tasks() -> None:
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    diagnostics = assign_multi_task_pcgrad(
        {
            "energy": parameter.sum(),
            "force": 2.0 * parameter.sum(),
            "hessian": 3.0 * parameter.sum(),
        },
        [parameter],
    )

    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 6.0))
    assert diagnostics["pcgrad/pairwise_conflict_count"] == 0.0


def test_two_task_pcgrad_projects_conflicting_gradients():
    parameter = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    diagnostics = assign_two_task_pcgrad(
        parameter[0],
        -parameter[0] + parameter[1],
        [parameter],
    )

    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.5, 1.5], dtype=torch.float64),
    )
    assert diagnostics["pcgrad/conflict"] == 1.0
    assert diagnostics["pcgrad/cosine_before"] == pytest.approx(-(2.0**-0.5))
    assert diagnostics["pcgrad/projected_task_dot"] == pytest.approx(0.5)


def test_two_task_pcgrad_matches_sum_when_gradients_align():
    parameter = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    diagnostics = assign_two_task_pcgrad(
        torch.sum(parameter**2),
        torch.sum(parameter),
        [parameter],
    )

    torch.testing.assert_close(
        parameter.grad,
        2.0 * parameter.detach() + torch.ones_like(parameter),
    )
    assert diagnostics["pcgrad/conflict"] == 0.0


def test_two_task_pcgrad_caps_second_task_gradient_norm() -> None:
    parameter = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    diagnostics = assign_two_task_pcgrad(
        parameter[0],
        100.0 * parameter[1],
        [parameter],
        max_second_to_first_norm_ratio=1.0,
    )

    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert diagnostics["pcgrad/second_task_balance_scale"] == pytest.approx(0.01)
    assert diagnostics[
        "pcgrad/balanced_second_to_first_norm_ratio"
    ] == pytest.approx(1.0)


def test_alternating_update_kind_supports_replay_sparse_schedule():
    assert [
        alternating_update_kind(
            step, hvp_update_period=2, replay_update_period=4
        )
        for step in range(1, 9)
    ] == ["hvp", "hvp", "hvp", "replay", "hvp", "hvp", "hvp", "replay"]
    assert [
        alternating_update_kind(step, hvp_update_period=2)
        for step in range(1, 5)
    ] == ["replay", "hvp", "replay", "hvp"]


def test_alternating_update_kind_counts_match_reported_schedule_metadata():
    schedule = [
        alternating_update_kind(
            step,
            hvp_update_period=1,
            replay_update_period=5,
        )
        for step in range(1, 11)
    ]

    assert schedule == [
        "hvp",
        "hvp",
        "hvp",
        "hvp",
        "replay",
        "hvp",
        "hvp",
        "hvp",
        "hvp",
        "replay",
    ]
    assert schedule.count("hvp") == 8
    assert schedule.count("replay") == 2


@pytest.mark.parametrize(
    ("cumulative_step", "hvp_period", "replay_period"),
    [(0, 1, None), (1, 0, None), (1, 1, 1)],
)
def test_alternating_update_kind_rejects_ambiguous_schedule_metadata(
    cumulative_step, hvp_period, replay_period
):
    with pytest.raises(ValueError):
        alternating_update_kind(
            cumulative_step,
            hvp_update_period=hvp_period,
            replay_update_period=replay_period,
        )


def test_central_energy_directional_curvature_is_scalar_energy_second_difference():
    h = 2.0e-3
    curvature = torch.tensor(3.25, dtype=torch.float64)
    slope = torch.tensor(-0.7, dtype=torch.float64)
    offset = torch.tensor(4.2, dtype=torch.float64)

    def energy(displacement):
        return offset + slope * displacement + 0.5 * curvature * displacement**2

    prediction = central_energy_directional_curvature(
        energy(0.0), energy(h), energy(-h), h
    )
    torch.testing.assert_close(prediction, curvature, rtol=1e-10, atol=1e-10)


def test_constrained_density_unroll_preserves_number_and_parameter_response():
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([1.0, 1.0], dtype=torch.float64)
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def energy(coefficients):
        target = torch.stack([parameter, -parameter])
        return 0.5 * torch.sum((coefficients - target) ** 2)

    coefficients = differentiable_constrained_density_unroll(
        initial,
        normalization,
        2.0,
        energy,
        steps=1,
        learning_rate=1.0,
    )
    assert torch.allclose(
        torch.dot(normalization, coefficients),
        torch.tensor(2.0, dtype=torch.float64),
    )
    response = torch.autograd.grad(coefficients[0], parameter)[0]
    assert torch.allclose(response, torch.tensor(1.0, dtype=torch.float64))


def test_implicit_stationary_density_map_has_exact_parameter_response():
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stationary = torch.stack((parameter.detach(), -parameter.detach()))

    def energy(coefficients):
        target = torch.stack((parameter, -parameter))
        return 0.5 * torch.sum((coefficients - target) ** 2)

    coefficients = implicit_stationary_density_parameter_response(
        stationary,
        normalization,
        0.0,
        energy,
        (parameter,),
        tolerance=1.0e-12,
        max_iterations=10,
        damping=0.0,
        diagonal_probes=0,
    )
    response = torch.autograd.grad(coefficients[0], parameter)[0]
    diagnostics = consume_implicit_response_diagnostics()

    torch.testing.assert_close(coefficients, stationary)
    torch.testing.assert_close(response, torch.tensor(1.0, dtype=torch.float64))
    assert diagnostics[0]["right_hand_side_norm"] > 0.0
    assert diagnostics[0]["scaled_residual"] <= diagnostics[0]["relative_residual"]


def test_implicit_stationary_density_direct_parameter_adjoint():
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stationary = torch.stack((parameter.detach(), -parameter.detach()))

    def energy(coefficients):
        target = torch.stack((parameter, -parameter))
        return 0.5 * torch.sum((coefficients - target) ** 2)

    coefficients = implicit_stationary_density_parameter_response(
        stationary,
        normalization,
        0.0,
        energy,
        (parameter,),
        tolerance=1.0e-12,
        max_iterations=10,
        damping=0.0,
        solver="direct",
    )
    response = torch.autograd.grad(coefficients[0], parameter)[0]
    diagnostics = consume_implicit_response_diagnostics()

    torch.testing.assert_close(response, torch.tensor(1.0, dtype=torch.float64))
    assert diagnostics[0]["converged"] is True
    assert diagnostics[0]["solver"] == "direct"
    assert diagnostics[0]["method"] == "dense_direct_parameter_adjoint"
    assert diagnostics[0]["iterations"] == 1


def test_implicit_stationary_density_parameter_response_reuses_adjoint():
    clear_implicit_response_warm_starts()
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stationary = torch.stack((parameter.detach(), -parameter.detach()))

    def energy(coefficients):
        target = torch.stack((parameter, -parameter))
        return 0.5 * torch.sum((coefficients - target) ** 2)

    iterations = []
    for _ in range(2):
        coefficients = implicit_stationary_density_parameter_response(
            stationary,
            normalization,
            0.0,
            energy,
            (parameter,),
            tolerance=1.0e-12,
            max_iterations=10,
            damping=0.0,
            warm_start_key="quadratic-test",
        )
        torch.autograd.grad(coefficients[0], parameter)
        iterations.append(consume_implicit_response_diagnostics()[-1]["iterations"])

    assert iterations == [1, 0]
    clear_implicit_response_warm_starts()


def test_stationary_density_parameter_step_prediction_matches_exact_update():
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    parameter_step = torch.tensor(-0.03, dtype=torch.float64)
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stationary = torch.stack(
        (parameter.detach(), -parameter.detach())
    )

    def energy(coefficients):
        target = torch.stack((parameter, -parameter))
        return 0.5 * torch.sum((coefficients - target).square())

    prediction = stationary_density_parameter_step_prediction(
        stationary,
        normalization,
        energy,
        (parameter,),
        (parameter_step,),
    )
    torch.testing.assert_close(
        prediction,
        torch.stack((parameter_step, -parameter_step)),
    )
    torch.testing.assert_close(
        torch.dot(normalization, prediction),
        torch.zeros((), dtype=torch.float64),
    )


def test_central_force_secant_hvp_preserves_parameter_gradient():
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([1.0, -2.0], dtype=torch.float64)
    step = 1.0e-3
    plus_force = -parameter * step * direction
    minus_force = parameter * step * direction

    hvp = central_force_secant_hvp(plus_force, minus_force, step)
    assert torch.allclose(hvp, parameter * direction)
    torch.sum(hvp).backward()
    assert torch.allclose(parameter.grad, direction.sum())


def test_relaxed_force_secant_hvp_implicit_parameter_gradient_matches_reoptimization_fd():
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    coefficient_curvature = 2.3
    coordinate_curvature = 4.1
    displacement = 1.0e-3
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def graph_force(coordinate_value: float) -> torch.Tensor:
        coordinate = torch.tensor(
            coordinate_value, dtype=torch.float64, requires_grad=True
        )
        stationary_value = -float(theta.detach()) * coordinate_value / coefficient_curvature
        stationary = torch.tensor(
            [stationary_value, -stationary_value], dtype=torch.float64
        )

        def energy(coefficients: torch.Tensor) -> torch.Tensor:
            return (
                0.5 * coefficient_curvature * torch.sum(coefficients**2)
                + theta * coordinate * (coefficients[0] - coefficients[1])
                + 0.5 * coordinate_curvature * coordinate**2
            )

        coefficients = implicit_stationary_density_parameter_response(
            stationary,
            normalization,
            0.0,
            energy,
            (theta,),
            tolerance=1.0e-12,
            max_iterations=10,
            damping=0.0,
        )
        return -torch.autograd.grad(
            energy(coefficients), coordinate, create_graph=True
        )[0]

    graph_hvp = central_force_secant_hvp(
        graph_force(displacement),
        graph_force(-displacement),
        displacement,
    )
    graph_parameter_gradient = torch.autograd.grad(graph_hvp, theta)[0]

    def strict_hvp(theta_value: float) -> float:
        def force(coordinate_value: float) -> float:
            stationary_value = (
                -theta_value * coordinate_value / coefficient_curvature
            )
            return -(
                theta_value * (2.0 * stationary_value)
                + coordinate_curvature * coordinate_value
            )

        return -(
            force(displacement) - force(-displacement)
        ) / (2.0 * displacement)

    parameter_step = 1.0e-5
    finite_difference_gradient = (
        strict_hvp(float(theta.detach()) + parameter_step)
        - strict_hvp(float(theta.detach()) - parameter_step)
    ) / (2.0 * parameter_step)
    expected_hvp = coordinate_curvature - (
        2.0 * float(theta.detach()) ** 2 / coefficient_curvature
    )
    expected_parameter_gradient = -4.0 * float(theta.detach()) / coefficient_curvature

    torch.testing.assert_close(
        graph_hvp,
        torch.tensor(expected_hvp, dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        graph_parameter_gradient,
        torch.tensor(finite_difference_gradient, dtype=torch.float64),
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    torch.testing.assert_close(
        graph_parameter_gradient,
        torch.tensor(expected_parameter_gradient, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_analytic_relaxed_hvp_parameter_gradient_includes_stationary_density_response():
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    coefficient_curvature = 2.3
    cubic_coefficient = 0.1
    coordinate_curvature = 4.1
    coordinate_value = 0.2
    coordinate = torch.tensor(
        [coordinate_value], dtype=torch.float64, requires_grad=True
    )
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def stationary_x(theta_value: float, position_value: float) -> float:
        discriminant = (
            coefficient_curvature**2
            - 16.0 * cubic_coefficient * theta_value * position_value
        )
        return (
            -coefficient_curvature + discriminant**0.5
        ) / (8.0 * cubic_coefficient)

    x_value = stationary_x(float(theta.detach()), coordinate_value)
    stationary = torch.tensor(
        [x_value, -x_value], dtype=torch.float64
    )

    def energy_function(coefficients: torch.Tensor) -> torch.Tensor:
        difference = coefficients[0] - coefficients[1]
        return (
            0.5
            * coefficient_curvature
            * torch.sum(coefficients.square())
            + theta * coordinate[0] * difference
            + (cubic_coefficient / 3.0) * difference**3
            + 0.5 * coordinate_curvature * coordinate[0].square()
        )

    coefficients = implicit_stationary_density_parameter_response(
        stationary,
        normalization,
        0.0,
        energy_function,
        (theta,),
        tolerance=1.0e-12,
        max_iterations=20,
        damping=0.0,
    )
    system = ConstrainedResponseSystem(
        total_energy=energy_function(coefficients),
        coeffs=coefficients,
        positions=coordinate,
        normalization=normalization,
        n_electron=0.0,
        multiplier=0.0,
    )
    direction = torch.ones_like(coordinate)
    response = system.solve_tangent_direct_implicit(
        direction, create_graph=True
    )
    hvp = system.relaxed_hvp(direction, response, create_graph=True)
    graph_gradient = torch.autograd.grad(hvp.sum(), theta)[0]

    def strict_hvp(theta_value: float) -> float:
        x = stationary_x(theta_value, coordinate_value)
        return coordinate_curvature - (
            2.0
            * theta_value**2
            / (coefficient_curvature + 8.0 * cubic_coefficient * x)
        )

    parameter_step = 1.0e-5
    finite_difference_gradient = (
        strict_hvp(float(theta.detach()) + parameter_step)
        - strict_hvp(float(theta.detach()) - parameter_step)
    ) / (2.0 * parameter_step)

    torch.testing.assert_close(
        hvp,
        torch.tensor(
            [strict_hvp(float(theta.detach()))], dtype=torch.float64
        ),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        graph_gradient,
        torch.tensor(finite_difference_gradient, dtype=torch.float64),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_mixed_absolute_relative_l1_uses_reference_floor():
    prediction = torch.tensor([0.02, -0.02], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    loss = mixed_absolute_relative_l1(
        prediction,
        target,
        absolute_scale=0.1,
        relative_floor=0.01,
        relative_fraction=0.5,
    )
    assert torch.allclose(loss, torch.tensor(1.1, dtype=torch.float64))


def test_normalized_energy_l1_is_dimensionless_mean_absolute_total_energy_error():
    prediction = torch.tensor([-75.2, -74.7], dtype=torch.float64)
    target = torch.tensor([-75.0, -75.0], dtype=torch.float64)

    loss = normalized_energy_l1(
        prediction,
        target,
        absolute_scale_hartree=0.1,
    )

    # mean(|-0.2|, |+0.3|) / 0.1 Ha
    torch.testing.assert_close(loss, torch.tensor(2.5, dtype=torch.float64))


def test_electron_number_tangent_projection_removes_only_chemical_potential_gauge():
    normalization = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    tangent = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    vector = tangent + 7.0 * normalization

    projected = electron_number_tangent_projection(vector, normalization)

    torch.testing.assert_close(projected, tangent)
    torch.testing.assert_close(
        torch.dot(projected, normalization),
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        electron_number_tangent_projection(projected, normalization),
        projected,
    )


def test_structures25_gradient_loss_matches_label_difference_not_center_diagnostic():
    normalization = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    target = torch.tensor([-3.0, 4.0, 1.0], dtype=torch.float64)
    tangent_error = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    prediction = target + tangent_error + 5.0 * normalization

    loss = structures25_projected_gradient_mse(
        prediction,
        target,
        normalization,
        absolute_scale=2.0,
    )

    # The chemical-potential component is projected out.  Only the three
    # Structures25 label-error components enter the mean; a self-consistent
    # center residual is deliberately not an argument to this canonical loss.
    expected = torch.mean((tangent_error / 2.0) ** 2)
    torch.testing.assert_close(loss, expected)


def test_structures25_gradient_loss_has_no_gradient_along_number_constraint():
    normalization = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    prediction = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(prediction)

    loss = structures25_projected_gradient_mse(
        prediction,
        target,
        normalization,
        absolute_scale=0.01,
    )
    gradient = torch.autograd.grad(loss, prediction)[0]

    torch.testing.assert_close(
        torch.dot(gradient, normalization),
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_mixed_absolute_relative_rmse_matches_relative_column_l2():
    prediction = torch.tensor([3.0, 0.0], dtype=torch.float64)
    target = torch.tensor([0.0, 4.0], dtype=torch.float64)
    loss = mixed_absolute_relative_rmse(
        prediction,
        target,
        absolute_scale=1.0,
        relative_floor=0.01,
        relative_fraction=1.0,
    )
    expected = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target)
    torch.testing.assert_close(loss, expected)


def test_hessian_column_assembly_and_metrics_report_antisymmetry():
    prediction = torch.tensor([[1.0, 2.0], [1.0, 3.0]], dtype=torch.float64)
    assembled = assemble_hessian_columns([prediction[:, 0], prediction[:, 1]])
    assert torch.equal(assembled, prediction)
    metrics = hessian_error_metrics(assembled, torch.eye(2, dtype=torch.float64))
    assert metrics["relative_frobenius"] > 0
    assert metrics["antisymmetric_over_symmetric_frobenius"] > 0
    assert torch.allclose(
        metrics["symmetry_max_abs"], torch.tensor(1.0, dtype=torch.float64)
    )


def test_low_mode_curvature_penalizes_wrong_sign_more():
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target = torch.tensor([2.0, 0.0], dtype=torch.float64)
    right = low_mode_curvature_loss(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        target,
        direction,
        curvature_floor=0.1,
    )
    wrong = low_mode_curvature_loss(
        torch.tensor([-1.0, 0.0], dtype=torch.float64),
        target,
        direction,
        curvature_floor=0.1,
    )
    assert wrong > right


def test_parameter_gradient_diagnostics_does_not_write_parameter_grad():
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    losses = {
        "left": torch.sum(parameter**2),
        "right": torch.sum(parameter),
    }
    diagnostics = parameter_gradient_diagnostics(losses, [parameter])
    assert parameter.grad is None
    assert diagnostics["gradient_norm/left"] > diagnostics["gradient_norm/right"]
    assert 0.0 < diagnostics["gradient_cosine/left_vs_right"] <= 1.0


def test_parameter_gradient_diagnostics_reports_group_contributions():
    left = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    diagnostics = parameter_gradient_diagnostics(
        {"loss": torch.sum(left**2) + torch.sum(right**2)},
        [left, right],
        parameter_names=["encoder.weight", "readout.weight"],
    )

    fractions = [
        diagnostics["gradient_group_squared_fraction/loss/encoder"],
        diagnostics["gradient_group_squared_fraction/loss/readout"],
    ]
    assert sum(fractions) == pytest.approx(1.0)
    assert diagnostics["gradient_group_norm/loss/readout"] > 0.0


def test_parameter_gradient_diagnostics_accepts_released_microbatch_graphs():
    parameter = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    unused_parameter = torch.tensor(
        [3.0], dtype=torch.float64, requires_grad=True
    )
    replay_loss = torch.sum(parameter**2)
    microbatch_losses = [
        torch.sum((parameter * scale) ** 2) for scale in (1.0, 2.0, 3.0)
    ]
    expected_hvp = torch.autograd.grad(
        torch.stack(microbatch_losses).mean(),
        parameter,
        retain_graph=True,
    )[0]
    accumulated_hvp = torch.zeros_like(parameter)
    for loss in microbatch_losses:
        gradient = torch.autograd.grad(loss / len(microbatch_losses), parameter)[0]
        accumulated_hvp.add_(gradient.detach())

    diagnostics = parameter_gradient_diagnostics(
        {"replay": replay_loss},
        [parameter, unused_parameter],
        precomputed_gradients={"hvp": (accumulated_hvp, None)},
    )

    torch.testing.assert_close(accumulated_hvp, expected_hvp)
    assert diagnostics["gradient_norm/hvp"] == pytest.approx(
        float(torch.linalg.vector_norm(expected_hvp))
    )
    assert "gradient_cosine/replay_vs_hvp" in diagnostics
    assert parameter.grad is None
    assert unused_parameter.grad is None
    # The microbatch graphs are gone. A separately constructed replay/force
    # graph must still support its one real backward pass after diagnostics.
    replay_loss.backward()
    torch.testing.assert_close(parameter.grad, 2.0 * parameter.detach())
    assert unused_parameter.grad is None


def test_internal_coordinate_projector_removes_translations_and_is_idempotent():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 0.8, 0.4]],
        dtype=torch.float64,
    )
    masses = torch.tensor([1.0, 12.0, 16.0], dtype=torch.float64)
    projector = internal_coordinate_projector(positions, masses)
    assert torch.allclose(projector, projector.T, atol=1.0e-12)
    assert torch.allclose(projector @ projector, projector, atol=1.0e-12)
    for axis in range(3):
        translation = torch.zeros_like(positions)
        translation[:, axis] = torch.sqrt(masses)
        assert torch.linalg.vector_norm(projector @ translation.reshape(-1)) < 1.0e-10
