import numpy as np
import torch

from mldft.ofdft.implicit_response import (
    block_preconditioned_conjugate_gradient,
    ConstrainedResponseSystem,
    FiniteDifferenceGeometryResponseSystem,
    deflated_preconditioned_conjugate_gradient,
    implicit_symmetric_linear_solve,
    preconditioned_conjugate_gradient,
    preconditioned_minimum_residual,
)


def test_preconditioned_conjugate_gradient():
    matrix = torch.tensor(
        [[5.0, 1.0, 0.0], [1.0, 4.0, 0.5], [0.0, 0.5, 3.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    inverse_diagonal = 1.0 / torch.diag(matrix)
    result = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        preconditioner=lambda vector: inverse_diagonal * vector,
        tolerance=1e-12,
    )

    assert result.converged
    assert result.relative_residual < 1e-12
    assert torch.allclose(result.solution, torch.linalg.solve(matrix, rhs), atol=1e-12, rtol=1e-12)


def test_preconditioned_conjugate_gradient_accepts_exact_initial_guess():
    matrix = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
    rhs = torch.tensor([4.0, -10.0], dtype=torch.float64)
    exact = torch.linalg.solve(matrix, rhs)

    result = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        tolerance=1e-12,
        initial_guess=exact,
    )

    assert result.converged
    assert result.iterations == 0
    torch.testing.assert_close(result.solution, exact)


def test_block_preconditioned_conjugate_gradient_matches_dense_solve():
    matrix = torch.tensor(
        [[5.0, 1.0, 0.0], [1.0, 4.0, 0.5], [0.0, 0.5, 3.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [[1.0, -0.5], [-2.0, 0.2], [0.5, 1.2]],
        dtype=torch.float64,
    )
    inverse_diagonal = 1.0 / torch.diag(matrix)
    result = block_preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        preconditioner=lambda vector: inverse_diagonal * vector,
        tolerance=1.0e-11,
        max_iterations=12,
    )

    assert result.converged
    assert result.relative_residual < 1.0e-11
    torch.testing.assert_close(
        result.solution,
        torch.linalg.solve(matrix, rhs),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_deflated_pcg_resolves_ill_conditioned_low_modes():
    eigenvalues = torch.tensor(
        [1e-8, 1e-7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64
    )
    rhs = torch.arange(1, 9, dtype=torch.float64)
    deflation_vectors = torch.eye(8, dtype=torch.float64)[:, :2]
    result = deflated_preconditioned_conjugate_gradient(
        lambda vector: eigenvalues * vector,
        rhs,
        deflation_vectors,
        tolerance=1e-11,
        max_iterations=12,
    )

    assert result.converged
    assert result.relative_residual < 1e-11
    torch.testing.assert_close(result.solution, rhs / eigenvalues, rtol=1e-11, atol=1e-7)


def test_preconditioned_minres_handles_indefinite_system():
    matrix = torch.tensor(
        [[2.0, 0.5, 0.0], [0.5, -1.0, 0.25], [0.0, 0.25, 3.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    inverse_absolute_diagonal = 1.0 / torch.abs(torch.diag(matrix))
    result = preconditioned_minimum_residual(
        lambda vector: matrix @ vector,
        rhs,
        preconditioner=lambda vector: inverse_absolute_diagonal * vector,
        tolerance=1e-12,
        max_iterations=20,
    )

    assert result.converged
    assert result.relative_residual < 1e-11
    torch.testing.assert_close(
        result.solution, torch.linalg.solve(matrix, rhs), rtol=1e-11, atol=1e-11
    )


def test_implicit_symmetric_linear_solve_parameter_gradient_matches_reference():
    parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    matrix = torch.stack(
        (
            torch.stack((2.0 + parameter, 0.2 + 0.0 * parameter)),
            torch.stack((0.2 + 0.0 * parameter, 1.5 - 0.1 * parameter)),
        )
    )
    rhs = torch.stack((1.0 + parameter, -0.4 * parameter))
    solution = implicit_symmetric_linear_solve(matrix, rhs)
    loss = torch.sum(solution.square())
    actual = torch.autograd.grad(loss, parameter)[0]

    parameter_reference = torch.tensor(
        0.3, dtype=torch.float64, requires_grad=True
    )
    matrix_reference = torch.stack(
        (
            torch.stack(
                (
                    2.0 + parameter_reference,
                    0.2 + 0.0 * parameter_reference,
                )
            ),
            torch.stack(
                (
                    0.2 + 0.0 * parameter_reference,
                    1.5 - 0.1 * parameter_reference,
                )
            ),
        )
    )
    rhs_reference = torch.stack(
        (1.0 + parameter_reference, -0.4 * parameter_reference)
    )
    expected_gradient = torch.autograd.grad(
        torch.sum(
            torch.linalg.solve(matrix_reference, rhs_reference).square()
        ),
        parameter_reference,
    )[0]

    assert torch.allclose(
        solution,
        torch.linalg.solve(matrix.detach(), rhs.detach()),
    )
    torch.testing.assert_close(actual, expected_gradient)


def _quadratic_response_problem():
    coefficient_hessian = torch.tensor(
        [[4.0, 0.5, 0.2], [0.5, 3.0, -0.1], [0.2, -0.1, 2.5]],
        dtype=torch.float64,
    )
    mixed = torch.tensor(
        [[0.2, -0.3], [0.5, 0.1], [-0.4, 0.6]], dtype=torch.float64
    )
    coordinate_hessian = torch.tensor(
        [[1.2, 0.15], [0.15, 0.9]], dtype=torch.float64
    )
    q = torch.tensor([1.0, 1.5, -0.5], dtype=torch.float64)
    coeffs = torch.tensor([0.3, 0.4, -0.2], dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    normalization = q + positions.sum() * 0.0
    energy = (
        0.5 * coeffs @ coefficient_hessian @ coeffs
        + coeffs @ mixed @ positions
        + 0.5 * positions @ coordinate_hessian @ positions
    )
    direction = torch.tensor([0.7, -0.4], dtype=torch.float64)
    system = ConstrainedResponseSystem(
        total_energy=energy,
        coeffs=coeffs,
        positions=positions,
        normalization=normalization,
        n_electron=float(q @ coeffs.detach()),
        multiplier=0.25,
    )
    kkt = torch.zeros((4, 4), dtype=torch.float64)
    kkt[:3, :3] = coefficient_hessian
    kkt[:3, 3] = q
    kkt[3, :3] = q
    rhs = torch.cat((-(mixed @ direction), torch.zeros(1, dtype=torch.float64)))
    expected = torch.linalg.solve(kkt, rhs)
    return system, direction, expected, mixed


def test_tangent_pcg_matches_dense_kkt_and_response_correction():
    system, direction, expected, mixed = _quadratic_response_problem()
    result = system.solve_tangent_pcg(direction, tolerance=1e-12)

    assert result.krylov.converged
    assert result.stationarity_direction_residual < 1e-11
    assert result.constraint_direction_residual < 1e-12
    assert torch.allclose(result.density_response, expected[:3], atol=1e-11, rtol=1e-11)
    assert torch.allclose(result.multiplier_response, expected[3], atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        system.response_correction(
            result.density_response, result.multiplier_response
        ),
        mixed.T @ expected[:3],
        atol=1e-11,
        rtol=1e-11,
    )


def test_direct_implicit_relaxed_hvp_and_parameter_gradient_are_exact():
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    coefficient_curvature = 2.3
    coordinate_curvature = 4.1
    coordinate = torch.tensor(
        [0.2], dtype=torch.float64, requires_grad=True
    )
    normalization = torch.tensor([1.0, 1.0], dtype=torch.float64)
    stationary_value = (
        -float(theta.detach())
        * float(coordinate.detach())
        / coefficient_curvature
    )
    coefficients = torch.tensor(
        [stationary_value, -stationary_value],
        dtype=torch.float64,
        requires_grad=True,
    )
    energy = (
        0.5 * coefficient_curvature * torch.sum(coefficients.square())
        + theta * coordinate[0] * (coefficients[0] - coefficients[1])
        + 0.5 * coordinate_curvature * coordinate[0].square()
    )
    system = ConstrainedResponseSystem(
        total_energy=energy,
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
    parameter_gradient = torch.autograd.grad(hvp.sum(), theta)[0]

    expected_hvp = coordinate_curvature - (
        2.0 * theta.detach().square() / coefficient_curvature
    )
    expected_gradient = -4.0 * theta.detach() / coefficient_curvature
    torch.testing.assert_close(hvp, expected_hvp.reshape_as(hvp))
    torch.testing.assert_close(parameter_gradient, expected_gradient)
    assert response.krylov.method == "dense_direct_implicit_adjoint"
    assert response.stationarity_direction_residual < 1.0e-12
    assert response.constraint_direction_residual < 1.0e-12


def test_minres_matches_dense_kkt():
    system, direction, expected, _ = _quadratic_response_problem()
    result = system.solve_kkt_minres(direction, tolerance=1e-12, max_iterations=100)

    assert result.krylov.converged
    assert result.krylov.relative_residual < 1e-10
    assert np.allclose(
        result.density_response.detach().numpy(), expected[:3].numpy(), atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(result.multiplier_response, expected[3], atol=1e-10, rtol=1e-10)


def test_finite_difference_geometry_response_matches_dense_kkt():
    coefficient_hessian = torch.tensor(
        [[4.0, 0.5, 0.2], [0.5, 3.0, -0.1], [0.2, -0.1, 2.5]],
        dtype=torch.float64,
    )
    mixed = torch.tensor(
        [[0.2, -0.3], [0.5, 0.1], [-0.4, 0.6]], dtype=torch.float64
    )
    coordinate_hessian = torch.tensor(
        [[1.2, 0.15], [0.15, 0.9]], dtype=torch.float64
    )
    normalization = torch.tensor([1.0, 1.5, -0.5], dtype=torch.float64)
    coeffs = torch.tensor([0.3, 0.4, -0.2], dtype=torch.float64)
    positions = np.asarray([0.1, -0.2])
    direction = np.asarray([0.7, -0.4])

    def energy_function(variable, geometry):
        geometry_tensor = torch.as_tensor(geometry, dtype=variable.dtype, device=variable.device)
        return (
            0.5 * variable @ coefficient_hessian @ variable
            + variable @ mixed @ geometry_tensor
            + 0.5 * geometry_tensor @ coordinate_hessian @ geometry_tensor
        )

    system = FiniteDifferenceGeometryResponseSystem(
        energy_function=energy_function,
        coeffs=coeffs,
        positions_bohr=positions,
        normalization=normalization,
        n_electron=float(normalization @ coeffs),
    )
    kkt = torch.zeros((4, 4), dtype=torch.float64)
    kkt[:3, :3] = coefficient_hessian
    kkt[:3, 3] = normalization
    kkt[3, :3] = normalization
    expected = torch.linalg.solve(
        kkt,
        torch.cat(
            (
                -(mixed @ torch.as_tensor(direction)),
                torch.zeros(1, dtype=torch.float64),
            )
        ),
    )

    inverse_diagonal = system.estimate_tangent_inverse_diagonal(probes=64)
    pcg = system.solve_tangent_pcg(
        direction,
        geometry_step_bohr=1e-5,
        tolerance=1e-12,
        tangent_inverse_diagonal=inverse_diagonal,
    )
    minres_result = system.solve_kkt_minres(
        direction, geometry_step_bohr=1e-5, tolerance=1e-12
    )
    tangent_minres_result = system.solve_tangent_minres(
        direction,
        geometry_step_bohr=1e-5,
        tolerance=1e-12,
        tangent_inverse_diagonal=inverse_diagonal,
    )
    dense_result = system.solve_tangent_dense_reference(
        direction, geometry_step_bohr=1e-5
    )

    assert pcg.krylov.converged
    assert minres_result.krylov.converged
    assert tangent_minres_result.krylov.converged
    assert torch.allclose(pcg.density_response, expected[:3], atol=2e-11, rtol=2e-11)
    assert torch.allclose(
        minres_result.density_response, expected[:3], atol=2e-10, rtol=2e-10
    )
    assert torch.allclose(
        tangent_minres_result.density_response,
        expected[:3],
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.allclose(
        dense_result.density_response, expected[:3], atol=2e-11, rtol=2e-11
    )
    assert pcg.stationarity_direction_residual < 1e-10
    assert pcg.constraint_direction_residual < 1e-12
