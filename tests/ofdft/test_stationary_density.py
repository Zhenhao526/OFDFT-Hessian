import torch

from mldft.ofdft.stationary_density import (
    refine_constrained_coefficients_lbfgs,
    refine_constrained_coefficients_newton_pcg,
)


def test_tangent_lbfgs_reaches_constrained_quadratic_stationary_point():
    hessian = torch.tensor(
        [[4.0, 0.3, -0.2], [0.3, 2.5, 0.4], [-0.2, 0.4, 3.0]],
        dtype=torch.float64,
    )
    linear = torch.tensor([-1.0, 0.7, -0.2], dtype=torch.float64)
    normalization = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float64)
    target = 1.7
    initial = torch.tensor([0.2, 0.6, 1.2], dtype=torch.float64)

    result = refine_constrained_coefficients_lbfgs(
        initial,
        normalization,
        target,
        lambda coeffs: 0.5 * coeffs @ hessian @ coeffs + linear @ coeffs,
        tolerance=1e-8,
        max_iterations=100,
    )
    kkt = torch.zeros((4, 4), dtype=torch.float64)
    kkt[:3, :3] = hessian
    kkt[:3, 3] = normalization
    kkt[3, :3] = normalization
    rhs = torch.cat((-linear, torch.tensor([target], dtype=torch.float64)))
    expected = torch.linalg.solve(kkt, rhs)[:3]

    assert result.converged
    assert result.final_projected_gradient_norm < 1e-8
    assert abs(result.constraint_residual) < 1e-12
    torch.testing.assert_close(result.coeffs, expected, atol=1e-9, rtol=1e-9)


def test_newton_pcg_reaches_constrained_quadratic_stationary_point():
    hessian = torch.tensor(
        [[4.0, 0.3, -0.2], [0.3, 2.5, 0.4], [-0.2, 0.4, 3.0]],
        dtype=torch.float64,
    )
    linear = torch.tensor([-1.0, 0.7, -0.2], dtype=torch.float64)
    normalization = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float64)
    target = 1.7
    initial = torch.tensor([0.2, 0.6, 1.2], dtype=torch.float64)

    result = refine_constrained_coefficients_newton_pcg(
        initial,
        normalization,
        target,
        lambda coeffs: 0.5 * coeffs @ hessian @ coeffs + linear @ coeffs,
        tolerance=1e-10,
        max_newton_iterations=3,
        krylov_tolerance=1e-12,
        diagonal_probes=4,
        damping=0.0,
    )

    assert result.converged
    assert result.final_projected_gradient_norm < 1e-10
    assert abs(result.constraint_residual) < 1e-12
    assert result.krylov_iterations
