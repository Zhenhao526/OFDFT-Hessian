import numpy as np
import pytest
import torch

from mldft.ofdft.geometry_integrals import (
    AutodiffPySCFIntegralProvider,
    FiniteDifferencePySCFIntegralProvider,
    classical_energy_from_bundle,
    linearized_integral_tensor,
)


def test_pyscfad_directional_integral_curvature_matches_scalar_oracle():
    pytest.importorskip("pyscfad")
    positions_np = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    direction = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    direction /= np.linalg.norm(direction)
    provider = AutodiffPySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
        derivative_step_bohr=0.0,
    )
    bundle = provider.evaluate_with_directional_second_derivatives(
        positions_np,
        direction,
        directional_step_bohr=0.0,
    )
    assert bundle.derivative_backend == "pyscfad_jax_autodiff"
    assert bundle.directional_second_backend == (
        "pyscfad_jacfwd_of_jvp_with_traceable_rinv"
    )
    assert bundle.derivative_step_bohr == 0.0
    assert bundle.directional_second_step_bohr == 0.0

    numerical_reference = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
        derivative_step_bohr=1.0e-4,
    ).evaluate_with_derivatives(positions_np)
    np.testing.assert_allclose(
        bundle.values.nuclear_attraction,
        numerical_reference.values.nuclear_attraction,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        bundle.derivatives.nuclear_attraction,
        numerical_reference.derivatives.nuclear_attraction,
        atol=2.0e-7,
        rtol=2.0e-7,
    )

    coeffs_np = np.asarray([0.35, 0.42])
    positions = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True
    )
    energy = classical_energy_from_bundle(
        torch.tensor(coeffs_np, dtype=torch.float64),
        positions,
        torch.tensor([1, 1]),
        bundle,
    )
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hvp = torch.autograd.grad(
        gradient,
        positions,
        grad_outputs=torch.as_tensor(direction, dtype=torch.float64),
    )[0]
    autodiff_curvature = float(
        torch.sum(hvp * torch.as_tensor(direction, dtype=torch.float64))
    )

    oracle_step = 3.0e-4
    plus_values = provider.evaluate(positions_np + oracle_step * direction)
    center_values = provider.evaluate(positions_np)
    minus_values = provider.evaluate(positions_np - oracle_step * direction)
    direction_flat = direction.reshape(-1)
    directional = bundle.directional_second_derivatives
    assert directional is not None
    component_curvatures = {
        "coulomb_ad": float(
            0.5
            * coeffs_np
            @ np.tensordot(direction_flat, directional.coulomb, axes=([0], [0]))
            @ coeffs_np
        ),
        "coulomb_fd": float(
            0.5
            * coeffs_np
            @ (
                plus_values.coulomb
                - 2.0 * center_values.coulomb
                + minus_values.coulomb
            )
            @ coeffs_np
            / oracle_step**2
        ),
        "attraction_ad": float(
            coeffs_np
            @ np.tensordot(
                direction_flat, directional.nuclear_attraction, axes=([0], [0])
            )
        ),
        "attraction_fd": float(
            coeffs_np
            @ (
                plus_values.nuclear_attraction
                - 2.0 * center_values.nuclear_attraction
                + minus_values.nuclear_attraction
            )
            / oracle_step**2
        ),
        "nuclear_ad": float(
            direction_flat @ directional.nuclear_repulsion
        ),
        "nuclear_fd": float(
            (
                plus_values.nuclear_repulsion
                - 2.0 * center_values.nuclear_repulsion
                + minus_values.nuclear_repulsion
            )
            / oracle_step**2
        ),
    }
    scalar_oracle = (
        _classical_energy_direct(
            provider, positions_np + oracle_step * direction, coeffs_np
        )
        - 2.0 * _classical_energy_direct(provider, positions_np, coeffs_np)
        + _classical_energy_direct(
            provider, positions_np - oracle_step * direction, coeffs_np
        )
    ) / oracle_step**2
    assert np.isclose(
        autodiff_curvature, scalar_oracle, atol=2.0e-5, rtol=2.0e-5
    ), component_curvatures


def test_pyscfad_backend_rejects_coordinate_difference_steps():
    pytest.importorskip("pyscfad")
    with pytest.raises(ValueError, match="finite differences are forbidden"):
        AutodiffPySCFIntegralProvider(
            atomic_numbers=np.asarray([1, 1]),
            basis="sto-3g",
            derivative_step_bohr=1.0e-4,
        )


def _classical_energy_direct(provider, positions, coeffs):
    values = provider.evaluate(positions)
    return float(
        0.5 * coeffs @ values.coulomb @ coeffs
        + coeffs @ values.nuclear_attraction
        + values.nuclear_repulsion
    )


def test_directional_quadratic_integral_tensor_has_requested_hvp():
    generator = np.random.default_rng(13)
    dimension = 5
    output_shape = (2, 3)
    value = generator.normal(size=output_shape)
    first_derivative = generator.normal(size=(dimension, *output_shape))
    symmetric_hessian = generator.normal(
        size=(dimension, dimension, *output_shape)
    )
    symmetric_hessian = 0.5 * (
        symmetric_hessian
        + np.swapaxes(symmetric_hessian, 0, 1)
    )
    direction = generator.normal(size=dimension)
    requested_hvp = np.tensordot(
        symmetric_hessian, direction, axes=([1], [0])
    )
    positions = torch.zeros(dimension, dtype=torch.float64, requires_grad=True)
    tensor = linearized_integral_tensor(
        value,
        first_derivative,
        positions,
        directional_second_derivative=requested_hvp,
        direction=direction,
    )

    first = torch.autograd.grad(
        tensor,
        positions,
        grad_outputs=torch.ones_like(tensor),
        create_graph=True,
    )[0]
    actual = torch.autograd.grad(
        first,
        positions,
        grad_outputs=torch.as_tensor(direction, dtype=torch.float64),
    )[0]
    expected = torch.as_tensor(
        requested_hvp.reshape(dimension, -1).sum(axis=1),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        tensor,
        torch.as_tensor(value, dtype=torch.float64),
    )
    torch.testing.assert_close(
        first,
        torch.as_tensor(
            first_derivative.reshape(dimension, -1).sum(axis=1),
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(actual, expected)


def test_geometry_integral_bundle_total_classical_gradient_matches_direct_fd():
    positions_np = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
        derivative_step_bohr=1e-4,
    )
    bundle = provider.evaluate_with_derivatives(positions_np)
    coeffs_np = np.asarray([0.35, 0.42])
    positions = torch.tensor(positions_np, dtype=torch.float64, requires_grad=True)
    coeffs = torch.tensor(coeffs_np, dtype=torch.float64)

    energy = classical_energy_from_bundle(
        coeffs,
        positions,
        torch.tensor([1, 1]),
        bundle,
    )
    gradient = torch.autograd.grad(energy, positions)[0].detach().numpy()

    outer_step = 3e-4
    finite_difference = np.zeros_like(positions_np)
    for coordinate in range(positions_np.size):
        plus = positions_np.reshape(-1).copy()
        minus = positions_np.reshape(-1).copy()
        plus[coordinate] += outer_step
        minus[coordinate] -= outer_step
        finite_difference.reshape(-1)[coordinate] = (
            _classical_energy_direct(
                provider, plus.reshape(positions_np.shape), coeffs_np
            )
            - _classical_energy_direct(
                provider, minus.reshape(positions_np.shape), coeffs_np
            )
        ) / (2.0 * outer_step)

    assert np.allclose(gradient, finite_difference, atol=2e-7, rtol=2e-6)
    assert np.allclose(gradient.sum(axis=0), 0.0, atol=2e-9, rtol=0)


def test_physical_basis_classical_directional_curvature_matches_scalar_fd():
    positions_np = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    direction = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    direction /= np.linalg.norm(direction)
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
        derivative_step_bohr=1.0e-4,
    )
    outer_step = 3.0e-4
    bundle = provider.evaluate_with_directional_second_derivatives(
        positions_np,
        direction,
        directional_step_bohr=outer_step,
    )
    coeffs_np = np.asarray([0.35, 0.42])
    positions = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True
    )
    energy = classical_energy_from_bundle(
        torch.tensor(coeffs_np, dtype=torch.float64),
        positions,
        torch.tensor([1, 1]),
        bundle,
    )
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hvp = torch.autograd.grad(
        gradient,
        positions,
        grad_outputs=torch.as_tensor(direction, dtype=torch.float64),
    )[0]
    autograd_curvature = float(
        torch.sum(hvp * torch.as_tensor(direction, dtype=torch.float64))
    )
    scalar_curvature = (
        _classical_energy_direct(
            provider, positions_np + outer_step * direction, coeffs_np
        )
        - 2.0 * _classical_energy_direct(provider, positions_np, coeffs_np)
        + _classical_energy_direct(
            provider, positions_np - outer_step * direction, coeffs_np
        )
    ) / outer_step**2

    assert np.isclose(
        autograd_curvature, scalar_curvature, atol=2.0e-5, rtol=2.0e-5
    )


def test_integral_derivatives_include_translation_invariant_constraint_and_pulay_terms():
    positions = np.asarray(
        [[-0.4, 0.1, -0.8], [0.3, -0.2, 0.7], [0.8, 0.4, 0.1]]
    )
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1, 1]),
        basis="sto-3g",
        charge=1,
        derivative_step_bohr=1e-4,
    )
    bundle = provider.evaluate_with_derivatives(positions)

    assert bundle.derivatives.normalization.shape == (9, 3)
    assert bundle.derivatives.overlap.shape == (9, 3, 3)
    assert bundle.derivatives.coulomb.shape == (9, 3, 3)
    assert bundle.derivatives.nuclear_attraction.shape == (9, 3)
    assert bundle.derivatives.nuclear_repulsion.shape == (9,)

    for derivative in (
        bundle.derivatives.normalization,
        bundle.derivatives.overlap,
        bundle.derivatives.coulomb,
        bundle.derivatives.nuclear_attraction,
    ):
        reshaped = derivative.reshape(3, 3, *derivative.shape[1:])
        assert np.allclose(reshaped.sum(axis=0), 0.0, atol=2e-7, rtol=0)

    # Normalized atom-centred auxiliary basis integrals are individually translation invariant.
    assert np.max(np.abs(bundle.derivatives.normalization)) < 1e-8
    # The nonzero off-centre overlap derivative is an explicit moving-basis/Pulay contribution.
    assert np.max(np.abs(bundle.derivatives.overlap)) > 1e-4

    finite_difference_bundle = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers=np.asarray([1, 1, 1]),
        basis="sto-3g",
        charge=1,
        derivative_step_bohr=1e-4,
        analytic_overlap_derivative=False,
    ).evaluate_with_derivatives(positions)
    assert np.allclose(
        bundle.derivatives.overlap,
        finite_difference_bundle.derivatives.overlap,
        atol=2e-9,
        rtol=2e-8,
    )


def test_parallel_integral_derivatives_match_serial():
    positions = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    common = {
        "atomic_numbers": np.asarray([1, 1]),
        "basis": "sto-3g",
        "derivative_step_bohr": 1e-4,
    }
    serial = FiniteDifferencePySCFIntegralProvider(
        **common, derivative_workers=1
    ).evaluate_with_derivatives(positions)
    parallel = FiniteDifferencePySCFIntegralProvider(
        **common, derivative_workers=2
    ).evaluate_with_derivatives(positions)

    for field in ("normalization", "overlap", "coulomb", "nuclear_attraction"):
        np.testing.assert_allclose(
            getattr(parallel.values, field), getattr(serial.values, field), atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            getattr(parallel.derivatives, field),
            getattr(serial.derivatives, field),
            atol=0.0,
            rtol=0.0,
        )
    assert parallel.values.nuclear_repulsion == serial.values.nuclear_repulsion
    np.testing.assert_allclose(
        parallel.derivatives.nuclear_repulsion,
        serial.derivatives.nuclear_repulsion,
        atol=0.0,
        rtol=0.0,
    )
