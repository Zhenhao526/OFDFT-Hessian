from types import SimpleNamespace

import numpy as np
import torch

from mldft.ofdft.conservative_force import (
    DifferentiableGeometry,
    evaluate_total_ofdft_force,
    replace_coordinate_gradient,
)
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.geometry_integrals import (
    GeometryIntegralBundle,
    GeometryIntegralDerivatives,
    GeometryIntegralValues,
)


def test_replace_coordinate_gradient_preserves_value_and_other_first_derivatives():
    coordinate = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    original = coordinate.square().sum() + coefficient * coordinate[0]
    target = torch.tensor([3.0, -4.0], dtype=torch.float64)

    replaced = replace_coordinate_gradient(original, coordinate, target)
    coordinate_gradient, coefficient_gradient = torch.autograd.grad(
        replaced, (coordinate, coefficient)
    )

    torch.testing.assert_close(replaced, original)
    torch.testing.assert_close(coordinate_gradient, target)
    torch.testing.assert_close(coefficient_gradient, coordinate[0])


def test_total_force_rebuilds_classical_energy_in_physical_basis():
    positions_np = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    positions = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True
    )
    coeffs = torch.tensor([0.35, 0.42], dtype=torch.float64, requires_grad=True)
    values = GeometryIntegralValues(
        normalization=np.asarray([1.0, 1.0]),
        overlap=np.eye(2),
        coulomb=np.asarray([[1.2, 0.1], [0.1, 0.8]]),
        nuclear_attraction=np.asarray([-0.7, -0.9]),
        nuclear_repulsion=1.0 / 1.4,
    )
    derivatives = GeometryIntegralDerivatives(
        normalization=np.zeros((6, 2)),
        overlap=np.zeros((6, 2, 2)),
        coulomb=np.zeros((6, 2, 2)),
        nuclear_attraction=np.zeros((6, 2)),
        nuclear_repulsion=np.zeros(6),
    )
    bundle = GeometryIntegralBundle(
        values=values,
        derivatives=derivatives,
        positions_bohr=positions_np,
        derivative_step_bohr=1.0e-4,
    )

    # Deliberately inconsistent transformed classical inputs make it observable
    # whether the force path uses the registered physical-basis bundle.
    sample = SimpleNamespace(
        coeffs=torch.tensor(
            [9.0, -4.0], dtype=torch.float64, requires_grad=True
        ),
        coulomb_matrix=torch.eye(2, dtype=torch.float64) * 17.0,
        nuclear_attraction_vector=torch.tensor(
            [5.0, 6.0], dtype=torch.float64
        ),
        pos=positions,
        atomic_numbers=torch.tensor([1, 1]),
    )
    geometry = DifferentiableGeometry(
        sample=sample,
        positions=positions,
        coeffs=coeffs,
        bundle=bundle,
    )
    result = evaluate_total_ofdft_force(
        FunctionalFactory("hartree", "nuclear_attraction"),
        geometry,
        n_electron=float(values.normalization @ coeffs.detach().numpy()),
        create_graph=True,
    )

    expected_hartree = 0.5 * coeffs @ (
        torch.as_tensor(values.coulomb, dtype=torch.float64) @ coeffs
    )
    expected_attraction = coeffs @ torch.as_tensor(
        values.nuclear_attraction, dtype=torch.float64
    )
    torch.testing.assert_close(result.energies["hartree"], expected_hartree)
    torch.testing.assert_close(
        result.energies["nuclear_attraction"], expected_attraction
    )
    assert torch.isfinite(result.force).all()
