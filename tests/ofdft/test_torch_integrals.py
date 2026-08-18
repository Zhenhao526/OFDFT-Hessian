import os

import numpy as np
import pytest
import torch

from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_normalization_vector,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
)
from mldft.ofdft.complete_total_training import assign_parameter_only_gradients
from mldft.ofdft.torch_integrals import (
    TORCH_DIRECTIONAL_SECOND_BACKEND,
    TORCH_INTEGRAL_BACKEND,
    TorchAutogradLibcintIntegralProvider,
)


def _require_overlay() -> None:
    if not os.environ.get("MLDFT_DQC_OVERLAY"):
        pytest.skip("MLDFT_DQC_OVERLAY is not configured")


def test_torch_integrals_match_pyscf_and_support_double_backward():
    _require_overlay()
    from pyscf import gto

    positions_np = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    positions = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True
    )
    provider = TorchAutogradLibcintIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
        derivative_step_bohr=0.0,
    )
    values = provider.evaluate(positions)
    assert values.derivative_backend == TORCH_INTEGRAL_BACKEND
    assert values.directional_second_backend == TORCH_DIRECTIONAL_SECOND_BACKEND

    molecule = gto.M(
        atom=[("H", tuple(position)) for position in positions_np],
        unit="Bohr",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    torch.testing.assert_close(
        values.normalization.detach(),
        torch.as_tensor(get_normalization_vector(molecule)),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    torch.testing.assert_close(
        values.overlap.detach(),
        torch.as_tensor(get_overlap_matrix(molecule)),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    torch.testing.assert_close(
        values.coulomb.detach(),
        torch.as_tensor(get_coulomb_matrix(molecule)),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    torch.testing.assert_close(
        values.nuclear_attraction.detach(),
        torch.as_tensor(get_nuclear_attraction_vector(molecule)),
        atol=2.0e-14,
        rtol=2.0e-14,
    )

    direction = torch.arange(1, 7, dtype=positions.dtype).reshape_as(positions)
    normalization_gradient = torch.autograd.grad(
        values.normalization.sum(), positions, create_graph=True, retain_graph=True
    )[0]
    normalization_hvp = torch.autograd.grad(
        torch.sum(normalization_gradient * direction),
        positions,
        retain_graph=True,
    )[0]
    assert torch.max(torch.abs(normalization_gradient)) < 2.0e-14
    assert torch.max(torch.abs(normalization_hvp)) < 2.0e-14

    scalar = (
        0.13 * values.overlap.square().sum()
        + 0.07 * values.coulomb.square().sum()
        + 0.11 * values.normalization.square().sum()
        + 0.17 * values.nuclear_attraction.square().sum()
    )
    gradient = torch.autograd.grad(scalar, positions, create_graph=True)[0]
    hvp = torch.autograd.grad(torch.sum(gradient * direction), positions)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()
    assert torch.linalg.vector_norm(hvp) > 0.0


def test_torch_integral_backend_rejects_coordinate_difference_steps():
    with pytest.raises(ValueError, match="coordinate finite differences are forbidden"):
        TorchAutogradLibcintIntegralProvider(
            atomic_numbers=np.asarray([1, 1]),
            basis="sto-3g",
            derivative_step_bohr=1.0e-4,
        )


def test_torch_coulomb_hvp_backpropagates_only_to_model_parameter():
    _require_overlay()
    positions = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    values = TorchAutogradLibcintIntegralProvider(
        atomic_numbers=np.asarray([1, 1]),
        basis="sto-3g",
    ).evaluate(positions)
    energy = parameter * values.coulomb.square().sum()
    coordinate_gradient = torch.autograd.grad(
        energy, positions, create_graph=True
    )[0]
    direction = torch.arange(1, 7, dtype=positions.dtype).reshape_as(positions)
    hvp = torch.autograd.grad(
        torch.sum(coordinate_gradient * direction),
        positions,
        create_graph=True,
    )[0]

    gradients = assign_parameter_only_gradients(hvp.square().sum(), [parameter])

    assert gradients[0] is not None
    assert torch.isfinite(gradients[0])
    assert positions.grad is None
