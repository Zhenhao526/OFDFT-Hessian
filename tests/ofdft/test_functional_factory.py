import pytest
import torch
from lightning import LightningModule
from pyscf import dft, gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.energies import Energies, TensorEnergies
from mldft.ofdft.functional_factory import (
    FunctionalFactory,
    constrained_energy_lagrangian,
    nuclear_repulsion_energy_tensor,
)
from mldft.utils.molecules import build_mol_with_even_tempered_basis
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
def test_functional_factory():
    """Basis test for the functional factory using a dummy torch module."""

    mol = gto.M(atom="O 0 0 0; C 0 1 0", basis="6-31G(2df,p)")
    mol = build_mol_with_even_tempered_basis(mol, beta=2.5)
    n = mol.nao

    grid = dft.Grids(mol)
    grid.build()
    ao = torch.Tensor(dft.numint.eval_ao(mol, grid.coords))

    class DummyModule(LightningModule):
        """Dummy module for testing the functional factory."""

        def __init__(self, target_key: str):
            """Initialize the dummy module."""
            super().__init__()
            self.target_key = target_key

        def sample_forward(self, sample: OFData):
            """Apply the DummyModule to the sample.

            Use 42 as energy and a tensor full of ones in the correct size as gradient.
            """
            sample.add_item("pred_energy", torch.tensor(42), Representation.SCALAR)
            sample.add_item(
                "pred_gradient", torch.full(size=(n,), fill_value=1), Representation.GRADIENT
            )
            return sample

    model = DummyModule(target_key="dummy")
    model.training = True

    func_factory = FunctionalFactory(model, "libxc_LDA")
    functional = func_factory.construct(mol, torch.zeros((n, n)), torch.zeros(n), grid, ao)

    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis([1, 6, 8])
    pos = mol.atom_coords()
    atomic_numbers = mol.atom_charges()
    sample = OFData.construct_new(
        basis_info, pos, atomic_numbers, torch.zeros((n,)), dual_basis_integrals="infer_from_basis"
    )
    energies, grad = functional(sample)

    assert grad.shape == (n,)
    assert isinstance(energies, Energies)
    assert energies["dummy"] == 42
    assert list(energies.energies_dict.keys()) == ["nuclear_repulsion", "libxc_LDA", "dummy"]

    with pytest.raises(ValueError):
        FunctionalFactory("hartree", model, "hartree")


@set_default_torch_dtype(torch.float64)
def test_tensor_functional_preserves_density_and_coordinate_graphs():
    class DifferentiableDummy(torch.nn.Module):
        target_key = "kin_plus_xc"

        def __init__(self):
            super().__init__()
            self.register_buffer("dtype_marker", torch.tensor(0.0, dtype=torch.float32))

        @property
        def dtype(self):
            return self.dtype_marker.dtype

        def forward_predictions(
            self, sample, compute_density_gradients=True, compute_forces=None
        ):
            energy = sample.coeffs.square().sum() + 0.25 * sample.pos.square().sum()
            return energy.reshape(1), None, None, None

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coeffs = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    sample = OFData(
        pos=positions,
        atomic_numbers=torch.tensor([1, 8]),
        coeffs=coeffs,
    )
    coulomb = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float64)
    attraction = torch.tensor([-1.5, -2.0], dtype=torch.float64)

    factory = FunctionalFactory.from_module(DifferentiableDummy())
    energies = factory.evaluate_tensor_functional(sample, coulomb, attraction)
    assert isinstance(energies, TensorEnergies)

    coeff_gradient, position_gradient = torch.autograd.grad(
        energies.total_energy, (coeffs, positions)
    )
    expected_coeff_gradient = 2.0 * coeffs + coulomb @ coeffs + attraction
    assert torch.allclose(coeff_gradient, expected_coeff_gradient, atol=2e-7, rtol=2e-7)

    expected_model_position_gradient = 0.5 * positions
    nuclear_energy = nuclear_repulsion_energy_tensor(
        positions, sample.atomic_numbers
    )
    expected_nuclear_gradient = torch.autograd.grad(nuclear_energy, positions)[0]
    assert torch.allclose(
        position_gradient,
        expected_model_position_gradient + expected_nuclear_gradient,
        atol=2e-7,
        rtol=2e-7,
    )


@set_default_torch_dtype(torch.float64)
def test_nuclear_repulsion_force_matches_finite_difference_and_is_translation_invariant():
    positions = torch.tensor(
        [[-0.3, 0.1, 0.2], [0.5, -0.2, 1.7], [1.1, 0.4, -0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([1, 6, 8])
    energy = nuclear_repulsion_energy_tensor(positions, atomic_numbers)
    gradient = torch.autograd.grad(energy, positions)[0]

    step = 1e-5
    plus = positions.detach().clone()
    minus = positions.detach().clone()
    plus[1, 2] += step
    minus[1, 2] -= step
    finite_difference = (
        nuclear_repulsion_energy_tensor(plus, atomic_numbers)
        - nuclear_repulsion_energy_tensor(minus, atomic_numbers)
    ) / (2.0 * step)

    assert torch.allclose(gradient[1, 2], finite_difference, atol=1e-8, rtol=1e-8)
    assert torch.allclose(gradient.sum(dim=0), torch.zeros(3), atol=1e-12, rtol=0)


def test_constrained_energy_lagrangian_gradient():
    coeffs = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    dual = torch.tensor([1.0, 1.0], dtype=torch.float64)
    multiplier = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    energy = coeffs.square().sum()

    lagrangian = constrained_energy_lagrangian(
        energy, coeffs, dual, n_electron=1.0, multiplier=multiplier
    )
    coeff_gradient, constraint = torch.autograd.grad(lagrangian, (coeffs, multiplier))

    assert torch.allclose(coeff_gradient, 2.0 * coeffs + multiplier * dual)
    assert torch.allclose(constraint, torch.tensor(0.0, dtype=torch.float64))
