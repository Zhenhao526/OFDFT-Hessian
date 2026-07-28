"""Provides energy functionals."""

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from loguru import logger
from pyscf import dft, gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft import libxc_functionals, torch_functionals
from mldft.ofdft.energies import Energies, TensorEnergies
from mldft.utils.coeffs_to_grid import coeffs_to_rho
from mldft.utils.molecules import build_molecule_ofdata

LIBXC_PREFIX = "libxc_"


def hartree_energy_tensor(
    coeffs: torch.Tensor, coulomb_matrix: torch.Tensor
) -> torch.Tensor:
    """Return the differentiable Hartree energy ``0.5 * c.T @ J @ c``."""
    return 0.5 * coeffs @ (coulomb_matrix @ coeffs)


def nuclear_attraction_energy_tensor(
    coeffs: torch.Tensor, nuclear_attraction_vector: torch.Tensor
) -> torch.Tensor:
    """Return the differentiable electron-nuclear attraction energy ``c.T @ v_ext``."""
    return coeffs @ nuclear_attraction_vector


def nuclear_repulsion_energy_tensor(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return nuclear repulsion without leaving the coordinate autograd graph.

    Atom pairs are constructed without diagonal entries, so this path never differentiates
    ``norm(0)`` self interactions. ``batch`` prevents interactions between different molecules.
    Positions are expected in Bohr and the returned energy is in Hartree.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (n_atom, 3), got {positions.shape}")
    if atomic_numbers.shape != (positions.shape[0],):
        raise ValueError(
            "atomic_numbers must have shape (n_atom,), got "
            f"{atomic_numbers.shape} for {positions.shape[0]} atoms"
        )

    pair_i, pair_j = torch.triu_indices(
        positions.shape[0], positions.shape[0], offset=1, device=positions.device
    )
    if batch is not None:
        batch = batch.to(device=positions.device)
        same_molecule = batch[pair_i] == batch[pair_j]
        pair_i = pair_i[same_molecule]
        pair_j = pair_j[same_molecule]
    if pair_i.numel() == 0:
        return positions.sum() * 0.0

    displacement = positions[pair_i] - positions[pair_j]
    squared_distance = torch.sum(displacement * displacement, dim=-1)
    if bool(torch.any(squared_distance == 0).detach().cpu()):
        raise ValueError("Coincident nuclei make the nuclear repulsion singular")
    distance = torch.sqrt(squared_distance)
    charges = atomic_numbers.to(dtype=positions.dtype, device=positions.device)
    return torch.sum(charges[pair_i] * charges[pair_j] / distance)


def electron_number_tensor(
    coeffs: torch.Tensor, dual_basis_integrals: torch.Tensor
) -> torch.Tensor:
    """Return the electron number represented by auxiliary-density coefficients."""
    return dual_basis_integrals @ coeffs


def constrained_energy_lagrangian(
    total_energy: torch.Tensor,
    coeffs: torch.Tensor,
    dual_basis_integrals: torch.Tensor,
    n_electron: torch.Tensor | float,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Return ``E + mu * (q.T @ c - N)`` for constrained response calculations."""
    target = torch.as_tensor(n_electron, dtype=coeffs.dtype, device=coeffs.device)
    return total_energy + multiplier * (
        electron_number_tensor(coeffs, dual_basis_integrals) - target
    )


def requires_grid(
    target_key: str, negative_integrated_density_penalty_weight: float | None = None
) -> bool:
    """Check if the functional requires a grid for the evaluation.

    Returns:
        True if the functional requires a grid, False otherwise.
    """
    if target_key in ["kin_plus_xc", "tot"] and (
        negative_integrated_density_penalty_weight is None
        or negative_integrated_density_penalty_weight == 0
    ):
        return False
    else:
        return True


def hartree_functional(
    coeffs: torch.Tensor, coulomb_matrix: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """Get the Hartree energy for the given coefficients.

    Args:
        coulomb_matrix: Coulomb matrix of the system.
        coeffs: The coefficients p.

    Returns:
        The Hartree energy and its gradient.
    """
    hartree_potential_vector = coulomb_matrix @ coeffs
    hartree_energy = 0.5 * coeffs @ hartree_potential_vector

    return hartree_energy.item(), hartree_potential_vector


def nuclear_attraction_functional(
    coeffs: torch.Tensor, nuclear_attraction_vector: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """Get the nuclear attraction energy for the given coefficients.

    Args:
        nuclear_attraction_vector: The nuclear attraction vector.
        coeffs: The coefficients p.

    Returns:
        The nuclear attraction energy and its gradient.
    """
    nuclear_attraction_energy = coeffs @ nuclear_attraction_vector
    return nuclear_attraction_energy.item(), nuclear_attraction_vector


class NegativeIntegratedDensity(torch.nn.Module):
    """Class to compute a penalty term on the negative integrated density."""

    def __init__(self, grid_weights: torch.Tensor, ao: torch.Tensor, gamma: float = 1000.0):
        """Initialize the negative integrated density penalty.

        Args:
            grid_weights: The grid_weights.
            ao: The atomic orbital values on the grid.
            sample: The OFData object.
            gamma: The penalty factor.
        """
        super().__init__()
        self.gamma = gamma
        self.__name__ = "integrated_negative_density"

        self.weights = grid_weights
        self.ao = ao

    def forward(self, sample: OFData) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the negative integrated density penalty.

        Args:
            sample: The OFData object.

        Returns:
            The energy (0.0) and gradient of the penalty term.
        """
        coeffs = sample.coeffs
        coeffs.requires_grad = True
        density = coeffs_to_rho(coeffs, self.ao)
        density[density > 0] = 0
        integrated_negative_density = self.gamma * (density * density) @ self.weights

        # get the gradient
        grad = torch.autograd.grad(integrated_negative_density, coeffs, create_graph=False)[0]

        sample.coeffs = coeffs.detach()

        return torch.tensor([0.0], dtype=torch.float32, device=sample.coeffs.device), grad.detach()


class FunctionalFactory:
    """Class to construct energy functionals.

    In the init, only contributions to the energy functional need to be passed. One instance can be
    used to construct multiple functionals that are specialized to multiple molecules.
    """

    def __init__(
        self,
        *contributions: str | torch.nn.Module | Callable[[OFData], tuple[float, torch.Tensor]],
    ):
        """Initialize the energy functional with contributions that will be summed up.

        Args:
            contributions: The contributions (addends) to the energy functional. Allowed strings
                are either libxc functionals, torch functionals, "hartree" or "nuclear_attraction".
                Can also be a torch.nn.Module or a callable.

        Raises:
            ValueError: If there are duplicate contributions. This is important since energies are
                stored in a dict, see :class:`~mldft.ofdft.energies.Energies`.
        """

        libxc_functionals_list = []  # Used to collect all libxc functionals to compute at once
        torch_functionals_list = []  # Used to collect all torch functionals to compute at once

        for contribution in contributions:
            if isinstance(contribution, str):
                if (
                    contribution == "hartree"
                    or contribution == "nuclear_attraction"
                    or "integrated_negative_density" in contribution
                ):
                    continue
                elif contribution.startswith(LIBXC_PREFIX):
                    functional_name = contribution[len(LIBXC_PREFIX) :]
                    libxc_functionals.check_kxc_implementation(functional_name)
                    libxc_functionals_list.append(functional_name)
                else:
                    assert contribution in torch_functionals.str_to_torch_functionals, (
                        f"{contribution} not in supported torch functionals. Supported functionals"
                        f" are: {list(torch_functionals.str_to_torch_functionals.keys())}"
                    )
                    torch_functionals_list.append(contribution)
            elif isinstance(contribution, torch.nn.Module):
                contribution.__name__ = contribution.target_key
                if contribution.training:
                    contribution.eval()
                    logger.warning(f"Setting {contribution.__class__.__name__} to eval mode.")
            elif callable(contribution):
                pass
            else:
                raise ValueError(f"Invalid functional contribution: {contribution}")

        # throw error if there are duplicate names
        # this is important since energies are stored in a dict
        names = []
        for c in contributions:
            if isinstance(c, str):
                names.append(c)
            else:
                names.append(c.__name__)
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate contributions: {names}")

        self.contributions = list(contributions)
        self.torch_functionals = torch_functionals_list
        self.libxc_functionals = libxc_functionals_list

        classical_functionals = self.libxc_functionals + self.torch_functionals
        self.max_derivative = libxc_functionals.required_derivative(classical_functionals)

    @classmethod
    def get_vw_functional(cls, xc_functional: str):
        """Construct a functional with the von Weizsäcker kinetic energy functional.

        The von Weizsäcker kinetic energy functional is exact for two-electron molecules. The
        remaining erroneous contribution is the exchange part of the xc functional (there should
        be no correlation part for two-electron molecules).

        Args:
            xc_functional: The xc functional.
        """
        return cls("libxc_GGA_K_VW", xc_functional, "hartree", "nuclear_attraction")

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        xc_functional: str = None,
        negative_integrated_density_penalty_weight: float = 0.0,
    ):
        """Construct a functional from a torch.nn.Module.

        Args:
            module: The torch.nn.Module.
            xc_functional: The xc functional. Can be omitted if the module learned the xc
                functional. Note that torch_PBE will be converted to the PBE X and PBE C functional. Other
                torch functionals will be used as is (but can thus only be one functional).

        Returns:
            The functional factory.
        """
        if not hasattr(module, "target_key"):
            raise ValueError("Module has no target key.")

        target_key = module.target_key

        if target_key == "kin":
            assert xc_functional is not None, f"xc_functional must be given for {target_key=}"
            contributions = [module, xc_functional, "hartree", "nuclear_attraction"]
        elif target_key == "kin_minus_apbe":
            assert xc_functional is not None, f"xc_functional must be given for {target_key=}"
            contributions = [
                module,
                "APBE",
                xc_functional,
                "hartree",
                "nuclear_attraction",
            ]
        elif target_key == "kin_plus_xc":
            contributions = [module, "hartree", "nuclear_attraction"]
        elif target_key == "tot":
            contributions = [module]
        else:
            raise ValueError(f"Invalid target key: {target_key}")

        if negative_integrated_density_penalty_weight > 0:
            contributions.append(
                f"{negative_integrated_density_penalty_weight}*integrated_negative_density"
            )

        return cls(*contributions)

    def evaluate_functional(
        self,
        sample: OFData,
        mol: gto.Mole,
        coulomb_matrix: torch.Tensor,
        nuclear_attraction_vector: torch.Tensor,
        grid: dft.Grids | None = None,
        grid_weights: torch.Tensor | None = None,
        ao: np.ndarray | None = None,
        max_xc_memory: int = 4000,
    ) -> tuple[Energies, torch.Tensor]:
        """Evaluate the energy functional for the given coefficients.

        The main use of this function is to be partially called by :func:`construct()`.

        Args:
            sample: The OFData object for a torch.nn.Module. The coeffs attribute is updated.
            mol: The molecule of the current optimization.
            coulomb_matrix: The Coulomb matrix.
            nuclear_attraction_vector: The nuclear attraction vector.
            grid: The grid for the evaluation of the xc functional.
            grid_weights: The weights of the grid points.
            ao: The atomic orbital values on the grid.
            max_xc_memory: The maximum (additional to ao memory usage) memory in MB for the evaluation of the xc functional.

        Returns:
            The energies and gradient.
        """
        energies = Energies(mol)
        gradient = torch.zeros_like(sample.coeffs)

        # Run torch and libxc functionals all at once to not have to recompute the densities
        if len(self.torch_functionals) > 0:
            torch_functional_outs = torch_functionals.eval_torch_functionals_blocked_fast(
                ao,
                grid_weights,
                sample.coeffs,
                self.torch_functionals,
                max_memory=max_xc_memory,
            )
            for name, (energy, grad) in torch_functional_outs.items():
                gradient += grad
                energies[name] = energy.item()
            sample.coeffs = sample.coeffs.detach()

        if len(self.libxc_functionals) > 0:
            libxc_functionals_outs = libxc_functionals.eval_libxc_functionals(
                sample.coeffs.numpy(force=True),
                self.libxc_functionals,
                grid,
                ao.cpu().numpy(),
                self.max_derivative,
            )
            grad = libxc_functionals_outs[1]
            gradient += torch.as_tensor(grad, dtype=torch.float64, device=sample.coeffs.device)
            for name, energy in zip(self.libxc_functionals, libxc_functionals_outs[0]):
                energies[LIBXC_PREFIX + name] = energy

        for contribution in self.contributions:
            # Hartree energy
            if contribution == "hartree":
                energy, grad = hartree_functional(sample.coeffs, coulomb_matrix=coulomb_matrix)
            # Nuclear attraction energy
            elif contribution == "nuclear_attraction":
                energy, grad = nuclear_attraction_functional(
                    sample.coeffs, nuclear_attraction_vector=nuclear_attraction_vector
                )
            # Negative integrated density
            elif isinstance(contribution, str):
                if "integrated_negative_density" in contribution:
                    negative_integrated_density = NegativeIntegratedDensity(
                        grid_weights,
                        ao,
                        gamma=float(contribution[: -len("integrated_negative_density") - 1]),
                    )
                    energy, grad = negative_integrated_density(sample.detach())
                    energy = energy.item()
                    grad = grad
                    sample.coeffs = sample.coeffs.detach()
                else:
                    # Torch and libxc functionals are already evaluated
                    continue
            # torch.nn.Module
            elif isinstance(contribution, torch.nn.Module):
                # If we do float32 model forward, copy the sample, then overwrite new coeffs at the end
                if contribution.dtype != sample.coeffs.dtype:
                    converted_sample = to_torch(
                        sample.detach().clone(),
                        float_dtype=contribution.dtype,
                        device=sample.coeffs.device,
                    )
                else:
                    converted_sample = sample
                converted_sample = contribution.sample_forward(converted_sample)
                energy = converted_sample.pred_energy.item()
                grad = converted_sample.pred_gradient.to(torch.float64)
                sample.coeffs = converted_sample.coeffs.detach().to(sample.coeffs.dtype)
            # other callable
            else:
                energy, grad = contribution(sample)

            if isinstance(contribution, str):
                energies[contribution] = energy
            else:
                energies[contribution.__name__] = energy
            gradient += grad

        assert not sample.coeffs.requires_grad, "Coeffs should not require grad"
        assert not gradient.requires_grad, "Gradient should not require grad"

        return energies, gradient

    def evaluate_tensor_functional(
        self,
        sample: OFData,
        coulomb_matrix: torch.Tensor,
        nuclear_attraction_vector: torch.Tensor,
        nuclear_repulsion: torch.Tensor | None = None,
    ) -> TensorEnergies:
        """Evaluate a scalar total energy while preserving its full torch graph.

        This is the force/response API. Unlike :meth:`evaluate_functional`, it never converts
        energy contributions to Python or NumPy scalars. At present it deliberately supports the
        learned OFDFT path plus Hartree and electron-nuclear terms. LibXC, grid penalties, and the
        historical callable API must first receive tensor-native implementations before they can
        be used here.

        Integral tensors may themselves depend on geometry. Keeping them as tensors here allows
        an analytic, custom-autograd, or audited finite-difference integral provider to supply the
        moving-basis/Pulay derivatives without changing this energy assembly layer.
        """
        unsupported = []
        for contribution in self.contributions:
            if isinstance(contribution, str) and contribution not in {
                "hartree",
                "nuclear_attraction",
            }:
                unsupported.append(contribution)
            elif not isinstance(contribution, (str, torch.nn.Module)):
                unsupported.append(getattr(contribution, "__name__", repr(contribution)))
        if unsupported:
            raise NotImplementedError(
                "Tensor functional does not yet support contributions: "
                + ", ".join(unsupported)
            )

        if nuclear_repulsion is None:
            nuclear_repulsion = nuclear_repulsion_energy_tensor(
                sample.pos,
                sample.atomic_numbers,
                getattr(sample, "batch", None),
            )
        energies = TensorEnergies(nuclear_repulsion=nuclear_repulsion)

        for contribution in self.contributions:
            if contribution == "hartree":
                energy = hartree_energy_tensor(sample.coeffs, coulomb_matrix)
                name = contribution
            elif contribution == "nuclear_attraction":
                energy = nuclear_attraction_energy_tensor(
                    sample.coeffs, nuclear_attraction_vector
                )
                name = contribution
            elif isinstance(contribution, torch.nn.Module):
                model_dtype = getattr(contribution, "dtype", sample.coeffs.dtype)
                if model_dtype != sample.coeffs.dtype:
                    converted_sample = to_torch(
                        sample.clone(),
                        float_dtype=model_dtype,
                        device=sample.coeffs.device,
                    )
                else:
                    converted_sample = sample

                if hasattr(contribution, "forward_predictions"):
                    prediction = contribution.forward_predictions(
                        converted_sample,
                        compute_density_gradients=False,
                        compute_forces=False,
                    )[0]
                else:
                    converted_sample = contribution.sample_forward(converted_sample)
                    prediction = converted_sample.pred_energy
                energy = prediction.sum()
                name = contribution.target_key
            else:  # pragma: no cover - guarded by the unsupported check above
                raise AssertionError(f"Unhandled contribution {contribution}")

            energies[name] = energy

        return energies

    def construct(
        self,
        mol: gto.Mole,
        coulomb_matrix: torch.Tensor,
        nuclear_attraction_vector: torch.Tensor,
        grid: dft.Grids | None = None,
        ao: torch.Tensor | None = None,
        max_xc_memory: int = 4000,
    ) -> Callable[[OFData], tuple[Energies, torch.Tensor]]:
        """Construct the energy functional.

        Args:
            mol: The molecule of the current optimization.
            coulomb_matrix: The Coulomb matrix.
            nuclear_attraction_vector: The nuclear attraction vector.
            grid: The grid for the evaluation of the xc functional.
            ao: The atomic orbital values on the grid.
            max_xc_memory: The maximum (additional to ao memory usage) memory in MB for the evaluation of the xc functional.

        Returns:
            The energy functional.
        """
        if grid is not None:
            grid_weights = torch.as_tensor(
                grid.weights, dtype=torch.float64, device=coulomb_matrix.device
            )
        else:
            grid_weights = None

        return partial(
            self.evaluate_functional,
            mol=mol,
            coulomb_matrix=coulomb_matrix,
            nuclear_attraction_vector=nuclear_attraction_vector,
            grid=grid,
            grid_weights=grid_weights,
            ao=ao,
            max_xc_memory=max_xc_memory,
        )

    def construct_tensor(
        self,
        coulomb_matrix: torch.Tensor,
        nuclear_attraction_vector: torch.Tensor,
        nuclear_repulsion: torch.Tensor | None = None,
    ) -> Callable[[OFData], TensorEnergies]:
        """Construct a graph-preserving energy functional for one molecular geometry."""
        return partial(
            self.evaluate_tensor_functional,
            coulomb_matrix=coulomb_matrix,
            nuclear_attraction_vector=nuclear_attraction_vector,
            nuclear_repulsion=nuclear_repulsion,
        )

    def get_energies_label(self, sample: OFData, basis_info: BasisInfo) -> Energies:
        """Get the energies label from OFData object.

        Args:
            sample: OFData object with the ground state energy labels.

        Returns:
            Energies object initialized with the energies from OFData object corresponding to the
            functional contributions.
        """
        mol = build_molecule_ofdata(sample, basis_info.basis_dict)

        energies = Energies(mol)

        for contribution in self.contributions:
            if contribution == "hartree":
                energies[contribution] = sample["of_labels/energies/ground_state_e_hartree"]
            elif contribution == "nuclear_attraction":
                energies[contribution] = sample["of_labels/energies/ground_state_e_ext"]
            elif isinstance(contribution, str) and ("integrated_negative_density" in contribution):
                pass
            elif isinstance(contribution, str) and contribution.startswith(LIBXC_PREFIX):
                functional_name = contribution[len(LIBXC_PREFIX) :]
                libxc_functionals.check_kxc_implementation(functional_name)
                if "_K_" in contribution:
                    energies[contribution] = sample["of_labels/energies/ground_state_e_kin"]
                elif "_XC_" in contribution or functional_name in dft.libxc.XC_ALIAS:
                    energies[contribution] = sample["of_labels/energies/ground_state_e_xc"]
                else:
                    raise ValueError(f"No label for {contribution} available.")
            elif isinstance(contribution, str):
                if contribution == "APBE":
                    energies[contribution] = sample["of_labels/energies/ground_state_e_kinapbe"]
                elif contribution == "PBE":
                    energies[contribution] = sample["of_labels/energies/ground_state_e_xc"]
                else:
                    raise ValueError(f"No label for {contribution} available.")
            elif isinstance(contribution, torch.nn.Module):
                energies[contribution.target_key] = sample[
                    f"of_labels/energies/ground_state_e_{contribution.target_key}"
                ]
            elif isinstance(contribution, Callable):
                energies[contribution.__name__] = sample[
                    f"of_labels/energies/ground_state_e_{contribution.__name__}"
                ]
            else:
                raise ValueError(f"No label for {contribution} available.")
        for key, value in energies.energies_dict.items():
            if isinstance(value, torch.Tensor):
                energies.energies_dict[key] = value.item()
        return energies

    def __str__(self):
        """Return a string that contains the contribution names."""
        names = [c if isinstance(c, str) else c.__name__ for c in self.contributions]
        return "functional = " + " + ".join(names)
