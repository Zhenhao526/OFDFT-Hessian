"""Conservative total-OFDFT scalar energy and nuclear force assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.convert_transforms import (
    PRESERVE_INJECTED_OVERLAP,
    to_torch,
)
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.ofdft.energies import TensorEnergies
from mldft.ofdft.functional_factory import (
    FunctionalFactory,
    constrained_energy_lagrangian,
    electron_number_tensor,
    hartree_energy_tensor,
    nuclear_attraction_energy_tensor,
    nuclear_repulsion_energy_tensor,
)
from mldft.ofdft.geometry_integrals import (
    FiniteDifferencePySCFIntegralProvider,
    GeometryIntegralBundle,
    integral_tensor_from_bundle,
)


@dataclass
class DifferentiableGeometry:
    """Prepared model sample and the untransformed independent variables behind it."""

    sample: OFData
    positions: torch.Tensor
    coeffs: torch.Tensor
    bundle: GeometryIntegralBundle


@dataclass
class TotalForceResult:
    """A graph-preserving total energy/force evaluation at one density and geometry."""

    energies: TensorEnergies
    force: torch.Tensor
    electron_number: torch.Tensor
    constraint_residual: torch.Tensor
    projected_density_gradient_norm: torch.Tensor
    lagrange_multiplier: torch.Tensor
    geometry: DifferentiableGeometry
    model_geometry_derivative_mode: str


@dataclass
class FixedGeometryScalarEnergy:
    """Direct scalar total energy at one rebuilt geometry in the physical coefficient basis."""

    sample: OFData
    functional_factory: FunctionalFactory
    normalization_untransformed: torch.Tensor

    def __call__(self, coeffs_untransformed: torch.Tensor) -> torch.Tensor:
        coeffs = coeffs_untransformed.to(
            device=self.sample.coeffs.device, dtype=self.sample.coeffs.dtype
        )
        self.sample.coeffs = transform_tensor_with_sample(
            self.sample, coeffs, Representation.VECTOR
        )
        return self.functional_factory.evaluate_tensor_functional(
            self.sample,
            self.sample.coulomb_matrix,
            self.sample.nuclear_attraction_vector,
        ).total_energy


def prepare_fixed_geometry_scalar_energy(
    sample_generator: Any,
    functional_factory: FunctionalFactory,
    atomic_numbers: np.ndarray | torch.Tensor,
    positions_bohr: np.ndarray | torch.Tensor,
    charge: int = 0,
) -> FixedGeometryScalarEnergy:
    """Rebuild all geometry-dependent inputs for direct scalar ``E(c_raw, R)`` evaluation."""
    atomic_numbers_np = np.asarray(
        atomic_numbers.detach().cpu() if isinstance(atomic_numbers, torch.Tensor) else atomic_numbers,
        dtype=np.int64,
    )
    positions_np = np.asarray(
        positions_bohr.detach().cpu() if isinstance(positions_bohr, torch.Tensor) else positions_bohr,
        dtype=np.float64,
    )
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers_np,
        sample_generator.basis_info.basis_dict,
        charge=charge,
    )
    mol = provider.build_molecule(positions_np)
    sample = sample_generator.get_sample_from_mol(mol)
    normalization = torch.as_tensor(
        get_normalization_vector(mol),
        dtype=torch.float64,
        device=sample.coeffs.device,
    )
    return FixedGeometryScalarEnergy(
        sample=sample,
        functional_factory=functional_factory,
        normalization_untransformed=normalization,
    )


def prepare_differentiable_geometry(
    sample_generator: Any,
    atomic_numbers: np.ndarray | torch.Tensor,
    positions_bohr: np.ndarray | torch.Tensor,
    coeffs_untransformed: np.ndarray | torch.Tensor,
    integral_provider: FiniteDifferencePySCFIntegralProvider | None = None,
    integral_bundle: GeometryIntegralBundle | None = None,
    integral_derivative_step_bohr: float = 1e-4,
    charge: int = 0,
    require_coefficient_grad: bool = True,
    detach_coefficients: bool = True,
) -> DifferentiableGeometry:
    """Build a transformed model sample with every geometry dependency connected.

    ``coeffs_untransformed`` are components in the moving atom-centred auxiliary basis. Holding
    them fixed while differentiating nuclear coordinates is the explicit derivative required by
    the variational envelope theorem. Natural/local basis transforms are then recomputed inside
    torch from the geometry-dependent overlap and positions.
    """
    atomic_numbers_np = np.asarray(
        atomic_numbers.detach().cpu() if isinstance(atomic_numbers, torch.Tensor) else atomic_numbers,
        dtype=np.int64,
    )
    positions_np = np.asarray(
        positions_bohr.detach().cpu() if isinstance(positions_bohr, torch.Tensor) else positions_bohr,
        dtype=np.float64,
    )
    if integral_provider is None:
        integral_provider = FiniteDifferencePySCFIntegralProvider(
            atomic_numbers=atomic_numbers_np,
            basis=sample_generator.basis_info.basis_dict,
            charge=charge,
            derivative_step_bohr=integral_derivative_step_bohr,
        )
    if integral_bundle is None:
        integral_bundle = integral_provider.evaluate_with_derivatives(positions_np)

    # The configured transforms may begin on CPU and move the finished sample to the model device.
    # Autograd tracks those device/dtype copies back to these independent leaves.
    positions = torch.tensor(positions_np, dtype=torch.float64, requires_grad=True)
    coeffs = torch.as_tensor(coeffs_untransformed)
    if detach_coefficients:
        coeffs = coeffs.detach().to(device=positions.device, dtype=torch.float64).clone()
    else:
        coeffs = coeffs.to(device=positions.device, dtype=torch.float64)
    if require_coefficient_grad and not coeffs.requires_grad:
        coeffs.requires_grad_(True)
    values = integral_bundle.values
    derivatives = integral_bundle.derivatives
    directional = integral_bundle.directional_second_derivatives
    normalization = integral_tensor_from_bundle(
        values.normalization,
        derivatives.normalization,
        positions,
        integral_bundle,
        directional_second_derivative=(
            None if directional is None else directional.normalization
        ),
    )
    overlap = integral_tensor_from_bundle(
        values.overlap,
        derivatives.overlap,
        positions,
        integral_bundle,
        directional_second_derivative=(
            None if directional is None else directional.overlap
        ),
    )
    coulomb = integral_tensor_from_bundle(
        values.coulomb,
        derivatives.coulomb,
        positions,
        integral_bundle,
        directional_second_derivative=(
            None if directional is None else directional.coulomb
        ),
    )
    attraction = integral_tensor_from_bundle(
        values.nuclear_attraction,
        derivatives.nuclear_attraction,
        positions,
        integral_bundle,
        directional_second_derivative=(
            None if directional is None else directional.nuclear_attraction
        ),
    )

    sample = OFData.construct_new(
        basis_info=sample_generator.basis_info,
        pos=positions,
        atomic_numbers=atomic_numbers_np,
        coeffs=coeffs,
        dual_basis_integrals=normalization,
        add_irreps=True,
    )
    sample.add_item("overlap_matrix", overlap, Representation.BILINEAR_FORM)
    sample.add_item(PRESERVE_INJECTED_OVERLAP, True, Representation.NONE)
    sample.add_item("coulomb_matrix", coulomb, Representation.BILINEAR_FORM)
    sample.add_item(
        "nuclear_attraction_vector", attraction, Representation.DUAL_VECTOR
    )
    sample = sample_generator.transforms.forward(sample)
    sample.mol.charge = charge
    sample = to_torch(sample, device=sample_generator.model.device)
    return DifferentiableGeometry(
        sample=sample,
        positions=positions,
        coeffs=coeffs,
        bundle=integral_bundle,
    )


def _stationarity_diagnostics(
    energy: torch.Tensor,
    coeffs: torch.Tensor,
    normalization: torch.Tensor,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    density_gradient = torch.autograd.grad(
        energy, coeffs, create_graph=create_graph, retain_graph=True
    )[0]
    normalization_norm_squared = torch.dot(normalization, normalization)
    multiplier = -torch.dot(normalization, density_gradient) / normalization_norm_squared
    projected = density_gradient + multiplier * normalization
    return density_gradient, multiplier, torch.linalg.vector_norm(projected)


def finite_difference_model_geometry_gradient(
    sample_generator: Any,
    atomic_numbers: np.ndarray | torch.Tensor,
    positions_bohr: np.ndarray | torch.Tensor,
    coeffs_untransformed: np.ndarray | torch.Tensor,
    step_bohr: float,
    charge: int = 0,
    richardson: bool = False,
) -> torch.Tensor:
    """Differentiate the complete model preprocessing/energy map at fixed physical density.

    This is the numerical reference VJP for local-frame plus natural-reparametrization Pulay
    effects. It remains an acceptance oracle for the transform-autograd implementation.
    """
    if step_bohr <= 0:
        raise ValueError("step_bohr must be positive")
    atomic_numbers_np = np.asarray(
        atomic_numbers.detach().cpu() if isinstance(atomic_numbers, torch.Tensor) else atomic_numbers,
        dtype=np.int64,
    )
    positions_np = np.asarray(
        positions_bohr.detach().cpu() if isinstance(positions_bohr, torch.Tensor) else positions_bohr,
        dtype=np.float64,
    )
    coeffs = torch.as_tensor(coeffs_untransformed).detach()
    provider = FiniteDifferencePySCFIntegralProvider(
        atomic_numbers_np,
        sample_generator.basis_info.basis_dict,
        charge=charge,
    )

    def energy_at(displaced_positions: np.ndarray) -> float:
        sample = sample_generator.get_sample_from_mol(
            provider.build_molecule(displaced_positions)
        )
        untransformed = coeffs.to(device=sample.coeffs.device, dtype=sample.coeffs.dtype)
        sample.coeffs = transform_tensor_with_sample(
            sample, untransformed, Representation.VECTOR
        )
        with torch.no_grad():
            prediction, _ = sample_generator.model.net(sample)
        return float(prediction.sum().detach().cpu())

    def central_gradient(step: float) -> np.ndarray:
        gradient = np.zeros_like(positions_np)
        flat = positions_np.reshape(-1)
        for coordinate in range(flat.size):
            plus = flat.copy()
            minus = flat.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            gradient.reshape(-1)[coordinate] = (
                energy_at(plus.reshape(positions_np.shape))
                - energy_at(minus.reshape(positions_np.shape))
            ) / (2.0 * step)
        return gradient

    gradient = central_gradient(step_bohr)
    if richardson:
        half_step_gradient = central_gradient(0.5 * step_bohr)
        gradient = (4.0 * half_step_gradient - gradient) / 3.0
    return torch.as_tensor(gradient, dtype=torch.float64)


def replace_coordinate_gradient(
    scalar: torch.Tensor,
    coordinates: torch.Tensor,
    target_gradient: torch.Tensor,
) -> torch.Tensor:
    """Keep a scalar's value/coefficient graph but replace its first coordinate derivative."""
    current_gradient = torch.autograd.grad(
        scalar, coordinates, create_graph=True, retain_graph=True
    )[0]
    target_gradient = target_gradient.to(
        device=coordinates.device, dtype=coordinates.dtype
    )
    correction = torch.sum(
        (coordinates - coordinates.detach())
        * (target_gradient - current_gradient).detach()
    )
    return scalar + correction.to(device=scalar.device, dtype=scalar.dtype)


def evaluate_total_ofdft_force(
    functional_factory: FunctionalFactory,
    geometry: DifferentiableGeometry,
    n_electron: float | torch.Tensor,
    create_graph: bool = False,
    model_geometry_fd_step_bohr: float | None = None,
    model_geometry_fd_richardson: bool = False,
    sample_generator: Any | None = None,
    charge: int = 0,
    detach_lagrange_multiplier: bool = True,
) -> TotalForceResult:
    """Evaluate complete total energy and its conservative explicit nuclear force.

    The force is the coordinate derivative of the constrained Lagrangian at the supplied density.
    At a converged density this equals the derivative of the relaxed scalar total energy. The
    returned stationarity residual quantifies whether use of that envelope-theorem identity is
    justified. By default the multiplier is held fixed for the coordinate derivative, matching
    the strict evaluator. Setting ``detach_lagrange_multiplier=False`` is reserved for training
    graphs that need its model-parameter response; the density constraint must then be satisfied
    tightly so multiplier coordinate-response terms remain numerically negligible.
    """
    sample = geometry.sample
    energies = functional_factory.evaluate_tensor_functional(
        sample,
        sample.coulomb_matrix,
        sample.nuclear_attraction_vector,
    )
    reference_energy = next(iter(energies.energies_dict.values()))

    def to_energy_device(value: torch.Tensor) -> torch.Tensor:
        return value.to(
            device=reference_energy.device, dtype=reference_energy.dtype
        )

    directional = geometry.bundle.directional_second_derivatives
    if "hartree" in energies.energies_dict:
        coulomb_untransformed = integral_tensor_from_bundle(
            geometry.bundle.values.coulomb,
            geometry.bundle.derivatives.coulomb,
            geometry.positions,
            geometry.bundle,
            directional_second_derivative=(
                None if directional is None else directional.coulomb
            ),
        )
        energies["hartree"] = to_energy_device(
            hartree_energy_tensor(geometry.coeffs, coulomb_untransformed)
        )
    if "nuclear_attraction" in energies.energies_dict:
        attraction_untransformed = integral_tensor_from_bundle(
            geometry.bundle.values.nuclear_attraction,
            geometry.bundle.derivatives.nuclear_attraction,
            geometry.positions,
            geometry.bundle,
            directional_second_derivative=(
                None
                if directional is None
                else directional.nuclear_attraction
            ),
        )
        energies["nuclear_attraction"] = to_energy_device(
            nuclear_attraction_energy_tensor(
                geometry.coeffs, attraction_untransformed
            )
        )
    energies["nuclear_repulsion"] = to_energy_device(
        nuclear_repulsion_energy_tensor(
            geometry.positions,
            sample.atomic_numbers,
            getattr(sample, "batch", None),
        )
    )
    model_geometry_derivative_mode = "transform_autograd"
    if model_geometry_fd_step_bohr is not None:
        if sample_generator is None:
            raise ValueError(
                "sample_generator is required for the numerical model-geometry VJP"
            )
        model_contributions = [
            contribution
            for contribution in functional_factory.contributions
            if isinstance(contribution, torch.nn.Module)
        ]
        if len(model_contributions) != 1:
            raise ValueError(
                "Numerical model-geometry VJP requires exactly one torch model contribution"
            )
        model_name = model_contributions[0].target_key
        target_gradient = finite_difference_model_geometry_gradient(
            sample_generator,
            sample.atomic_numbers,
            geometry.positions,
            geometry.coeffs,
            model_geometry_fd_step_bohr,
            charge=charge,
            richardson=model_geometry_fd_richardson,
        )
        energies[model_name] = replace_coordinate_gradient(
            energies[model_name], geometry.positions, target_gradient
        )
        model_geometry_derivative_mode = (
            "fixed-density scalar finite difference "
            f"h={model_geometry_fd_step_bohr:g} Bohr"
            + (" with Richardson h/2" if model_geometry_fd_richardson else "")
        )
    normalization_untransformed = integral_tensor_from_bundle(
        geometry.bundle.values.normalization,
        geometry.bundle.derivatives.normalization,
        geometry.positions,
        geometry.bundle,
        directional_second_derivative=(
            None if directional is None else directional.normalization
        ),
    )
    _, multiplier, projected_norm = _stationarity_diagnostics(
        energies.total_energy,
        geometry.coeffs,
        normalization_untransformed,
        create_graph=True,
    )
    force_multiplier = (
        multiplier.detach() if detach_lagrange_multiplier else multiplier
    )
    lagrangian = constrained_energy_lagrangian(
        energies.total_energy,
        geometry.coeffs,
        normalization_untransformed,
        n_electron,
        force_multiplier,
    )
    force = -torch.autograd.grad(
        lagrangian,
        geometry.positions,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    electron_number = electron_number_tensor(
        geometry.coeffs, normalization_untransformed
    )
    target = torch.as_tensor(
        n_electron, dtype=electron_number.dtype, device=electron_number.device
    )
    return TotalForceResult(
        energies=energies,
        force=force,
        electron_number=electron_number,
        constraint_residual=electron_number - target,
        projected_density_gradient_norm=projected_norm,
        lagrange_multiplier=multiplier,
        geometry=geometry,
        model_geometry_derivative_mode=model_geometry_derivative_mode,
    )
