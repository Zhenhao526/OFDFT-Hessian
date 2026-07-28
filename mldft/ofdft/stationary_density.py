"""High-accuracy electron-number-constrained density stationarity refiners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.energies import TensorEnergies
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.implicit_response import tangent_basis
from mldft.ofdft.implicit_response import preconditioned_conjugate_gradient


@dataclass
class StationaryDensityResult:
    coeffs: torch.Tensor
    converged: bool
    final_projected_gradient_norm: float
    constraint_residual: float
    closure_evaluations: int
    energy_trace: list[float] = field(default_factory=list)
    gradient_norm_trace: list[float] = field(default_factory=list)
    krylov_iterations: list[int] = field(default_factory=list)
    krylov_relative_residuals: list[float] = field(default_factory=list)


def refine_constrained_coefficients_lbfgs(
    initial_coeffs: torch.Tensor,
    normalization: torch.Tensor,
    n_electron: float | torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    history_size: int = 50,
) -> StationaryDensityResult:
    """Minimize a scalar energy in an orthonormal electron-number tangent basis."""
    initial_coeffs = initial_coeffs.detach()
    normalization = normalization.detach().to(initial_coeffs)
    target = torch.as_tensor(n_electron, dtype=initial_coeffs.dtype, device=initial_coeffs.device)
    normalization_norm_squared = torch.dot(normalization, normalization)
    feasible_base = initial_coeffs + normalization * (
        (target - torch.dot(normalization, initial_coeffs)) / normalization_norm_squared
    )
    tangent = tangent_basis(normalization)
    tangent_parameters = torch.zeros(
        tangent.shape[1],
        dtype=initial_coeffs.dtype,
        device=initial_coeffs.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [tangent_parameters],
        lr=1.0,
        max_iter=max_iterations,
        max_eval=max_iterations * 2,
        tolerance_grad=tolerance,
        tolerance_change=1e-15,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    energy_trace: list[float] = []
    gradient_trace: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        coeffs = feasible_base + tangent @ tangent_parameters
        energy = energy_function(coeffs)
        gradient_parameters = torch.autograd.grad(energy, tangent_parameters)[0]
        tangent_parameters.grad = gradient_parameters
        energy_trace.append(float(energy.detach().cpu()))
        gradient_trace.append(float(torch.linalg.vector_norm(gradient_parameters).detach().cpu()))
        return energy

    optimizer.step(closure)
    final_coeffs = (feasible_base + tangent @ tangent_parameters).detach()
    final_coeffs_for_gradient = final_coeffs.clone().requires_grad_(True)
    final_energy = energy_function(final_coeffs_for_gradient)
    final_gradient = torch.autograd.grad(final_energy, final_coeffs_for_gradient)[0]
    projected_gradient = tangent @ (tangent.T @ final_gradient)
    final_norm = float(torch.linalg.vector_norm(projected_gradient).detach().cpu())
    constraint_residual = float(
        (torch.dot(normalization, final_coeffs) - target).detach().cpu()
    )
    return StationaryDensityResult(
        coeffs=final_coeffs,
        converged=final_norm < tolerance,
        final_projected_gradient_norm=final_norm,
        constraint_residual=constraint_residual,
        closure_evaluations=len(energy_trace),
        energy_trace=energy_trace,
        gradient_norm_trace=gradient_trace,
    )


def refine_sample_density_lbfgs(
    sample: OFData,
    functional_factory: FunctionalFactory,
    n_electron: float | torch.Tensor,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    history_size: int = 50,
) -> tuple[TensorEnergies, StationaryDensityResult]:
    """Refine an already transformed OFDFT sample and update ``sample.coeffs`` in place."""
    def energy_function(coeffs: torch.Tensor) -> torch.Tensor:
        sample.coeffs = coeffs
        return functional_factory.evaluate_tensor_functional(
            sample,
            sample.coulomb_matrix,
            sample.nuclear_attraction_vector,
        ).total_energy

    result = refine_constrained_coefficients_lbfgs(
        sample.coeffs,
        sample.dual_basis_integrals,
        n_electron,
        energy_function,
        tolerance=tolerance,
        max_iterations=max_iterations,
        history_size=history_size,
    )
    sample.coeffs = result.coeffs.detach()
    energies = functional_factory.evaluate_tensor_functional(
        sample,
        sample.coulomb_matrix,
        sample.nuclear_attraction_vector,
    )
    return energies, result


def refine_constrained_coefficients_newton_pcg(
    initial_coeffs: torch.Tensor,
    normalization: torch.Tensor,
    n_electron: float | torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    tolerance: float = 1e-8,
    max_newton_iterations: int = 6,
    max_krylov_iterations: int = 200,
    krylov_tolerance: float = 1e-10,
    diagonal_probes: int = 8,
    damping: float = 1e-8,
) -> StationaryDensityResult:
    """Refine stationarity with preconditioned coefficient-Hessian Newton steps."""
    normalization = normalization.detach().to(initial_coeffs)
    target = torch.as_tensor(n_electron, dtype=initial_coeffs.dtype, device=initial_coeffs.device)
    norm_squared = torch.dot(normalization, normalization)
    coeffs = initial_coeffs.detach() + normalization * (
        (target - torch.dot(normalization, initial_coeffs.detach())) / norm_squared
    )
    tangent = tangent_basis(normalization)
    energy_trace: list[float] = []
    gradient_trace: list[float] = []
    krylov_iterations: list[int] = []
    krylov_residuals: list[float] = []

    def energy_and_projected_gradient(candidate: torch.Tensor, create_graph: bool):
        variable = candidate.detach().clone().requires_grad_(True)
        energy = energy_function(variable)
        gradient = torch.autograd.grad(
            energy, variable, create_graph=create_graph, retain_graph=create_graph
        )[0]
        tangent_gradient = tangent.T @ gradient
        return variable, energy, gradient, tangent_gradient

    for newton_iteration in range(max_newton_iterations):
        variable, energy, gradient, tangent_gradient = energy_and_projected_gradient(
            coeffs, create_graph=True
        )
        gradient_norm = float(torch.linalg.vector_norm(tangent_gradient).detach().cpu())
        energy_trace.append(float(energy.detach().cpu()))
        gradient_trace.append(gradient_norm)
        if gradient_norm < tolerance:
            break

        def operator(tangent_vector: torch.Tensor) -> torch.Tensor:
            full_vector = tangent @ tangent_vector
            hessian_vector = torch.autograd.grad(
                torch.dot(gradient, full_vector), variable, retain_graph=True
            )[0]
            return tangent.T @ hessian_vector + damping * tangent_vector

        diagonal = torch.zeros_like(tangent_gradient)
        generator = torch.Generator(device=diagonal.device)
        generator.manual_seed(1729 + newton_iteration)
        for _ in range(diagonal_probes):
            signs = torch.randint(
                0,
                2,
                diagonal.shape,
                generator=generator,
                device=diagonal.device,
                dtype=torch.int64,
            ).to(diagonal.dtype)
            signs = 2.0 * signs - 1.0
            diagonal += signs * operator(signs)
        diagonal = torch.abs(diagonal / diagonal_probes).clamp_min(1e-8)
        krylov = preconditioned_conjugate_gradient(
            operator,
            -tangent_gradient.detach(),
            preconditioner=lambda vector: vector / diagonal,
            tolerance=krylov_tolerance,
            max_iterations=max_krylov_iterations,
        )
        krylov_iterations.append(krylov.iterations)
        krylov_residuals.append(krylov.relative_residual)
        if not krylov.converged and krylov.breakdown == "non_positive_curvature":
            break
        full_step = tangent @ krylov.solution

        accepted = False
        best = (gradient_norm, coeffs)
        for line_search_index in range(9):
            step_scale = 0.5**line_search_index
            candidate = coeffs + step_scale * full_step
            _, candidate_energy, _, candidate_tangent_gradient = energy_and_projected_gradient(
                candidate, create_graph=False
            )
            candidate_norm = float(
                torch.linalg.vector_norm(candidate_tangent_gradient).detach().cpu()
            )
            if candidate_norm < best[0]:
                best = (candidate_norm, candidate.detach())
            if candidate_norm < gradient_norm and float(candidate_energy) <= float(energy) + 1e-10:
                coeffs = candidate.detach()
                accepted = True
                break
        if not accepted:
            if best[0] < gradient_norm:
                coeffs = best[1]
            else:
                break

    final_variable, final_energy, _, final_tangent_gradient = energy_and_projected_gradient(
        coeffs, create_graph=False
    )
    final_norm = float(torch.linalg.vector_norm(final_tangent_gradient).detach().cpu())
    if not gradient_trace or final_norm != gradient_trace[-1]:
        energy_trace.append(float(final_energy.detach().cpu()))
        gradient_trace.append(final_norm)
    constraint_residual = float(
        (torch.dot(normalization, final_variable.detach()) - target).detach().cpu()
    )
    return StationaryDensityResult(
        coeffs=final_variable.detach(),
        converged=final_norm < tolerance,
        final_projected_gradient_norm=final_norm,
        constraint_residual=constraint_residual,
        closure_evaluations=len(energy_trace),
        energy_trace=energy_trace,
        gradient_norm_trace=gradient_trace,
        krylov_iterations=krylov_iterations,
        krylov_relative_residuals=krylov_residuals,
    )


def refine_sample_density_newton_pcg(
    sample: OFData,
    functional_factory: FunctionalFactory,
    n_electron: float | torch.Tensor,
    **kwargs,
) -> tuple[TensorEnergies, StationaryDensityResult]:
    """Apply :func:`refine_constrained_coefficients_newton_pcg` to a transformed sample."""
    def energy_function(coeffs: torch.Tensor) -> torch.Tensor:
        sample.coeffs = coeffs
        return functional_factory.evaluate_tensor_functional(
            sample,
            sample.coulomb_matrix,
            sample.nuclear_attraction_vector,
        ).total_energy

    result = refine_constrained_coefficients_newton_pcg(
        sample.coeffs,
        sample.dual_basis_integrals,
        n_electron,
        energy_function,
        **kwargs,
    )
    sample.coeffs = result.coeffs.detach()
    energies = functional_factory.evaluate_tensor_functional(
        sample,
        sample.coulomb_matrix,
        sample.nuclear_attraction_vector,
    )
    return energies, result
