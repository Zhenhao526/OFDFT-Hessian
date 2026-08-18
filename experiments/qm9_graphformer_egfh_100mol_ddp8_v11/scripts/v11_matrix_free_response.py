#!/usr/bin/env python3
"""v11-only matrix-free implicit adjoint for analytic geometry response.

The canonical dense reference materializes every constrained coefficient-Hessian
column with ``create_graph=True``.  That is exact but does not fit the largest
QM9 molecules.  This overlay keeps the same constrained KKT equation and
float64 autograd HVPs, solves it in the tangent space with matrix-free MINRES, and
uses a second matrix-free solve in backward to apply the implicit-function VJP.

No iterative step is unrolled into the model-parameter graph and no response is
detached from its parameter derivative.  The overlay is installed only in the
v11 trainer process; it does not modify the shared canonical source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch

from mldft.ofdft.implicit_response import (
    DensityResponseResult,
    KrylovResult,
    preconditioned_minimum_residual,
)


@dataclass(frozen=True)
class MatrixFreeSettings:
    tolerance: float
    max_iterations: int
    preconditioner: str = "streaming_exact_abs_block_diagonal"
    preconditioner_block_size: int = 512

    def validate(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("matrix-free tolerance must be positive")
        if self.max_iterations <= 0:
            raise ValueError("matrix-free max_iterations must be positive")
        if self.preconditioner != "streaming_exact_abs_block_diagonal":
            raise ValueError("unsupported matrix-free preconditioner")
        if self.preconditioner_block_size <= 0:
            raise ValueError("matrix-free preconditioner block size must be positive")


_ACTIVE_PARAMETERS: ContextVar[tuple[torch.Tensor, ...] | None] = ContextVar(
    "v11_matrix_free_parameters", default=None
)
_ACTIVE_SETTINGS: ContextVar[MatrixFreeSettings | None] = ContextVar(
    "v11_matrix_free_settings", default=None
)
_FORWARD_DIAGNOSTICS: list[dict[str, Any]] = []
_BACKWARD_DIAGNOSTICS: list[dict[str, Any]] = []
_INSTALLED = False


@contextmanager
def matrix_free_parameter_context(
    parameters: Iterable[torch.Tensor], settings: MatrixFreeSettings
) -> Iterator[None]:
    """Bind the explicit optimizer parameters consumed by the implicit VJP."""
    settings.validate()
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("matrix-free response requires model parameters")
    parameter_token = _ACTIVE_PARAMETERS.set(parameter_tuple)
    settings_token = _ACTIVE_SETTINGS.set(settings)
    try:
        yield
    finally:
        _ACTIVE_SETTINGS.reset(settings_token)
        _ACTIVE_PARAMETERS.reset(parameter_token)


def consume_matrix_free_diagnostics() -> dict[str, list[dict[str, Any]]]:
    result = {
        "forward": list(_FORWARD_DIAGNOSTICS),
        "backward": list(_BACKWARD_DIAGNOSTICS),
    }
    _FORWARD_DIAGNOSTICS.clear()
    _BACKWARD_DIAGNOSTICS.clear()
    return result


@dataclass
class _KktSolve:
    density: torch.Tensor
    multiplier: torch.Tensor
    krylov: KrylovResult
    coefficient_residual_norm: float
    constraint_residual_abs: float
    preconditioner_diagonal_min_abs: float
    preconditioner_diagonal_max_abs: float
    residual_replacements: int


@dataclass(frozen=True)
class _BlockPreconditioner:
    ranges: tuple[tuple[int, int], ...]
    inverse_blocks: tuple[torch.Tensor, ...]
    spectrum_min_abs: float
    spectrum_max_abs: float

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(vector)
        for (start, stop), inverse in zip(self.ranges, self.inverse_blocks):
            result[start:stop] = inverse @ vector[start:stop]
        return result


def _streaming_abs_block_preconditioner(
    system: Any,
    operator: Any,
    template: torch.Tensor,
    *,
    block_size: int,
    damping: float,
) -> _BlockPreconditioner:
    """Build a detached SPD block-Jacobi inverse without a global matrix.

    Only one operator column is live at a time.  Each small diagonal block is
    symmetrized and inverted through its absolute eigenspectrum, making the
    preconditioner admissible for MINRES even when the exact tangent Hessian is
    indefinite.  The detached blocks are reusable by the implicit adjoint at
    the same stationary center.
    """
    cache_key = (
        int(template.numel()),
        int(block_size),
        float(damping),
        str(template.dtype),
        str(template.device),
    )
    cache = getattr(system, "_v11_matrix_free_block_preconditioners", None)
    if cache is None:
        cache = {}
        setattr(system, "_v11_matrix_free_block_preconditioners", cache)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    ranges: list[tuple[int, int]] = []
    inverse_blocks: list[torch.Tensor] = []
    minimum = float("inf")
    maximum = 0.0
    basis_vector = torch.zeros_like(template)
    for start in range(0, template.numel(), block_size):
        stop = min(start + block_size, template.numel())
        width = stop - start
        block = torch.empty(
            (width, width), dtype=template.dtype, device=template.device
        )
        for local_column, global_column in enumerate(range(start, stop)):
            basis_vector.zero_()
            basis_vector[global_column] = 1.0
            applied = operator(basis_vector)
            block[:, local_column] = applied[start:stop]
        block = 0.5 * (block + block.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(block)
        absolute = torch.abs(eigenvalues)
        local_scale = float(torch.max(absolute).detach().cpu())
        floor = max(1.0e-12, 1.0e-12 * max(local_scale, 1.0))
        inverse = (
            eigenvectors
            @ torch.diag(1.0 / absolute.clamp_min(floor))
            @ eigenvectors.T
        ).detach()
        ranges.append((start, stop))
        inverse_blocks.append(inverse)
        minimum = min(minimum, float(torch.min(absolute).detach().cpu()))
        maximum = max(maximum, local_scale)
    result = _BlockPreconditioner(
        ranges=tuple(ranges),
        inverse_blocks=tuple(inverse_blocks),
        spectrum_min_abs=minimum,
        spectrum_max_abs=maximum,
    )
    cache[cache_key] = result
    return result


def _matrix_free_kkt_solve(
    system: Any,
    rhs_coefficients: torch.Tensor,
    rhs_constraint: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    damping: float,
    preconditioner_mode: str,
    preconditioner_block_size: int,
) -> _KktSolve:
    """Solve ``[L_cc+dP,q;q^T,0] y = rhs`` without forming ``L_cc``."""
    if damping < 0.0:
        raise ValueError("matrix-free damping must be nonnegative")
    q = system.constraint_gradient.detach()
    q_norm_squared = torch.dot(q, q)
    if not bool(torch.isfinite(q_norm_squared)) or float(q_norm_squared) <= 0.0:
        raise RuntimeError("invalid electron-number constraint gradient")
    tangent = system.tangent.detach()
    rhs_coefficients = rhs_coefficients.detach().to(q)
    rhs_constraint = rhs_constraint.detach().reshape(()).to(q)

    def coefficient_hvp(vector: torch.Tensor) -> torch.Tensor:
        return system.coefficient_hvp(
            vector.detach(), create_graph=False
        ).detach()

    particular = q * (rhs_constraint / q_norm_squared)

    def tangent_operator(vector: torch.Tensor) -> torch.Tensor:
        applied = tangent.T @ coefficient_hvp(tangent @ vector)
        return applied + damping * vector

    tangent_rhs = tangent.T @ (
        rhs_coefficients - coefficient_hvp(particular)
    )
    if preconditioner_mode != "streaming_exact_abs_block_diagonal":
        raise ValueError("unsupported matrix-free preconditioner")
    block_preconditioner = _streaming_abs_block_preconditioner(
        system,
        tangent_operator,
        tangent_rhs,
        block_size=preconditioner_block_size,
        damping=damping,
    )

    krylov = preconditioned_minimum_residual(
        tangent_operator,
        tangent_rhs,
        preconditioner=block_preconditioner.apply,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    solution = krylov.solution
    total_iterations = int(krylov.iterations)
    residual_replacements = 0
    rhs_norm = float(torch.linalg.vector_norm(tangent_rhs).detach().cpu())
    target = tolerance * max(rhs_norm, 1.0)
    residual = tangent_rhs - tangent_operator(solution)
    residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())
    while residual_norm > target and total_iterations < max_iterations:
        remaining = max_iterations - total_iterations
        scale = residual_norm
        normalized_rhs = residual / scale
        # MINRES tests a recurrence residual.  Normalize the explicit residual
        # correction so its requested relative tolerance corresponds directly
        # to the remaining absolute KKT target, then explicitly replace the
        # residual before any subsequent restart.
        correction_tolerance = min(0.5, 0.5 * target / scale)
        correction = preconditioned_minimum_residual(
            tangent_operator,
            normalized_rhs,
            preconditioner=block_preconditioner.apply,
            tolerance=correction_tolerance,
            max_iterations=remaining,
        )
        if correction.iterations <= 0:
            break
        solution = solution + scale * correction.solution
        total_iterations += int(correction.iterations)
        residual_replacements += 1
        residual = tangent_rhs - tangent_operator(solution)
        residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())

    krylov = KrylovResult(
        solution=solution,
        converged=bool(torch.isfinite(residual).all() and residual_norm <= target),
        iterations=total_iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(rhs_norm, 1.0e-300),
        method="tangent_minres_explicit_residual_replacement",
        breakdown=None if residual_norm <= target else "maximum_total_iterations",
    )
    if not krylov.converged:
        raise RuntimeError(
            "matrix-free tangent MINRES failed: "
            f"iterations={krylov.iterations} "
            f"relative_residual={krylov.relative_residual:.3e} "
            f"residual_replacements={residual_replacements} "
            f"breakdown={krylov.breakdown}"
        )
    density = particular + tangent @ krylov.solution
    projected_density = tangent @ (tangent.T @ density)
    hessian_density = coefficient_hvp(density)
    multiplier = torch.dot(
        q,
        rhs_coefficients - hessian_density - damping * projected_density,
    ) / q_norm_squared
    coefficient_residual = (
        hessian_density
        + damping * projected_density
        + q * multiplier
        - rhs_coefficients
    )
    constraint_residual = torch.dot(q, density) - rhs_constraint
    coefficient_residual_norm = float(
        torch.linalg.vector_norm(coefficient_residual).detach().cpu()
    )
    constraint_residual_abs = float(
        torch.abs(constraint_residual).detach().cpu()
    )
    rhs_norm = float(
        torch.linalg.vector_norm(
            torch.cat((rhs_coefficients.reshape(-1), rhs_constraint.reshape(1)))
        )
        .detach()
        .cpu()
    )
    explicit_target = tolerance * max(rhs_norm, 1.0)
    if (
        not torch.isfinite(coefficient_residual).all()
        or not torch.isfinite(constraint_residual)
        or coefficient_residual_norm > explicit_target
        or constraint_residual_abs > explicit_target
    ):
        raise RuntimeError(
            "matrix-free KKT explicit residual failed: "
            f"coefficient={coefficient_residual_norm:.3e} "
            f"constraint={constraint_residual_abs:.3e} "
            f"target={explicit_target:.3e}"
        )
    return _KktSolve(
        density=density,
        multiplier=multiplier,
        krylov=krylov,
        coefficient_residual_norm=coefficient_residual_norm,
        constraint_residual_abs=constraint_residual_abs,
        preconditioner_diagonal_min_abs=float(
            block_preconditioner.spectrum_min_abs
        ),
        preconditioner_diagonal_max_abs=block_preconditioner.spectrum_max_abs,
        residual_replacements=residual_replacements,
    )


class _ImplicitMatrixFreeGeometryResponse(torch.autograd.Function):
    """Parameter-differentiable KKT solve with a matrix-free implicit VJP."""

    @staticmethod
    def forward(
        ctx: Any,
        system: Any,
        position_direction: torch.Tensor,
        tolerance: float,
        max_iterations: int,
        preconditioner_mode: str,
        preconditioner_block_size: int,
        damping: float,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            mixed = system.mixed_coefficient_direction(
                position_direction, create_graph=False
            )
            constraint_rhs = -system.constraint_position_direction(
                position_direction, create_graph=False
            )
            solved = _matrix_free_kkt_solve(
                system,
                -mixed,
                constraint_rhs,
                tolerance=float(tolerance),
                max_iterations=int(max_iterations),
                damping=float(damping),
                preconditioner_mode=str(preconditioner_mode),
                preconditioner_block_size=int(preconditioner_block_size),
            )
        ctx.system = system
        ctx.tolerance = float(tolerance)
        ctx.max_iterations = int(max_iterations)
        ctx.preconditioner_mode = str(preconditioner_mode)
        ctx.preconditioner_block_size = int(preconditioner_block_size)
        ctx.damping = float(damping)
        ctx.parameter_count = len(parameters)
        ctx.save_for_backward(
            solved.density.detach(),
            solved.multiplier.detach(),
            position_direction.detach(),
            *parameters,
        )
        _FORWARD_DIAGNOSTICS.append(
            {
                "method": "matrix_free_tangent_minres_implicit_adjoint",
                "iterations": solved.krylov.iterations,
                "relative_residual": solved.krylov.relative_residual,
                "coefficient_residual_norm": solved.coefficient_residual_norm,
                "constraint_residual_abs": solved.constraint_residual_abs,
                "preconditioner_diagonal_min_abs": (
                    solved.preconditioner_diagonal_min_abs
                ),
                "preconditioner_diagonal_max_abs": (
                    solved.preconditioner_diagonal_max_abs
                ),
                "residual_replacements": solved.residual_replacements,
            }
        )
        return solved.density.detach(), solved.multiplier.detach()

    @staticmethod
    def backward(
        ctx: Any,
        density_output_gradient: torch.Tensor | None,
        multiplier_output_gradient: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        density = saved[0]
        multiplier = saved[1]
        position_direction = saved[2]
        parameters = saved[3:]
        system = ctx.system
        density_gradient = (
            torch.zeros_like(density)
            if density_output_gradient is None
            else density_output_gradient.detach().to(density)
        )
        multiplier_gradient = (
            torch.zeros_like(multiplier)
            if multiplier_output_gradient is None
            else multiplier_output_gradient.detach().reshape(()).to(multiplier)
        )
        with torch.enable_grad():
            adjoint = _matrix_free_kkt_solve(
                system,
                density_gradient,
                multiplier_gradient,
                tolerance=ctx.tolerance,
                max_iterations=ctx.max_iterations,
                damping=ctx.damping,
                preconditioner_mode=ctx.preconditioner_mode,
                preconditioner_block_size=ctx.preconditioner_block_size,
            )
            q = system.constraint_gradient
            q_norm_squared = torch.dot(q, q)
            projected_density = density - q * (
                torch.dot(q, density) / q_norm_squared
            )
            mixed = system.mixed_coefficient_direction(
                position_direction, create_graph=True
            )
            constraint_rhs = -system.constraint_position_direction(
                position_direction, create_graph=True
            )
            coefficient_equation = (
                system.coefficient_hvp(density, create_graph=True)
                + ctx.damping * projected_density
                + mixed
                + q * multiplier
            )
            parameter_gradients = torch.autograd.grad(
                outputs=coefficient_equation,
                inputs=parameters,
                grad_outputs=-adjoint.density,
                allow_unused=True,
                retain_graph=True,
            )
        _BACKWARD_DIAGNOSTICS.append(
            {
                "method": "matrix_free_tangent_minres_implicit_adjoint",
                "iterations": adjoint.krylov.iterations,
                "relative_residual": adjoint.krylov.relative_residual,
                "coefficient_residual_norm": adjoint.coefficient_residual_norm,
                "constraint_residual_abs": adjoint.constraint_residual_abs,
                "preconditioner_diagonal_min_abs": (
                    adjoint.preconditioner_diagonal_min_abs
                ),
                "preconditioner_diagonal_max_abs": (
                    adjoint.preconditioner_diagonal_max_abs
                ),
                "residual_replacements": adjoint.residual_replacements,
            }
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *parameter_gradients,
        )


def _solve_tangent_matrix_free_implicit(
    self: Any,
    position_direction: torch.Tensor,
    *,
    damping: float = 0.0,
    create_graph: bool = False,
) -> DensityResponseResult:
    settings = _ACTIVE_SETTINGS.get()
    parameters = _ACTIVE_PARAMETERS.get()
    if settings is None or parameters is None:
        raise RuntimeError(
            "matrix-free geometry response called without registered v11 context"
        )
    if create_graph:
        density_response, multiplier_response = (
            _ImplicitMatrixFreeGeometryResponse.apply(
                self,
                position_direction,
                settings.tolerance,
                settings.max_iterations,
                settings.preconditioner,
                settings.preconditioner_block_size,
                damping,
                *parameters,
            )
        )
        forward = _FORWARD_DIAGNOSTICS[-1]
        iterations = int(forward["iterations"])
        relative_residual = float(forward["relative_residual"])
    else:
        with torch.enable_grad():
            mixed = self.mixed_coefficient_direction(
                position_direction, create_graph=False
            )
            constraint_rhs = -self.constraint_position_direction(
                position_direction, create_graph=False
            )
            solved = _matrix_free_kkt_solve(
                self,
                -mixed,
                constraint_rhs,
                tolerance=settings.tolerance,
                max_iterations=settings.max_iterations,
                damping=damping,
                preconditioner_mode=settings.preconditioner,
                preconditioner_block_size=settings.preconditioner_block_size,
            )
        density_response = solved.density
        multiplier_response = solved.multiplier
        iterations = solved.krylov.iterations
        relative_residual = solved.krylov.relative_residual

    first_equation_without_multiplier = (
        self.coefficient_hvp(
            density_response, create_graph=create_graph
        )
        + self.mixed_coefficient_direction(
            position_direction, create_graph=create_graph
        )
    )
    q = self.constraint_gradient
    stationarity_residual = (
        first_equation_without_multiplier + q * multiplier_response
    )
    constraint_rhs = -self.constraint_position_direction(
        position_direction, create_graph=create_graph
    )
    constraint_residual = torch.dot(q, density_response) - constraint_rhs
    residual_norm = float(
        torch.linalg.vector_norm(stationarity_residual).detach().cpu()
    )
    rhs_norm = float(
        torch.linalg.vector_norm(first_equation_without_multiplier).detach().cpu()
    )
    return DensityResponseResult(
        density_response=density_response,
        multiplier_response=multiplier_response,
        krylov=KrylovResult(
            solution=density_response,
            converged=bool(
                torch.isfinite(stationarity_residual).all()
                and torch.isfinite(constraint_residual)
            ),
            iterations=iterations,
            residual_norm=residual_norm,
            relative_residual=(
                relative_residual
                if rhs_norm == 0.0
                else residual_norm / max(rhs_norm, 1.0e-300)
            ),
            method="matrix_free_tangent_minres_implicit_adjoint",
        ),
        stationarity_direction_residual=residual_norm,
        constraint_direction_residual=float(
            torch.abs(constraint_residual).detach().cpu()
        ),
    )


def install_into_canonical_core(core_module: Any) -> None:
    """Install the overlay in this process without editing canonical files."""
    global _INSTALLED
    if _INSTALLED:
        return
    response_class = core_module.ConstrainedResponseSystem
    response_class.solve_tangent_direct_implicit = (
        _solve_tangent_matrix_free_implicit
    )
    _INSTALLED = True
