"""Constrained density response and Krylov solvers for total-OFDFT Hessian products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, minres

from mldft.ofdft.functional_factory import constrained_energy_lagrangian


TensorOperator = Callable[[torch.Tensor], torch.Tensor]
GeometryEnergyFunction = Callable[[torch.Tensor, np.ndarray], torch.Tensor]


@dataclass
class KrylovResult:
    solution: torch.Tensor
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    method: str
    breakdown: str | None = None


@dataclass
class BlockKrylovResult:
    solution: torch.Tensor
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    method: str
    breakdown: str | None = None


@dataclass
class DensityResponseResult:
    density_response: torch.Tensor
    multiplier_response: torch.Tensor
    krylov: KrylovResult
    stationarity_direction_residual: float
    constraint_direction_residual: float


class _ImplicitSymmetricLinearSolve(torch.autograd.Function):
    """Dense linear solve whose backward reuses the forward LU factors.

    For ``X=A^-1 B`` the adjoint is ``Lambda=A^-T dL/dX`` with
    ``dL/dB=Lambda`` and ``dL/dA=-Lambda X^T``.  This avoids differentiating
    through an iterative solver while retaining exact parameter gradients.
    """

    @staticmethod
    def forward(ctx, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != matrix.shape[0]:
            raise ValueError("rhs must have shape (n,) or (n, nrhs)")
        rhs_matrix = rhs.unsqueeze(1) if rhs.ndim == 1 else rhs
        lu, pivots = torch.linalg.lu_factor(matrix.detach())
        solution = torch.linalg.lu_solve(lu, pivots, rhs_matrix.detach())
        ctx.rhs_was_vector = rhs.ndim == 1
        ctx.save_for_backward(lu, pivots, solution)
        return solution[:, 0] if rhs.ndim == 1 else solution

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        lu, pivots, solution = ctx.saved_tensors
        gradient_matrix = (
            output_gradient.unsqueeze(1)
            if ctx.rhs_was_vector
            else output_gradient
        )
        adjoint = torch.linalg.lu_solve(
            lu, pivots, gradient_matrix, adjoint=True
        )
        matrix_gradient = -(adjoint @ solution.T)
        rhs_gradient = adjoint[:, 0] if ctx.rhs_was_vector else adjoint
        return matrix_gradient, rhs_gradient


def implicit_symmetric_linear_solve(
    matrix: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Solve a dense response system with an exact implicit-adjoint backward."""
    return _ImplicitSymmetricLinearSolve.apply(matrix, rhs)


def preconditioned_conjugate_gradient(
    operator: TensorOperator,
    rhs: torch.Tensor,
    preconditioner: TensorOperator | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    initial_guess: torch.Tensor | None = None,
) -> KrylovResult:
    """Solve an SPD linear system with optional left preconditioning."""
    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    if max_iterations is None:
        max_iterations = max(20, 2 * rhs.numel())
    if initial_guess is not None and initial_guess.shape != rhs.shape:
        raise ValueError("initial_guess must have the same shape as rhs")
    x = (
        torch.zeros_like(rhs)
        if initial_guess is None
        else initial_guess.detach().clone().to(rhs)
    )
    residual = rhs - operator(x)
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    target = tolerance * max(rhs_norm, 1.0)
    z = residual if preconditioner is None else preconditioner(residual)
    direction = z.clone()
    rz = torch.dot(residual, z)
    residual_norm = float(torch.linalg.vector_norm(residual))
    if residual_norm <= target:
        return KrylovResult(x, True, 0, residual_norm, residual_norm / max(rhs_norm, 1e-300), "pcg")

    for iteration in range(1, max_iterations + 1):
        operator_direction = operator(direction)
        curvature = torch.dot(direction, operator_direction)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
            return KrylovResult(
                x,
                False,
                iteration - 1,
                residual_norm,
                residual_norm / max(rhs_norm, 1e-300),
                "pcg",
                breakdown="non_positive_curvature",
            )
        alpha = rz / curvature
        x = x + alpha * direction
        residual = residual - alpha * operator_direction
        residual_norm = float(torch.linalg.vector_norm(residual))
        if residual_norm <= target:
            return KrylovResult(
                x,
                True,
                iteration,
                residual_norm,
                residual_norm / max(rhs_norm, 1e-300),
                "pcg",
            )
        z = residual if preconditioner is None else preconditioner(residual)
        next_rz = torch.dot(residual, z)
        if not bool(torch.isfinite(next_rz)):
            return KrylovResult(
                x,
                False,
                iteration,
                residual_norm,
                residual_norm / max(rhs_norm, 1e-300),
                "pcg",
                breakdown="non_finite_preconditioned_residual",
            )
        beta = next_rz / rz
        direction = z + beta * direction
        rz = next_rz

    return KrylovResult(
        x,
        False,
        max_iterations,
        residual_norm,
        residual_norm / max(rhs_norm, 1e-300),
        "pcg",
        breakdown="maximum_iterations",
    )


def block_preconditioned_conjugate_gradient(
    operator: TensorOperator,
    rhs: torch.Tensor,
    preconditioner: TensorOperator | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
    initial_guess: torch.Tensor | None = None,
) -> BlockKrylovResult:
    """Solve an SPD system for several right-hand sides with block CG."""
    if rhs.ndim != 2:
        raise ValueError("block rhs must have shape (n, nrhs)")
    if max_iterations is None:
        max_iterations = max(20, 2 * rhs.shape[0])
    if initial_guess is not None and initial_guess.shape != rhs.shape:
        raise ValueError("initial_guess must have the same shape as rhs")

    def apply_columns(function: TensorOperator, matrix: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [function(matrix[:, column]) for column in range(matrix.shape[1])],
            dim=1,
        )

    solution = (
        torch.zeros_like(rhs)
        if initial_guess is None
        else initial_guess.detach().clone().to(rhs)
    )
    residual = rhs - apply_columns(operator, solution)
    rhs_norm = float(torch.linalg.matrix_norm(rhs))
    target = tolerance * max(rhs_norm, 1.0)
    residual_norm = float(torch.linalg.matrix_norm(residual))
    if residual_norm <= target:
        return BlockKrylovResult(
            solution,
            True,
            0,
            residual_norm,
            residual_norm / max(rhs_norm, 1.0e-300),
            "block_pcg",
        )
    preconditioned = (
        residual
        if preconditioner is None
        else apply_columns(preconditioner, residual)
    )
    search = preconditioned.clone()
    gram = residual.T @ preconditioned

    for iteration in range(1, max_iterations + 1):
        operator_search = apply_columns(operator, search)
        curvature = search.T @ operator_search
        if not bool(torch.isfinite(curvature).all()):
            return BlockKrylovResult(
                solution,
                False,
                iteration - 1,
                residual_norm,
                residual_norm / max(rhs_norm, 1.0e-300),
                "block_pcg",
                breakdown="non_finite_block_curvature",
            )
        alpha = torch.linalg.pinv(curvature) @ gram
        solution = solution + search @ alpha
        residual = residual - operator_search @ alpha
        residual_norm = float(torch.linalg.matrix_norm(residual))
        if residual_norm <= target:
            return BlockKrylovResult(
                solution,
                True,
                iteration,
                residual_norm,
                residual_norm / max(rhs_norm, 1.0e-300),
                "block_pcg",
            )
        next_preconditioned = (
            residual
            if preconditioner is None
            else apply_columns(preconditioner, residual)
        )
        next_gram = residual.T @ next_preconditioned
        if not bool(torch.isfinite(next_gram).all()):
            return BlockKrylovResult(
                solution,
                False,
                iteration,
                residual_norm,
                residual_norm / max(rhs_norm, 1.0e-300),
                "block_pcg",
                breakdown="non_finite_preconditioned_block_residual",
            )
        beta = torch.linalg.pinv(gram) @ next_gram
        search = next_preconditioned + search @ beta
        preconditioned = next_preconditioned
        gram = next_gram

    return BlockKrylovResult(
        solution,
        False,
        max_iterations,
        residual_norm,
        residual_norm / max(rhs_norm, 1.0e-300),
        "block_pcg",
        breakdown="maximum_iterations",
    )


def preconditioned_minimum_residual(
    operator: TensorOperator,
    rhs: torch.Tensor,
    preconditioner: TensorOperator | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    initial_guess: torch.Tensor | None = None,
) -> KrylovResult:
    """Solve a symmetric, possibly indefinite system with SciPy MINRES."""
    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    if max_iterations is None:
        max_iterations = max(20, 2 * rhs.numel())
    dtype = rhs.dtype
    device = rhs.device

    def to_numpy(vector: torch.Tensor) -> np.ndarray:
        return vector.detach().cpu().numpy().astype(np.float64, copy=False)

    def matvec(vector_np: np.ndarray) -> np.ndarray:
        vector = torch.as_tensor(vector_np, dtype=dtype, device=device)
        return to_numpy(operator(vector))

    linear_operator = LinearOperator(
        (rhs.numel(), rhs.numel()), matvec=matvec, dtype=np.float64
    )
    scipy_preconditioner = None
    if preconditioner is not None:

        def preconditioner_matvec(vector_np: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_np, dtype=dtype, device=device)
            return to_numpy(preconditioner(vector))

        scipy_preconditioner = LinearOperator(
            (rhs.numel(), rhs.numel()),
            matvec=preconditioner_matvec,
            dtype=np.float64,
        )

    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solution_np, info = minres(
        linear_operator,
        to_numpy(rhs),
        x0=None if initial_guess is None else to_numpy(initial_guess),
        rtol=tolerance,
        maxiter=max_iterations,
        M=scipy_preconditioner,
        callback=callback,
        show=False,
    )
    solution = torch.as_tensor(solution_np, dtype=dtype, device=device)
    residual = operator(solution) - rhs
    residual_norm = float(torch.linalg.vector_norm(residual))
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    target = tolerance * max(rhs_norm, 1.0)
    converged = info == 0 and residual_norm <= target
    breakdown = None
    if not converged:
        breakdown = (
            f"scipy_info_{info}" if info != 0 else "explicit_residual_above_tolerance"
        )
    return KrylovResult(
        solution=solution,
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(rhs_norm, 1e-300),
        method="tangent_minres",
        breakdown=breakdown,
    )


def deflated_preconditioned_conjugate_gradient(
    operator: TensorOperator,
    rhs: torch.Tensor,
    deflation_vectors: torch.Tensor,
    preconditioner: TensorOperator | None = None,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
) -> KrylovResult:
    """Solve an SPD system after a Galerkin coarse solve in supplied low-mode vectors."""
    if deflation_vectors.ndim != 2 or deflation_vectors.shape[0] != rhs.numel():
        raise ValueError("deflation_vectors must have shape (rhs.numel(), n_modes)")
    if deflation_vectors.shape[1] == 0:
        return preconditioned_conjugate_gradient(
            operator,
            rhs,
            preconditioner=preconditioner,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    vectors, _ = torch.linalg.qr(deflation_vectors, mode="reduced")

    def project(vector: torch.Tensor) -> torch.Tensor:
        return vector - vectors @ (vectors.T @ vector)

    applied_vectors = torch.stack(
        [operator(vectors[:, column]) for column in range(vectors.shape[1])],
        dim=1,
    )
    coarse_matrix = vectors.T @ applied_vectors
    coarse_matrix = 0.5 * (coarse_matrix + coarse_matrix.T)
    coarse_solution = vectors @ torch.linalg.solve(coarse_matrix, vectors.T @ rhs)
    projected_rhs = project(rhs - operator(coarse_solution))

    def projected_operator(vector: torch.Tensor) -> torch.Tensor:
        return project(operator(project(vector)))

    projected_preconditioner = None
    if preconditioner is not None:
        projected_preconditioner = lambda vector: project(preconditioner(project(vector)))
    fine = preconditioned_conjugate_gradient(
        projected_operator,
        projected_rhs,
        preconditioner=projected_preconditioner,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    solution = coarse_solution + project(fine.solution)
    residual = rhs - operator(solution)
    residual_norm = float(torch.linalg.vector_norm(residual))
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    target = tolerance * max(rhs_norm, 1.0)
    return KrylovResult(
        solution=solution,
        converged=fine.converged and residual_norm <= target,
        iterations=fine.iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(rhs_norm, 1e-300),
        method="deflated_pcg",
        breakdown=fine.breakdown if residual_norm > target else None,
    )


def tangent_basis(constraint_gradient: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for vectors orthogonal to one constraint gradient."""
    if constraint_gradient.ndim != 1:
        raise ValueError("constraint_gradient must be one-dimensional")
    if float(torch.linalg.vector_norm(constraint_gradient)) == 0:
        raise ValueError("constraint_gradient must be nonzero")
    complete_q, _ = torch.linalg.qr(constraint_gradient.reshape(-1, 1), mode="complete")
    return complete_q[:, 1:]


class ConstrainedResponseSystem:
    """Reusable HVP/KKT operator at one density and molecular geometry."""

    def __init__(
        self,
        total_energy: torch.Tensor,
        coeffs: torch.Tensor,
        positions: torch.Tensor,
        normalization: torch.Tensor,
        n_electron: float | torch.Tensor,
        multiplier: torch.Tensor | float,
    ) -> None:
        self.coeffs = coeffs
        self.positions = positions
        self.normalization = normalization
        if isinstance(multiplier, torch.Tensor):
            self.multiplier = multiplier.detach().clone().requires_grad_(True)
        else:
            self.multiplier = torch.tensor(
                multiplier, dtype=coeffs.dtype, device=coeffs.device, requires_grad=True
            )
        self.lagrangian = constrained_energy_lagrangian(
            total_energy,
            coeffs,
            normalization,
            n_electron,
            self.multiplier,
        )
        self.constraint = normalization @ coeffs - torch.as_tensor(
            n_electron, dtype=coeffs.dtype, device=coeffs.device
        )
        self.gradient_coeffs = torch.autograd.grad(
            self.lagrangian, coeffs, create_graph=True, retain_graph=True
        )[0]
        self.gradient_positions = torch.autograd.grad(
            self.lagrangian, positions, create_graph=True, retain_graph=True
        )[0]
        self.constraint_gradient = torch.autograd.grad(
            self.constraint, coeffs, create_graph=True, retain_graph=True
        )[0]
        self._tangent_basis: torch.Tensor | None = None

    @property
    def tangent(self) -> torch.Tensor:
        if self._tangent_basis is None:
            self._tangent_basis = tangent_basis(self.constraint_gradient)
        return self._tangent_basis

    def coefficient_hvp(
        self, vector: torch.Tensor, *, create_graph: bool = False
    ) -> torch.Tensor:
        """Apply the constrained coefficient Hessian ``L_cc``."""
        return torch.autograd.grad(
            self.gradient_coeffs,
            self.coeffs,
            grad_outputs=vector,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    def mixed_coefficient_direction(
        self,
        position_direction: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``L_cR @ v_R``."""
        return torch.autograd.grad(
            self.gradient_positions,
            self.coeffs,
            grad_outputs=position_direction,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    def constraint_position_direction(
        self,
        position_direction: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        constraint_position_gradient = torch.autograd.grad(
            self.constraint,
            self.positions,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if constraint_position_gradient is None:
            return torch.zeros(
                (), dtype=self.positions.dtype, device=self.positions.device
            )
        return torch.dot(
            constraint_position_gradient.reshape(-1), position_direction.reshape(-1)
        )

    def response_correction(
        self,
        density_response: torch.Tensor,
        multiplier_response: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``L_Rc dc + L_Rmu dmu`` for an energy-Hessian product."""
        coefficient_term = torch.autograd.grad(
            self.gradient_coeffs,
            self.positions,
            grad_outputs=density_response,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        constraint_term = torch.autograd.grad(
            self.constraint,
            self.positions,
            grad_outputs=multiplier_response,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if constraint_term is None:
            constraint_term = torch.zeros_like(coefficient_term)
        return coefficient_term + constraint_term

    def partial_position_hvp(
        self,
        position_direction: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return the fixed-KKT-variable partial product ``L_RR @ v``."""
        return torch.autograd.grad(
            self.gradient_positions,
            self.positions,
            grad_outputs=position_direction,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    def relaxed_hvp(
        self,
        position_direction: torch.Tensor,
        response: DensityResponseResult,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``L_RR v + L_Ry y_v`` from one stationary center graph."""
        return self.partial_position_hvp(
            position_direction, create_graph=create_graph
        ) + self.response_correction(
            response.density_response,
            response.multiplier_response,
            create_graph=create_graph,
        )

    def tangent_matrix(self, *, create_graph: bool = False) -> torch.Tensor:
        """Assemble the reusable constrained coefficient Hessian."""
        tangent = self.tangent
        columns = [
            tangent.T
            @ self.coefficient_hvp(
                tangent[:, column], create_graph=create_graph
            )
            for column in range(tangent.shape[1])
        ]
        matrix = torch.stack(columns, dim=1)
        return 0.5 * (matrix + matrix.T)

    def solve_tangent_direct_implicit(
        self,
        position_direction: torch.Tensor,
        *,
        damping: float = 0.0,
        create_graph: bool = False,
    ) -> DensityResponseResult:
        """Solve the tangent KKT response with a reusable implicit adjoint.

        This dense path is the correctness reference for a single small
        molecule.  Its backward uses a second solve rather than unrolling the
        linear solver.
        """
        if damping < 0:
            raise ValueError("damping must be nonnegative")
        q = self.constraint_gradient
        q_norm_squared = torch.dot(q, q)
        mixed = self.mixed_coefficient_direction(
            position_direction, create_graph=create_graph
        )
        constraint_rhs = -self.constraint_position_direction(
            position_direction, create_graph=create_graph
        )
        particular = q * (constraint_rhs / q_norm_squared)
        tangent = self.tangent
        matrix = self.tangent_matrix(create_graph=create_graph)
        if damping:
            matrix = matrix + damping * torch.eye(
                matrix.shape[0], dtype=matrix.dtype, device=matrix.device
            )
        rhs = tangent.T @ (
            -mixed - self.coefficient_hvp(
                particular, create_graph=create_graph
            )
        )
        tangent_response = implicit_symmetric_linear_solve(matrix, rhs)
        density_response = particular + tangent @ tangent_response
        first_equation_without_multiplier = (
            self.coefficient_hvp(
                density_response, create_graph=create_graph
            )
            + mixed
        )
        multiplier_response = (
            -torch.dot(q, first_equation_without_multiplier) / q_norm_squared
        )
        stationarity_residual = (
            first_equation_without_multiplier + q * multiplier_response
        )
        constraint_residual = torch.dot(q, density_response) - constraint_rhs
        residual_norm = float(
            torch.linalg.vector_norm(stationarity_residual).detach().cpu()
        )
        rhs_norm = float(torch.linalg.vector_norm(mixed).detach().cpu())
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=KrylovResult(
                solution=tangent_response,
                converged=bool(
                    torch.isfinite(stationarity_residual).all()
                    and torch.isfinite(constraint_residual)
                ),
                iterations=1,
                residual_norm=residual_norm,
                relative_residual=residual_norm / max(rhs_norm, 1.0e-300),
                method="dense_direct_implicit_adjoint",
            ),
            stationarity_direction_residual=residual_norm,
            constraint_direction_residual=float(
                torch.abs(constraint_residual).detach().cpu()
            ),
        )

    def solve_tangent_pcg(
        self,
        position_direction: torch.Tensor,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
        tangent_inverse_diagonal: torch.Tensor | None = None,
    ) -> DensityResponseResult:
        """Solve the one-constraint KKT response in its SPD tangent subspace."""
        q = self.constraint_gradient
        q_norm_squared = torch.dot(q, q)
        mixed = self.mixed_coefficient_direction(position_direction)
        constraint_rhs = -self.constraint_position_direction(position_direction)
        particular = q * (constraint_rhs / q_norm_squared)
        tangent = self.tangent

        def tangent_operator(vector: torch.Tensor) -> torch.Tensor:
            return tangent.T @ self.coefficient_hvp(tangent @ vector)

        rhs = tangent.T @ (-mixed - self.coefficient_hvp(particular))
        preconditioner = None
        if tangent_inverse_diagonal is not None:
            if tangent_inverse_diagonal.shape != rhs.shape:
                raise ValueError("tangent_inverse_diagonal has incompatible shape")
            preconditioner = lambda vector: tangent_inverse_diagonal * vector
        krylov = preconditioned_conjugate_gradient(
            tangent_operator,
            rhs,
            preconditioner=preconditioner,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        density_response = particular + tangent @ krylov.solution
        first_equation_without_multiplier = self.coefficient_hvp(density_response) + mixed
        multiplier_response = -torch.dot(q, first_equation_without_multiplier) / q_norm_squared
        stationarity_residual = (
            first_equation_without_multiplier + q * multiplier_response
        )
        constraint_residual = torch.dot(q, density_response) - constraint_rhs
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=krylov,
            stationarity_direction_residual=float(torch.linalg.vector_norm(stationarity_residual)),
            constraint_direction_residual=float(torch.abs(constraint_residual)),
        )

    def solve_kkt_minres(
        self,
        position_direction: torch.Tensor,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
    ) -> DensityResponseResult:
        """Solve the full symmetric-indefinite KKT system with SciPy MINRES."""
        mixed = self.mixed_coefficient_direction(position_direction)
        constraint_rhs = -self.constraint_position_direction(position_direction)
        rhs_torch = torch.cat((-mixed, constraint_rhs.reshape(1)))
        n_coeff = self.coeffs.numel()
        dtype = self.coeffs.dtype
        device = self.coeffs.device

        def matvec(vector_np: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_np, dtype=dtype, device=device)
            density_vector = vector[:n_coeff]
            multiplier_vector = vector[n_coeff]
            result = torch.cat(
                (
                    self.coefficient_hvp(density_vector)
                    + self.constraint_gradient * multiplier_vector,
                    torch.dot(self.constraint_gradient, density_vector).reshape(1),
                )
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)

        iterations = 0

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        operator = LinearOperator(
            (n_coeff + 1, n_coeff + 1), matvec=matvec, dtype=np.float64
        )
        solution_np, info = minres(
            operator,
            rhs_torch.detach().cpu().numpy().astype(np.float64),
            rtol=tolerance,
            maxiter=max_iterations,
            callback=callback,
            show=False,
        )
        solution = torch.as_tensor(solution_np, dtype=dtype, device=device)
        residual = torch.as_tensor(
            matvec(solution_np), dtype=dtype, device=device
        ) - rhs_torch
        residual_norm = float(torch.linalg.vector_norm(residual))
        rhs_norm = float(torch.linalg.vector_norm(rhs_torch))
        target = tolerance * max(rhs_norm, 1.0)
        converged = info == 0 and residual_norm <= target
        breakdown = None
        if not converged:
            breakdown = (
                f"scipy_info_{info}"
                if info != 0
                else "explicit_residual_above_tolerance"
            )
        krylov = KrylovResult(
            solution=solution,
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
            relative_residual=residual_norm / max(rhs_norm, 1e-300),
            method="minres",
            breakdown=breakdown,
        )
        return DensityResponseResult(
            density_response=solution[:n_coeff],
            multiplier_response=solution[n_coeff],
            krylov=krylov,
            stationarity_direction_residual=float(torch.linalg.vector_norm(residual[:n_coeff])),
            constraint_direction_residual=float(torch.abs(residual[n_coeff])),
        )


class FiniteDifferenceGeometryResponseSystem:
    """KKT response with autograd ``E_cc`` and numerical geometry mixed derivatives.

    The coefficient variables are components in the physical moving auxiliary basis. For the
    normalized atom-centred basis currently used by MLDFT, the electron-number vector is invariant
    to nuclear translation of its own centres, so the tangent constraint has no geometry RHS.
    Geometry-dependent model preprocessing is rebuilt independently at ``R +/- h v``; this avoids
    relying on the numerically unstable eigensystem derivative of the natural reparametrization.
    """

    def __init__(
        self,
        energy_function: GeometryEnergyFunction,
        coeffs: torch.Tensor,
        positions_bohr: np.ndarray | torch.Tensor,
        normalization: torch.Tensor,
        n_electron: float | torch.Tensor,
    ) -> None:
        self.energy_function = energy_function
        self.coeffs = coeffs.detach().clone().requires_grad_(True)
        self.positions_bohr = np.asarray(
            positions_bohr.detach().cpu()
            if isinstance(positions_bohr, torch.Tensor)
            else positions_bohr,
            dtype=np.float64,
        )
        self.normalization = normalization.detach().to(self.coeffs)
        target = torch.as_tensor(
            n_electron, dtype=self.coeffs.dtype, device=self.coeffs.device
        )
        self.constraint = torch.dot(self.normalization, self.coeffs) - target
        self.total_energy = self.energy_function(self.coeffs, self.positions_bohr)
        self.gradient_coeffs = torch.autograd.grad(
            self.total_energy, self.coeffs, create_graph=True, retain_graph=True
        )[0]
        norm_squared = torch.dot(self.normalization, self.normalization)
        self.multiplier = -torch.dot(
            self.normalization, self.gradient_coeffs
        ) / norm_squared
        self.projected_gradient = (
            self.gradient_coeffs + self.multiplier * self.normalization
        )
        self._tangent_basis: torch.Tensor | None = None

    @property
    def tangent(self) -> torch.Tensor:
        if self._tangent_basis is None:
            self._tangent_basis = tangent_basis(self.normalization)
        return self._tangent_basis

    def coefficient_hvp(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            torch.dot(self.gradient_coeffs, vector),
            self.coeffs,
            retain_graph=True,
        )[0]

    def _coefficient_gradient_at(self, positions_bohr: np.ndarray) -> torch.Tensor:
        variable = self.coeffs.detach().clone().requires_grad_(True)
        energy = self.energy_function(variable, positions_bohr)
        return torch.autograd.grad(energy, variable)[0].detach()

    def mixed_coefficient_direction(
        self, position_direction: np.ndarray | torch.Tensor, step_bohr: float
    ) -> torch.Tensor:
        """Return ``E_cR @ v`` from rebuilt-geometry coefficient gradients."""
        if step_bohr <= 0:
            raise ValueError("step_bohr must be positive")
        direction = np.asarray(
            position_direction.detach().cpu()
            if isinstance(position_direction, torch.Tensor)
            else position_direction,
            dtype=np.float64,
        )
        if direction.shape != self.positions_bohr.shape:
            raise ValueError(
                f"position_direction must have shape {self.positions_bohr.shape}, got {direction.shape}"
            )
        plus = self.positions_bohr + step_bohr * direction
        minus = self.positions_bohr - step_bohr * direction
        return (
            self._coefficient_gradient_at(plus)
            - self._coefficient_gradient_at(minus)
        ) / (2.0 * step_bohr)

    def estimate_tangent_inverse_diagonal(
        self, probes: int = 8, damping: float = 1e-8, seed: int = 1729
    ) -> torch.Tensor:
        """Estimate a positive Jacobi preconditioner with Hutchinson probes."""
        if probes <= 0:
            raise ValueError("probes must be positive")
        tangent = self.tangent
        diagonal = torch.zeros(
            tangent.shape[1], dtype=self.coeffs.dtype, device=self.coeffs.device
        )
        generator = torch.Generator(device=diagonal.device)
        generator.manual_seed(seed)
        for _ in range(probes):
            signs = torch.randint(
                0,
                2,
                diagonal.shape,
                generator=generator,
                device=diagonal.device,
                dtype=torch.int64,
            ).to(diagonal.dtype)
            signs = 2.0 * signs - 1.0
            applied = tangent.T @ self.coefficient_hvp(tangent @ signs)
            diagonal += signs * applied
        diagonal = torch.abs(diagonal / probes).clamp_min(damping)
        return 1.0 / diagonal

    def estimate_low_tangent_modes(
        self,
        modes: int,
        tolerance: float = 1e-6,
        max_iterations: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the smallest tangent-Hessian modes with matrix-free Lanczos."""
        tangent = self.tangent
        dimension = tangent.shape[1]
        if modes <= 0 or modes >= dimension:
            raise ValueError(f"modes must be in [1, {dimension - 1}]")
        dtype = self.coeffs.dtype
        device = self.coeffs.device

        def matvec(vector_np: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_np, dtype=dtype, device=device)
            applied = tangent.T @ self.coefficient_hvp(tangent @ vector)
            return applied.detach().cpu().numpy().astype(np.float64, copy=False)

        operator = LinearOperator(
            (dimension, dimension), matvec=matvec, dtype=np.float64
        )
        eigenvalues, eigenvectors = eigsh(
            operator,
            k=modes,
            which="SA",
            tol=tolerance,
            maxiter=max_iterations,
        )
        order = np.argsort(eigenvalues)
        return (
            torch.as_tensor(eigenvalues[order], dtype=dtype, device=device),
            torch.as_tensor(eigenvectors[:, order], dtype=dtype, device=device),
        )

    def solve_tangent_pcg(
        self,
        position_direction: np.ndarray | torch.Tensor,
        geometry_step_bohr: float,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
        tangent_inverse_diagonal: torch.Tensor | None = None,
    ) -> DensityResponseResult:
        mixed = self.mixed_coefficient_direction(
            position_direction, geometry_step_bohr
        )
        tangent = self.tangent

        def tangent_operator(vector: torch.Tensor) -> torch.Tensor:
            return tangent.T @ self.coefficient_hvp(tangent @ vector)

        rhs = -tangent.T @ mixed
        preconditioner = None
        if tangent_inverse_diagonal is not None:
            if tangent_inverse_diagonal.shape != rhs.shape:
                raise ValueError("tangent_inverse_diagonal has incompatible shape")
            preconditioner = lambda vector: tangent_inverse_diagonal * vector
        krylov = preconditioned_conjugate_gradient(
            tangent_operator,
            rhs,
            preconditioner=preconditioner,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        density_response = tangent @ krylov.solution
        first_equation_without_multiplier = (
            self.coefficient_hvp(density_response) + mixed
        )
        norm_squared = torch.dot(self.normalization, self.normalization)
        multiplier_response = -torch.dot(
            self.normalization, first_equation_without_multiplier
        ) / norm_squared
        stationarity_residual = (
            first_equation_without_multiplier
            + self.normalization * multiplier_response
        )
        constraint_residual = torch.dot(self.normalization, density_response)
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=krylov,
            stationarity_direction_residual=float(
                torch.linalg.vector_norm(stationarity_residual)
            ),
            constraint_direction_residual=float(torch.abs(constraint_residual)),
        )

    def solve_tangent_minres(
        self,
        position_direction: np.ndarray | torch.Tensor,
        geometry_step_bohr: float,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
        tangent_inverse_diagonal: torch.Tensor | None = None,
        initial_guess: torch.Tensor | None = None,
    ) -> DensityResponseResult:
        """Solve the tangent response with MINRES and an optional SPD preconditioner."""
        mixed = self.mixed_coefficient_direction(
            position_direction, geometry_step_bohr
        )
        tangent = self.tangent

        def tangent_operator(vector: torch.Tensor) -> torch.Tensor:
            return tangent.T @ self.coefficient_hvp(tangent @ vector)

        rhs = -tangent.T @ mixed
        preconditioner = None
        if tangent_inverse_diagonal is not None:
            if tangent_inverse_diagonal.shape != rhs.shape:
                raise ValueError("tangent_inverse_diagonal has incompatible shape")
            preconditioner = lambda vector: tangent_inverse_diagonal * vector
        krylov = preconditioned_minimum_residual(
            tangent_operator,
            rhs,
            preconditioner=preconditioner,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=initial_guess,
        )
        density_response = tangent @ krylov.solution
        first_equation_without_multiplier = (
            self.coefficient_hvp(density_response) + mixed
        )
        norm_squared = torch.dot(self.normalization, self.normalization)
        multiplier_response = -torch.dot(
            self.normalization, first_equation_without_multiplier
        ) / norm_squared
        stationarity_residual = (
            first_equation_without_multiplier
            + self.normalization * multiplier_response
        )
        constraint_residual = torch.dot(self.normalization, density_response)
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=krylov,
            stationarity_direction_residual=float(
                torch.linalg.vector_norm(stationarity_residual)
            ),
            constraint_direction_residual=float(torch.abs(constraint_residual)),
        )

    def solve_tangent_deflated_pcg(
        self,
        position_direction: np.ndarray | torch.Tensor,
        geometry_step_bohr: float,
        deflation_vectors: torch.Tensor,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
        tangent_inverse_diagonal: torch.Tensor | None = None,
    ) -> DensityResponseResult:
        """Solve tangent response after removing ill-conditioned low Hessian modes."""
        mixed = self.mixed_coefficient_direction(
            position_direction, geometry_step_bohr
        )
        tangent = self.tangent

        def tangent_operator(vector: torch.Tensor) -> torch.Tensor:
            return tangent.T @ self.coefficient_hvp(tangent @ vector)

        preconditioner = None
        if tangent_inverse_diagonal is not None:
            preconditioner = lambda vector: tangent_inverse_diagonal * vector
        krylov = deflated_preconditioned_conjugate_gradient(
            tangent_operator,
            -tangent.T @ mixed,
            deflation_vectors,
            preconditioner=preconditioner,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        density_response = tangent @ krylov.solution
        first_equation_without_multiplier = (
            self.coefficient_hvp(density_response) + mixed
        )
        norm_squared = torch.dot(self.normalization, self.normalization)
        multiplier_response = -torch.dot(
            self.normalization, first_equation_without_multiplier
        ) / norm_squared
        stationarity_residual = (
            first_equation_without_multiplier
            + self.normalization * multiplier_response
        )
        constraint_residual = torch.dot(self.normalization, density_response)
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=krylov,
            stationarity_direction_residual=float(
                torch.linalg.vector_norm(stationarity_residual)
            ),
            constraint_direction_residual=float(torch.abs(constraint_residual)),
        )

    def solve_kkt_minres(
        self,
        position_direction: np.ndarray | torch.Tensor,
        geometry_step_bohr: float,
        tolerance: float = 1e-10,
        max_iterations: int | None = None,
    ) -> DensityResponseResult:
        mixed = self.mixed_coefficient_direction(
            position_direction, geometry_step_bohr
        )
        rhs_torch = torch.cat((-mixed, torch.zeros(1, dtype=mixed.dtype, device=mixed.device)))
        n_coeff = self.coeffs.numel()
        dtype = self.coeffs.dtype
        device = self.coeffs.device

        def matvec(vector_np: np.ndarray) -> np.ndarray:
            vector = torch.as_tensor(vector_np, dtype=dtype, device=device)
            result = torch.cat(
                (
                    self.coefficient_hvp(vector[:n_coeff])
                    + self.normalization * vector[n_coeff],
                    torch.dot(self.normalization, vector[:n_coeff]).reshape(1),
                )
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)

        iterations = 0

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        operator = LinearOperator(
            (n_coeff + 1, n_coeff + 1), matvec=matvec, dtype=np.float64
        )
        solution_np, info = minres(
            operator,
            rhs_torch.detach().cpu().numpy().astype(np.float64),
            rtol=tolerance,
            maxiter=max_iterations,
            callback=callback,
            show=False,
        )
        solution = torch.as_tensor(solution_np, dtype=dtype, device=device)
        residual = torch.as_tensor(matvec(solution_np), dtype=dtype, device=device) - rhs_torch
        residual_norm = float(torch.linalg.vector_norm(residual))
        rhs_norm = float(torch.linalg.vector_norm(rhs_torch))
        target = tolerance * max(rhs_norm, 1.0)
        converged = info == 0 and residual_norm <= target
        breakdown = None
        if not converged:
            breakdown = (
                f"scipy_info_{info}"
                if info != 0
                else "explicit_residual_above_tolerance"
            )
        krylov = KrylovResult(
            solution=solution,
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
            relative_residual=residual_norm / max(rhs_norm, 1e-300),
            method="minres",
            breakdown=breakdown,
        )
        return DensityResponseResult(
            density_response=solution[:n_coeff],
            multiplier_response=solution[n_coeff],
            krylov=krylov,
            stationarity_direction_residual=float(
                torch.linalg.vector_norm(residual[:n_coeff])
            ),
            constraint_direction_residual=float(torch.abs(residual[n_coeff])),
        )

    def solve_tangent_dense_reference(
        self,
        position_direction: np.ndarray | torch.Tensor,
        geometry_step_bohr: float,
        compute_spectrum: bool = True,
    ) -> DensityResponseResult:
        """Assemble and solve the tangent Hessian; intended only as a small-system audit."""
        mixed = self.mixed_coefficient_direction(
            position_direction, geometry_step_bohr
        )
        tangent = self.tangent
        columns = [
            tangent.T @ self.coefficient_hvp(tangent[:, column])
            for column in range(tangent.shape[1])
        ]
        matrix = torch.stack(columns, dim=1)
        matrix = 0.5 * (matrix + matrix.T)
        rhs = -tangent.T @ mixed
        tangent_response = torch.linalg.solve(matrix, rhs)
        density_response = tangent @ tangent_response
        first_equation_without_multiplier = (
            self.coefficient_hvp(density_response) + mixed
        )
        norm_squared = torch.dot(self.normalization, self.normalization)
        multiplier_response = -torch.dot(
            self.normalization, first_equation_without_multiplier
        ) / norm_squared
        stationarity_residual = (
            first_equation_without_multiplier
            + self.normalization * multiplier_response
        )
        constraint_residual = torch.dot(self.normalization, density_response)
        residual_norm = float(torch.linalg.vector_norm(stationarity_residual))
        rhs_norm = float(torch.linalg.vector_norm(mixed))
        if compute_spectrum:
            eigenvalues = torch.linalg.eigvalsh(matrix)
            self.last_dense_tangent_eigenvalues = eigenvalues.detach()
        self.last_dense_tangent_matrix_symmetry = float(
            torch.linalg.vector_norm(matrix - matrix.T)
        )
        krylov = KrylovResult(
            solution=tangent_response,
            converged=True,
            iterations=tangent.shape[1],
            residual_norm=residual_norm,
            relative_residual=residual_norm / max(rhs_norm, 1e-300),
            method="dense_reference",
        )
        return DensityResponseResult(
            density_response=density_response,
            multiplier_response=multiplier_response,
            krylov=krylov,
            stationarity_direction_residual=residual_norm,
            constraint_direction_residual=float(torch.abs(constraint_residual)),
        )
