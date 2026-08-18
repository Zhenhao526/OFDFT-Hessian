"""Training utilities for conservative complete-total OFDFT curvature objectives.

This module contains only tensor algebra. Geometry construction, density relaxation, and the
scalar total-energy owner remain in :mod:`mldft.ofdft.conservative_force` so training and strict
evaluation cannot silently acquire different energy definitions.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from collections.abc import Callable

import torch

from mldft.ofdft.implicit_response import (
    KrylovResult,
    implicit_symmetric_linear_solve,
    preconditioned_conjugate_gradient,
    tangent_basis,
)


_IMPLICIT_RESPONSE_DIAGNOSTICS: list[dict[str, float | int | bool | str | None]] = []
_IMPLICIT_RESPONSE_WARM_STARTS: OrderedDict[str, torch.Tensor] = OrderedDict()
_IMPLICIT_RESPONSE_WARM_START_LIMIT = 256


def alternating_update_kind(
    cumulative_step: int,
    *,
    hvp_update_period: int,
    replay_update_period: int | None = None,
) -> str:
    """Return ``hvp`` or ``replay`` for an alternating multitask schedule."""
    if cumulative_step <= 0 or hvp_update_period <= 0:
        raise ValueError("cumulative_step and hvp_update_period must be positive")
    if replay_update_period is not None:
        if replay_update_period <= 1:
            raise ValueError("replay_update_period must be greater than one")
        return "replay" if cumulative_step % replay_update_period == 0 else "hvp"
    return "hvp" if cumulative_step % hvp_update_period == 0 else "replay"


def assign_parameter_only_gradients(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    """Differentiate a training loss only with respect to optimizer parameters.

    ``loss.backward()`` also requests gradients for unrelated leaf tensors such
    as nuclear coordinates.  For an HVP loss that asks an integral backend for
    one unnecessary additional coordinate derivative.  Optimizer semantics
    require only model-parameter gradients, so explicitly target that set and
    leave all other leaves untouched.
    """
    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("parameter-only backward requires at least one parameter")
    gradients = torch.autograd.grad(
        loss,
        parameter_list,
        allow_unused=True,
    )
    for parameter, gradient in zip(parameter_list, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach()
    return gradients


def assign_two_task_pcgrad(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    *,
    max_second_to_first_norm_ratio: float | None = None,
) -> dict[str, float]:
    """Write symmetric two-task PCGrad gradients and report their alignment.

    The task losses must already include their configured scalar weights. Missing parameter
    gradients are treated as zero. The merged gradient is the sum of both projected task
    gradients, matching the scale of an ordinary backward pass when no conflict is present.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("PCGrad requires at least one parameter")
    first = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    if (
        max_second_to_first_norm_ratio is not None
        and max_second_to_first_norm_ratio <= 0.0
    ):
        raise ValueError("max_second_to_first_norm_ratio must be positive")
    raw_dot = first_loss.new_zeros(())
    first_norm_squared = first_loss.new_zeros(())
    raw_second_norm_squared = first_loss.new_zeros(())
    for first_gradient, second_gradient in zip(first, second):
        if first_gradient is not None:
            first_norm_squared = first_norm_squared + torch.sum(first_gradient**2)
        if second_gradient is not None:
            raw_second_norm_squared = raw_second_norm_squared + torch.sum(
                second_gradient**2
            )
        if first_gradient is not None and second_gradient is not None:
            raw_dot = raw_dot + torch.sum(first_gradient * second_gradient)

    epsilon = torch.finfo(raw_dot.dtype).tiny
    balance_scale = raw_dot.new_ones(())
    if max_second_to_first_norm_ratio is not None:
        allowed = (
            max_second_to_first_norm_ratio
            * torch.sqrt(first_norm_squared.clamp_min(epsilon))
            / torch.sqrt(raw_second_norm_squared.clamp_min(epsilon))
        )
        balance_scale = torch.clamp(allowed, max=1.0)
    dot = raw_dot * balance_scale
    second_norm_squared = raw_second_norm_squared * balance_scale.square()
    conflicting = bool((dot < 0).detach().cpu())
    first_scale = (
        dot / second_norm_squared.clamp_min(epsilon)
        if conflicting
        else dot.new_zeros(())
    )
    second_scale = (
        dot / first_norm_squared.clamp_min(epsilon)
        if conflicting
        else dot.new_zeros(())
    )
    projected_dot = dot.new_zeros(())
    for parameter, first_gradient, second_gradient in zip(parameters, first, second):
        first_value = (
            torch.zeros_like(parameter) if first_gradient is None else first_gradient
        )
        second_value = balance_scale * (
            torch.zeros_like(parameter) if second_gradient is None else second_gradient
        )
        projected_first = first_value - first_scale * second_value
        projected_second = second_value - second_scale * first_value
        projected_dot = projected_dot + torch.sum(projected_first * projected_second)
        parameter.grad = (projected_first + projected_second).detach()

    raw_denominator = torch.sqrt(
        first_norm_squared * raw_second_norm_squared
    ).clamp_min(epsilon)
    balanced_ratio = torch.sqrt(second_norm_squared.clamp_min(epsilon)) / torch.sqrt(
        first_norm_squared.clamp_min(epsilon)
    )
    return {
        "pcgrad/conflict": float(conflicting),
        "pcgrad/cosine_before": float((raw_dot / raw_denominator).detach().cpu()),
        "pcgrad/projected_task_dot": float(projected_dot.detach().cpu()),
        "pcgrad/second_task_balance_scale": float(balance_scale.detach().cpu()),
        "pcgrad/balanced_second_to_first_norm_ratio": float(
            balanced_ratio.detach().cpu()
        ),
    }


def assign_multi_task_pcgrad(
    task_losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.Tensor],
) -> dict[str, float]:
    """Write deterministic PCGrad gradients for three or more named tasks.

    Tasks are projected in mapping insertion order against the original gradients of every
    other task. The caller is responsible for applying any detached GradNorm scales first.
    Missing parameter gradients are represented by zeros, and the projected task gradients
    are summed to preserve ordinary-backward scale when no conflicts exist.
    """
    names = list(task_losses)
    if len(names) < 3:
        raise ValueError("multi-task PCGrad requires at least three tasks")
    parameters = list(parameters)
    if not parameters:
        raise ValueError("PCGrad requires at least one parameter")

    raw: dict[str, list[torch.Tensor]] = {}
    for name in names:
        gradients = torch.autograd.grad(
            task_losses[name],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        raw[name] = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]

    first_loss = task_losses[names[0]]
    epsilon = torch.finfo(first_loss.dtype).tiny
    norms_squared = {
        name: sum(
            (torch.sum(gradient * gradient) for gradient in raw[name]),
            first_loss.new_zeros(()),
        )
        for name in names
    }
    diagnostics: dict[str, float] = {}
    conflict_count = 0
    for first_index, first_name in enumerate(names):
        diagnostics[f"pcgrad/{first_name}_gradient_norm"] = float(
            torch.sqrt(norms_squared[first_name].clamp_min(epsilon)).detach().cpu()
        )
        for second_name in names[first_index + 1 :]:
            dot = sum(
                (
                    torch.sum(first_gradient * second_gradient)
                    for first_gradient, second_gradient in zip(
                        raw[first_name], raw[second_name], strict=True
                    )
                ),
                first_loss.new_zeros(()),
            )
            denominator = torch.sqrt(
                norms_squared[first_name] * norms_squared[second_name]
            ).clamp_min(epsilon)
            conflict = bool((dot < 0).detach().cpu())
            conflict_count += int(conflict)
            pair = f"{first_name}_vs_{second_name}"
            diagnostics[f"pcgrad/{pair}_cosine_before"] = float(
                (dot / denominator).detach().cpu()
            )
            diagnostics[f"pcgrad/{pair}_conflict"] = float(conflict)

    projected: dict[str, list[torch.Tensor]] = {
        name: [gradient.clone() for gradient in raw[name]] for name in names
    }
    for name in names:
        for other_name in names:
            if other_name == name:
                continue
            dot = sum(
                (
                    torch.sum(projected_gradient * other_gradient)
                    for projected_gradient, other_gradient in zip(
                        projected[name], raw[other_name], strict=True
                    )
                ),
                first_loss.new_zeros(()),
            )
            if bool((dot < 0).detach().cpu()):
                scale = dot / norms_squared[other_name].clamp_min(epsilon)
                projected[name] = [
                    projected_gradient - scale * other_gradient
                    for projected_gradient, other_gradient in zip(
                        projected[name], raw[other_name], strict=True
                    )
                ]

    for parameter_index, parameter in enumerate(parameters):
        parameter.grad = sum(
            (projected[name][parameter_index] for name in names),
            torch.zeros_like(parameter),
        ).detach()
    diagnostics["pcgrad/pairwise_conflict_count"] = float(conflict_count)
    return diagnostics


def consume_implicit_response_diagnostics() -> list[dict[str, float | int | bool | str | None]]:
    """Return and clear matrix-free implicit-response diagnostics for the current process."""
    diagnostics = list(_IMPLICIT_RESPONSE_DIAGNOSTICS)
    _IMPLICIT_RESPONSE_DIAGNOSTICS.clear()
    return diagnostics


def clear_implicit_response_warm_starts() -> None:
    """Clear detached adjoint warm starts, primarily for independent runs and tests."""
    _IMPLICIT_RESPONSE_WARM_STARTS.clear()


class _ImplicitStationaryDensityParameterResponse(torch.autograd.Function):
    """Identity-valued stationary density with an implicit model-parameter VJP."""

    @staticmethod
    def forward(
        ctx,
        coefficients: torch.Tensor,
        normalization: torch.Tensor,
        n_electron: float,
        energy_function: Callable[[torch.Tensor], torch.Tensor],
        tolerance: float,
        max_iterations: int,
        damping: float,
        diagonal_probes: int,
        solver: str,
        warm_start_key: str | None,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        del n_electron  # Feasibility is checked by the public wrapper.
        ctx.energy_function = energy_function
        ctx.tolerance = tolerance
        ctx.max_iterations = max_iterations
        ctx.damping = damping
        ctx.diagonal_probes = diagonal_probes
        ctx.solver = solver
        ctx.warm_start_key = warm_start_key
        ctx.save_for_backward(
            coefficients.detach(), normalization.detach(), *parameters
        )
        return coefficients.detach().clone()

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        saved = ctx.saved_tensors
        coefficients = saved[0].detach().requires_grad_(True)
        normalization = saved[1].to(coefficients)
        parameters = saved[2:]
        parameter_gradients: tuple[torch.Tensor | None, ...]

        with torch.enable_grad():
            normalization_norm_squared = torch.dot(normalization, normalization)

            def project(vector: torch.Tensor) -> torch.Tensor:
                return vector - normalization * (
                    torch.dot(normalization, vector) / normalization_norm_squared
                )

            right_hand_side = project(output_gradient.detach().to(coefficients))
            right_hand_side_norm = float(
                torch.linalg.vector_norm(right_hand_side).detach().cpu()
            )
            if right_hand_side_norm <= 1.0e-8:
                parameter_gradients = tuple(torch.zeros_like(item) for item in parameters)
            else:
                energy = ctx.energy_function(coefficients)
                density_gradient = torch.autograd.grad(
                    energy, coefficients, create_graph=True, retain_graph=True
                )[0]

                def tangent_hessian(vector: torch.Tensor) -> torch.Tensor:
                    tangent_vector = project(vector)
                    hessian_vector = torch.autograd.grad(
                        density_gradient,
                        coefficients,
                        grad_outputs=tangent_vector,
                        retain_graph=True,
                    )[0]
                    return project(hessian_vector) + ctx.damping * tangent_vector

                initial_guess = None
                if ctx.solver == "pcg" and ctx.warm_start_key is not None:
                    cached = _IMPLICIT_RESPONSE_WARM_STARTS.get(ctx.warm_start_key)
                    if cached is not None and cached.shape == right_hand_side.shape:
                        initial_guess = project(cached.to(right_hand_side))
                        _IMPLICIT_RESPONSE_WARM_STARTS.move_to_end(ctx.warm_start_key)

                if ctx.solver == "direct":
                    tangent = tangent_basis(normalization)
                    columns = [
                        tangent.T @ tangent_hessian(tangent[:, column])
                        for column in range(tangent.shape[1])
                    ]
                    matrix = torch.stack(columns, dim=1)
                    matrix = 0.5 * (matrix + matrix.T)
                    tangent_rhs = tangent.T @ right_hand_side
                    tangent_adjoint = torch.linalg.solve(matrix, tangent_rhs)
                    solution = tangent @ tangent_adjoint
                    residual = right_hand_side - tangent_hessian(solution)
                    residual_norm = float(
                        torch.linalg.vector_norm(residual).detach().cpu()
                    )
                    target = ctx.tolerance * max(right_hand_side_norm, 1.0)
                    adjoint = KrylovResult(
                        solution=solution,
                        converged=bool(
                            torch.isfinite(solution).all()
                            and torch.isfinite(residual).all()
                            and residual_norm <= target
                        ),
                        iterations=1,
                        residual_norm=residual_norm,
                        relative_residual=(
                            residual_norm / max(right_hand_side_norm, 1.0e-300)
                        ),
                        method="dense_direct_parameter_adjoint",
                        breakdown=(
                            None
                            if residual_norm <= target
                            else "explicit_residual_above_tolerance"
                        ),
                    )
                else:
                    preconditioner = None
                    if ctx.diagonal_probes > 0:
                        diagonal = torch.zeros_like(coefficients)
                        generator = torch.Generator(device=coefficients.device)
                        generator.manual_seed(20260717)
                        for _ in range(ctx.diagonal_probes):
                            signs = torch.randint(
                                0,
                                2,
                                coefficients.shape,
                                generator=generator,
                                device=coefficients.device,
                                dtype=torch.int64,
                            ).to(coefficients.dtype)
                            signs = project(2.0 * signs - 1.0)
                            diagonal = diagonal + signs * tangent_hessian(signs)
                        diagonal = torch.abs(
                            diagonal / float(ctx.diagonal_probes)
                        ).clamp_min(max(ctx.damping, 1.0e-8))
                        inverse_diagonal = 1.0 / diagonal

                        def preconditioner(vector: torch.Tensor) -> torch.Tensor:
                            return project(inverse_diagonal * project(vector))

                    adjoint = preconditioned_conjugate_gradient(
                        tangent_hessian,
                        right_hand_side,
                        preconditioner=preconditioner,
                        tolerance=ctx.tolerance,
                        max_iterations=ctx.max_iterations,
                        initial_guess=initial_guess,
                    )
                if (
                    ctx.solver == "pcg"
                    and ctx.warm_start_key is not None
                    and adjoint.converged
                ):
                    _IMPLICIT_RESPONSE_WARM_STARTS[ctx.warm_start_key] = (
                        adjoint.solution.detach().clone()
                    )
                    _IMPLICIT_RESPONSE_WARM_STARTS.move_to_end(ctx.warm_start_key)
                    while (
                        len(_IMPLICIT_RESPONSE_WARM_STARTS)
                        > _IMPLICIT_RESPONSE_WARM_START_LIMIT
                    ):
                        _IMPLICIT_RESPONSE_WARM_STARTS.popitem(last=False)
                _IMPLICIT_RESPONSE_DIAGNOSTICS.append(
                    {
                        "converged": adjoint.converged,
                        "iterations": adjoint.iterations,
                        "right_hand_side_norm": right_hand_side_norm,
                        "residual_norm": adjoint.residual_norm,
                        "relative_residual": adjoint.relative_residual,
                        "scaled_residual": adjoint.residual_norm
                        / max(right_hand_side_norm, 1.0),
                        "breakdown": adjoint.breakdown,
                        "method": adjoint.method,
                        "solver": ctx.solver,
                        "diagonal_probes": ctx.diagonal_probes,
                        "warm_start_used": initial_guess is not None,
                    }
                )
                if not adjoint.converged:
                    raise RuntimeError(
                        "Implicit density parameter-response solve failed: "
                        f"iterations={adjoint.iterations} "
                        f"relative_residual={adjoint.relative_residual:.3e} "
                        f"breakdown={adjoint.breakdown}"
                    )
                mixed = torch.autograd.grad(
                    density_gradient,
                    parameters,
                    grad_outputs=adjoint.solution.detach(),
                    allow_unused=True,
                )
                parameter_gradients = tuple(
                    None if value is None else -value for value in mixed
                )

        # No warm-start, normalization, target, closure, or solver-setting gradient is needed.
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *parameter_gradients,
        )


def implicit_stationary_density_parameter_response(
    coefficients: torch.Tensor,
    normalization: torch.Tensor,
    n_electron: float | torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    parameters: Iterable[torch.Tensor],
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 500,
    damping: float = 1.0e-8,
    diagonal_probes: int = 0,
    solver: str = "pcg",
    warm_start_key: str | None = None,
) -> torch.Tensor:
    """Attach the exact stationary-density model-parameter VJP to converged coefficients.

    The forward value is unchanged. During backward, a matrix-free constrained coefficient-Hessian
    solve supplies ``dc_star/dtheta`` through the implicit-function theorem. This handles parameter
    response only; the caller must supply its separately defined geometry-response
    path (the canonical v4 branch uses the constrained analytic KKT response).
    """
    if (
        tolerance <= 0
        or max_iterations <= 0
        or damping < 0
        or diagonal_probes < 0
        or solver not in {"pcg", "direct"}
    ):
        raise ValueError("invalid implicit-response solver settings")
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("at least one differentiable parameter is required")
    coefficients = coefficients.detach()
    normalization = normalization.detach().to(coefficients)
    target = torch.as_tensor(n_electron, dtype=coefficients.dtype, device=coefficients.device)
    residual = torch.dot(normalization, coefficients) - target
    feasibility_tolerance = 100.0 * torch.finfo(coefficients.dtype).eps * max(
        1.0, abs(float(target.detach().cpu()))
    )
    if abs(float(residual.detach().cpu())) > feasibility_tolerance:
        raise ValueError(
            "implicit stationary density is not electron-number feasible: "
            f"residual={float(residual.detach().cpu()):.3e}"
        )
    return _ImplicitStationaryDensityParameterResponse.apply(
        coefficients,
        normalization,
        float(target.detach().cpu()),
        energy_function,
        tolerance,
        max_iterations,
        damping,
        diagonal_probes,
        solver,
        warm_start_key,
        *parameter_tuple,
    )


def stationary_density_parameter_step_prediction(
    coefficients: torch.Tensor,
    normalization: torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    parameters: Iterable[torch.Tensor],
    parameter_steps: Iterable[torch.Tensor],
    *,
    damping: float = 0.0,
) -> torch.Tensor:
    """Predict ``dc*`` for one model-parameter step from the KKT response.

    The mixed product is evaluated as
    ``G_theta dtheta = d/dc[(dE/dtheta) dot dtheta]``.  The result lies in the
    electron-number tangent space and is intended only as the initial guess for
    a strict corrector.
    """
    if damping < 0:
        raise ValueError("damping must be nonnegative")
    parameter_tuple = tuple(parameters)
    step_tuple = tuple(parameter_steps)
    if not parameter_tuple or len(parameter_tuple) != len(step_tuple):
        raise ValueError("parameters and parameter_steps must have equal nonzero length")
    coefficients = coefficients.detach().clone().requires_grad_(True)
    normalization = normalization.detach().to(coefficients)
    tangent = tangent_basis(normalization)
    energy = energy_function(coefficients)
    density_gradient = torch.autograd.grad(
        energy, coefficients, create_graph=True, retain_graph=True
    )[0]
    parameter_gradients = torch.autograd.grad(
        energy,
        parameter_tuple,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    directional_parameter_derivative = sum(
        (
            torch.sum(gradient * step.detach().to(gradient))
            for gradient, step in zip(
                parameter_gradients, step_tuple, strict=True
            )
            if gradient is not None
        ),
        energy * 0.0,
    )
    mixed = torch.autograd.grad(
        directional_parameter_derivative,
        coefficients,
        retain_graph=True,
    )[0]
    columns = [
        tangent.T
        @ torch.autograd.grad(
            density_gradient,
            coefficients,
            grad_outputs=tangent[:, column],
            retain_graph=True,
        )[0]
        for column in range(tangent.shape[1])
    ]
    matrix = torch.stack(columns, dim=1)
    matrix = 0.5 * (matrix + matrix.T)
    if damping:
        matrix = matrix + damping * torch.eye(
            matrix.shape[0], dtype=matrix.dtype, device=matrix.device
        )
    tangent_step = implicit_symmetric_linear_solve(
        matrix, -(tangent.T @ mixed)
    )
    prediction = tangent @ tangent_step
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("stationary density parameter-step prediction is non-finite")
    return prediction


def differentiable_constrained_density_unroll(
    initial_coefficients: torch.Tensor,
    normalization: torch.Tensor,
    n_electron: float | torch.Tensor,
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> torch.Tensor:
    """Unroll projected density-gradient steps without detaching the parameter response.

    This is a truncated response prototype, not a replacement for strict density relaxation. The
    caller should start from independently converged coefficients and use the returned tensor only
    on the training graph. Electron number is restored after every step.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    coefficients = initial_coefficients
    if not coefficients.requires_grad:
        coefficients = coefficients.clone().requires_grad_(True)
    normalization = normalization.to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    target = torch.as_tensor(
        n_electron, device=coefficients.device, dtype=coefficients.dtype
    )
    normalization_norm_squared = torch.dot(normalization, normalization)

    def make_feasible(value: torch.Tensor) -> torch.Tensor:
        residual = target - torch.dot(normalization, value)
        return value + normalization * (residual / normalization_norm_squared)

    coefficients = make_feasible(coefficients)
    for _ in range(steps):
        energy = energy_function(coefficients)
        gradient = torch.autograd.grad(
            energy, coefficients, create_graph=True, retain_graph=True
        )[0]
        projected = gradient - normalization * (
            torch.dot(normalization, gradient) / normalization_norm_squared
        )
        coefficients = make_feasible(coefficients - learning_rate * projected)
    return coefficients


def central_force_secant_hvp(
    plus_force: torch.Tensor,
    minus_force: torch.Tensor,
    displacement_bohr: float,
) -> torch.Tensor:
    """Return ``H v`` from scalar-derived forces at ``R +/- h v``.

    ``F = -dE/dR``, hence ``H v = -(F_plus - F_minus)/(2h)``. The operation preserves the
    parameter graph of both force tensors.
    """
    if displacement_bohr <= 0:
        raise ValueError("displacement_bohr must be positive")
    if plus_force.shape != minus_force.shape:
        raise ValueError(
            f"force shapes differ: {plus_force.shape} versus {minus_force.shape}"
        )
    return -(plus_force - minus_force) / (2.0 * displacement_bohr)


def central_energy_directional_curvature(
    base_energy: torch.Tensor,
    plus_energy: torch.Tensor,
    minus_energy: torch.Tensor,
    displacement_bohr: float,
) -> torch.Tensor:
    """Return ``v.T @ H @ v`` from relaxed scalar energies at ``R`` and ``R +/- h v``.

    This scalar contraction is useful as an envelope-theorem gradient audit. It is not a vector
    Hessian-vector product and callers must not report it as one.
    """
    if displacement_bohr <= 0:
        raise ValueError("displacement_bohr must be positive")
    if base_energy.numel() != 1 or plus_energy.numel() != 1 or minus_energy.numel() != 1:
        raise ValueError("all energy inputs must be scalar tensors")
    return (plus_energy - 2.0 * base_energy + minus_energy) / (
        displacement_bohr * displacement_bohr
    )


def mixed_absolute_relative_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    absolute_scale: float,
    relative_floor: float,
    relative_fraction: float = 0.5,
) -> torch.Tensor:
    """Combine a dimensionless absolute MAE with a reference-scaled relative MAE.

    The reference scale is the target RMS with a fixed floor. A floor prevents nearly-zero HVPs
    or forces from dominating an update while the absolute term keeps large-reference samples from
    becoming irrelevant.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if absolute_scale <= 0 or relative_floor <= 0:
        raise ValueError("absolute_scale and relative_floor must be positive")
    if not 0.0 <= relative_fraction <= 1.0:
        raise ValueError("relative_fraction must be in [0, 1]")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be non-empty")

    target = target.to(device=prediction.device, dtype=prediction.dtype)
    mae = torch.mean(torch.abs(prediction - target))
    reference_rms = torch.sqrt(torch.mean(target * target)).clamp_min(relative_floor)
    absolute = mae / absolute_scale
    relative = mae / reference_rms
    return (1.0 - relative_fraction) * absolute + relative_fraction * relative


def mixed_absolute_relative_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    absolute_scale: float,
    relative_floor: float,
    relative_fraction: float = 0.5,
) -> torch.Tensor:
    """Combine absolute and reference-scaled RMSE for Frobenius-aligned HVP fitting.

    For a Cartesian Hessian column, the relative term is exactly its relative Euclidean
    error. Applying the same loss to all columns therefore aligns the optimization norm with
    the full-Hessian relative Frobenius acceptance metric more directly than elementwise L1.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if absolute_scale <= 0 or relative_floor <= 0:
        raise ValueError("absolute_scale and relative_floor must be positive")
    if not 0.0 <= relative_fraction <= 1.0:
        raise ValueError("relative_fraction must be in [0, 1]")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be non-empty")

    target = target.to(device=prediction.device, dtype=prediction.dtype)
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
    reference_rms = torch.sqrt(torch.mean(target * target)).clamp_min(relative_floor)
    absolute = rmse / absolute_scale
    relative = rmse / reference_rms
    return (1.0 - relative_fraction) * absolute + relative_fraction * relative


def hutchinson_internal_frobenius_squared_loss(
    prediction_hvp: torch.Tensor,
    target_hvp: torch.Tensor,
    *,
    internal_dimension: int,
    reduction: str = "sum",
) -> torch.Tensor:
    """Return an unbiased internal-space Hessian Frobenius estimator.

    ``prediction_hvp`` and ``target_hvp`` must be products with the same
    unnormalized probe ``v = B.T @ z``, where ``B`` is a complete orthonormal
    internal basis and ``z`` is Rademacher.  With ``reduction="sum"`` the
    expectation is exactly ``||(H_pred-H_ref) B.T||_F^2``.  The
    ``mean_internal_matrix`` reduction divides this estimator by
    ``n_cartesian * internal_dimension`` without changing its unbiasedness for
    the corresponding matrix mean-square error.
    """
    if prediction_hvp.shape != target_hvp.shape:
        raise ValueError(
            f"prediction shape {prediction_hvp.shape} != target shape {target_hvp.shape}"
        )
    if prediction_hvp.numel() == 0:
        raise ValueError("prediction and target must be non-empty")
    if internal_dimension <= 0:
        raise ValueError("internal_dimension must be positive")
    target = target_hvp.to(
        device=prediction_hvp.device, dtype=prediction_hvp.dtype
    )
    squared_error = torch.sum((prediction_hvp - target) ** 2)
    if reduction == "sum":
        return squared_error
    if reduction == "mean_internal_matrix":
        return squared_error / (
            float(prediction_hvp.numel()) * float(internal_dimension)
        )
    raise ValueError(
        "reduction must be 'sum' or 'mean_internal_matrix', "
        f"got {reduction!r}"
    )


def electron_number_tangent_projection(
    vector: torch.Tensor,
    normalization: torch.Tensor,
) -> torch.Tensor:
    """Project one coefficient-space vector onto the fixed-electron-number tangent space.

    ``normalization`` is the coefficient-space gradient of the electron-number
    constraint ``q.T @ c = N``.  The operation is deliberately defined here,
    rather than reconstructed independently by each E/G/F/H caller, so label
    gradient training and density-stationarity diagnostics use the same gauge.
    """
    if vector.ndim != 1 or normalization.ndim != 1:
        raise ValueError("vector and normalization must be one-dimensional")
    if vector.shape != normalization.shape:
        raise ValueError(
            f"vector shape {vector.shape} != normalization shape {normalization.shape}"
        )
    normalization = normalization.to(device=vector.device, dtype=vector.dtype)
    if not bool(torch.isfinite(vector.detach()).all().cpu()) or not bool(
        torch.isfinite(normalization.detach()).all().cpu()
    ):
        raise ValueError("vector and normalization must be finite")
    norm_squared = torch.dot(normalization, normalization)
    if bool((norm_squared <= 0).detach().cpu()):
        raise ValueError("normalization must be nonzero")
    return vector - normalization * (
        torch.dot(normalization, vector) / norm_squared
    )


def structures25_projected_gradient_mse(
    predicted_kin_plus_xc_gradient: torch.Tensor,
    target_kin_plus_xc_gradient: torch.Tensor,
    normalization: torch.Tensor,
    *,
    absolute_scale: float,
) -> torch.Tensor:
    """Match the canonical Structures25 ``kin_plus_xc`` derivative modulo chemical potential.

    Both gradients are derivatives with respect to the same untransformed
    auxiliary-density coefficients.  Only their electron-number-tangent
    difference is observable under ``q.T @ c = N``.  The mean-square reduction
    makes the loss independent of the number of auxiliary coefficients.
    """
    if predicted_kin_plus_xc_gradient.shape != target_kin_plus_xc_gradient.shape:
        raise ValueError(
            "predicted and target kin_plus_xc gradient shapes differ: "
            f"{predicted_kin_plus_xc_gradient.shape} versus "
            f"{target_kin_plus_xc_gradient.shape}"
        )
    if absolute_scale <= 0:
        raise ValueError("absolute_scale must be positive")
    target = target_kin_plus_xc_gradient.to(
        device=predicted_kin_plus_xc_gradient.device,
        dtype=predicted_kin_plus_xc_gradient.dtype,
    )
    projected_error = electron_number_tangent_projection(
        predicted_kin_plus_xc_gradient - target,
        normalization,
    )
    return torch.mean((projected_error / absolute_scale) ** 2)


def hutchinson_internal_projected_frobenius_squared_loss(
    prediction_hvp: torch.Tensor,
    target_hvp: torch.Tensor,
    internal_basis: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """Return a Hutchinson estimator aligned with the two-sided internal Hessian.

    The rows of ``internal_basis`` are an orthonormal Cartesian internal basis
    ``B`` and both HVPs use ``v = B.T @ z``.  Projecting the output with ``B``
    gives ``B (H_pred-H_ref) B.T z``.  Its expected squared norm is exactly the
    Frobenius-square error of the internal matrix used by final evaluation.
    """
    if prediction_hvp.shape != target_hvp.shape:
        raise ValueError(
            f"prediction shape {prediction_hvp.shape} != target shape {target_hvp.shape}"
        )
    if prediction_hvp.numel() == 0:
        raise ValueError("prediction and target must be non-empty")
    if internal_basis.ndim != 2:
        raise ValueError("internal_basis must be a rank-two tensor")
    if internal_basis.shape[1] != prediction_hvp.numel():
        raise ValueError(
            "internal basis Cartesian dimension does not match the HVP: "
            f"{internal_basis.shape[1]} != {prediction_hvp.numel()}"
        )
    if internal_basis.shape[0] <= 0:
        raise ValueError("internal_basis must contain at least one direction")
    basis = internal_basis.to(device=prediction_hvp.device, dtype=prediction_hvp.dtype)
    target = target_hvp.to(device=prediction_hvp.device, dtype=prediction_hvp.dtype)
    if (
        not bool(torch.isfinite(prediction_hvp.detach()).all().cpu())
        or not bool(torch.isfinite(target.detach()).all().cpu())
        or not bool(torch.isfinite(basis.detach()).all().cpu())
    ):
        raise ValueError("prediction, target, and internal_basis must be finite")
    gram = basis @ basis.T
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    orthonormal_tolerance = (
        1.0e-10 if gram.dtype == torch.float64 else 1.0e-5
    )
    if not bool(
        torch.allclose(
            gram,
            identity,
            atol=orthonormal_tolerance,
            rtol=orthonormal_tolerance,
        )
    ):
        raise ValueError("internal_basis rows must be orthonormal")
    internal_error = basis @ (prediction_hvp - target).reshape(-1)
    squared_error = torch.sum(internal_error**2)
    if reduction == "sum":
        return squared_error
    if reduction == "mean_internal_matrix":
        internal_dimension = float(basis.shape[0])
        return squared_error / (internal_dimension * internal_dimension)
    raise ValueError(
        "reduction must be 'sum' or 'mean_internal_matrix', "
        f"got {reduction!r}"
    )


def normalized_energy_l1(
    prediction: torch.Tensor,
    target: torch.Tensor | float,
    *,
    absolute_scale_hartree: float,
) -> torch.Tensor:
    """Return a dimensionless absolute total-energy error."""
    if absolute_scale_hartree <= 0:
        raise ValueError("absolute_scale_hartree must be positive")
    target_tensor = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
    return torch.mean(torch.abs(prediction - target_tensor)) / absolute_scale_hartree


def low_mode_curvature_loss(
    prediction_hvp: torch.Tensor,
    target_hvp: torch.Tensor,
    direction: torch.Tensor,
    *,
    curvature_floor: float,
    wrong_curvature_multiplier: float = 2.0,
) -> torch.Tensor:
    """Penalize low-mode directional curvature error and an incorrect curvature sign."""
    if curvature_floor <= 0:
        raise ValueError("curvature_floor must be positive")
    if wrong_curvature_multiplier < 0:
        raise ValueError("wrong_curvature_multiplier must be non-negative")
    if not (
        prediction_hvp.shape == target_hvp.shape == direction.shape
    ):
        raise ValueError("prediction_hvp, target_hvp, and direction must have equal shapes")

    direction = direction.to(device=prediction_hvp.device, dtype=prediction_hvp.dtype)
    target_hvp = target_hvp.to(device=prediction_hvp.device, dtype=prediction_hvp.dtype)
    norm_squared = torch.sum(direction * direction).clamp_min(
        torch.finfo(prediction_hvp.dtype).tiny
    )
    predicted_curvature = torch.sum(direction * prediction_hvp) / norm_squared
    target_curvature = torch.sum(direction * target_hvp) / norm_squared
    scale = torch.abs(target_curvature).clamp_min(curvature_floor)
    relative_error = torch.abs(predicted_curvature - target_curvature) / scale
    target_sign = torch.sign(target_curvature.detach())
    wrong_sign = torch.relu(-target_sign * predicted_curvature) / scale
    return relative_error + wrong_curvature_multiplier * wrong_sign


def assemble_hessian_columns(columns: Iterable[torch.Tensor]) -> torch.Tensor:
    """Assemble flattened Cartesian HVP columns into a square Hessian."""
    flattened = [column.reshape(-1) for column in columns]
    if not flattened:
        raise ValueError("at least one Hessian column is required")
    size = flattened[0].numel()
    if len(flattened) != size:
        raise ValueError(f"received {len(flattened)} columns for a {size}x{size} Hessian")
    if any(column.numel() != size for column in flattened):
        raise ValueError("all Hessian columns must have equal sizes")
    return torch.stack(flattened, dim=1)


def hessian_error_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return raw/symmetric Hessian errors and numerical self-consistency metrics."""
    if prediction.ndim != 2 or prediction.shape[0] != prediction.shape[1]:
        raise ValueError("prediction must be square")
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != reference shape {reference.shape}"
        )
    reference = reference.to(device=prediction.device, dtype=prediction.dtype)
    difference = prediction - reference
    symmetric = 0.5 * (prediction + prediction.T)
    antisymmetric = 0.5 * (prediction - prediction.T)
    symmetric_difference = symmetric - 0.5 * (reference + reference.T)
    reference_norm = torch.linalg.matrix_norm(reference).clamp_min(
        torch.finfo(prediction.dtype).tiny
    )
    symmetric_norm = torch.linalg.matrix_norm(symmetric).clamp_min(
        torch.finfo(prediction.dtype).tiny
    )
    return {
        "mae": torch.mean(torch.abs(difference)),
        "rmse": torch.sqrt(torch.mean(difference * difference)),
        "relative_frobenius": torch.linalg.matrix_norm(difference) / reference_norm,
        "symmetric_mae": torch.mean(torch.abs(symmetric_difference)),
        "symmetric_rmse": torch.sqrt(torch.mean(symmetric_difference * symmetric_difference)),
        "symmetric_relative_frobenius": (
            torch.linalg.matrix_norm(symmetric_difference) / reference_norm
        ),
        "antisymmetric_over_symmetric_frobenius": (
            torch.linalg.matrix_norm(antisymmetric) / symmetric_norm
        ),
        "symmetry_max_abs": torch.max(torch.abs(prediction - prediction.T)),
    }


def parameter_gradient_diagnostics(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    *,
    parameter_names: Iterable[str] | None = None,
    precomputed_gradients: Mapping[
        str, Iterable[torch.Tensor | None]
    ] | None = None,
) -> dict[str, float]:
    """Measure per-loss gradient norms, module contributions, and cosine conflicts.

    The caller must invoke this before the final backward pass. Graphs are retained and parameter
    ``.grad`` buffers are not modified. ``precomputed_gradients`` accepts memory-safe
    microbatch gradients after their individual graphs have been released. When names are
    supplied, the first dotted name component defines a module group and its squared-norm
    fraction is reported without another autograd pass.
    """
    parameter_list = list(parameters)
    if parameter_names is None:
        name_list = [str(index) for index in range(len(parameter_list))]
        report_groups = False
    else:
        name_list = list(parameter_names)
        if len(name_list) != len(parameter_list):
            raise ValueError("parameter_names and parameters must have equal lengths")
        report_groups = True
    trainable_pairs = [
        (name, parameter)
        for name, parameter in zip(name_list, parameter_list)
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in trainable_pairs]
    groups = [name.split(".", 1)[0] for name, _ in trainable_pairs]
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    norms: dict[str, torch.Tensor] = {}
    output: dict[str, float] = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(
            loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[name] = values

    for name, supplied in (precomputed_gradients or {}).items():
        if name in gradients:
            raise ValueError(
                f"gradient diagnostics received duplicate task {name!r}"
            )
        values = tuple(supplied)
        if len(values) != len(trainable):
            raise ValueError(
                "precomputed gradient tuples must align with trainable parameters"
            )
        gradients[name] = values

    if not gradients:
        return {}
    reference = next(iter(losses.values()), None)
    if reference is None:
        reference = next(
            (
                value
                for values in gradients.values()
                for value in values
                if value is not None
            ),
            None,
        )
    if reference is None:
        raise ValueError("gradient diagnostics require at least one tensor")

    for name, values in gradients.items():
        squared = reference.new_zeros(())
        group_squared: dict[str, torch.Tensor] = {}
        for group, value in zip(groups, values):
            if value is not None:
                contribution = torch.sum(value.detach() ** 2)
                squared = squared + contribution
                group_squared[group] = group_squared.get(
                    group, reference.new_zeros(())
                ) + contribution
        norm = torch.sqrt(squared)
        norms[name] = norm
        output[f"gradient_norm/{name}"] = float(norm.detach().cpu())
        if report_groups:
            denominator = squared.clamp_min(torch.finfo(squared.dtype).tiny)
            for group, contribution in group_squared.items():
                output[f"gradient_group_norm/{name}/{group}"] = float(
                    torch.sqrt(contribution).detach().cpu()
                )
                output[f"gradient_group_squared_fraction/{name}/{group}"] = float(
                    (contribution / denominator).detach().cpu()
                )

    for left, right in itertools.combinations(gradients, 2):
        dot = reference.new_zeros(())
        for left_gradient, right_gradient in zip(gradients[left], gradients[right]):
            if left_gradient is not None and right_gradient is not None:
                dot = dot + torch.sum(left_gradient.detach() * right_gradient.detach())
        denominator = norms[left] * norms[right]
        cosine = dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny)
        output[f"gradient_cosine/{left}_vs_{right}"] = float(cosine.detach().cpu())
    return output


def internal_coordinate_projector(
    positions_bohr: torch.Tensor,
    atomic_masses: torch.Tensor,
    *,
    rank_tolerance: float = 1.0e-10,
) -> torch.Tensor:
    """Return the Cartesian projector orthogonal to mass-weighted translations/rotations.

    The projector acts on ordinary Cartesian vectors. Linear molecules naturally have rank five
    external motion; nonlinear molecules have rank six.
    """
    if positions_bohr.ndim != 2 or positions_bohr.shape[1] != 3:
        raise ValueError("positions_bohr must have shape (natoms, 3)")
    if atomic_masses.shape != (positions_bohr.shape[0],):
        raise ValueError("atomic_masses must have shape (natoms,)")
    if bool(torch.any(atomic_masses <= 0)):
        raise ValueError("atomic masses must be positive")

    dtype = positions_bohr.dtype
    device = positions_bohr.device
    masses = atomic_masses.to(device=device, dtype=dtype)
    center = torch.sum(masses[:, None] * positions_bohr, dim=0) / torch.sum(masses)
    centered = positions_bohr - center
    sqrt_mass = torch.sqrt(masses)
    external = []
    for axis in range(3):
        translation = torch.zeros_like(positions_bohr)
        translation[:, axis] = sqrt_mass
        external.append(translation.reshape(-1))
    axes = torch.eye(3, dtype=dtype, device=device)
    mass_weighted_positions = sqrt_mass[:, None] * centered
    for axis in axes:
        rotation = torch.linalg.cross(
            axis.expand_as(mass_weighted_positions), mass_weighted_positions, dim=1
        )
        external.append(rotation.reshape(-1))
    external_matrix = torch.stack(external, dim=1)
    u, singular_values, _ = torch.linalg.svd(external_matrix, full_matrices=False)
    threshold = rank_tolerance * singular_values.max().clamp_min(
        torch.finfo(dtype).tiny
    )
    rank = int(torch.sum(singular_values > threshold).detach().cpu())
    q = u[:, :rank]
    identity = torch.eye(positions_bohr.numel(), dtype=dtype, device=device)
    return identity - q @ q.T
