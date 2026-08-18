#!/usr/bin/env python3
"""Synthetic float64 direct-versus-matrix-free value and VJP gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import qm9_complete_total_capacity_train as core
import v11_matrix_free_response as matrix_free


torch.set_default_dtype(torch.float64)


def build(theta: torch.Tensor):
    coefficients = torch.tensor(
        [0.3, -0.2, 0.4, 0.1, -0.1, 0.2], requires_grad=True
    )
    positions = torch.tensor(
        [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]], requires_grad=True
    )
    base = torch.diag(torch.tensor([2.0, 2.4, 2.8, 3.2, 3.6, 4.0]))
    coupling = torch.tensor(
        [
            [0.0, 0.2, -0.1, 0.0, 0.1, 0.0],
            [0.2, 0.0, 0.1, -0.1, 0.0, 0.1],
            [-0.1, 0.1, 0.0, 0.2, 0.0, 0.0],
            [0.0, -0.1, 0.2, 0.0, 0.1, -0.1],
            [0.1, 0.0, 0.0, 0.1, 0.0, 0.2],
            [0.0, 0.1, 0.0, -0.1, 0.2, 0.0],
        ]
    )
    mixed = torch.arange(36, dtype=torch.float64).reshape(6, 6) / 200.0
    hessian = base + theta * coupling
    geometry_coupling = mixed * (1.0 + 0.1 * theta)
    flat_positions = positions.reshape(-1)
    total_energy = (
        0.5 * coefficients @ hessian @ coefficients
        + coefficients @ geometry_coupling @ flat_positions
        + 0.25 * (1.0 + theta) * torch.dot(flat_positions, flat_positions)
    )
    normalization = torch.tensor([1.0, 0.7, 0.5, 0.4, 0.3, 0.2])
    target = torch.dot(normalization, coefficients.detach())
    system = core.ConstrainedResponseSystem(
        total_energy=total_energy,
        coeffs=coefficients,
        positions=positions,
        normalization=normalization,
        n_electron=target,
        multiplier=0.0,
    )
    direction = torch.tensor(
        [[0.2, -0.3], [0.1, 0.4], [-0.2, 0.5]]
    )
    return system, direction


def objective(system, direction, response):
    relaxed = system.relaxed_hvp(direction, response, create_graph=True)
    density_weight = torch.linspace(0.2, 0.7, response.density_response.numel())
    geometry_weight = torch.linspace(0.1, 0.6, relaxed.numel()).reshape_as(relaxed)
    value = (
        torch.dot(response.density_response, density_weight)
        + 0.3 * response.multiplier_response
        + torch.sum(relaxed * geometry_weight)
        + 0.1 * torch.sum(relaxed.square())
    )
    return value, relaxed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    theta_direct = torch.tensor(0.17, requires_grad=True)
    direct_system, direction = build(theta_direct)
    direct_response = direct_system.solve_tangent_direct_implicit(
        direction, damping=0.0, create_graph=True
    )
    direct_loss, direct_hvp = objective(
        direct_system, direction, direct_response
    )
    direct_gradient = torch.autograd.grad(direct_loss, theta_direct)[0]

    matrix_free.install_into_canonical_core(core)
    theta_matrix_free = torch.tensor(0.17, requires_grad=True)
    matrix_free_system, matrix_free_direction = build(theta_matrix_free)
    settings = matrix_free.MatrixFreeSettings(
        tolerance=1.0e-12, max_iterations=200
    )
    with matrix_free.matrix_free_parameter_context(
        [theta_matrix_free], settings
    ):
        matrix_free_response = (
            matrix_free_system.solve_tangent_direct_implicit(
                matrix_free_direction, damping=0.0, create_graph=True
            )
        )
    matrix_free_loss, matrix_free_hvp = objective(
        matrix_free_system, matrix_free_direction, matrix_free_response
    )
    matrix_free_gradient = torch.autograd.grad(
        matrix_free_loss, theta_matrix_free
    )[0]

    metrics = {
        "density_response_max_abs": float(
            torch.max(
                torch.abs(
                    matrix_free_response.density_response
                    - direct_response.density_response
                )
            )
        ),
        "multiplier_response_abs": float(
            torch.abs(
                matrix_free_response.multiplier_response
                - direct_response.multiplier_response
            )
        ),
        "relaxed_hvp_max_abs": float(
            torch.max(torch.abs(matrix_free_hvp - direct_hvp))
        ),
        "objective_abs": float(torch.abs(matrix_free_loss - direct_loss)),
        "parameter_vjp_abs": float(
            torch.abs(matrix_free_gradient - direct_gradient)
        ),
        "direct_parameter_vjp": float(direct_gradient),
        "matrix_free_parameter_vjp": float(matrix_free_gradient),
        "diagnostics": matrix_free.consume_matrix_free_diagnostics(),
        "dtype": str(matrix_free_loss.dtype),
    }
    limits = {
        "density_response_max_abs": 1.0e-10,
        "multiplier_response_abs": 1.0e-10,
        "relaxed_hvp_max_abs": 1.0e-10,
        "objective_abs": 1.0e-10,
        "parameter_vjp_abs": 1.0e-9,
    }
    failures = {
        name: metrics[name]
        for name, limit in limits.items()
        if metrics[name] > limit
    }
    report = {
        "status": "pass" if not failures else "fail",
        "metrics": metrics,
        "limits": limits,
        "failures": failures,
        "test_accessed": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered)
        os.replace(temporary, args.output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
