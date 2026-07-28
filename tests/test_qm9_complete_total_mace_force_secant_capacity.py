from __future__ import annotations

import pytest
import torch

from scripts.qm9_complete_total_mace_force_secant_capacity import (
    apply_fixed_scale_pcgrad,
    apply_fixed_scale_pcgrad_from_gradients,
    assign_flat_parameter_gradient,
    backtracking_tasks_are_acceptable,
    backtracking_tasks_are_budget_acceptable,
    central_force_secant_hvp,
    project_conflicting_gradients,
    _validate_initial_checkpoint_metadata,
)


def test_central_force_secant_has_hessian_sign() -> None:
    hessian = torch.tensor([[2.0, -0.5], [-0.5, 1.0]], dtype=torch.float64)
    direction = torch.tensor([0.3, -0.8], dtype=torch.float64)
    step = 1e-3
    plus_force = -(hessian @ (step * direction))
    minus_force = -(hessian @ (-step * direction))

    actual = central_force_secant_hvp(plus_force, minus_force, step)

    torch.testing.assert_close(actual, hessian @ direction)


def test_central_force_secant_preserves_parameter_gradient() -> None:
    parameter = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    plus_force = torch.stack((parameter, 2.0 * parameter))
    minus_force = -plus_force

    value = central_force_secant_hvp(plus_force, minus_force, 0.5).sum()
    gradient = torch.autograd.grad(value, parameter)[0]

    torch.testing.assert_close(gradient, torch.tensor(-6.0, dtype=torch.float64))


def test_central_force_secant_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        central_force_secant_hvp(torch.ones(2), torch.ones(2), 0.0)
    with pytest.raises(ValueError, match="matching"):
        central_force_secant_hvp(torch.ones(2), torch.ones(3), 1e-3)


def test_pcgrad_removes_two_task_conflict() -> None:
    gradients = {
        "force": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "hvp": torch.tensor([-1.0, 1.0], dtype=torch.float64),
    }

    projected = project_conflicting_gradients(gradients)

    assert torch.dot(projected["force"], gradients["hvp"]) >= -1e-14
    assert torch.dot(projected["hvp"], gradients["force"]) >= -1e-14


def test_assign_flat_parameter_gradient_checks_size_and_shape() -> None:
    first = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    flat = torch.arange(6, dtype=torch.float64)

    assign_flat_parameter_gradient([first, second], flat)

    torch.testing.assert_close(first.grad, flat[:4].reshape(2, 2))
    torch.testing.assert_close(second.grad, flat[4:])
    with pytest.raises(ValueError, match="size"):
        assign_flat_parameter_gradient([first, second], flat[:-1])


def test_fixed_scale_pcgrad_fills_unused_parameter_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    terms = {
        "energy_loss": first**2,
        "force_loss": (first - second) ** 2,
        "hvp_loss": second**2,
        "parameter_loss": (first**2 + second**2) / 2.0,
    }
    training = {
        "lambda_energy": 1.0,
        "lambda_force": 1.0,
        "lambda_hvp": 1.0,
        "lambda_parameter": 0.0,
        "task_gradient_reference_norms": {
            "energy": 1.0,
            "force": 1.0,
            "hvp": 1.0,
        },
    }

    diagnostics = apply_fixed_scale_pcgrad(terms, [first, second], training)

    assert first.grad is not None and second.grad is not None
    assert torch.isfinite(first.grad) and torch.isfinite(second.grad)
    assert diagnostics["combined_gradient_norm_before_clip"] > 0.0


def test_fixed_scale_pcgrad_accepts_accumulated_task_gradients() -> None:
    direct_parameters = [
        torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64)),
    ]
    accumulated_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in direct_parameters
    ]
    first, second = direct_parameters
    terms = {
        "energy_loss": first**2,
        "force_loss": (first - second) ** 2,
        "hvp_loss": second**2,
        "parameter_loss": (first**2 + second**2) / 2.0,
    }
    training = {
        "lambda_energy": 1.0,
        "lambda_force": 1.0,
        "lambda_hvp": 1.0,
        "lambda_parameter": 0.0,
        "task_gradient_reference_norms": {
            "energy": 1.0,
            "force": 1.0,
            "hvp": 1.0,
        },
    }
    raw_gradients = {
        "energy": torch.tensor([4.0, 0.0], dtype=torch.float64),
        "force": torch.tensor([-2.0, 2.0], dtype=torch.float64),
        "hvp": torch.tensor([0.0, 6.0], dtype=torch.float64),
    }

    direct_diagnostics = apply_fixed_scale_pcgrad(
        terms, direct_parameters, training
    )
    accumulated_diagnostics = apply_fixed_scale_pcgrad_from_gradients(
        raw_gradients, accumulated_parameters, training
    )

    for direct, accumulated in zip(
        direct_parameters, accumulated_parameters, strict=True
    ):
        torch.testing.assert_close(direct.grad, accumulated.grad)
    assert direct_diagnostics == accumulated_diagnostics


def test_initial_checkpoint_metadata_is_fail_closed() -> None:
    inputs = {
        "initial_protocol_sha256": "protocol-sha",
        "initial_arm_id": "source-arm",
    }
    checkpoint = {
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "protocol_sha256": "protocol-sha",
        "arm_id": "source-arm",
        "selected_parent_ids": ["0028399"],
    }

    _validate_initial_checkpoint_metadata(checkpoint, inputs, ["0028399"])

    drifted = dict(checkpoint, test100_accessed=True)
    with pytest.raises(ValueError, match="frozen Test100"):
        _validate_initial_checkpoint_metadata(drifted, inputs, ["0028399"])
    with pytest.raises(ValueError, match="parent list drift"):
        _validate_initial_checkpoint_metadata(checkpoint, inputs, ["0031108"])


def test_backtracking_gate_requires_all_tasks_and_total_to_improve() -> None:
    baseline = {"energy": 1.0, "force": 2.0, "hvp": 3.0}

    assert backtracking_tasks_are_acceptable(
        baseline,
        {"energy": 0.9, "force": 1.9, "hvp": 2.9},
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )
    assert not backtracking_tasks_are_acceptable(
        baseline,
        {"energy": 0.9, "force": 1.9, "hvp": 3.1},
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )
    assert not backtracking_tasks_are_acceptable(
        baseline,
        {"energy": float("nan"), "force": 1.9, "hvp": 2.9},
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )


def test_budget_backtracking_allows_energy_trade_within_physical_gate() -> None:
    baseline = {"energy": 1e-8, "force": 0.5, "hvp": 3.0}
    budgets = {"energy": 4e-4, "force": 3.6e-3}

    assert backtracking_tasks_are_budget_acceptable(
        baseline,
        {"energy": 3e-4, "force": 0.49, "hvp": 2.9},
        budgets,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )
    assert not backtracking_tasks_are_budget_acceptable(
        baseline,
        {"energy": 5e-4, "force": 0.49, "hvp": 2.9},
        budgets,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )
    assert not backtracking_tasks_are_budget_acceptable(
        baseline,
        {"energy": 3e-4, "force": 0.51, "hvp": 2.9},
        budgets,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-12,
    )
