from __future__ import annotations

import hashlib

import pytest
import torch
import yaml

from scripts.qm9_complete_total_local_scalar_full_hessian_capacity import (
    _build_model_and_optimizer,
    _build_single_group_optimizer,
    _full_hessian_loss,
    _load_protocol,
    _periodic_evaluation_metrics,
    _separate_gradnorm_target_multiplier,
    _task_gradient_norm,
)


def test_full_hessian_loss_is_zero_only_for_exact_matrix() -> None:
    reference = torch.eye(4, dtype=torch.float64)
    exact = _full_hessian_loss(
        reference,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.1,
    )
    shifted = _full_hessian_loss(
        reference + 0.01,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.1,
    )

    assert float(exact) == 0.0
    assert float(shifted) > 0.0


def test_single_group_optimizer_supports_zero_momentum_sgd() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    optimizer = _build_single_group_optimizer(
        [parameter], {"optimizer": "sgd", "learning_rate": 1e-3, "momentum": 0.0}
    )

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    with pytest.raises(ValueError, match="zero momentum"):
        _build_single_group_optimizer(
            [parameter],
            {"optimizer": "sgd", "learning_rate": 1e-3, "momentum": 0.9},
        )


def test_task_gradient_norm_reports_parameter_space_l2_norm() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    loss = torch.sum(parameter * parameter)

    norm = _task_gradient_norm(loss, [parameter])

    assert torch.allclose(norm, torch.tensor(10.0, dtype=torch.float64))
    assert parameter.grad is None


def test_periodic_metrics_include_all_live_capacity_gates() -> None:
    final = {
        "hessian_relative_frobenius": {"median": 0.02, "p90": 0.03, "max": 0.04},
        "energy_abs_error_hartree": {"median": 0.1},
        "force_mae_hartree_per_bohr": {"median": 0.05},
        "energy_median_ratio_to_source": 0.8,
        "force_median_ratio_to_source": 0.9,
        "max_antisymmetric_over_symmetric_frobenius": 1e-8,
        "selection_score": 0.06,
        "gate": {"passed": True},
    }

    metrics = _periodic_evaluation_metrics(final)

    assert metrics["median_relative_frobenius"] == 0.02
    assert metrics["median_energy_abs_error_hartree"] == 0.1
    assert metrics["median_force_mae_hartree_per_bohr"] == 0.05
    assert metrics["energy_median_ratio_to_source"] == 0.8
    assert metrics["force_median_ratio_to_source"] == 0.9
    assert metrics["selection_score"] == 0.06
    assert metrics["capacity_gate_passed"] is True


def test_separate_gradnorm_can_preserve_hessian_warmup() -> None:
    assert _separate_gradnorm_target_multiplier(
        "hessian", hessian_warmup=0.2, preserve_hessian_warmup=True
    ) == 0.2
    assert _separate_gradnorm_target_multiplier(
        "force", hessian_warmup=0.2, preserve_hessian_warmup=True
    ) == 1.0
    assert _separate_gradnorm_target_multiplier(
        "hessian", hessian_warmup=0.2, preserve_hessian_warmup=False
    ) == 1.0


def test_tensor_equivariant_builder_binds_output_scale() -> None:
    arm = {
        "architecture": "tensor_product_quadrupole_message_passing",
        "scalar_channels": 4,
        "vector_channels": 2,
        "tensor_channels": 1,
        "radial_size": 4,
        "radial_hidden_size": 6,
        "interaction_layers": 1,
        "cutoff_bohr": 8.0,
        "output_scale": 37.0,
        "learning_rate": 1e-4,
        "seed": 3,
    }

    model, _, metadata = _build_model_and_optimizer(arm, torch.device("cpu"))

    assert model.output_scale == 37.0
    assert metadata["architecture"] == "tensor_product_quadrupole_message_passing"


def test_equivariant_quadratic_builder_binds_coefficient_scale() -> None:
    arm = {
        "architecture": "reference_local_equivariant_quadratic_scalar",
        "hidden_size": 8,
        "radial_size": 4,
        "cutoff_bohr": 8.0,
        "coefficient_scale": 3.5,
        "seed": 29,
        "learning_rate": 1e-3,
    }

    model, _, metadata = _build_model_and_optimizer(arm, torch.device("cpu"))

    assert model.network.coefficient_scale == 3.5
    assert metadata["architecture"] == arm["architecture"]


def test_three_task_protocol_stage_is_accepted(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    stable5 = tmp_path / "stable5.yaml"
    baseline.write_text("{}")
    stable5.write_text("stage: stable5\n")
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "stage": "stable5_fit_only_three_task_gradient_conflict_audit",
        "inputs": {
            "baseline_manifest": baseline.as_posix(),
            "baseline_manifest_sha256": hashlib.sha256(
                baseline.read_bytes()
            ).hexdigest(),
            "stable5_protocol": stable5.as_posix(),
            "stable5_protocol_sha256": hashlib.sha256(
                stable5.read_bytes()
            ).hexdigest(),
        },
        "arms": [{"id": "K5"}],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))

    loaded, arm = _load_protocol(protocol_path, "K5", "smoke")

    assert loaded["stage"] == "stable5_fit_only_three_task_gradient_conflict_audit"
    assert arm["id"] == "K5"


def test_nonlinear_local_scalar_protocol_stage_is_accepted(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    stable5 = tmp_path / "stable5.yaml"
    baseline.write_text("{}")
    stable5.write_text("stage: stable5\n")
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "stage": "stable5_fit_only_nonlinear_local_scalar_capacity",
        "inputs": {
            "baseline_manifest": baseline.as_posix(),
            "baseline_manifest_sha256": hashlib.sha256(
                baseline.read_bytes()
            ).hexdigest(),
            "stable5_protocol": stable5.as_posix(),
            "stable5_protocol_sha256": hashlib.sha256(
                stable5.read_bytes()
            ).hexdigest(),
        },
        "arms": [{"id": "K9"}],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))

    loaded, arm = _load_protocol(protocol_path, "K9", "smoke")

    assert loaded["stage"] == "stable5_fit_only_nonlinear_local_scalar_capacity"
    assert arm["id"] == "K9"


def test_force_secant_pcgrad_stage_binds_gradient_audit(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    stable5 = tmp_path / "stable5.yaml"
    gradient_audit = tmp_path / "gradient_audit.json"
    baseline.write_text("{}")
    stable5.write_text("stage: stable5\n")
    gradient_audit.write_text('{"test100_accessed": false}\n')
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "stage": "stable5_fit_only_force_secant_gradient_conflict_intervention",
        "inputs": {
            "baseline_manifest": baseline.as_posix(),
            "baseline_manifest_sha256": hashlib.sha256(
                baseline.read_bytes()
            ).hexdigest(),
            "stable5_protocol": stable5.as_posix(),
            "stable5_protocol_sha256": hashlib.sha256(
                stable5.read_bytes()
            ).hexdigest(),
            "gradient_audit": gradient_audit.as_posix(),
            "gradient_audit_sha256": hashlib.sha256(
                gradient_audit.read_bytes()
            ).hexdigest(),
        },
        "arms": [{"id": "E13"}],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))

    loaded, arm = _load_protocol(protocol_path, "E13", "smoke")

    assert loaded["stage"].endswith("gradient_conflict_intervention")
    assert arm["id"] == "E13"
    gradient_audit.write_text("drifted\n")
    with pytest.raises(ValueError, match="gradient audit hash drift"):
        _load_protocol(protocol_path, "E13", "smoke")
