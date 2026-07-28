from __future__ import annotations

import argparse

import pytest
import torch

from scripts.expand_qm9_complete_total_hidden_width import expand
from scripts.qm9_complete_total_geometry_mlp_capacity import SmoothDescriptorResidual


def _checkpoint(path, *, frozen: bool = True) -> None:
    model = SmoothDescriptorResidual(
        feature_count=7,
        hidden_size=3,
        initial_linear=torch.linspace(-0.2, 0.3, 7, dtype=torch.float64),
        seed=17,
    )
    with torch.no_grad():
        model.output.copy_(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64))
    torch.save(
        {
            "step": 123,
            "test100_accessed": False if frozen else True,
            "test100_evaluations_used": 0,
            "config": {"hidden_size": 3},
            "feature_inventory_manifest_sha256": "inventory",
            "state_dict": model.state_dict(),
            "optimizer_state_dict": {"must_not_be_reused": True},
        },
        path,
    )


def test_hidden_expansion_preserves_energy_force_and_hessian(tmp_path) -> None:
    source_path = tmp_path / "source.ckpt"
    _checkpoint(source_path)
    result = expand(
        argparse.Namespace(
            source_checkpoint=source_path,
            target_hidden_size=9,
            seed=29,
            output_dir=tmp_path / "expanded",
        )
    )
    source = torch.load(source_path, weights_only=False)
    target = torch.load(result["output_checkpoint"], weights_only=False)
    source_model = SmoothDescriptorResidual(
        7, 3, torch.zeros(7, dtype=torch.float64), seed=1
    )
    target_model = SmoothDescriptorResidual(
        7, 9, torch.zeros(7, dtype=torch.float64), seed=1
    )
    source_model.load_state_dict(source["state_dict"])
    target_model.load_state_dict(target["state_dict"])

    generator = torch.Generator().manual_seed(31)
    descriptor = torch.randn(7, dtype=torch.float64, generator=generator)
    jacobian = torch.randn(6, 7, dtype=torch.float64, generator=generator)
    weighted_hessian = torch.randn(36, 7, dtype=torch.float64, generator=generator)
    source_values = source_model.derivatives(
        descriptor, jacobian, weighted_hessian, hessian_weight=0.7
    )
    target_values = target_model.derivatives(
        descriptor, jacobian, weighted_hessian, hessian_weight=0.7
    )
    for source_value, target_value in zip(source_values, target_values, strict=True):
        torch.testing.assert_close(target_value, source_value, rtol=0.0, atol=1e-14)

    assert target["state_dict"]["weight"].shape == (9, 7)
    assert torch.count_nonzero(target["state_dict"]["weight"][3:]) > 0
    assert torch.count_nonzero(target["state_dict"]["output"][3:]) == 0
    assert "optimizer_state_dict" not in target
    assert target["feature_inventory_manifest_sha256"] == "inventory"
    assert result["test100_accessed"] is False


def test_hidden_expansion_rejects_nonexpansion_and_test_access(tmp_path) -> None:
    source_path = tmp_path / "source.ckpt"
    _checkpoint(source_path)
    with pytest.raises(ValueError, match="must exceed"):
        expand(
            argparse.Namespace(
                source_checkpoint=source_path,
                target_hidden_size=3,
                seed=1,
                output_dir=tmp_path / "same",
            )
        )
    exposed_path = tmp_path / "exposed.ckpt"
    _checkpoint(exposed_path, frozen=False)
    with pytest.raises(ValueError, match="does not freeze Test100"):
        expand(
            argparse.Namespace(
                source_checkpoint=exposed_path,
                target_hidden_size=4,
                seed=1,
                output_dir=tmp_path / "exposed",
            )
        )


def test_canceling_pair_expansion_preserves_function_and_wakes_weight_gradients(
    tmp_path,
) -> None:
    source_path = tmp_path / "source.ckpt"
    _checkpoint(source_path)
    result = expand(
        argparse.Namespace(
            source_checkpoint=source_path,
            target_hidden_size=7,
            seed=37,
            new_unit_initialization="canceling_pairs",
            pair_output_magnitude=1.0e-3,
            output_dir=tmp_path / "paired",
        )
    )
    source = torch.load(source_path, weights_only=False)
    target = torch.load(result["output_checkpoint"], weights_only=False)
    source_model = SmoothDescriptorResidual(
        7, 3, torch.zeros(7, dtype=torch.float64), seed=1
    )
    target_model = SmoothDescriptorResidual(
        7, 7, torch.zeros(7, dtype=torch.float64), seed=1
    )
    source_model.load_state_dict(source["state_dict"])
    target_model.load_state_dict(target["state_dict"])
    generator = torch.Generator().manual_seed(41)
    descriptor = torch.randn(7, dtype=torch.float64, generator=generator)
    jacobian = torch.randn(6, 7, dtype=torch.float64, generator=generator)
    weighted_hessian = torch.randn(36, 7, dtype=torch.float64, generator=generator)
    source_values = source_model.derivatives(
        descriptor, jacobian, weighted_hessian, hessian_weight=0.7
    )
    target_values = target_model.derivatives(
        descriptor, jacobian, weighted_hessian, hessian_weight=0.7
    )
    for source_value, target_value in zip(source_values, target_values, strict=True):
        torch.testing.assert_close(target_value, source_value, rtol=0.0, atol=1e-14)

    target_model.zero_grad(set_to_none=True)
    energy, _ = target_model.energy_force(descriptor, jacobian)
    energy.backward()
    pair_gradients = target_model.weight.grad[3:].reshape(2, 2, 7)
    assert torch.count_nonzero(pair_gradients) > 0
    torch.testing.assert_close(
        pair_gradients[:, 0], -pair_gradients[:, 1], rtol=1e-13, atol=1e-15
    )
    assert result["new_pair_output_sum_max_abs"] == 0.0
    assert result["new_pair_weight_max_abs_difference"] == 0.0
