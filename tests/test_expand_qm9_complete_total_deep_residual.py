from __future__ import annotations

import argparse

import pytest
import torch

from scripts.expand_qm9_complete_total_deep_residual import expand
from scripts.qm9_complete_total_geometry_mlp_capacity import (
    DeepSmoothDescriptorResidual,
    SmoothDescriptorResidual,
)


def _checkpoint(path, *, exposed=False):
    model = SmoothDescriptorResidual(
        6,
        hidden_size=4,
        initial_linear=torch.randn(6, dtype=torch.float64),
        seed=71,
    )
    with torch.no_grad():
        model.output.copy_(torch.randn_like(model.output))
    torch.save(
        {
            "step": 17,
            "state_dict": model.state_dict(),
            "config": {"hidden_size": 4},
            "feature_inventory_manifest_sha256": "inventory",
            "test100_accessed": exposed,
            "test100_evaluations_used": int(exposed),
        },
        path,
    )


def test_deep_expansion_preserves_scalar_derivatives_and_wakes_pairs(tmp_path):
    source_path = tmp_path / "source.ckpt"
    _checkpoint(source_path)
    result = expand(
        argparse.Namespace(
            source_checkpoint=source_path,
            deep_hidden_size=8,
            seed=73,
            pair_output_magnitude=1.0e-3,
            output_dir=tmp_path / "deep",
        )
    )
    source = torch.load(source_path, weights_only=False)
    target = torch.load(result["output_checkpoint"], weights_only=False)
    shallow = SmoothDescriptorResidual(
        6, 4, torch.zeros(6, dtype=torch.float64), seed=1
    )
    deep = DeepSmoothDescriptorResidual(
        6, 4, 8, torch.zeros(6, dtype=torch.float64), seed=1
    )
    shallow.load_state_dict(source["state_dict"])
    deep.load_state_dict(target["state_dict"])
    generator = torch.Generator().manual_seed(79)
    descriptor = torch.randn(6, dtype=torch.float64, generator=generator)
    jacobian = torch.randn(5, 6, dtype=torch.float64, generator=generator)
    feature_hessian = torch.randn(25, 6, dtype=torch.float64, generator=generator)
    shallow_values = shallow.derivatives(
        descriptor, jacobian, feature_hessian, hessian_weight=0.6
    )
    deep_values = deep.derivatives(
        descriptor, jacobian, feature_hessian, hessian_weight=0.6
    )
    for source_value, target_value in zip(shallow_values, deep_values, strict=True):
        torch.testing.assert_close(target_value, source_value, rtol=0.0, atol=1e-14)

    deep.zero_grad(set_to_none=True)
    energy, _ = deep.energy_force(descriptor, jacobian)
    energy.backward()
    gradients = deep.deep_weight.grad.reshape(4, 2, 4)
    assert torch.count_nonzero(gradients) > 0
    torch.testing.assert_close(
        gradients[:, 0], -gradients[:, 1], rtol=1e-13, atol=1e-15
    )
    assert "optimizer_state_dict" not in target
    assert result["deep_pair_output_sum_max_abs"] == 0.0
    assert result["deep_pair_weight_max_abs_difference"] == 0.0


def test_deep_expansion_rejects_odd_width_and_test100_source(tmp_path):
    source_path = tmp_path / "source.ckpt"
    _checkpoint(source_path)
    with pytest.raises(ValueError, match="positive and even"):
        expand(
            argparse.Namespace(
                source_checkpoint=source_path,
                deep_hidden_size=7,
                seed=1,
                pair_output_magnitude=1.0e-3,
                output_dir=tmp_path / "odd",
            )
        )
    exposed_path = tmp_path / "exposed.ckpt"
    _checkpoint(exposed_path, exposed=True)
    with pytest.raises(ValueError, match="does not freeze Test100"):
        expand(
            argparse.Namespace(
                source_checkpoint=exposed_path,
                deep_hidden_size=8,
                seed=1,
                pair_output_magnitude=1.0e-3,
                output_dir=tmp_path / "exposed",
            )
        )
