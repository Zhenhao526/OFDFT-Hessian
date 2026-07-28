import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.qm9_complete_total_geometry_mlp_capacity import (
    DeepSmoothDescriptorResidual,
    FrozenReplayCache,
    SmoothDescriptorResidual,
    _active_stage,
    _design_provenance,
    _aggregate_parent_hvp_losses,
    _directional_hvp_loss,
    _low_mode_spectral_loss,
    _parent_cv_partition,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analytic_scalar_force_and_hessian_match_autograd():
    torch.manual_seed(47)
    feature_count = 5
    coordinate_count = 4
    model = SmoothDescriptorResidual(
        feature_count,
        hidden_size=3,
        initial_linear=torch.randn(feature_count, dtype=torch.float64),
        seed=53,
    )
    with torch.no_grad():
        model.output.copy_(torch.randn_like(model.output))
    base = torch.randn(feature_count, dtype=torch.float64)
    jacobian = torch.randn(coordinate_count, feature_count, dtype=torch.float64)
    feature_hessian = torch.randn(
        coordinate_count, coordinate_count, feature_count, dtype=torch.float64
    )
    feature_hessian = 0.5 * (
        feature_hessian + feature_hessian.transpose(0, 1)
    )
    coordinates = torch.zeros(coordinate_count, dtype=torch.float64)

    def descriptor(candidate):
        linear = jacobian.T @ candidate
        quadratic = 0.5 * torch.einsum(
            "i,ijf,j->f", candidate, feature_hessian, candidate
        )
        return base + linear + quadratic

    def scalar(candidate):
        value = descriptor(candidate)
        activation = model.weight @ value + model.bias
        return torch.dot(model.linear, value) + torch.dot(
            model.output, torch.nn.functional.softplus(activation)
        )

    energy, force, hessian = model.derivatives(
        base,
        jacobian,
        feature_hessian.reshape(coordinate_count**2, feature_count),
        hessian_weight=1.0,
    )
    expected_energy = scalar(coordinates)
    expected_gradient = torch.func.jacrev(scalar)(coordinates)
    expected_hessian = torch.func.hessian(scalar)(coordinates)
    energy_only, force_only = model.energy_force(base, jacobian)
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(force, -expected_gradient)
    torch.testing.assert_close(hessian, expected_hessian)
    torch.testing.assert_close(energy_only, energy)
    torch.testing.assert_close(force_only, force)

    scale = 3.0

    def extensive_scalar(candidate):
        value = descriptor(candidate) / scale
        activation = model.weight @ value + model.bias
        return scale * (
            torch.dot(model.linear, value)
            + torch.dot(model.output, torch.nn.functional.softplus(activation))
        )

    extensive_energy, extensive_force, extensive_hessian = model.derivatives(
        base,
        jacobian,
        feature_hessian.reshape(coordinate_count**2, feature_count),
        hessian_weight=1.0,
        extensivity_scale=scale,
    )
    torch.testing.assert_close(extensive_energy, extensive_scalar(coordinates))
    torch.testing.assert_close(
        extensive_force, -torch.func.jacrev(extensive_scalar)(coordinates)
    )
    torch.testing.assert_close(
        extensive_hessian, torch.func.hessian(extensive_scalar)(coordinates)
    )


def test_deep_analytic_scalar_force_and_hessian_match_autograd():
    torch.manual_seed(59)
    feature_count = 5
    coordinate_count = 4
    model = DeepSmoothDescriptorResidual(
        feature_count,
        hidden_size=4,
        deep_hidden_size=3,
        initial_linear=torch.randn(feature_count, dtype=torch.float64),
        seed=61,
    )
    with torch.no_grad():
        model.output.copy_(torch.randn_like(model.output))
        model.deep_output.copy_(torch.randn_like(model.deep_output))
        model.bias.copy_(torch.randn_like(model.bias))
        model.deep_bias.copy_(torch.randn_like(model.deep_bias))
    base = torch.randn(feature_count, dtype=torch.float64)
    jacobian = torch.randn(coordinate_count, feature_count, dtype=torch.float64)
    feature_hessian = torch.randn(
        coordinate_count, coordinate_count, feature_count, dtype=torch.float64
    )
    feature_hessian = 0.5 * (
        feature_hessian + feature_hessian.transpose(0, 1)
    )
    coordinates = torch.zeros(coordinate_count, dtype=torch.float64)

    def descriptor(candidate):
        return base + jacobian.T @ candidate + 0.5 * torch.einsum(
            "i,ijf,j->f", candidate, feature_hessian, candidate
        )

    def scalar(candidate):
        value = descriptor(candidate)
        first = torch.nn.functional.softplus(model.weight @ value + model.bias)
        deep = torch.nn.functional.softplus(
            model.deep_weight @ first + model.deep_bias
        )
        return (
            torch.dot(model.linear, value)
            + torch.dot(model.output, first)
            + torch.dot(model.deep_output, deep)
        )

    energy, force, hessian = model.derivatives(
        base,
        jacobian,
        feature_hessian.reshape(coordinate_count**2, feature_count),
        hessian_weight=1.0,
    )
    torch.testing.assert_close(energy, scalar(coordinates))
    torch.testing.assert_close(force, -torch.func.jacrev(scalar)(coordinates))
    torch.testing.assert_close(hessian, torch.func.hessian(scalar)(coordinates))
    energy_only, force_only = model.energy_force(base, jacobian)
    torch.testing.assert_close(energy_only, energy)
    torch.testing.assert_close(force_only, force)


def test_directional_hvp_loss_mixes_absolute_and_floored_relative_terms():
    prediction = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    target = torch.zeros((2, 2), dtype=torch.float64)
    reference = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    directions = torch.eye(2, dtype=torch.float64)

    loss = _directional_hvp_loss(
        prediction,
        target,
        reference,
        directions,
        hessian_weight=1.0,
        relative_fraction=0.5,
        absolute_scale=2.0,
        reference_floor=0.5,
    )

    # Direction 0: absolute=1/2, relative=4. Direction 1: absolute=1/8, relative=4.
    expected = 0.5 * (0.5 * (0.5 + 4.0) + 0.5 * (0.125 + 4.0))
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float64))


def test_low_mode_spectral_loss_penalizes_wrong_curvature_sign():
    directions = torch.eye(2, dtype=torch.float64)
    reference = torch.diag(torch.tensor([-0.5, 1.0], dtype=torch.float64))
    correct = reference.clone()
    wrong = torch.diag(torch.tensor([0.5, -1.0], dtype=torch.float64))

    correct_loss = _low_mode_spectral_loss(
        correct,
        reference,
        directions,
        reference_floor=0.1,
        wrong_curvature_multiplier=2.0,
    )
    wrong_loss = _low_mode_spectral_loss(
        wrong,
        reference,
        directions,
        reference_floor=0.1,
        wrong_curvature_multiplier=2.0,
    )

    torch.testing.assert_close(correct_loss, torch.zeros_like(correct_loss))
    assert wrong_loss > 0


def test_parent_hvp_tail_loss_emphasizes_only_the_largest_train_losses():
    losses = torch.tensor([1.0, 2.0, 3.0, 8.0], dtype=torch.float64)

    mean_only = _aggregate_parent_hvp_losses(
        losses, tail_fraction=0.25, tail_weight=0.0
    )
    with_tail = _aggregate_parent_hvp_losses(
        losses, tail_fraction=0.25, tail_weight=0.5
    )

    torch.testing.assert_close(mean_only, torch.mean(losses))
    torch.testing.assert_close(with_tail, torch.mean(losses) + 0.5 * losses[-1])


def test_frozen_replay_cache_uses_scalar_energy_force_path(tmp_path):
    cache_path = tmp_path / "record.npz"
    np.savez(
        cache_path,
        descriptor=np.asarray([2.0, 3.0]),
        descriptor_jacobian=np.ones((3, 2)),
        energy_target=np.asarray(1.0),
        force_target=np.ones(3),
    )
    success_csv = tmp_path / "success.csv"
    with success_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "molecule_id",
                "sample_id",
                "descriptor_cache",
                "descriptor_cache_sha256",
                "natoms",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "molecule_id": "0000001",
                "sample_id": 0,
                "descriptor_cache": cache_path.as_posix(),
                "descriptor_cache_sha256": _sha256(cache_path),
                "natoms": 1,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "test100_accessed": False,
                "active_feature_count": 2,
                "schema_manifest_sha256": "schema",
                "schema_checkpoint_sha256": "checkpoint",
                "artifacts": {
                    "success_csv": {
                        "path": success_csv.as_posix(),
                        "sha256": _sha256(success_csv),
                    }
                },
            }
        )
    )
    replay = FrozenReplayCache(manifest_path, 2, cpu_cache_size=1)
    model = SmoothDescriptorResidual(
        2,
        hidden_size=1,
        initial_linear=torch.zeros(2, dtype=torch.float64),
        seed=7,
    )

    energy_loss, force_loss = replay.batch_losses(
        model,
        [0],
        torch.device("cpu"),
        energy_scale=1.0,
        force_scale=1.0,
    )

    torch.testing.assert_close(energy_loss, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(force_loss, torch.tensor(1.0, dtype=torch.float64))
    assert replay.disk_load_count == 1
    replay.batch_losses(
        model,
        [0],
        torch.device("cpu"),
        energy_scale=1.0,
        force_scale=1.0,
    )
    assert replay.cache_hit_count == 1


def test_active_stage_unit_scale_falls_back_to_resolved_feature_keys(tmp_path):
    stage_dir = tmp_path / "three_body"
    stage_dir.mkdir()
    np.save(stage_dir / "weighted_hessian_design.npy", np.asarray([[3.0, 4.0]]))
    np.save(stage_dir / "anchor_constraints.npy", np.asarray([[1.0, 2.0]]))
    keys = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    np.savez(
        stage_dir / "shared_coefficients.npz",
        coefficients=np.asarray([5.0, 6.0]),
        feature_keys=keys,
    )

    stage = _active_stage(
        tmp_path,
        "three_body",
        None,
        cutoff=1.0e-8,
        chunk_rows=16,
        zero_linear_initialization=True,
        feature_scale_mode="unit",
    )

    np.testing.assert_array_equal(stage["keys"], keys)
    np.testing.assert_allclose(stage["source_column_norms"], [3.0, 4.0])
    np.testing.assert_allclose(stage["norms"], [1.0, 1.0])
    np.testing.assert_allclose(stage["initial_linear"], [0.0, 0.0])


def test_active_stage_floored_column_norm_limits_inverse_amplification(tmp_path):
    stage_dir = tmp_path / "three_body"
    stage_dir.mkdir()
    np.save(
        stage_dir / "weighted_hessian_design.npy",
        np.asarray([[1.0e-4, 2.0]]),
    )
    np.save(stage_dir / "anchor_constraints.npy", np.asarray([[1.0, 2.0]]))
    keys = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    np.savez(
        stage_dir / "shared_coefficients.npz",
        coefficients=np.asarray([5.0, 6.0]),
        feature_keys=keys,
    )

    stage = _active_stage(
        tmp_path,
        "three_body",
        None,
        cutoff=1.0e-8,
        chunk_rows=16,
        zero_linear_initialization=True,
        feature_scale_mode="floored_column_norm",
        feature_scale_floor=0.1,
    )

    np.testing.assert_allclose(stage["source_column_norms"], [1.0e-4, 2.0])
    np.testing.assert_allclose(stage["norms"], [0.1, 2.0])
    assert stage["feature_scale_floor"] == 0.1


def test_active_stage_floored_column_norm_requires_positive_floor(tmp_path):
    stage_dir = tmp_path / "three_body"
    stage_dir.mkdir()
    np.save(stage_dir / "weighted_hessian_design.npy", np.asarray([[1.0]]))
    np.save(stage_dir / "anchor_constraints.npy", np.asarray([[1.0]]))
    np.savez(
        stage_dir / "shared_coefficients.npz",
        coefficients=np.asarray([0.0]),
        feature_keys=np.asarray([[1, 2]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="feature_scale_floor must be positive"):
        _active_stage(
            tmp_path,
            "three_body",
            None,
            cutoff=1.0e-8,
            chunk_rows=16,
            zero_linear_initialization=True,
            feature_scale_mode="floored_column_norm",
            feature_scale_floor=0.0,
        )


def test_active_stage_maps_stage2_columns_into_larger_inventory(tmp_path):
    stage_dir = tmp_path / "three_body"
    stage_dir.mkdir()
    np.save(stage_dir / "weighted_hessian_design.npy", np.asarray([[2.0, 3.0]]))
    np.save(stage_dir / "anchor_constraints.npy", np.asarray([[4.0, 5.0]]))
    source_keys = np.asarray([[1, 1], [3, 3]], dtype=np.int64)
    np.save(stage_dir / "feature_keys.npy", source_keys)
    inventory = {
        "feature_keys": np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.int64),
        "column_norms": np.asarray([1.0, 7.0, 3.0]),
    }

    stage = _active_stage(
        tmp_path,
        "three_body",
        None,
        cutoff=0.0,
        chunk_rows=16,
        zero_linear_initialization=True,
        feature_scale_mode="floored_column_norm",
        feature_scale_floor=1.0,
        inventory_stage=inventory,
    )

    np.testing.assert_array_equal(stage["global_indices"], [0, 2])
    np.testing.assert_allclose(stage["norms"], [1.0, 3.0])
    assert stage["active_count"] == 3
    assert stage["source_design_feature_count"] == 2
    np.testing.assert_allclose(stage["initial_linear"], np.zeros(3))


def test_design_provenance_accepts_feature_keys_inside_coefficients_npz(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    for stage_name in ("three_body", "four_body"):
        stage_dir = tmp_path / stage_name
        stage_dir.mkdir()
        (stage_dir / "summary.json").write_text("{}")
        np.save(stage_dir / "weighted_hessian_design.npy", np.ones((1, 1)))
        np.save(stage_dir / "anchor_constraints.npy", np.ones((1, 1)))
        np.savez(
            stage_dir / "shared_coefficients.npz",
            coefficients=np.ones(1),
            feature_keys=np.ones((1, 2), dtype=np.int64),
        )

    provenance = _design_provenance(tmp_path)

    assert provenance["stages"]["three_body"]["feature_keys_dataset"] == "feature_keys"
    assert provenance["stages"]["four_body"]["feature_keys"].endswith(
        "shared_coefficients.npz"
    )


def _parent_cv_fixture(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    design_root = tmp_path / "design"
    design_root.mkdir()
    (design_root / "summary.json").write_text("{}")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "protocol_id: test_parent_cv\n"
        "scope:\n"
        "  validation_parent_access_allowed: false\n"
        "  test100_accessed: false\n"
        "  test100_evaluations_used: 0\n"
        "  parents_per_fit_fold: 1\n"
        "  parents_per_held_fold: 1\n"
        "frozen_inputs:\n"
        f"  baseline_manifest_sha256: {_sha256(baseline)}\n"
        f"  design_summary_sha256: {_sha256(design_root / 'summary.json')}\n"
        f"  feature_inventory_manifest_sha256: {_sha256(inventory)}\n"
        "  replay_cache_manifest_sha256: unused\n"
        "cross_validation:\n"
        "  base_seed: 100\n"
        "  base_replay_seed: 200\n"
        "  fold_assignment:\n"
        "    - ['0000001']\n"
        "    - ['0000002']\n"
        "preregistered_variants:\n"
        "  - id: no_replay\n"
        "    train800_energy_force_replay: false\n"
        "    runtime_parameters:\n"
        "      hidden_size: 8\n"
        "      learning_rate: 0.001\n"
    )
    args = SimpleNamespace(
        parent_cv_protocol=protocol,
        parent_cv_variant_id="no_replay",
        parent_cv_fold_index=0,
        baseline_manifest=baseline,
        design_root=design_root,
        feature_inventory_manifest=inventory,
        checkpoint=None,
        initial_root=None,
        zero_linear_initialization=True,
        direction_manifest=None,
        hidden_size=8,
        learning_rate=1.0e-3,
        replay_cache_manifest=None,
        seed=100,
        replay_seed=200,
    )
    parents = [
        SimpleNamespace(molecule_id="0000001"),
        SimpleNamespace(molecule_id="0000002"),
    ]
    return parents, args


def test_parent_cv_partition_is_hash_bound_and_holds_out_one_fold(tmp_path):
    parents, args = _parent_cv_fixture(tmp_path)

    fit, held, metadata = _parent_cv_partition(parents, args)

    assert [parent.molecule_id for parent in fit] == ["0000002"]
    assert [parent.molecule_id for parent in held] == ["0000001"]
    assert metadata["held_parent_hessian_used_for_gradient"] is False
    assert metadata["held_parent_hessian_used_for_checkpoint_selection"] is False
    assert metadata["test100_accessed"] is False
    assert metadata["test100_evaluations_used"] == 0


def test_parent_cv_partition_rejects_runtime_drift(tmp_path):
    parents, args = _parent_cv_fixture(tmp_path)
    args.hidden_size = 16

    with pytest.raises(ValueError, match="runtime parameter differs"):
        _parent_cv_partition(parents, args)
