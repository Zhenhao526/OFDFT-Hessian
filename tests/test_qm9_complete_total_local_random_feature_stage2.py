from __future__ import annotations

import numpy as np
import pytest
import torch

from mldft.ml.models.components.local_body_order_residual import (
    build_body_order_topology,
)
from scripts.qm9_complete_total_local_random_feature_stage2 import (
    CapacityParent,
    _load_kernel_checkpoint,
    _resolve_effective_ridge,
    _random_feature_subset_mask,
    _extend_training_direction_bank,
    _expand_hashed_subspace_coefficients,
    _hashed_subspace_mapping,
    _project_design_to_hashed_subspace,
    _stage2_design_blocks,
    _streamed_normal_equation_solve,
)
from scripts.qm9_complete_total_structured_scalar_subspace import (
    build_structured_scalar_subspace_mapping,
    expand_structured_subspace_coefficients,
    project_design_to_structured_subspace,
)


def test_load_kernel_checkpoint_returns_float64_tensors(tmp_path) -> None:
    checkpoint = tmp_path / "kernel.pt"
    keys = (
        "environment_inverse_rms",
        "projection",
        "bias",
        "activation_scale",
    )
    torch.save(
        {
            "protocol_sha256": "frozen",
            **{key: torch.ones(2, dtype=torch.float32) for key in keys},
        },
        checkpoint,
    )
    tensors = _load_kernel_checkpoint(
        {
            "inputs": {
                "v9_checkpoint": checkpoint,
                "v9_protocol_sha256": "frozen",
            }
        },
        torch.device("cpu"),
    )
    assert len(tensors) == 4
    assert all(tensor.dtype == torch.float64 for tensor in tensors)


def test_load_kernel_checkpoint_rejects_protocol_drift(tmp_path) -> None:
    checkpoint = tmp_path / "kernel.pt"
    torch.save({"protocol_sha256": "unexpected"}, checkpoint)
    with pytest.raises(ValueError, match="protocol drift"):
        _load_kernel_checkpoint(
            {
                "inputs": {
                    "v9_checkpoint": checkpoint,
                    "v9_protocol_sha256": "frozen",
                }
            },
            torch.device("cpu"),
        )


def test_ridge_override_is_explicitly_diagnostic_only() -> None:
    assert _resolve_effective_ridge(1e-8, None, False) == 1e-8
    assert _resolve_effective_ridge(1e-8, 1e-3, True) == 1e-3
    with pytest.raises(ValueError, match="diagnostic-ridge-override"):
        _resolve_effective_ridge(1e-8, 1e-3, False)
    with pytest.raises(ValueError, match="finite and positive"):
        _resolve_effective_ridge(1e-8, 0.0, True)


def test_random_feature_subset_is_stratified_across_scales_and_elements() -> None:
    mask, diagnostics = _random_feature_subset_mask(
        global_feature_count=22,
        element_count=2,
        activation_scale=torch.tensor(
            [0.5, 0.5, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64
        ),
        width_per_scale=1,
    )

    assert torch.where(mask)[0].tolist() == list(range(10)) + [10, 12, 14, 16, 18, 20]
    assert diagnostics == {
        "base_feature_count": 10,
        "element_count": 2,
        "scale_count": 3,
        "full_width_per_scale": 2,
        "selected_width_per_scale": 1,
        "selected_global_feature_count": 16,
        "full_global_feature_count": 22,
    }


def test_hashed_scalar_subspace_projection_matches_expanded_coefficients() -> None:
    generator = torch.Generator().manual_seed(53)
    design = torch.randn(19, 31, dtype=torch.float64, generator=generator)
    subspace_coefficients = torch.randn(
        11, dtype=torch.float64, generator=generator
    )
    buckets, signs, metadata = _hashed_subspace_mapping(
        feature_count=31,
        subspace_dimension=11,
        repetitions=3,
        seed=20260729,
    )
    projected = _project_design_to_hashed_subspace(
        design,
        buckets,
        signs,
        subspace_dimension=11,
        column_chunk_size=7,
    )
    expanded = _expand_hashed_subspace_coefficients(
        subspace_coefficients, buckets, signs
    )

    assert torch.allclose(
        projected @ subspace_coefficients,
        design @ expanded,
        atol=1e-12,
        rtol=1e-12,
    )
    assert metadata["mapping_sha256"] == _hashed_subspace_mapping(
        feature_count=31,
        subspace_dimension=11,
        repetitions=3,
        seed=20260729,
    )[2]["mapping_sha256"]
    assert metadata["empty_bucket_count"] >= 0


def test_structured_scalar_subspace_projection_matches_expanded_coefficients() -> None:
    element_count = 2
    radial_size = 3
    angular_order = 1
    environment_count = element_count * radial_size + (
        element_count * (element_count + 1) // 2
        * radial_size
        * radial_size
        * (angular_order + 1)
    )
    angular_count = element_count + element_count * environment_count
    four_body_keys = (
        (1, 1, 1, 1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0, 1, 1, 0),
        (1, 6, 6, 1, 1, 0, 1, 1),
    )
    generator = torch.Generator().manual_seed(71)
    projection = torch.randn(
        environment_count, 8, dtype=torch.float64, generator=generator
    )
    mapping, metadata = build_structured_scalar_subspace_mapping(
        element_count=element_count,
        radial_size=radial_size,
        angular_order=angular_order,
        angular_feature_count=angular_count,
        four_body_keys=four_body_keys,
        four_body_center_count=2,
        angular_radial_modes=2,
        four_body_radial_modes=1,
        random_environment_radial_modes=1,
        random_bias_order=2,
        inverse_rms=torch.linspace(0.2, 1.4, environment_count, dtype=torch.float64),
        projection=projection,
        bias=torch.linspace(-0.4, 0.4, 8, dtype=torch.float64),
        activation_scale=torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
    )
    design = torch.randn(
        13, mapping.shape[0], dtype=torch.float64, generator=generator
    )
    subspace_coefficients = torch.randn(
        mapping.shape[1], dtype=torch.float64, generator=generator
    )
    projected = project_design_to_structured_subspace(
        design, mapping, row_chunk_size=4, device=torch.device("cpu")
    )
    expanded = expand_structured_subspace_coefficients(
        subspace_coefficients, mapping
    )

    assert torch.allclose(
        projected @ subspace_coefficients,
        design @ expanded,
        atol=1e-11,
        rtol=1e-11,
    )
    repeated, repeated_metadata = build_structured_scalar_subspace_mapping(
        element_count=element_count,
        radial_size=radial_size,
        angular_order=angular_order,
        angular_feature_count=angular_count,
        four_body_keys=four_body_keys,
        four_body_center_count=2,
        angular_radial_modes=2,
        four_body_radial_modes=1,
        random_environment_radial_modes=1,
        random_bias_order=2,
        inverse_rms=torch.linspace(0.2, 1.4, environment_count, dtype=torch.float64),
        projection=projection,
        bias=torch.linspace(-0.4, 0.4, 8, dtype=torch.float64),
        activation_scale=torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
    )
    assert metadata["mapping_sha256"] == repeated_metadata["mapping_sha256"]
    assert torch.equal(mapping.indices(), repeated.indices())
    assert torch.equal(mapping.values(), repeated.values())
    assert metadata["four_body_group_count"] == 2
    assert metadata["definition"].startswith("label-independent")


def test_direction_extension_fills_only_the_internal_train_complement() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.2, 1.1, 0.0]],
        dtype=np.float64,
    )
    from scripts.prepare_qm9_complete_total_direction_generalization import (
        _external_basis,
    )

    external = _external_basis(positions)
    _, _, vh = np.linalg.svd(external.T, full_matrices=True)
    internal = vh[external.shape[1] :]
    directions = internal[:2]
    roles = np.asarray(["train", "heldout"])
    kinds = np.asarray(["bond", "angle"])

    extended, extended_roles, extended_kinds, diagnostics = (
        _extend_training_direction_bank(
            positions,
            directions,
            roles,
            kinds,
            target_train_count=None,
            maximal_train_complement=True,
            seed=17,
        )
    )

    assert extended.shape == (3, 9)
    assert extended_roles.tolist() == ["train", "heldout", "train"]
    assert extended_kinds[-1] == "extension_random_internal"
    assert diagnostics["internal_dimension"] == 3
    assert diagnostics["final_train_count"] == 2
    assert diagnostics["heldout_count"] == 1
    assert diagnostics["orthonormality_max_abs"] < 1e-12
    assert diagnostics["external_overlap_max_abs"] < 1e-12


def test_streamed_normal_equation_matches_augmented_lstsq() -> None:
    generator = torch.Generator().manual_seed(41)
    design = torch.randn(70, 13, dtype=torch.float64, generator=generator)
    design[:, 5] *= 1e-4
    target = torch.randn(70, dtype=torch.float64, generator=generator)
    ridge = 1e-6

    actual, diagnostics = _streamed_normal_equation_solve(
        design,
        target,
        ridge=ridge,
        column_floor=1e-12,
        row_chunk_size=11,
        device=torch.device("cpu"),
    )
    scale = torch.linalg.vector_norm(design, dim=0)
    normalized = design / scale[None, :]
    augmented = torch.cat(
        (normalized, ridge**0.5 * torch.eye(13, dtype=torch.float64)), dim=0
    )
    augmented_target = torch.cat((target, torch.zeros(13, dtype=torch.float64)))
    expected = torch.linalg.lstsq(augmented, augmented_target).solution / scale

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert diagnostics["converged"]
    assert diagnostics["final_relative_normal_residual"] < 1e-12


def test_stage2_design_uses_only_train_directions() -> None:
    atomic_numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64
    )
    coordinate_count = positions.numel()
    parent = CapacityParent(
        molecule_id="toy",
        natoms=2,
        atomic_numbers=atomic_numbers,
        positions=positions,
        topology=build_body_order_topology(atomic_numbers, positions),
        pbe_energy=1.0,
        pbe_force=np.ones((2, 3)),
        pbe_hessian=2.0 * np.eye(coordinate_count),
        source_energy=0.5,
        source_force=np.zeros((2, 3)),
        source_hessian_raw=np.eye(coordinate_count),
        source_hessian_symmetric=np.eye(coordinate_count),
    )
    feature_count = 4
    jet = (
        torch.ones(feature_count, dtype=torch.float64),
        torch.ones((feature_count, coordinate_count), dtype=torch.float64),
        torch.ones(
            (feature_count, coordinate_count, coordinate_count), dtype=torch.float64
        ),
    )
    directions = np.stack(
        (np.eye(coordinate_count)[0], np.eye(coordinate_count)[1], np.eye(coordinate_count)[2])
    )
    roles = np.asarray(["train", "heldout", "train"])
    designs, targets = _stage2_design_blocks(
        parent,
        jet,
        directions,
        roles,
        {
            "energy_scale": 0.1,
            "force_scale": 0.05,
            "absolute_hvp_scale": 0.1,
            "relative_loss_fraction": 0.5,
            "hvp_reference_floor": 0.1,
        },
    )

    assert [block.shape[0] for block in designs] == [1, 6, 12, 12]
    assert [block.shape[0] for block in targets] == [1, 6, 12, 12]


def test_stage2_full_hessian_capacity_design_uses_complete_matrix() -> None:
    atomic_numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64
    )
    coordinate_count = positions.numel()
    parent = CapacityParent(
        molecule_id="toy-full",
        natoms=2,
        atomic_numbers=atomic_numbers,
        positions=positions,
        topology=build_body_order_topology(atomic_numbers, positions),
        pbe_energy=1.0,
        pbe_force=np.ones((2, 3)),
        pbe_hessian=2.0 * np.eye(coordinate_count),
        source_energy=0.5,
        source_force=np.zeros((2, 3)),
        source_hessian_raw=np.eye(coordinate_count),
        source_hessian_symmetric=np.eye(coordinate_count),
    )
    feature_count = 4
    jet = (
        torch.ones(feature_count, dtype=torch.float64),
        torch.ones((feature_count, coordinate_count), dtype=torch.float64),
        torch.ones(
            (feature_count, coordinate_count, coordinate_count), dtype=torch.float64
        ),
    )
    directions = np.eye(coordinate_count)[:3]
    roles = np.asarray(["train", "heldout", "train"])
    designs, targets = _stage2_design_blocks(
        parent,
        jet,
        directions,
        roles,
        {
            "energy_scale": 0.1,
            "force_scale": 0.05,
            "absolute_hvp_scale": 0.1,
            "relative_loss_fraction": 0.5,
            "hvp_reference_floor": 0.1,
        },
        fit_full_hessian=True,
    )

    assert [block.shape[0] for block in designs] == [1, 6, 36, 36]
    assert [block.shape[0] for block in targets] == [1, 6, 36, 36]
