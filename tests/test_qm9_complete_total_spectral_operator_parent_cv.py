from __future__ import annotations

import argparse
import hashlib

import numpy as np
import yaml

from scripts.qm9_complete_total_spectral_operator_parent_cv import (
    _cap_correction,
    _feature_normalization,
    _folds,
    _transform_parent_feature,
    _validate_protocol,
)


def _parent(molecule_id: str, atomic_numbers: list[int], scale: float):
    return argparse.Namespace(
        molecule_id=molecule_id,
        natoms=len(atomic_numbers),
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        spectral_scale=scale,
    )


def test_parent_folds_are_balanced_and_hold_every_parent_once() -> None:
    parents = [
        _parent(f"{index:07d}", [1] * (5 + index % 5) + [6, 7], 1.0 + index)
        for index in range(20)
    ]

    folds = _folds(parents, 5)

    assert [len(fold) for fold in folds] == [4, 4, 4, 4, 4]
    assert sorted(index for fold in folds for index in fold) == list(range(20))


def test_bounded_parent_feature_transforms_limit_unseen_z_scores() -> None:
    train = [
        _parent("0000001", [1, 1, 6], 1.0),
        _parent("0000002", [1, 6, 8, 8], 2.0),
        _parent("0000003", [1, 1, 1, 6, 7], 1.5),
    ]
    unseen = _parent("0000004", [9] * 20, 1000.0)
    normalization = _feature_normalization(train)

    standard = _transform_parent_feature(
        unseen, normalization, transform="standard", transform_scale=2.0
    )
    tanh = _transform_parent_feature(
        unseen, normalization, transform="tanh", transform_scale=2.0
    )
    clipped = _transform_parent_feature(
        unseen, normalization, transform="clip", transform_scale=2.0
    )
    constant = _transform_parent_feature(
        unseen, normalization, transform="constant", transform_scale=2.0
    )

    assert np.max(np.abs(standard[1:])) > 2.0
    assert np.max(np.abs(tanh[1:])) <= 1.0
    assert np.max(np.abs(clipped[1:])) <= 2.0
    assert np.count_nonzero(constant[1:]) == 0


def test_correction_cap_bounds_frobenius_relative_to_source() -> None:
    source = np.diag([2.0, 1.0])
    correction = 100.0 * np.ones((2, 2))

    capped, scale = _cap_correction(correction, source, 0.5)

    assert scale < 1.0
    assert np.isclose(np.linalg.norm(capped), 0.5 * np.linalg.norm(source))

    unchanged, scale = _cap_correction(correction, source, float("inf"))
    assert scale == 1.0
    assert np.array_equal(unchanged, correction)


def test_protocol_binds_variant_and_prohibits_external_parent_access(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    direction = tmp_path / "directions.json"
    baseline.write_text("baseline")
    direction.write_text("directions")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {
        "protocol_id": "unit",
        "scope": {
            "validation_parent_access_allowed": False,
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "fold_count": 5,
        },
        "frozen_inputs": {
            "baseline_manifest_sha256": sha(baseline),
            "direction_manifest_sha256": sha(direction),
        },
        "cross_validation": {"ridge_grid": [1e-4, 1.0]},
        "operator": {
            "source_hessian_spectral_polynomial_degree": 5,
            "source_hessian_spectral_rbf_centers": [-0.25, 0.0],
            "source_hessian_spectral_rbf_width": 0.5,
        },
        "preregistered_variants": [
            {
                "id": "bounded",
                "parent_feature_transform": "tanh",
                "parent_feature_transform_scale": 2.0,
                "block_parent_conditioning": True,
                "max_correction_to_source": 1.0,
            }
        ],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    args = argparse.Namespace(
        protocol=protocol_path,
        baseline_manifest=baseline,
        direction_manifest=direction,
        source_from_baseline=True,
        source_run_dir=None,
        fold_count=5,
        ridge_grid=[1e-4, 1.0],
        polynomial_degree=5,
        rbf_centers=[-0.25, 0.0],
        rbf_width=0.5,
        variant_id="bounded",
        parent_feature_transform="tanh",
        parent_feature_transform_scale=2.0,
        block_parent_conditioning=True,
        max_correction_to_source=1.0,
    )

    assert _validate_protocol(args)["protocol_id"] == "unit"

    args.max_correction_to_source = 2.0
    with np.testing.assert_raises(ValueError):
        _validate_protocol(args)
