from __future__ import annotations

import argparse

import numpy as np
import yaml

from scripts.qm9_complete_total_spectral_operator_external_eval import (
    _parent_features,
    _validate_protocol,
)


def test_parent_features_apply_frozen_training_normalization() -> None:
    atomic_numbers = np.asarray([1, 1, 6, 7, 8, 9], dtype=np.int64)
    raw = np.asarray(
        [
            6 / 20,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            np.mean(atomic_numbers) / 10,
            np.log(2.0),
        ]
    )
    mean = raw - np.arange(raw.size, dtype=np.float64)
    scale = np.arange(1, raw.size + 1, dtype=np.float64)

    features = _parent_features(
        atomic_numbers,
        2.0,
        {"mean": mean.tolist(), "scale": scale.tolist()},
    )

    assert np.array_equal(features[:1], np.ones(1))
    assert np.allclose(features[1:], np.arange(raw.size) / scale)


def test_parent_features_reject_invalid_frozen_scale() -> None:
    atomic_numbers = np.asarray([1, 6], dtype=np.int64)
    normalization = {"mean": [0.0] * 7, "scale": [1.0] * 6 + [0.0]}

    with np.testing.assert_raises(ValueError):
        _parent_features(atomic_numbers, 1.0, normalization)


def test_protocol_validation_binds_unseen_parents_and_artifact_hashes(tmp_path) -> None:
    protocol = {
        "protocol_id": "unit",
        "scope": {
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "validation_parent_ids": ["0000001", "0000002"],
        },
        "frozen_inputs": {
            "validation_baseline_manifest_sha256": "baseline",
            "h512_source_checkpoint_sha256": "source-a",
            "replay_source_checkpoint_sha256": "source-b",
            "stage2_candidate_summary_sha256": "candidate-a",
            "replay_source_candidate_summary_sha256": "candidate-b",
            "shared_coefficients_sha256": "coeff-a",
            "replay_source_shared_coefficients_sha256": "coeff-b",
        },
        "evaluation": {
            "exact_parent_hvp_interpolant_allowed": False,
            "per_parent_fitting_allowed": False,
            "validation_hvp_labels_allowed": False,
            "polynomial_degree": 5,
            "rbf_centers": [-0.25, 0.0, 0.25],
            "rbf_width": 0.5,
        },
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    args = argparse.Namespace(
        protocol=path,
        polynomial_degree=5,
        rbf_centers=[-0.25, 0.0, 0.25],
        rbf_width=0.5,
    )

    loaded = _validate_protocol(
        args,
        ["0000001", "0000002"],
        "baseline",
        "source-b",
        "candidate-b",
        "coeff-b",
    )
    assert loaded["protocol_id"] == "unit"

    with np.testing.assert_raises(ValueError):
        _validate_protocol(
            args,
            ["0000002", "0000001"],
            "baseline",
            "source-b",
            "candidate-b",
            "coeff-b",
        )
