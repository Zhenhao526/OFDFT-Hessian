from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.qm9_complete_total_relaxed_q_local_scalar import (
    _design_blocks,
    _dual_normalized_ridge_grid,
    _load_jet_audit,
    _q_relative,
    _resolve_feature_chunk_size,
)


class _Parent:
    positions = torch.zeros((2, 3), dtype=torch.float64)
    pbe_energy = 2.0
    source_energy = 1.5
    pbe_force = np.ones((2, 3), dtype=np.float64)
    source_force = np.zeros((2, 3), dtype=np.float64)


def test_q_relative_uses_reference_floor() -> None:
    result = _q_relative(
        np.asarray([0.01, 0.2]), np.asarray([0.001, 2.0]), floor=0.1
    )
    assert np.allclose(result, [0.1, 0.1])


def test_design_blocks_use_train_q_only_and_scalar_hessian_contraction() -> None:
    features = torch.tensor([1.0, 2.0], dtype=torch.float64)
    jacobian = torch.ones((2, 6), dtype=torch.float64)
    hessian = torch.zeros((2, 6, 6), dtype=torch.float64)
    hessian[0] = torch.eye(6, dtype=torch.float64)
    hessian[1] = 2.0 * torch.eye(6, dtype=torch.float64)
    directions = np.stack(
        [np.eye(6)[0].reshape(2, 3), np.eye(6)[1].reshape(2, 3)]
    )
    q = {
        "role": np.asarray(["train", "heldout"]),
        "direction": directions,
        "pbe_q": np.asarray([0.3, 9.0]),
        "baseline_q": np.asarray([0.1, -9.0]),
    }
    design, target = _design_blocks(
        _Parent(),
        q,
        (features, jacobian, hessian),
        {
            "energy_scale_hartree": 0.1,
            "force_scale_hartree_per_bohr": 0.05,
            "q_absolute_scale_hartree_per_bohr2": 0.1,
            "q_relative_loss_fraction": 0.5,
            "q_reference_floor_hartree_per_bohr2": 0.1,
        },
    )
    assert [block.shape[0] for block in design] == [1, 6, 1, 1]
    assert [block.shape[0] for block in target] == [1, 6, 1, 1]
    ratio = design[2][0] / design[2][0, 0]
    assert torch.allclose(ratio, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_dual_normalized_ridge_matches_primal_solution() -> None:
    generator = torch.Generator().manual_seed(17)
    design = torch.randn((7, 13), generator=generator, dtype=torch.float64)
    target = torch.randn(7, generator=generator, dtype=torch.float64)
    ridge = 1.0e-4
    result, diagnostics = _dual_normalized_ridge_grid(
        design,
        target,
        ridges=[ridge],
        column_floor=1.0e-12,
        device=torch.device("cpu"),
    )[ridge]
    scale = torch.linalg.vector_norm(design, dim=0)
    normalized = design / scale
    expected_normalized = torch.linalg.solve(
        normalized.T @ normalized + ridge * torch.eye(13, dtype=torch.float64),
        normalized.T @ target,
    )
    expected = expected_normalized / scale
    assert torch.allclose(result, expected, atol=1e-10, rtol=1e-9)
    assert diagnostics["dual_gram_dimension"] == 7


def test_feature_chunk_override_only_reduces_default() -> None:
    protocol = {
        "feature_jet": {
            "feature_chunk_size": 32,
            "large_parent_min_natoms": 20,
            "large_parent_feature_chunk_size": 24,
        }
    }
    assert _resolve_feature_chunk_size(protocol, 19, None) == (32, False)
    assert _resolve_feature_chunk_size(protocol, 27, None) == (24, False)
    assert _resolve_feature_chunk_size(protocol, 27, 8) == (8, True)
    with pytest.raises(ValueError, match="no larger"):
        _resolve_feature_chunk_size(protocol, 27, 32)


def test_load_jet_audit_binds_protocol_and_parent_identity(tmp_path) -> None:
    path = tmp_path / "feature_jet_manifest.json"
    manifest = {
        "protocol_sha256": "protocol-hash",
        "test100_accessed": False,
        "parent_count": 2,
        "entries": [
            {"molecule_id": "0000001"},
            {"molecule_id": "0000002"},
        ],
    }
    path.write_text(json.dumps(manifest))
    parents = [
        SimpleNamespace(molecule_id="0000001"),
        SimpleNamespace(molecule_id="0000002"),
    ]
    loaded, entries = _load_jet_audit(path, "protocol-hash", parents)
    assert loaded == manifest
    assert set(entries) == {"0000001", "0000002"}

    manifest["test100_accessed"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="opened Test100"):
        _load_jet_audit(path, "protocol-hash", parents)
