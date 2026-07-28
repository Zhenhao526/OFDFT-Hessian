from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from scripts.expand_qm9_complete_total_checkpoint_to_feature_inventory import expand


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path, name, rows):
    stages = []
    for stage, keys, scales in rows:
        path = tmp_path / f"{name}_{stage}.npz"
        np.savez(path, feature_keys=np.asarray(keys), column_norms=np.asarray(scales))
        stages.append({"stage": stage, "schema": str(path), "schema_sha256": _sha(path)})
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "stages": stages,
                "feature_scale_mode": "floored_column_norm",
                "feature_scale_floor": 1.0,
                "test100_accessed": False,
            }
        )
    )
    return path


def test_expand_preserves_scalar_under_feature_reordering_and_rescaling(tmp_path):
    source_schema = _manifest(
        tmp_path,
        "source",
        [("three_body", [[1, 2]], [2.0]), ("four_body", [[3, 4]], [4.0])],
    )
    target_inventory = _manifest(
        tmp_path,
        "target",
        [
            ("three_body", [[0, 0], [1, 2]], [1.0, 6.0]),
            ("four_body", [[3, 4], [9, 9]], [8.0, 1.0]),
        ],
    )
    source_checkpoint = tmp_path / "source.ckpt"
    torch.save(
        {
            "step": 7,
            "test100_accessed": False,
            "config": {},
            "state_dict": {
                "linear": torch.tensor([2.0, 3.0]),
                "weight": torch.tensor([[5.0, 7.0]]),
                "bias": torch.tensor([0.25]),
                "output": torch.tensor([11.0]),
            },
        },
        source_checkpoint,
    )
    result = expand(
        argparse.Namespace(
            source_checkpoint=source_checkpoint,
            source_schema_manifest=source_schema,
            feature_inventory_manifest=target_inventory,
            output_dir=tmp_path / "output",
        )
    )
    expanded = torch.load(result["output_checkpoint"], weights_only=False)
    state = expanded["state_dict"]
    source_phi = torch.tensor([10.0, 20.0])
    target_phi = torch.tensor([0.0, 10.0, 20.0, 0.0])
    source_x = source_phi / torch.tensor([2.0, 4.0])
    target_x = target_phi / torch.tensor([1.0, 6.0, 8.0, 1.0])
    source_linear = torch.dot(torch.tensor([2.0, 3.0]), source_x)
    target_linear = torch.dot(state["linear"], target_x)
    source_hidden = torch.dot(torch.tensor([5.0, 7.0]), source_x)
    target_hidden = torch.dot(state["weight"][0], target_x)
    torch.testing.assert_close(target_linear, source_linear)
    torch.testing.assert_close(target_hidden, source_hidden)
    assert state["linear"][0] == 0.0 and state["linear"][3] == 0.0
    assert result["source_feature_count"] == 2
    assert result["target_feature_count"] == 4
