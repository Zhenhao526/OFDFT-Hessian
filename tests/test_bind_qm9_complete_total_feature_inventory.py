import argparse
import hashlib
import json

import numpy as np
import torch

from scripts.bind_qm9_complete_total_feature_inventory import bind


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_binds_inventory_to_exact_checkpoint_dimensions(tmp_path):
    stages = []
    for name, keys in (
        ("three_body", np.asarray([[1, 2], [2, 3]], dtype=np.int64)),
        ("four_body", np.asarray([[1, 2, 3]], dtype=np.int64)),
    ):
        schema = tmp_path / f"{name}.npz"
        np.savez(schema, feature_keys=keys, column_norms=np.ones(len(keys)))
        stages.append(
            {
                "stage": name,
                "schema": schema.as_posix(),
                "schema_sha256": _sha256(schema),
            }
        )
    inventory_path = tmp_path / "inventory.json"
    inventory = {
        "active_feature_count": 3,
        "feature_scale_mode": "floored_column_norm",
        "feature_scale_floor": 1.0,
        "descriptor_settings": {},
        "stages": stages,
        "test100_accessed": False,
    }
    inventory_path.write_text(json.dumps(inventory))
    checkpoint_path = tmp_path / "seed.ckpt"
    torch.save(
        {
            "step": 0,
            "test100_accessed": False,
            "feature_inventory_manifest_sha256": _sha256(inventory_path),
            "state_dict": {
                "linear": torch.zeros(3),
                "weight": torch.zeros((2, 3)),
            },
        },
        checkpoint_path,
    )
    args = argparse.Namespace(
        inventory_manifest=inventory_path,
        checkpoint=checkpoint_path,
        output_dir=tmp_path / "bound",
    )

    result = bind(args)

    assert result["active_feature_count"] == 3
    assert result["test100_accessed"] is False
    assert result["checkpoint_sha256"] == _sha256(checkpoint_path)
