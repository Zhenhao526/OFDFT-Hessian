import argparse
import csv
import hashlib
import json

import pytest
import torch

from scripts.qm9_complete_total_stage3_training_preflight import prepare


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    payload = {
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **payload,
    }
    path.write_text(json.dumps(payload))
    return path


def _fixture(tmp_path):
    split_hash = "split"
    baseline = _write_json(
        tmp_path / "baseline.json",
        {
            "source_split_sha256": split_hash,
            "parents": [{"molecule_id": "train"}],
        },
    )
    directions = _write_json(
        tmp_path / "directions.json",
        {"candidates": [{"molecule_id": "train"}]},
    )
    validation = _write_json(
        tmp_path / "validation.json",
        {
            "source_split_sha256": split_hash,
            "complete": True,
            "expected_parent_count": 1,
            "parents": [{"molecule_id": "validation"}],
        },
    )
    success_csv = tmp_path / "replay.csv"
    with success_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("molecule_id", "sample_id"))
        writer.writeheader()
        writer.writerow({"molecule_id": "train", "sample_id": 0})
    checkpoint = tmp_path / "model.ckpt"
    inventory_sha = "inventory"
    torch.save(
        {
            "step": 3,
            "test100_accessed": False,
            "feature_inventory_manifest_sha256": inventory_sha,
            "state_dict": {
                "linear": torch.zeros(2),
                "weight": torch.zeros((3, 2)),
            },
        },
        checkpoint,
    )
    schema = _write_json(
        tmp_path / "schema.json",
        {
            "active_feature_count": 2,
            "inventory_manifest_sha256": inventory_sha,
        },
    )
    replay = _write_json(
        tmp_path / "replay.json",
        {
            "complete": True,
            "schema_checkpoint_sha256": _sha(checkpoint),
            "schema_manifest": schema.as_posix(),
            "schema_manifest_sha256": _sha(schema),
            "active_feature_count": 2,
            "counts": {"expected_tasks": 1},
            "artifacts": {
                "success_csv": {
                    "path": success_csv.as_posix(),
                    "sha256": _sha(success_csv),
                }
            },
        },
    )
    return argparse.Namespace(
        baseline_manifest=baseline,
        direction_manifest=directions,
        checkpoint=checkpoint,
        replay_cache_manifest=replay,
        validation_baseline_manifest=validation,
        output_dir=tmp_path / "output",
        expected_replay_parent_count=1,
    )


def test_preflight_accepts_disjoint_hash_bound_inputs(tmp_path):
    result = prepare(_fixture(tmp_path))
    assert result["ready"] is True
    assert result["parent_group_disjoint"] is True
    assert result["test100_evaluations_used"] == 0


def test_preflight_rejects_validation_replay_overlap(tmp_path):
    args = _fixture(tmp_path)
    replay_csv = tmp_path / "replay.csv"
    replay_csv.write_text("molecule_id,sample_id\ntrain,0\nvalidation,0\n")
    replay = json.loads(args.replay_cache_manifest.read_text())
    replay["artifacts"]["success_csv"]["sha256"] = _sha(replay_csv)
    replay["counts"]["expected_tasks"] = 2
    args.replay_cache_manifest.write_text(json.dumps(replay))
    args.expected_replay_parent_count = 2
    with pytest.raises(ValueError, match="replay and validation parents overlap"):
        prepare(args)
