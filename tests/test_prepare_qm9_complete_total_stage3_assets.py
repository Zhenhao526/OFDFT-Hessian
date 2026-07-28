from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_qm9_complete_total_stage3_assets import (
    _latest_baseline_rows,
    _parent_ids_from_difficulty,
    _replay_rows,
    _validation_capacity_manifest,
)


def test_replay_rows_require_all_grouped_samples(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    for sample_id in range(4):
        (labels / f"0000001.{sample_id:07d}.zarr.zip").write_bytes(
            f"sample-{sample_id}".encode()
        )
    rows = _replay_rows(tmp_path, ["0000001"], {0, 1, 2, 3})
    assert [row["sample_id"] for row in rows] == [0, 1, 2, 3]
    assert len({row["label_sha256"] for row in rows}) == 4

    (labels / "0000001.0000003.zarr.zip").unlink()
    with pytest.raises(ValueError, match="replay samples"):
        _replay_rows(tmp_path, ["0000001"], {0, 1, 2, 3})


def test_train_parent_inventory_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "difficulty.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parent_id", "natoms"])
        writer.writeheader()
        writer.writerow({"parent_id": "0000001", "natoms": 2})
        writer.writerow({"parent_id": "0000001", "natoms": 2})
    with pytest.raises(ValueError, match="duplicate parent IDs"):
        _parent_ids_from_difficulty(path)


def test_empty_baseline_run_root_is_not_an_error(tmp_path: Path) -> None:
    candidate_manifest = {"candidates": [{"molecule_id": "0000001"}]}
    assert _latest_baseline_rows(candidate_manifest, tmp_path) == {}


def test_validation_capacity_manifest_freezes_test100(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    stability = tmp_path / "validation_stability.json"
    protocol.write_text("protocol_id: test\n")
    stability.write_text('{"test100_accessed": false}\n')
    rows = [
        {
            "molecule_id": "0000001",
            "label_path": "/data/0000001.0000000.zarr.zip",
            "pbe_hessian_path": "/data/0000001.npz",
        }
    ]

    manifest = _validation_capacity_manifest(
        rows,
        protocol=protocol,
        validation_stability_summary=stability,
    )

    assert manifest["test100_accessed"] is False
    assert manifest["test100_evaluations_used"] == 0
    assert manifest["parent_count"] == 1
    assert manifest["parents"] == rows
