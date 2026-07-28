from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from scripts import prepare_qm9_train800_rebuild_assets as rebuild


def test_finalize_registers_the_written_train_only_split(
    monkeypatch, tmp_path: Path
) -> None:
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("frozen-source\n")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "dataset_info.yaml").write_text("name: test\n")
    statistics = dataset_root / rebuild.STATISTICS
    statistics.mkdir(parents=True)
    (statistics / ".zgroup").write_text("{}\n")

    rows = [
        {
            "label_path": f"/source/0000001.{sample_id:07d}.zarr.zip",
            "label_sha256": str(sample_id) * 64,
            "molecule_id": "0000001",
            "sample_id": str(sample_id),
        }
        for sample_id in range(4)
    ]
    entries = [
        ("TrainOnly", f"0000001.{sample_id:07d}.zarr.zip", 3)
        for sample_id in range(4)
    ]
    label_records = [
        {
            "molecule_id": "0000001",
            "sample_id": str(sample_id),
            "filename": entry[1],
            "source_label_sha256": str(sample_id) * 64,
            "new_label_sha256": str(sample_id + 1) * 64,
            "source_hash_matches": "false",
            "transformed_label_sha256": str(sample_id + 2) * 64,
            "n_scf_steps": "3",
        }
        for sample_id, entry in enumerate(entries)
    ]
    monkeypatch.setattr(rebuild, "_rows", lambda _: rows)
    monkeypatch.setattr(
        rebuild,
        "_build_entries",
        lambda *args, **kwargs: (
            entries,
            label_records,
            "a" * 64,
            "b" * 64,
        ),
    )

    rebuild.finalize(
        argparse.Namespace(
            source_csv=source_csv,
            dataset_root=dataset_root,
            dataset_name="TrainOnly",
            label_hash_policy="regenerated",
        )
    )

    split = pickle.loads((dataset_root / "split.pkl").read_bytes())
    manifest = json.loads(
        (
            dataset_root / "provenance" / "train_only_dataset_manifest.json"
        ).read_text()
    )
    assert split["train"] == entries
    assert split["val"] == []
    assert split["test"] == []
    assert split["sizes"] == {"train": 8, "val": 0, "test": 0}
    assert manifest["train_entries"] == 4
    assert manifest["validation_entries"] == 0
    assert manifest["test_entries"] == 0
    assert manifest["usable_training_samples"] == 8
    assert manifest["validation_accessed"] is False
    assert manifest["test_accessed"] is False


def test_training_sample_budget_is_deterministic_and_explicit() -> None:
    entries = [
        ("TrainOnly", f"000000{i}.0000000.zarr.zip", 5)
        for i in range(1, 5)
    ]
    adjusted, available, trimmed = rebuild._apply_training_sample_budget(
        entries,
        expected_samples=14,
        seed="frozen-seed",
    )
    repeated, repeated_available, repeated_trimmed = (
        rebuild._apply_training_sample_budget(
            entries,
            expected_samples=14,
            seed="frozen-seed",
        )
    )
    assert available == repeated_available == 16
    assert adjusted == repeated
    assert trimmed == repeated_trimmed
    assert sum(entry[2] - 1 for entry in adjusted) == 14
    assert len(trimmed) == 2
    assert all(
        row["physical_n_scf_steps"] - row["registered_n_scf_steps"] == 1
        for row in trimmed
    )
