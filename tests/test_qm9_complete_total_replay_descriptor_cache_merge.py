import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.qm9_complete_total_replay_descriptor_cache import _task_hash
from scripts.qm9_complete_total_replay_descriptor_cache_merge import merge


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    baseline = tmp_path / "baseline.npz"
    np.savez(baseline, value=np.asarray(1.0))
    task_csv = tmp_path / "tasks.csv"
    with task_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "molecule_id",
                "sample_id",
                "baseline_array",
                "baseline_array_sha256",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "molecule_id": "0000001",
                "sample_id": 0,
                "baseline_array": baseline.as_posix(),
                "baseline_array_sha256": _sha256(baseline),
            }
        )
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "active_feature_count": 3,
                "checkpoint_sha256": "checkpoint",
                "test100_accessed": False,
            }
        )
    )
    schema_sha = _sha256(schema)
    task_csv_sha = _sha256(task_csv)
    settings = {"example": 1}
    row = next(csv.DictReader(task_csv.open()))
    cache_root = tmp_path / "cache"
    task_dir = cache_root / "task_0000_0000001_0000000"
    task_dir.mkdir(parents=True)
    cache_path = task_dir / "descriptor_cache.npz"
    np.savez(
        cache_path,
        descriptor=np.asarray([1.0, 2.0, 3.0]),
        descriptor_jacobian=np.ones((6, 3)),
        energy_target=np.asarray(0.25),
        force_target=np.ones((2, 3)),
    )
    summary = {
        "success": True,
        "test100_accessed": False,
        "task_csv_sha256": task_csv_sha,
        "schema_manifest_sha256": schema_sha,
        "descriptor_settings": settings,
        "task_sha256": _task_hash(row, schema_sha, settings),
        "baseline_array_sha256": row["baseline_array_sha256"],
        "descriptor_cache": cache_path.as_posix(),
        "descriptor_cache_sha256": _sha256(cache_path),
        "natoms": 2,
        "coordinate_count": 6,
        "active_feature_count": 3,
        "energy_target_hartree": 0.25,
        "force_target_mae_hartree_per_bohr": 1.0,
        "wall_time_s": 2.0,
        "max_rss_mb": 10.0,
        "peak_gpu_memory_mb": 0.0,
        "three_body_matched_active_feature_fraction": 1.0,
        "four_body_matched_active_feature_fraction": 0.5,
    }
    (task_dir / "summary.json").write_text(json.dumps(summary))
    (cache_root / "shard_0000_summary.json").write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "task_csv_sha256": task_csv_sha,
                "schema_manifest_sha256": schema_sha,
                "shard_index": 0,
                "shard_count": 1,
                "descriptor_settings": settings,
            }
        )
    )
    args = argparse.Namespace(
        task_csv=task_csv,
        schema_manifest=schema,
        cache_root=cache_root,
        output_dir=tmp_path / "merged",
        shard_count=1,
        require_complete=True,
    )
    return args, cache_path


def test_merge_accepts_complete_hash_bound_cache(tmp_path):
    args, _ = _fixture(tmp_path)
    result = merge(args)
    assert result["complete"] is True
    assert result["counts"]["successful_tasks"] == 1
    assert result["metrics"]["four_body_matched_active_feature_fraction"]["median"] == 0.5
    assert result["test100_evaluations_used"] == 0


def test_merge_rejects_cache_changed_after_summary(tmp_path):
    args, cache_path = _fixture(tmp_path)
    with cache_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="audit failed"):
        merge(args)
