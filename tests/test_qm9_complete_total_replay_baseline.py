from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qm9_complete_total_replay_baseline import (
    _is_completed,
    _select_task_rows,
)


def test_task_selection_is_disjoint_and_complete() -> None:
    rows = [{"molecule_id": f"{index:07d}"} for index in range(11)]
    shards = [
        _select_task_rows(
            rows, task_index=None, shard_index=index, shard_count=3
        )
        for index in range(3)
    ]
    indices = [task for shard in shards for task, _ in shard]
    assert sorted(indices) == list(range(11))
    assert len(indices) == len(set(indices))

    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_task_rows(
            rows, task_index=0, shard_index=0, shard_count=3
        )


def test_resume_requires_success_frozen_test_and_matching_task_hash(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "success": True,
                "test100_accessed": False,
                "task_sha256": "expected",
            }
        )
    )
    assert _is_completed(tmp_path, "expected")
    assert not _is_completed(tmp_path, "different")
    payload = json.loads(summary.read_text())
    payload["test100_accessed"] = True
    summary.write_text(json.dumps(payload))
    assert not _is_completed(tmp_path, "expected")

