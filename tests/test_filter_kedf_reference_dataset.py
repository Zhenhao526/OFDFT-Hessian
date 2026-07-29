from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.filter_kedf_reference_dataset import filter_dataset


def write_source(root: Path, target_kedf: str = "lkt") -> tuple[Path, Path]:
    frames = root / "frames.jsonl"
    rows = [
        {"phase": "solid", "frame": 0},
        {"phase": "liquid", "frame": 1},
        {"phase": "liquid", "frame": 2},
        {"phase": "liquid", "frame": 3},
    ]
    frames.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"target_kedf": target_kedf}) + "\n",
        encoding="utf-8",
    )
    return frames, manifest


def test_filters_phase_and_records_provenance(tmp_path: Path):
    frames, manifest = write_source(tmp_path)
    output = tmp_path / "liquid"

    result = filter_dataset(
        frames,
        manifest,
        output,
        target_kedf="lkt",
        phase="liquid",
    )

    rows = [
        json.loads(line)
        for line in (output / "frames.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "verified"
    assert result["source_indices"] == [1, 2, 3]
    assert {row["phase"] for row in rows} == {"liquid"}
    assert result["output_frames"]["sha256"] == hashlib.sha256(
        (output / "frames.jsonl").read_bytes()
    ).hexdigest()


def test_rejects_target_kedf_mismatch(tmp_path: Path):
    frames, manifest = write_source(tmp_path, target_kedf="xwm")

    with pytest.raises(ValueError, match="target_kedf mismatch"):
        filter_dataset(
            frames,
            manifest,
            tmp_path / "liquid",
            target_kedf="lkt",
            phase="liquid",
        )
