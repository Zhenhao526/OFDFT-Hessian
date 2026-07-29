#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_dataset(
    source_frames: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    target_kedf: str,
    phase: str,
) -> Dict[str, Any]:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if str(manifest.get("target_kedf", "")).lower() != target_kedf:
        raise ValueError("source manifest target_kedf mismatch")
    rows = [
        json.loads(line)
        for line in source_frames.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("phase", "")).lower() == phase
    ]
    if len(selected) < 3:
        raise ValueError(f"source dataset has fewer than three {phase} frames")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")

    output_dir.mkdir(parents=True)
    output_frames = output_dir / "frames.jsonl"
    output_frames.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for _, row in selected),
        encoding="utf-8",
    )
    payload = {
        "schema": "kedf-phase-filtered-reference-dataset-v1",
        "status": "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_kedf": target_kedf,
        "phase": phase,
        "frames": len(selected),
        "source_indices": [index for index, _ in selected],
        "source": {
            "frames": {
                "path": str(source_frames.resolve()),
                "sha256": sha256(source_frames),
            },
            "manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": sha256(source_manifest),
            },
        },
        "output_frames": {
            "path": str(output_frames.resolve()),
            "sha256": sha256(output_frames),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a provenance-preserving phase subset of a KEDF dataset"
    )
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-kedf", choices=("xwm", "lkt"), required=True)
    parser.add_argument("--phase", choices=("solid", "liquid"), required=True)
    args = parser.parse_args()

    result = filter_dataset(
        args.source_frames,
        args.source_manifest,
        args.out,
        target_kedf=args.target_kedf,
        phase=args.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
