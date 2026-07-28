#!/usr/bin/env python3
"""Freeze the normalized active descriptor columns used by a geometry-residual checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scripts.qm9_complete_total_geometry_mlp_capacity import _active_stage
except ModuleNotFoundError:
    from qm9_complete_total_geometry_mlp_capacity import _active_stage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_key_container(stage_root: Path) -> tuple[Path, str | None]:
    feature_keys = stage_root / "feature_keys.npy"
    if feature_keys.is_file():
        return feature_keys, None
    shared = stage_root / "shared_coefficients.npz"
    if not shared.is_file():
        raise FileNotFoundError(f"no feature-key container under {stage_root}")
    with np.load(shared) as payload:
        if "feature_keys" not in payload:
            raise KeyError(f"shared coefficient archive has no feature_keys: {shared}")
    return shared, "feature_keys"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    active_total = 0
    for stage_name in ("three_body", "four_body"):
        stage = _active_stage(
            args.design_root,
            stage_name,
            None,
            args.column_norm_relative_cutoff,
            args.norm_chunk_rows,
            True,
            args.feature_scale_mode,
            args.feature_scale_floor,
        )
        output = args.output_dir / f"{stage_name}_active_schema.npz"
        np.savez_compressed(
            output,
            feature_keys=np.asarray(stage["keys"], dtype=np.int64),
            column_norms=np.asarray(stage["norms"], dtype=np.float64),
            source_column_norms=np.asarray(
                stage["source_column_norms"], dtype=np.float64
            ),
            total_feature_count=np.asarray(stage["total_count"], dtype=np.int64),
            active_feature_count=np.asarray(stage["active_count"], dtype=np.int64),
            column_norm_threshold=np.asarray(stage["threshold"], dtype=np.float64),
            column_norm_relative_cutoff=np.asarray(
                args.column_norm_relative_cutoff, dtype=np.float64
            ),
            feature_scale_floor=np.asarray(args.feature_scale_floor, dtype=np.float64),
        )
        feature_keys, feature_keys_dataset = _feature_key_container(
            args.design_root / stage_name
        )
        summary = args.design_root / stage_name / "summary.json"
        record = {
            "stage": stage_name,
            "total_feature_count": int(stage["total_count"]),
            "active_feature_count": int(stage["active_count"]),
            "column_norm_threshold": float(stage["threshold"]),
            "feature_scale_mode": stage["feature_scale_mode"],
            "feature_scale_floor": stage["feature_scale_floor"],
            "schema": output.resolve().as_posix(),
            "schema_sha256": _sha256(output),
            "source_feature_keys": feature_keys.resolve().as_posix(),
            "source_feature_keys_dataset": feature_keys_dataset,
            "source_feature_keys_sha256": _sha256(feature_keys),
            "source_stage_summary": summary.resolve().as_posix(),
            "source_stage_summary_sha256": _sha256(summary),
        }
        records.append(record)
        active_total += int(stage["active_count"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("geometry-residual checkpoint does not freeze Test100")
    checkpoint_feature_count = int(checkpoint["state_dict"]["linear"].numel())
    if checkpoint_feature_count != active_total:
        raise ValueError(
            f"checkpoint has {checkpoint_feature_count} features, schema has {active_total}"
        )
    design_summary = args.design_root / "summary.json"
    result = {
        "definition": (
            "Frozen active normalized scalar descriptor columns for train800 E/F replay "
            "and independent-parent evaluation."
        ),
        "design_root": args.design_root.resolve().as_posix(),
        "design_summary": design_summary.resolve().as_posix(),
        "design_summary_sha256": _sha256(design_summary),
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "column_norm_relative_cutoff": args.column_norm_relative_cutoff,
        "feature_scale_mode": args.feature_scale_mode,
        "feature_scale_floor": args.feature_scale_floor,
        "active_feature_count": active_total,
        "stages": records,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "wall_time_s": time.perf_counter() - started,
    }
    output = args.output_dir / "active_feature_schema_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column-norm-relative-cutoff", type=float, default=1.0e-8)
    parser.add_argument("--norm-chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--feature-scale-mode",
        choices=("column_norm", "floored_column_norm", "unit"),
        default="column_norm",
    )
    parser.add_argument("--feature-scale-floor", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
