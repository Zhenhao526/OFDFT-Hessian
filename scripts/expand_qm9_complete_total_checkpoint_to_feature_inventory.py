#!/usr/bin/env python3
"""Embed a trained scalar checkpoint into a larger descriptor-key inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stages(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError(f"manifest does not freeze Test100: {manifest_path}")
    stages = {}
    for row in manifest["stages"]:
        schema = Path(row["schema"])
        if _sha256(schema) != row["schema_sha256"]:
            raise ValueError(f"feature schema hash mismatch: {schema}")
        with np.load(schema) as payload:
            keys = np.asarray(payload["feature_keys"], dtype=np.int64)
            scales = np.asarray(payload["column_norms"], dtype=np.float64)
        if scales.shape != (keys.shape[0],):
            raise ValueError(f"feature scale shape mismatch: {schema}")
        stages[str(row["stage"])] = {"keys": keys, "scales": scales}
    if set(stages) != {"three_body", "four_body"}:
        raise ValueError("both three_body and four_body stages are required")
    return manifest, stages


def expand(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest, source_stages = _load_stages(args.source_schema_manifest)
    inventory_manifest, inventory_stages = _load_stages(args.feature_inventory_manifest)
    inventory_sha = _sha256(args.feature_inventory_manifest)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("test100_accessed") is not False:
        raise ValueError("source checkpoint does not freeze Test100")
    source_state = source["state_dict"]
    source_linear = source_state["linear"].detach().cpu()
    source_weight = source_state["weight"].detach().cpu()
    if source_weight.shape[1] != source_linear.numel():
        raise ValueError("source scalar state has inconsistent feature dimensions")

    source_total = sum(stage["keys"].shape[0] for stage in source_stages.values())
    target_total = sum(stage["keys"].shape[0] for stage in inventory_stages.values())
    if source_linear.numel() != source_total:
        raise ValueError("source checkpoint dimension does not match source schema")

    target_linear = torch.zeros(target_total, dtype=source_linear.dtype)
    target_weight = torch.zeros(
        (source_weight.shape[0], target_total), dtype=source_weight.dtype
    )
    source_offset = 0
    target_offset = 0
    stage_records = []
    for stage_name in ("three_body", "four_body"):
        source_stage = source_stages[stage_name]
        target_stage = inventory_stages[stage_name]
        target_lookup = {
            tuple(int(value) for value in key): index
            for index, key in enumerate(target_stage["keys"])
        }
        target_indices = []
        missing = []
        for key in source_stage["keys"]:
            key_tuple = tuple(int(value) for value in key)
            index = target_lookup.get(key_tuple)
            if index is None:
                missing.append(key_tuple)
            else:
                target_indices.append(index)
        if missing:
            raise ValueError(
                f"target inventory misses {len(missing)} {stage_name} source keys"
            )
        source_count = source_stage["keys"].shape[0]
        target_index = torch.as_tensor(
            np.asarray(target_indices) + target_offset, dtype=torch.int64
        )
        source_slice = slice(source_offset, source_offset + source_count)
        ratio = torch.as_tensor(
            target_stage["scales"][target_indices] / source_stage["scales"],
            dtype=source_linear.dtype,
        )
        if not bool(torch.all(torch.isfinite(ratio))) or bool(torch.any(ratio <= 0)):
            raise ValueError(f"invalid {stage_name} target/source feature-scale ratio")
        target_linear[target_index] = source_linear[source_slice] * ratio
        target_weight[:, target_index] = source_weight[:, source_slice] * ratio[None, :]
        stage_records.append(
            {
                "stage": stage_name,
                "source_feature_count": source_count,
                "target_feature_count": target_stage["keys"].shape[0],
                "mapped_feature_count": len(target_indices),
                "new_zero_feature_count": target_stage["keys"].shape[0] - len(target_indices),
                "scale_ratio_min": float(torch.min(ratio)),
                "scale_ratio_max": float(torch.max(ratio)),
            }
        )
        source_offset += source_count
        target_offset += target_stage["keys"].shape[0]

    state = {
        name: value.detach().cpu().clone() for name, value in source_state.items()
    }
    state["linear"] = target_linear
    state["weight"] = target_weight
    config = dict(source.get("config", {}))
    config.update(
        feature_inventory_manifest=args.feature_inventory_manifest.resolve().as_posix(),
        column_norm_relative_cutoff=0.0,
        feature_scale_mode=inventory_manifest["feature_scale_mode"],
        feature_scale_floor=float(inventory_manifest["feature_scale_floor"]),
        zero_linear_initialization=True,
    )
    payload = {
        "step": 0,
        "state_dict": state,
        "config": config,
        "test100_accessed": False,
        "feature_inventory_manifest_sha256": inventory_sha,
        "initialization_definition": (
            "stable scalar checkpoint embedded by exact descriptor key; weights are "
            "rescaled to preserve the source physical scalar and new inventory columns are zero"
        ),
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_checkpoint_step": int(source["step"]),
        "source_schema_manifest": args.source_schema_manifest.resolve().as_posix(),
        "source_schema_manifest_sha256": _sha256(args.source_schema_manifest),
        "feature_inventory_manifest": args.feature_inventory_manifest.resolve().as_posix(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "initial.ckpt"
    torch.save(payload, checkpoint)
    result = {
        "definition": payload["initialization_definition"],
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_checkpoint_step": payload["source_checkpoint_step"],
        "source_schema_manifest": payload["source_schema_manifest"],
        "source_schema_manifest_sha256": payload["source_schema_manifest_sha256"],
        "feature_inventory_manifest": payload["feature_inventory_manifest"],
        "feature_inventory_manifest_sha256": inventory_sha,
        "output_checkpoint": checkpoint.resolve().as_posix(),
        "output_checkpoint_sha256": _sha256(checkpoint),
        "source_feature_count": source_total,
        "target_feature_count": target_total,
        "hidden_size": int(source_weight.shape[0]),
        "stages": stage_records,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest = args.output_dir / "expanded_checkpoint_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-schema-manifest", type=Path, required=True)
    parser.add_argument("--feature-inventory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    expand(parse_args())
