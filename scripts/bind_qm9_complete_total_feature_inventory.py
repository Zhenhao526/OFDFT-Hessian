#!/usr/bin/env python3
"""Bind a train-only feature inventory to its exact scalar checkpoint dimensions."""

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


def bind(args: argparse.Namespace) -> dict[str, Any]:
    inventory_sha = _sha256(args.inventory_manifest)
    inventory = json.loads(args.inventory_manifest.read_text())
    if inventory.get("test100_accessed") is not False:
        raise ValueError("feature inventory does not freeze Test100")
    stages = []
    feature_count = 0
    for row in inventory["stages"]:
        schema = Path(row["schema"])
        if _sha256(schema) != row["schema_sha256"]:
            raise ValueError(f"feature inventory schema hash mismatch: {schema}")
        with np.load(schema) as payload:
            local_count = int(np.asarray(payload["feature_keys"]).shape[0])
            if np.asarray(payload["column_norms"]).shape != (local_count,):
                raise ValueError(f"feature scale shape mismatch: {schema}")
        stages.append(dict(row))
        feature_count += local_count
    if feature_count != int(inventory["active_feature_count"]):
        raise ValueError("inventory feature count mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("checkpoint does not freeze Test100")
    if checkpoint.get("feature_inventory_manifest_sha256") != inventory_sha:
        raise ValueError("checkpoint feature inventory hash mismatch")
    state = checkpoint["state_dict"]
    if int(state["linear"].numel()) != feature_count:
        raise ValueError("checkpoint linear feature dimension mismatch")
    if int(state["weight"].shape[1]) != feature_count:
        raise ValueError("checkpoint nonlinear feature dimension mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "definition": "checkpoint-bound train800-union scalar descriptor schema",
        "inventory_manifest": args.inventory_manifest.resolve().as_posix(),
        "inventory_manifest_sha256": inventory_sha,
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "active_feature_count": feature_count,
        "feature_scale_mode": inventory["feature_scale_mode"],
        "feature_scale_floor": inventory["feature_scale_floor"],
        "descriptor_settings": inventory["descriptor_settings"],
        "stages": stages,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output = args.output_dir / "active_feature_schema_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    bind(parse_args())
