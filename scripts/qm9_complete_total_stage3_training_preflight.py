#!/usr/bin/env python3
"""Freeze and audit Stage-3 replay/HVP training inputs before submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("test100_accessed") is not False:
        raise ValueError(f"{label} does not freeze Test100")
    if int(payload.get("test100_evaluations_used", 0)) != 0:
        raise ValueError(f"{label} records nonzero Test100 evaluations")
    return payload


def _artifact_csv(payload: dict[str, Any], name: str) -> tuple[Path, list[dict[str, str]]]:
    artifact = payload["artifacts"][name]
    path = Path(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return path, rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _json(args.baseline_manifest, "training baseline manifest")
    directions = _json(args.direction_manifest, "training direction manifest")
    replay = _json(args.replay_cache_manifest, "replay cache manifest")
    validation = _json(args.validation_baseline_manifest, "validation baseline manifest")
    if replay.get("complete") is not True:
        raise ValueError("replay descriptor cache is incomplete")
    if validation.get("complete") is not True:
        raise ValueError("validation complete-total baseline set is incomplete")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test100_accessed") is not False:
        raise ValueError("source checkpoint does not freeze Test100")
    checkpoint_sha = _sha256(args.checkpoint)
    schema_path = Path(replay["schema_manifest"])
    if _sha256(schema_path) != replay["schema_manifest_sha256"]:
        raise ValueError("replay active-feature schema hash mismatch")
    schema = _json(schema_path, "replay active-feature schema")
    if int(replay["active_feature_count"]) != int(schema["active_feature_count"]):
        raise ValueError("replay and schema active feature counts differ")
    state = checkpoint["state_dict"]
    feature_count = int(schema["active_feature_count"])
    if int(state["linear"].numel()) != feature_count:
        raise ValueError("checkpoint linear dimension differs from replay schema")
    if int(state["weight"].shape[1]) != feature_count:
        raise ValueError("checkpoint nonlinear dimension differs from replay schema")
    if (
        checkpoint.get("feature_inventory_manifest_sha256")
        != schema.get("inventory_manifest_sha256")
    ):
        raise ValueError("checkpoint and replay schema feature inventories differ")
    source_matches_schema = checkpoint_sha == replay["schema_checkpoint_sha256"]
    resume_matches_schema = (
        checkpoint.get("active_schema_manifest_sha256")
        == replay["schema_manifest_sha256"]
        and checkpoint.get("replay_cache_manifest_sha256")
        == _sha256(args.replay_cache_manifest)
    )
    if not (source_matches_schema or resume_matches_schema):
        raise ValueError("checkpoint is not bound to replay active-feature schema")

    selected_train_ids = {
        str(row["molecule_id"]) for row in baseline.get("parents", [])
    }
    if not selected_train_ids:
        raise ValueError("training baseline manifest has no parents")
    direction_ids = {
        str(row["molecule_id"]) for row in directions.get("candidates", [])
    }
    if not selected_train_ids.issubset(direction_ids):
        raise ValueError("one or more selected HVP parents are absent from direction manifest")
    validation_ids = {
        str(row["molecule_id"]) for row in validation.get("parents", [])
    }
    if len(validation_ids) != int(validation["expected_parent_count"]):
        raise ValueError("validation parent count is inconsistent")
    replay_csv, replay_rows = _artifact_csv(replay, "success_csv")
    replay_ids = {str(row["molecule_id"]) for row in replay_rows}
    expected_replay = int(replay["counts"]["expected_tasks"])
    if len(replay_rows) != expected_replay:
        raise ValueError("replay success CSV does not cover every expected task")
    if len(replay_ids) != args.expected_replay_parent_count:
        raise ValueError(
            f"expected {args.expected_replay_parent_count} replay parents, "
            f"found {len(replay_ids)}"
        )
    if not selected_train_ids.issubset(replay_ids):
        raise ValueError("one or more HVP training parents are absent from E/F replay")
    if selected_train_ids & validation_ids:
        raise ValueError("HVP training and validation parents overlap")
    if replay_ids & validation_ids:
        raise ValueError("energy/force replay and validation parents overlap")
    if baseline.get("source_split_sha256") != validation.get("source_split_sha256"):
        raise ValueError("training and validation source split hashes differ")

    result = {
        "definition": (
            "Frozen Stage-3 train800 E/F replay plus stable train-parent complete-total "
            "HVP inputs; independent validation parents remain evaluation-only."
        ),
        "inputs": {
            "baseline_manifest": args.baseline_manifest.resolve().as_posix(),
            "baseline_manifest_sha256": _sha256(args.baseline_manifest),
            "direction_manifest": args.direction_manifest.resolve().as_posix(),
            "direction_manifest_sha256": _sha256(args.direction_manifest),
            "checkpoint": args.checkpoint.resolve().as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": int(checkpoint["step"]),
            "replay_cache_manifest": args.replay_cache_manifest.resolve().as_posix(),
            "replay_cache_manifest_sha256": _sha256(args.replay_cache_manifest),
            "replay_success_csv": replay_csv.resolve().as_posix(),
            "replay_success_csv_sha256": _sha256(replay_csv),
            "active_schema_manifest": schema_path.resolve().as_posix(),
            "active_schema_manifest_sha256": _sha256(schema_path),
            "feature_inventory_manifest_sha256": schema[
                "inventory_manifest_sha256"
            ],
            "validation_baseline_manifest": (
                args.validation_baseline_manifest.resolve().as_posix()
            ),
            "validation_baseline_manifest_sha256": _sha256(
                args.validation_baseline_manifest
            ),
            "source_split_sha256": baseline["source_split_sha256"],
        },
        "counts": {
            "hvp_training_parents": len(selected_train_ids),
            "replay_geometries": len(replay_rows),
            "replay_parents": len(replay_ids),
            "validation_parents": len(validation_ids),
        },
        "hvp_training_parent_ids": sorted(selected_train_ids),
        "validation_parent_ids": sorted(validation_ids),
        "parent_group_disjoint": True,
        "checkpoint_schema_binding": (
            "exact_schema_source" if source_matches_schema else "same_schema_resume"
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "ready": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "stage3_training_preflight_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-cache-manifest", type=Path, required=True)
    parser.add_argument("--validation-baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-replay-parent-count", type=int, default=800)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
