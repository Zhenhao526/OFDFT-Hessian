#!/usr/bin/env python3
"""Build a train800-only descriptor-key inventory for robust replay training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from mldft.ml.models.components.three_body_geometry_residual import (
    build_triplet_groups,
    make_three_body_feature_function,
)
from scripts.qm9_complete_total_geometry_four_body_capacity import (
    build_bonded_chain_groups,
    make_four_body_feature_function,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_task_csv(asset_manifest: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(asset_manifest.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("asset manifest does not freeze Test100")
    row = manifest["output_artifacts"]["train800_replay_labels"]
    path = Path(row["path"])
    if _sha256(path) != row["sha256"]:
        raise ValueError("train800 replay task CSV hash mismatch")
    return manifest, path


def _local_keys(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, list[tuple[int, ...]]]:
    three_groups = build_triplet_groups(atomic_numbers)
    _, three_keys = make_three_body_feature_function(
        three_groups,
        np.linspace(
            args.three_body_center_min,
            args.three_body_center_max,
            args.three_body_center_count,
        ),
        args.three_body_sigma,
        args.three_body_angular_order,
    )
    four_groups, _ = build_bonded_chain_groups(
        atomic_numbers, positions_bohr, args.four_body_bond_scale
    )
    if four_groups:
        _, four_keys = make_four_body_feature_function(
            four_groups,
            np.linspace(
                args.four_body_center_min,
                args.four_body_center_max,
                args.four_body_center_count,
            ),
            args.four_body_sigma,
            args.four_body_torsion_order,
        )
    else:
        four_keys = []
    return {"three_body": three_keys, "four_body": four_keys}


def _source_schema(path: Path) -> tuple[dict[str, Any], dict[str, dict[tuple[int, ...], float]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("source schema does not freeze Test100")
    mappings = {}
    for row in manifest["stages"]:
        schema = Path(row["schema"])
        if _sha256(schema) != row["schema_sha256"]:
            raise ValueError(f"source schema hash mismatch: {schema}")
        with np.load(schema) as payload:
            keys = np.asarray(payload["feature_keys"], dtype=np.int64)
            if "source_column_norms" in payload:
                norms = np.asarray(payload["source_column_norms"], dtype=np.float64)
            else:
                norms = np.asarray(payload["column_norms"], dtype=np.float64)
        mappings[str(row["stage"])] = {
            tuple(int(value) for value in key): float(norm)
            for key, norm in zip(keys, norms, strict=True)
        }
    return manifest, mappings


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    asset_manifest, task_csv = _asset_task_csv(args.asset_manifest)
    with task_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3200:
        raise ValueError(f"expected 3200 train800 geometries, found {len(rows)}")
    parent_ids = {str(row["molecule_id"]) for row in rows}
    if len(parent_ids) != 800:
        raise ValueError(f"expected 800 train parents, found {len(parent_ids)}")
    if len({(row["molecule_id"], int(row["sample_id"])) for row in rows}) != len(rows):
        raise ValueError("duplicate train800 molecule/sample record")
    _, source_mappings = _source_schema(args.source_schema_manifest)

    inventories: dict[str, set[tuple[int, ...]]] = {
        "three_body": set(),
        "four_body": set(),
    }
    molecule_sample_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        label = Path(row["label_path"])
        if _sha256(label) != row["label_sha256"]:
            raise ValueError(f"label hash mismatch: {label}")
        root = zarr.open(label, mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        for stage, keys in _local_keys(atomic_numbers, positions, args).items():
            inventories[stage].update(keys)
        molecule_id = str(row["molecule_id"])
        molecule_sample_counts[molecule_id] = molecule_sample_counts.get(molecule_id, 0) + 1
        if args.progress_interval and index % args.progress_interval == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "three_body_keys": len(inventories["three_body"]),
                        "four_body_keys": len(inventories["four_body"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if set(molecule_sample_counts.values()) != {4}:
        raise ValueError("every train parent must contribute exactly four geometries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_records = []
    active_total = 0
    for stage in ("three_body", "four_body"):
        source = source_mappings[stage]
        train_keys = inventories[stage]
        source_keys = set(source)
        combined_keys = train_keys | source_keys
        keys = np.asarray(sorted(combined_keys), dtype=np.int64)
        source_norms = np.asarray(
            [source.get(tuple(int(value) for value in key), 0.0) for key in keys],
            dtype=np.float64,
        )
        scales = np.maximum(source_norms, args.feature_scale_floor)
        in_source = source_norms > 0.0
        output = args.output_dir / f"{stage}_feature_schema.npz"
        np.savez_compressed(
            output,
            feature_keys=keys,
            column_norms=scales,
            source_column_norms=source_norms,
            present_in_stage2_design=in_source,
            feature_scale_floor=np.asarray(args.feature_scale_floor, dtype=np.float64),
        )
        stage_records.append(
            {
                "stage": stage,
                "feature_count": int(len(keys)),
                "train800_feature_count": len(train_keys),
                "stage2_design_feature_count": len(source_keys),
                "present_in_stage2_design_count": int(np.sum(in_source)),
                "new_train800_feature_count": len(train_keys - source_keys),
                "stage2_only_feature_count": len(source_keys - train_keys),
                "schema": output.resolve().as_posix(),
                "schema_sha256": _sha256(output),
            }
        )
        active_total += int(len(keys))

    settings = {
        name: getattr(args, name)
        for name in (
            "three_body_center_min",
            "three_body_center_max",
            "three_body_center_count",
            "three_body_sigma",
            "three_body_angular_order",
            "four_body_center_min",
            "four_body_center_max",
            "four_body_center_count",
            "four_body_sigma",
            "four_body_torsion_order",
            "four_body_bond_scale",
        )
    }
    result = {
        "definition": "train800-only descriptor-key inventory with floored Stage2 preconditioning",
        "provisional_unbound_checkpoint": True,
        "asset_manifest": args.asset_manifest.resolve().as_posix(),
        "asset_manifest_sha256": _sha256(args.asset_manifest),
        "task_csv": task_csv.resolve().as_posix(),
        "task_csv_sha256": _sha256(task_csv),
        "source_schema_manifest": args.source_schema_manifest.resolve().as_posix(),
        "source_schema_manifest_sha256": _sha256(args.source_schema_manifest),
        "parent_count": len(parent_ids),
        "geometry_count": len(rows),
        "feature_scale_mode": "floored_column_norm",
        "feature_scale_floor": args.feature_scale_floor,
        "active_feature_count": active_total,
        "descriptor_settings": settings,
        "stages": stage_records,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "wall_time_s": time.perf_counter() - started,
    }
    manifest = args.output_dir / "train800_feature_inventory_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--source-schema-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-scale-floor", type=float, default=1.0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--three-body-center-min", type=float, default=0.5)
    parser.add_argument("--three-body-center-max", type=float, default=8.0)
    parser.add_argument("--three-body-center-count", type=int, default=6)
    parser.add_argument("--three-body-sigma", type=float, default=0.5)
    parser.add_argument("--three-body-angular-order", type=int, default=4)
    parser.add_argument("--four-body-center-min", type=float, default=1.0)
    parser.add_argument("--four-body-center-max", type=float, default=4.0)
    parser.add_argument("--four-body-center-count", type=int, default=4)
    parser.add_argument("--four-body-sigma", type=float, default=0.75)
    parser.add_argument("--four-body-torsion-order", type=int, default=5)
    parser.add_argument("--four-body-bond-scale", type=float, default=1.35)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
