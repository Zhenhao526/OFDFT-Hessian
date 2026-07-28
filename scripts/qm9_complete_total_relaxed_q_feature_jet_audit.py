#!/usr/bin/env python3
"""Audit every relaxed-q feature jet before allowing the Stage-2.5 fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch

from scripts.qm9_complete_total_relaxed_q_local_scalar import (
    _load_manifest,
    _load_protocol,
    _sha256,
)


FEATURE_HESSIAN_RELATIVE_ASYMMETRY_LIMIT = 1.0e-10


def _hessian_symmetry_metrics(
    hessian: torch.Tensor, chunk_size: int = 256
) -> dict[str, float]:
    transpose_difference_maximum = 0.0
    symmetric_maximum = 0.0
    antisymmetric_squared = 0.0
    symmetric_squared = 0.0
    for start in range(0, hessian.shape[0], chunk_size):
        block = hessian[start : start + chunk_size]
        symmetric = 0.5 * (block + block.transpose(1, 2))
        antisymmetric = 0.5 * (block - block.transpose(1, 2))
        transpose_difference_maximum = max(
            transpose_difference_maximum,
            float(torch.max(torch.abs(block - block.transpose(1, 2)))),
        )
        symmetric_maximum = max(
            symmetric_maximum, float(torch.max(torch.abs(symmetric)))
        )
        antisymmetric_squared += float(torch.sum(antisymmetric * antisymmetric))
        symmetric_squared += float(torch.sum(symmetric * symmetric))
    tiny = torch.finfo(torch.float64).tiny
    return {
        "transpose_difference_max_abs": transpose_difference_maximum,
        "asymmetry_max_over_symmetry_max": (
            0.5 * transpose_difference_maximum / max(symmetric_maximum, tiny)
        ),
        "asymmetry_fro_over_symmetry_fro": math.sqrt(
            antisymmetric_squared / max(symmetric_squared, tiny)
        ),
    }


def _hessian_symmetry_max(hessian: torch.Tensor, chunk_size: int = 256) -> float:
    return _hessian_symmetry_metrics(hessian, chunk_size)[
        "transpose_difference_max_abs"
    ]


def _feature_chunk_metadata(
    payload: dict[str, Any], summary: dict[str, Any], molecule_id: str
) -> tuple[int, bool]:
    summary_size = int(summary["feature_chunk_size"])
    summary_override = bool(summary.get("feature_chunk_size_override", False))
    if "feature_chunk_size" not in payload:
        if summary_override:
            raise ValueError(
                f"legacy feature jet cannot claim chunk override: {molecule_id}"
            )
        return summary_size, False
    payload_size = int(payload["feature_chunk_size"])
    payload_override = bool(payload.get("feature_chunk_size_override", False))
    if summary_size != payload_size:
        raise ValueError(f"feature jet chunk-size drift: {molecule_id}")
    if summary_override != payload_override:
        raise ValueError(f"feature jet chunk-override drift: {molecule_id}")
    return payload_size, payload_override


def audit(
    protocol_path: Path, jet_dir: Path, output_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _load_protocol(protocol_path)
    _, rows = _load_manifest(protocol)
    protocol_hash = _sha256(protocol_path)
    expected_global_count: int | None = None
    entries = []
    for index, row in enumerate(rows):
        molecule_id = str(row["molecule_id"])
        artifact = jet_dir / f"{index:02d}_{molecule_id}.pt"
        summary_path = jet_dir / f"{index:02d}_{molecule_id}.json"
        if not artifact.is_file() or not summary_path.is_file():
            raise ValueError(f"missing feature jet artifact: {index} {molecule_id}")
        artifact_hash = _sha256(artifact)
        summary = json.loads(summary_path.read_text())
        if summary.get("artifact_sha256") != artifact_hash:
            raise ValueError(f"feature jet summary hash drift: {molecule_id}")
        if Path(summary.get("artifact", "")).resolve() != artifact.resolve():
            raise ValueError(f"feature jet summary path drift: {molecule_id}")
        if (
            summary.get("molecule_id") != molecule_id
            or int(summary.get("parent_index", -1)) != index
        ):
            raise ValueError(f"feature jet summary identity drift: {molecule_id}")
        if summary.get("test100_accessed") is not False:
            raise ValueError(f"feature jet opened Test100: {molecule_id}")
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"feature jet protocol drift: {molecule_id}")
        if payload.get("molecule_id") != molecule_id or int(payload.get("parent_index", -1)) != index:
            raise ValueError(f"feature jet identity drift: {molecule_id}")
        if payload.get("test100_accessed") is not False:
            raise ValueError(f"feature jet payload opened Test100: {molecule_id}")
        features = payload["features"].to(dtype=torch.float64)
        jacobian = payload["jacobian"].to(dtype=torch.float64)
        hessian = payload["hessian"].to(dtype=torch.float64)
        columns = payload["global_columns"].to(dtype=torch.long)
        coordinate_count = 3 * int(row["natoms"])
        feature_count = int(features.numel())
        feature_chunk_size, chunk_override = _feature_chunk_metadata(
            payload, summary, molecule_id
        )
        if int(summary["local_feature_count"]) != feature_count:
            raise ValueError(f"feature jet feature-count drift: {molecule_id}")
        if tuple(jacobian.shape) != (feature_count, coordinate_count):
            raise ValueError(f"feature jet Jacobian shape drift: {molecule_id}")
        if tuple(hessian.shape) != (
            feature_count,
            coordinate_count,
            coordinate_count,
        ):
            raise ValueError(f"feature jet Hessian shape drift: {molecule_id}")
        if tuple(columns.shape) != (feature_count,):
            raise ValueError(f"feature jet column shape drift: {molecule_id}")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (features, jacobian, hessian)
        ):
            raise FloatingPointError(f"non-finite feature jet: {molecule_id}")
        global_count = int(payload["global_feature_count"])
        if int(summary["global_feature_count"]) != global_count:
            raise ValueError(f"feature jet global-count drift: {molecule_id}")
        expected_global_count = (
            global_count if expected_global_count is None else expected_global_count
        )
        if global_count != expected_global_count:
            raise ValueError("global feature count differs across parent jets")
        if int(torch.min(columns)) < 0 or int(torch.max(columns)) >= global_count:
            raise ValueError(f"feature jet column out of bounds: {molecule_id}")
        if int(torch.unique(columns).numel()) != feature_count:
            raise ValueError(f"duplicate feature jet columns: {molecule_id}")
        symmetry = _hessian_symmetry_metrics(hessian)
        if max(
            symmetry["asymmetry_max_over_symmetry_max"],
            symmetry["asymmetry_fro_over_symmetry_fro"],
        ) > FEATURE_HESSIAN_RELATIVE_ASYMMETRY_LIMIT:
            raise FloatingPointError(
                f"feature Hessian relative symmetry drift: {molecule_id} "
                f"{symmetry}"
            )
        entries.append(
            {
                "parent_index": index,
                "molecule_id": molecule_id,
                "natoms": int(row["natoms"]),
                "feature_count": feature_count,
                "coordinate_count": coordinate_count,
                "global_feature_count": global_count,
                "artifact": artifact.resolve().as_posix(),
                "artifact_sha256": artifact_hash,
                "summary": summary_path.resolve().as_posix(),
                "summary_sha256": _sha256(summary_path),
                "feature_chunk_size": feature_chunk_size,
                "feature_chunk_size_override": chunk_override,
                "hessian_transpose_difference_max_abs": symmetry[
                    "transpose_difference_max_abs"
                ],
                "hessian_asymmetry_max_over_symmetry_max": symmetry[
                    "asymmetry_max_over_symmetry_max"
                ],
                "hessian_asymmetry_fro_over_symmetry_fro": symmetry[
                    "asymmetry_fro_over_symmetry_fro"
                ],
                "wall_time_s": float(summary["wall_time_s"]),
                "gpu_peak_memory_mb": float(summary["gpu_peak_memory_mb"]),
            }
        )
        del payload, features, jacobian, hessian, columns

    manifest = {
        "definition": (
            "Complete hash/shape/finite/symmetry inventory for the 59 train-only "
            "relaxed-q local-scalar feature jets"
        ),
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": protocol_hash,
        "parent_count": len(entries),
        "global_feature_count": expected_global_count,
        "chunk_override_count": sum(
            int(entry["feature_chunk_size_override"]) for entry in entries
        ),
        "total_artifact_bytes": sum(
            Path(entry["artifact"]).stat().st_size for entry in entries
        ),
        "sum_gpu_task_wall_time_s": sum(entry["wall_time_s"] for entry in entries),
        "maximum_gpu_peak_memory_mb": max(
            entry["gpu_peak_memory_mb"] for entry in entries
        ),
        "feature_hessian_relative_asymmetry_limit": (
            FEATURE_HESSIAN_RELATIVE_ASYMMETRY_LIMIT
        ),
        "maximum_hessian_transpose_difference_abs": max(
            entry["hessian_transpose_difference_max_abs"] for entry in entries
        ),
        "maximum_hessian_asymmetry_max_over_symmetry_max": max(
            entry["hessian_asymmetry_max_over_symmetry_max"] for entry in entries
        ),
        "maximum_hessian_asymmetry_fro_over_symmetry_fro": max(
            entry["hessian_asymmetry_fro_over_symmetry_fro"] for entry in entries
        ),
        "entries": entries,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "feature_jet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        "manifest": manifest_path.resolve().as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "parent_count": len(entries),
        "global_feature_count": expected_global_count,
        "chunk_override_count": manifest["chunk_override_count"],
        "total_artifact_bytes": manifest["total_artifact_bytes"],
        "sum_gpu_task_wall_time_s": manifest["sum_gpu_task_wall_time_s"],
        "maximum_gpu_peak_memory_mb": manifest["maximum_gpu_peak_memory_mb"],
        "feature_hessian_relative_asymmetry_limit": manifest[
            "feature_hessian_relative_asymmetry_limit"
        ],
        "maximum_hessian_transpose_difference_abs": manifest[
            "maximum_hessian_transpose_difference_abs"
        ],
        "maximum_hessian_asymmetry_max_over_symmetry_max": manifest[
            "maximum_hessian_asymmetry_max_over_symmetry_max"
        ],
        "maximum_hessian_asymmetry_fro_over_symmetry_fro": manifest[
            "maximum_hessian_asymmetry_fro_over_symmetry_fro"
        ],
        "audit_wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--jet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    audit(args.protocol, args.jet_dir, args.output_dir)
