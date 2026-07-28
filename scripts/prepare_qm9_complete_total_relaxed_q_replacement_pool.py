#!/usr/bin/env python3
"""Freeze train700 replacement candidates before computing replacement labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.prepare_qm9_hvp100_experiment import _composition_class, _percentile_ranks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("stage") != "train800_only_replacement_candidate_pool_prelabel":
        raise ValueError("unexpected replacement-pool stage")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("replacement protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("replacement protocol Test100 count is not zero")
    for key, expected in protocol["inputs"].items():
        if not key.endswith("_sha256"):
            continue
        source_key = key.removesuffix("_sha256")
        if _sha256(Path(protocol["inputs"][source_key])) != str(expected):
            raise ValueError(f"input hash drift: {source_key}")
    return protocol


def _enriched_difficulty(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 800:
        raise ValueError("difficulty inventory must contain train800")
    for row in rows:
        row["natoms"] = int(row["natoms"])
        row["force_component_mae"] = float(row["force_component_mae"])
        row["paired_hvp_component_mae"] = float(row["paired_hvp_component_mae"])
        row["composition_class"] = _composition_class(str(row["composition"]))
    force_rank = _percentile_ranks(
        np.asarray([row["force_component_mae"] for row in rows])
    )
    hvp_rank = _percentile_ranks(
        np.asarray([row["paired_hvp_component_mae"] for row in rows])
    )
    natom_rank = _percentile_ranks(
        np.asarray([row["natoms"] for row in rows], dtype=np.float64)
    )
    for index, row in enumerate(rows):
        row["force_difficulty_percentile"] = float(force_rank[index])
        row["hvp_difficulty_percentile"] = float(hvp_rank[index])
        row["difficulty_score"] = float((force_rank[index] + hvp_rank[index]) / 2.0)
        row["difficulty_bin"] = min(3, int(4 * row["difficulty_score"]))
        row["natoms_bin"] = min(3, int(4 * natom_rank[index]))
    return rows


def _rank_key(
    slot: dict[str, Any], candidate: dict[str, Any], seed: int
) -> tuple[int, int, int, int, str]:
    token = f"{seed}:{slot['parent_id']}:{candidate['parent_id']}".encode()
    return (
        int(candidate["composition_class"] != slot["composition_class"]),
        abs(int(candidate["natoms_bin"]) - int(slot["natoms_bin"])),
        abs(int(candidate["difficulty_bin"]) - int(slot["difficulty_bin"])),
        abs(int(candidate["natoms"]) - int(slot["natoms"])),
        hashlib.sha256(token).hexdigest(),
    )


def prepare(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    original_manifest = json.loads(Path(inputs["original_selection_manifest"]).read_text())
    if original_manifest.get("test_accessed_for_selection") is not False:
        raise ValueError("original selection does not freeze Test100")
    if original_manifest.get("parent_overlap") != {
        "train_test": [],
        "train_val": [],
        "val_test": [],
    }:
        raise ValueError("original parent split is not disjoint")
    selected_rows = json.loads(Path(inputs["original_selection"]).read_text())
    selected = {str(row["parent_id"]): row for row in selected_rows}
    if len(selected) != 100:
        raise ValueError("original selection must contain 100 parents")
    with Path(inputs["parent_stability_csv"]).open(newline="") as handle:
        stability = list(csv.DictReader(handle))
    failed_ids = sorted(
        str(row["molecule_id"])
        for row in stability
        if str(row["parent_gate_failures"]).strip()
    )
    expected_failed = int(protocol["selection"]["failed_slot_count"])
    if len(failed_ids) != expected_failed:
        raise ValueError("failed parent-gate slot count drift")
    if not set(failed_ids) <= set(selected):
        raise ValueError("failed parent is absent from original selection")

    difficulty = _enriched_difficulty(Path(inputs["difficulty_csv"]))
    difficulty_lookup = {str(row["parent_id"]): row for row in difficulty}
    if len(difficulty_lookup) != 800:
        raise ValueError("duplicate train800 parent in difficulty inventory")
    for molecule_id, original in selected.items():
        current = difficulty_lookup[molecule_id]
        for key in ("natoms", "natoms_bin", "difficulty_bin", "composition_class"):
            if current[key] != original[key]:
                raise ValueError(f"original stratum drift: {molecule_id} {key}")

    available = [row for row in difficulty if str(row["parent_id"]) not in selected]
    used: set[str] = set()
    per_slot = int(protocol["selection"]["candidates_per_slot"])
    slots = []
    for failed_id in failed_ids:
        slot = selected[failed_id]
        candidates = [row for row in available if str(row["parent_id"]) not in used]
        ranked = sorted(candidates, key=lambda row: _rank_key(slot, row, int(protocol["selection"]["seed"])))
        chosen = ranked[:per_slot]
        if len(chosen) != per_slot:
            raise ValueError(f"insufficient replacement candidates: {failed_id}")
        candidate_rows = []
        for rank, candidate in enumerate(chosen, start=1):
            candidate_id = str(candidate["parent_id"])
            used.add(candidate_id)
            key = _rank_key(slot, candidate, int(protocol["selection"]["seed"]))
            candidate_rows.append(
                {
                    "rank": rank,
                    "parent_id": candidate_id,
                    "natoms": int(candidate["natoms"]),
                    "natoms_bin": int(candidate["natoms_bin"]),
                    "composition": str(candidate["composition"]),
                    "composition_class": str(candidate["composition_class"]),
                    "difficulty_bin": int(candidate["difficulty_bin"]),
                    "difficulty_score": float(candidate["difficulty_score"]),
                    "composition_class_mismatch": key[0],
                    "natoms_bin_distance": key[1],
                    "difficulty_bin_distance": key[2],
                    "natoms_distance": key[3],
                    "exact_stratum": key[:3] == (0, 0, 0),
                }
            )
        slots.append(
            {
                "failed_parent_id": failed_id,
                "failed_parent_gate_reasons": next(
                    row["parent_gate_failures"]
                    for row in stability
                    if str(row["molecule_id"]) == failed_id
                ),
                "source_stratum": {
                    "natoms": int(slot["natoms"]),
                    "natoms_bin": int(slot["natoms_bin"]),
                    "composition": str(slot["composition"]),
                    "composition_class": str(slot["composition_class"]),
                    "difficulty_bin": int(slot["difficulty_bin"]),
                    "difficulty_score": float(slot["difficulty_score"]),
                },
                "candidates": candidate_rows,
            }
        )

    expected_candidates = int(protocol["selection"]["expected_candidate_count"])
    if len(used) != expected_candidates:
        raise ValueError("replacement candidate count drift")
    manifest = {
        "definition": protocol["purpose"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "failed_slot_count": len(slots),
        "candidate_count": len(used),
        "candidate_ids": sorted(used),
        "exact_stratum_candidate_count": sum(
            int(candidate["exact_stratum"])
            for slot in slots
            for candidate in slot["candidates"]
        ),
        "fallback_candidate_count": sum(
            int(not candidate["exact_stratum"])
            for slot in slots
            for candidate in slot["candidates"]
        ),
        "slots": slots,
        "labels_authorized": False,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "replacement_pool_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        "manifest": manifest_path.resolve().as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "failed_slot_count": len(slots),
        "candidate_count": len(used),
        "exact_stratum_candidate_count": manifest["exact_stratum_candidate_count"],
        "fallback_candidate_count": manifest["fallback_candidate_count"],
        "labels_authorized": False,
        "formal_stage3_authorized": False,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    prepare(args.protocol, args.output_dir)
