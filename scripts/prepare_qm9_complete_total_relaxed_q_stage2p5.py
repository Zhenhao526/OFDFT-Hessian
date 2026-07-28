#!/usr/bin/env python3
"""Freeze a train-only unseen-direction split over eligible relaxed q labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("stage") != "train59_relaxed_q_unseen_direction_diagnostic_split":
        raise ValueError("unexpected relaxed-q split stage")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    for hash_key, expected in protocol["inputs"].items():
        if not hash_key.endswith("_sha256"):
            continue
        source_key = hash_key.removesuffix("_sha256")
        source = Path(protocol["inputs"][source_key])
        if _sha256(source) != str(expected):
            raise ValueError(f"input hash drift: {source_key}")
    return protocol


def _baseline_lookup(path: Path, sample_id: int) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        str(row["molecule_id"]): row
        for row in rows
        if int(row["sample_id"]) == sample_id
    }
    if len(selected) != len(
        [row for row in rows if int(row["sample_id"]) == sample_id]
    ):
        raise ValueError("duplicate sample0 replay baseline parent")
    return selected


def _pbe_lookup(path: Path, sample_id: int) -> dict[str, dict[str, Any]]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError("PBE Hessian manifest must be a list")
    selected = {
        str(row["molecule_id"]): row
        for row in rows
        if int(row["sample_id"]) == sample_id and row.get("success") is True
    }
    if len(selected) != len(
        [
            row
            for row in rows
            if int(row["sample_id"]) == sample_id and row.get("success") is True
        ]
    ):
        raise ValueError("duplicate successful PBE Hessian parent")
    return selected


def _heldout_index(
    seed: int, molecule_id: str, indices: np.ndarray, kinds: np.ndarray
) -> int:
    ranked = []
    for index, kind in zip(indices.tolist(), kinds.tolist(), strict=True):
        token = f"{seed}:{molecule_id}:{int(index)}:{str(kind)}".encode()
        ranked.append((hashlib.sha256(token).hexdigest(), int(index)))
    return min(ranked)[1]


def prepare(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    summary = _read_json(Path(inputs["eligibility_summary"]))
    if summary.get("test100_accessed") is not False:
        raise ValueError("eligibility summary does not freeze Test100")
    if summary.get("train100_q_ready") is not False:
        raise ValueError("Stage-2.5 split is only valid for the failed train100 gate")
    expected_parent_count = int(protocol["data"]["expected_parent_count"])
    expected_audit_direction_count = int(
        protocol["data"]["expected_audit_eligible_direction_count"]
    )
    expected_selected_direction_count = int(
        protocol["data"]["expected_selected_parent_direction_count"]
    )
    if int(summary["eligible_parent_count"]) != expected_parent_count:
        raise ValueError("eligible parent count drift")
    if int(summary["eligible_direction_count"]) != expected_audit_direction_count:
        raise ValueError("eligible direction count drift")

    eligibility_path = Path(inputs["eligibility_manifest"])
    eligibility = _read_json(eligibility_path)
    entries = list(eligibility["entries"])
    if len(entries) != expected_parent_count:
        raise ValueError("eligibility manifest parent count drift")
    sample_id = int(protocol["data"]["sample_id"])
    baselines = _baseline_lookup(
        Path(inputs["replay_baseline_success_csv"]), sample_id
    )
    pbe = _pbe_lookup(Path(inputs["pbe_hessian_manifest"]), sample_id)
    sidecar_dir = eligibility_path.parent / "sidecars"
    seed = int(protocol["direction_split"]["seed"])
    minimum = int(protocol["data"]["minimum_eligible_directions_per_parent"])
    held_count = int(protocol["direction_split"]["heldout_count_per_parent"])
    if held_count != 1:
        raise ValueError("v1 supports exactly one held direction per parent")

    parents = []
    train_count = 0
    heldout_count = 0
    kind_role_counts: dict[str, int] = {}
    for entry in entries:
        molecule_id = str(entry["molecule_id"])
        sidecar = sidecar_dir / str(entry["filename"])
        if _sha256(sidecar) != str(entry["sha256"]):
            raise ValueError(f"sidecar hash drift: {molecule_id}")
        with np.load(sidecar) as payload:
            mask = np.asarray(payload["eligibility_mask"], dtype=np.bool_)
            indices = np.asarray(payload["direction_index"], dtype=np.int64)[mask]
            kinds = np.asarray(payload["direction_kind"]).astype(str)[mask]
            natoms = int(np.asarray(payload["atomic_numbers"]).size)
            parent_eligible = bool(np.asarray(payload["parent_eligibility"]).item())
        if not parent_eligible or indices.size < minimum:
            raise ValueError(f"ineligible parent in manifest: {molecule_id}")
        if int(entry["eligible_direction_count"]) != int(indices.size):
            raise ValueError(f"eligible direction count drift: {molecule_id}")
        heldout = _heldout_index(seed, molecule_id, indices, kinds)
        direction_rows = []
        for index, kind in zip(indices.tolist(), kinds.tolist(), strict=True):
            role = "heldout" if int(index) == heldout else "train"
            direction_rows.append(
                {"direction_index": int(index), "direction_kind": str(kind), "role": role}
            )
            kind_role_counts[f"{role}:{kind}"] = (
                kind_role_counts.get(f"{role}:{kind}", 0) + 1
            )
            train_count += int(role == "train")
            heldout_count += int(role == "heldout")

        baseline = baselines.get(molecule_id)
        if baseline is None or baseline.get("strict_converged") != "True":
            raise ValueError(f"missing strict sample0 baseline: {molecule_id}")
        baseline_array = Path(baseline["baseline_array"])
        if _sha256(baseline_array) != baseline["baseline_array_sha256"]:
            raise ValueError(f"baseline array hash drift: {molecule_id}")
        pbe_row = pbe.get(molecule_id)
        if pbe_row is None:
            raise ValueError(f"missing PBE Hessian: {molecule_id}")
        pbe_array = Path(pbe_row["cache_path"])
        if not pbe_array.is_file():
            raise ValueError(f"missing PBE Hessian array: {molecule_id}")
        parents.append(
            {
                "molecule_id": molecule_id,
                "natoms": natoms,
                "sidecar": sidecar.resolve().as_posix(),
                "sidecar_sha256": _sha256(sidecar),
                "baseline_array": baseline_array.resolve().as_posix(),
                "baseline_array_sha256": _sha256(baseline_array),
                "pbe_hessian": pbe_array.resolve().as_posix(),
                "pbe_hessian_sha256": _sha256(pbe_array),
                "directions": direction_rows,
            }
        )

    if train_count + heldout_count != expected_selected_direction_count:
        raise ValueError("split direction total drift")
    if heldout_count != expected_parent_count:
        raise ValueError("heldout direction count drift")
    output = {
        "definition": protocol["scope"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "source_eligibility_manifest": eligibility_path.resolve().as_posix(),
        "source_eligibility_manifest_sha256": _sha256(eligibility_path),
        "split_seed": seed,
        "parent_count": len(parents),
        "train_direction_count": train_count,
        "heldout_direction_count": heldout_count,
        "kind_role_counts": dict(sorted(kind_role_counts.items())),
        "parents": parents,
        "formal_stage3_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "direction_split_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    result = {
        "manifest": manifest_path.resolve().as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "parent_count": len(parents),
        "train_direction_count": train_count,
        "heldout_direction_count": heldout_count,
        "kind_role_counts": output["kind_role_counts"],
        "formal_stage3_authorized": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    prepare(arguments.protocol, arguments.output_dir)
