#!/usr/bin/env python3
"""Materialize hash-bound relaxed vector-HVP targets from existing train tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
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


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text())
    if (
        protocol.get("stage")
        != "train20_complete_total_relaxed_vector_target_materialization"
    ):
        raise ValueError("unexpected relaxed-vector target stage")
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("protocol does not freeze Test100")
    if int(protocol.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("protocol Test100 count is not zero")
    for key, expected in protocol["inputs"].items():
        if not key.endswith("_sha256"):
            continue
        source = Path(protocol["inputs"][key.removesuffix("_sha256")])
        if _sha256(source) != str(expected):
            raise ValueError(f"input hash drift: {source}")
    return protocol


def _selected_step(steps: np.ndarray, requested: float) -> int:
    index = int(np.argmin(np.abs(steps - requested)))
    if not math.isclose(
        float(steps[index]), requested, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"requested vector-HVP step is missing: {requested}")
    return index


def _vector_spread(vectors: np.ndarray, floor: float) -> float:
    mean = np.mean(vectors, axis=0)
    denominator = max(float(np.linalg.norm(mean)), floor)
    return max(float(np.linalg.norm(value - mean)) for value in vectors) / denominator


def _load_branch(
    task_root: Path,
    task_row: dict[str, str],
    expected_run: str,
    selected_step: float,
) -> dict[str, Any]:
    index = int(task_row["task_index"])
    molecule_id = str(task_row["molecule_id"])
    direction_index = int(task_row["direction_index"])
    branch = str(task_row["branch"])
    task_dir = (
        task_root
        / f"task_{index:04d}_{molecule_id}_d{direction_index}_{branch}"
    )
    summary_path = task_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing branch summary: {task_dir}")
    summary = json.loads(summary_path.read_text())
    if summary.get("runs") != [expected_run]:
        raise ValueError(f"raw-task source run drift: {task_dir}")
    if summary.get("molecules") != [molecule_id]:
        raise ValueError(f"raw-task molecule drift: {task_dir}")
    arrays = list(task_dir.glob("*_hvp_arrays.npz"))
    if len(arrays) != 1:
        raise ValueError(f"missing or ambiguous HVP arrays: {task_dir}")
    with np.load(arrays[0]) as payload:
        steps = np.asarray(payload["curvature_steps_bohr"], dtype=np.float64)
        step_index = _selected_step(steps, selected_step)
        result = {
            "branch": branch,
            "artifact": arrays[0].resolve().as_posix(),
            "artifact_sha256": _sha256(arrays[0]),
            "summary": summary_path.resolve().as_posix(),
            "summary_sha256": _sha256(summary_path),
            "atomic_numbers": np.asarray(
                payload["atomic_numbers"], dtype=np.int64
            ),
            "positions_bohr": np.asarray(
                payload["positions_bohr"], dtype=np.float64
            ),
            "direction": np.asarray(payload["direction"], dtype=np.float64),
            "direction_kind": str(np.asarray(payload["direction_kind"]).item()),
            "source_hvp": np.asarray(
                payload["relaxed_hvp_step_scan"], dtype=np.float64
            )[step_index],
            "selected_payload_hvp": np.asarray(
                payload["relaxed_hvp"], dtype=np.float64
            ),
            "pbe_hvp": np.asarray(payload["pbe_hvp"], dtype=np.float64),
        }
    if not np.allclose(
        result["source_hvp"],
        result["selected_payload_hvp"],
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise ValueError(f"selected source HVP payload drift: {task_dir}")
    return result


def materialize(
    protocol_path: Path, output_dir: Path
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    inventory = protocol["inventory"]
    gate = protocol["vector_gate"]
    selection = json.loads(Path(inputs["selection_manifest"]).read_text())
    if selection.get("protocol_sha256") != str(
        inputs["selection_protocol_sha256"]
    ):
        raise ValueError("selection protocol drift")
    if (
        selection.get("validation_accessed") is not False
        or selection.get("test100_accessed") is not False
    ):
        raise ValueError("selection opened validation/Test100")
    expected_counts = {
        "selected_parent_count": int(inventory["parent_count"]),
        "train_direction_count": int(inventory["train_direction_count"]),
        "heldout_direction_count": int(inventory["heldout_direction_count"]),
    }
    for key, expected in expected_counts.items():
        if int(selection[key]) != expected:
            raise ValueError(f"selection {key} drift")
    if selection.get("source_checkpoint_sha256") != str(
        inputs["source_checkpoint_sha256"]
    ):
        raise ValueError("selection source checkpoint drift")

    stability_rows = _read_csv(Path(inputs["direction_stability_csv"]))
    stable_keys = {
        (str(row["molecule_id"]), int(row["direction_index"]))
        for row in stability_rows
        if str(row["stable"]) == "True"
    }
    task_rows = _read_csv(Path(inputs["tasks_tsv"]), delimiter="\t")
    task_lookup: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        task_lookup[
            (str(row["molecule_id"]), int(row["direction_index"]))
        ].append(row)

    expected_branches = list(gate["branches"])
    selected_step = float(gate["selected_step_bohr"])
    floor = float(gate["vector_norm_floor_hartree_per_bohr2"])
    sidecar_dir = output_dir / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    maximum_source_spread = 0.0
    maximum_pbe_spread = 0.0
    maximum_direction_mismatch = 0.0
    maximum_direction_norm_error = 0.0
    for parent in selection["parents"]:
        molecule_id = str(parent["molecule_id"])
        directions = []
        parent_artifacts = []
        reference_atomic_numbers = None
        reference_positions = None
        for direction_row in parent["directions"]:
            direction_index = int(direction_row["direction_index"])
            key = (molecule_id, direction_index)
            if key not in stable_keys:
                raise ValueError(f"selection contains unstable vector direction: {key}")
            rows = task_lookup.get(key, [])
            rows.sort(key=lambda row: expected_branches.index(str(row["branch"])))
            if [str(row["branch"]) for row in rows] != expected_branches:
                raise ValueError(f"branch inventory drift: {key}")
            branches = [
                _load_branch(
                    Path(inputs["task_root"]),
                    row,
                    str(inputs["raw_task_run"]).strip(),
                    selected_step,
                )
                for row in rows
            ]
            atomic_numbers = branches[0]["atomic_numbers"]
            positions = branches[0]["positions_bohr"]
            direction = branches[0]["direction"]
            direction_kind = branches[0]["direction_kind"]
            for branch in branches[1:]:
                if not np.array_equal(atomic_numbers, branch["atomic_numbers"]):
                    raise ValueError(f"branch atomic-number drift: {key}")
                if not np.allclose(
                    positions, branch["positions_bohr"], atol=1.0e-12, rtol=0.0
                ):
                    raise ValueError(f"branch position drift: {key}")
                maximum_direction_mismatch = max(
                    maximum_direction_mismatch,
                    float(np.max(np.abs(direction - branch["direction"]))),
                )
                if direction_kind != branch["direction_kind"]:
                    raise ValueError(f"branch direction-kind drift: {key}")
            source_vectors = np.stack(
                [branch["source_hvp"] for branch in branches]
            )
            pbe_vectors = np.stack([branch["pbe_hvp"] for branch in branches])
            source_spread = _vector_spread(source_vectors, floor)
            pbe_spread = _vector_spread(pbe_vectors, floor)
            maximum_source_spread = max(maximum_source_spread, source_spread)
            maximum_pbe_spread = max(maximum_pbe_spread, pbe_spread)
            norm_error = abs(float(np.linalg.norm(direction)) - 1.0)
            maximum_direction_norm_error = max(
                maximum_direction_norm_error, norm_error
            )
            if source_spread > float(
                gate["maximum_source_branch_relative_spread"]
            ):
                raise ValueError(f"source vector branch instability: {key}")
            if pbe_spread > float(gate["maximum_pbe_branch_relative_spread"]):
                raise ValueError(f"PBE vector branch drift: {key}")
            if maximum_direction_mismatch > float(
                gate["maximum_direction_mismatch"]
            ):
                raise ValueError(f"direction branch mismatch: {key}")
            if norm_error > float(gate["maximum_direction_norm_error"]):
                raise ValueError(f"direction norm drift: {key}")
            source_hvp = np.mean(source_vectors, axis=0)
            pbe_hvp = np.mean(pbe_vectors, axis=0)
            directions.append(
                {
                    "direction_index": direction_index,
                    "direction_kind": direction_kind,
                    "role": str(direction_row["role"]),
                    "direction": direction,
                    "source_hvp": source_hvp,
                    "pbe_hvp": pbe_hvp,
                    "correction_hvp": pbe_hvp - source_hvp,
                    "source_branch_relative_spread": source_spread,
                    "pbe_branch_relative_spread": pbe_spread,
                }
            )
            parent_artifacts.extend(
                {
                    key: branch[key]
                    for key in (
                        "branch",
                        "artifact",
                        "artifact_sha256",
                        "summary",
                        "summary_sha256",
                    )
                }
                for branch in branches
            )
            if reference_atomic_numbers is None:
                reference_atomic_numbers = atomic_numbers
                reference_positions = positions
            elif not np.array_equal(reference_atomic_numbers, atomic_numbers):
                raise ValueError(f"direction atomic-number drift: {molecule_id}")
            elif not np.allclose(
                reference_positions, positions, atol=1.0e-12, rtol=0.0
            ):
                raise ValueError(f"direction position drift: {molecule_id}")

        sidecar = sidecar_dir / f"{molecule_id}.0000000.npz"
        np.savez_compressed(
            sidecar,
            atomic_numbers=reference_atomic_numbers,
            positions_bohr=reference_positions,
            direction_index=np.asarray(
                [row["direction_index"] for row in directions], dtype=np.int64
            ),
            direction_kind=np.asarray(
                [row["direction_kind"] for row in directions]
            ),
            role=np.asarray([row["role"] for row in directions]),
            direction=np.stack([row["direction"] for row in directions]),
            source_hvp_hartree_per_bohr2=np.stack(
                [row["source_hvp"] for row in directions]
            ),
            pbe_hvp_hartree_per_bohr2=np.stack(
                [row["pbe_hvp"] for row in directions]
            ),
            correction_hvp_hartree_per_bohr2=np.stack(
                [row["correction_hvp"] for row in directions]
            ),
            source_branch_relative_spread=np.asarray(
                [row["source_branch_relative_spread"] for row in directions]
            ),
            pbe_branch_relative_spread=np.asarray(
                [row["pbe_branch_relative_spread"] for row in directions]
            ),
        )
        entries.append(
            {
                "molecule_id": molecule_id,
                "natoms": int(reference_atomic_numbers.size),
                "sidecar": sidecar.resolve().as_posix(),
                "sidecar_sha256": _sha256(sidecar),
                "direction_count": len(directions),
                "train_direction_count": sum(
                    row["role"] == "train" for row in directions
                ),
                "heldout_direction_count": sum(
                    row["role"] == "heldout" for row in directions
                ),
                "source_artifacts": parent_artifacts,
            }
        )

    manifest = {
        "definition": protocol["definition"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "selection_manifest": str(inputs["selection_manifest"]),
        "selection_manifest_sha256": str(inputs["selection_manifest_sha256"]),
        "source_checkpoint": str(inputs["source_checkpoint"]),
        "source_checkpoint_sha256": str(inputs["source_checkpoint_sha256"]),
        "parent_count": len(entries),
        "train_direction_count": sum(
            row["train_direction_count"] for row in entries
        ),
        "heldout_direction_count": sum(
            row["heldout_direction_count"] for row in entries
        ),
        "maximum_source_branch_relative_spread": maximum_source_spread,
        "maximum_pbe_branch_relative_spread": maximum_pbe_spread,
        "maximum_direction_mismatch": maximum_direction_mismatch,
        "maximum_direction_norm_error": maximum_direction_norm_error,
        "entries": entries,
        "formal_stage3_authorized": False,
        "replacement_labels_authorized": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        key: value
        for key, value in manifest.items()
        if key not in {"entries", "definition"}
    }
    summary["manifest"] = manifest_path.resolve().as_posix()
    summary["manifest_sha256"] = _sha256(manifest_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    materialize(args.protocol, args.output_dir)
