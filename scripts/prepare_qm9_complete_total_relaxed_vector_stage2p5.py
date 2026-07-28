#!/usr/bin/env python3
"""Freeze the train20 vector-HVP Stage-2.5 split without reading model metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ELEMENT_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


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
        != "train20_complete_total_relaxed_vector_unseen_direction_split"
    ):
        raise ValueError("unexpected relaxed-vector split stage")
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


def _rank(seed: int, *parts: object) -> str:
    token = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(token.encode()).hexdigest()


def _atom_count_bin(natoms: int, bins: list[dict[str, Any]]) -> str:
    matches = [
        str(item["name"])
        for item in bins
        if int(item["minimum"]) <= natoms <= int(item["maximum"])
    ]
    if len(matches) != 1:
        raise ValueError(f"atom count {natoms} has {len(matches)} bins")
    return matches[0]


def _composition(atomic_numbers: np.ndarray) -> str:
    heavy = sorted(set(int(value) for value in atomic_numbers if int(value) != 1))
    if not heavy or any(value not in ELEMENT_SYMBOL for value in heavy):
        raise ValueError(f"unsupported QM9 composition: {heavy}")
    return "".join(ELEMENT_SYMBOL[value] for value in heavy)


def _round_robin_select(
    candidates: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["stratum"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: _rank(seed, "parent", row["molecule_id"]))
    strata = sorted(grouped, key=lambda value: _rank(seed, "stratum", value))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        advanced = False
        for stratum in strata:
            if grouped[stratum] and len(selected) < count:
                selected.append(grouped[stratum].pop(0))
                advanced = True
        if not advanced:
            raise ValueError("not enough candidates for requested selection")
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    inventory = protocol["inventory"]
    parent_rows = _read_csv(Path(inputs["parent_stability_csv"]))
    direction_rows = _read_csv(Path(inputs["direction_stability_csv"]))
    task_rows = _read_csv(Path(inputs["tasks_tsv"]), delimiter="\t")
    if len(parent_rows) != int(inventory["train100_parent_count"]):
        raise ValueError("parent inventory drift")
    if len(direction_rows) != int(inventory["direction_count"]):
        raise ValueError("direction inventory drift")
    if len(task_rows) != 3 * int(inventory["direction_count"]):
        raise ValueError("branch task inventory drift")

    selection = json.loads(Path(inputs["train100_selection_manifest"]).read_text())
    if selection.get("test_accessed_for_selection") is not False:
        raise ValueError("train100 selection accessed Test100")
    if selection.get("parent_overlap") != {
        "train_test": [],
        "train_val": [],
        "val_test": [],
    }:
        raise ValueError("source parent split overlap")
    tasks_json = json.loads(Path(inputs["tasks_json"]).read_text())
    if (
        tasks_json.get("test100_accessed") is not False
        or tasks_json.get("subset") not in {"file", "train"}
    ):
        raise ValueError("branch tasks are not frozen train-only tasks")

    parent_by_id = {str(row["molecule_id"]): row for row in parent_rows}
    if len(parent_by_id) != len(parent_rows):
        raise ValueError("duplicate parent stability rows")
    parent_gate_pass = {
        molecule_id
        for molecule_id, row in parent_by_id.items()
        if str(row["parent_gate_failures"]) == ""
    }
    if len(parent_gate_pass) != int(inventory["parent_gate_pass_count"]):
        raise ValueError("parent-gate count drift")

    stable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_keys = set()
    for row in direction_rows:
        key = (str(row["molecule_id"]), int(row["direction_index"]))
        if key in direction_keys:
            raise ValueError(f"duplicate direction stability row: {key}")
        direction_keys.add(key)
        if key[0] in parent_gate_pass and str(row["stable"]) == "True":
            stable[key[0]].append(
                {
                    "direction_index": key[1],
                    "direction_kind": str(row["direction_kind"]),
                }
            )
    if sum(len(rows) for rows in stable.values()) != int(
        inventory["parent_gate_and_vector_stable_direction_count"]
    ):
        raise ValueError("stable vector-direction count drift")

    q59 = json.loads(Path(inputs["q59_direction_split_manifest"]).read_text())
    if (
        q59.get("validation_accessed") is not False
        or q59.get("test100_accessed") is not False
    ):
        raise ValueError("q59 inventory opened validation/Test100")
    q59_parents = {str(row["molecule_id"]): row for row in q59["parents"]}
    feature_audit = json.loads(Path(inputs["feature_jet_manifest"]).read_text())
    if feature_audit.get("test100_accessed") is not False:
        raise ValueError("feature audit opened Test100")
    feature_by_id = {
        str(row["molecule_id"]): row for row in feature_audit["entries"]
    }

    minimum = int(inventory["minimum_stable_directions_per_parent"])
    candidate_ids = {
        molecule_id
        for molecule_id, rows in stable.items()
        if len(rows) >= minimum
    }
    if len(candidate_ids) != int(inventory["candidate_parent_count"]):
        raise ValueError("vector-stable candidate-parent count drift")
    if sum(len(stable[value]) for value in candidate_ids) != int(
        inventory["candidate_stable_direction_count"]
    ):
        raise ValueError("candidate stable-direction count drift")
    if not candidate_ids <= set(q59_parents) or not candidate_ids <= set(feature_by_id):
        raise ValueError("vector candidates lack audited q59 feature jets")

    replay_rows = [
        row
        for row in _read_csv(Path(inputs["replay_baseline_success_csv"]))
        if int(row["sample_id"]) == 0
    ]
    replay_by_id = {str(row["molecule_id"]): row for row in replay_rows}
    if len(replay_by_id) != len(replay_rows):
        raise ValueError("duplicate sample0 replay rows")

    bins = list(protocol["selection"]["atom_count_bins"])
    seed = int(protocol["selection"]["seed"])
    candidates = []
    for molecule_id in sorted(candidate_ids):
        row = q59_parents[molecule_id]
        with np.load(row["sidecar"]) as payload:
            atomic_numbers = np.asarray(payload["atomic_numbers"], dtype=np.int64)
        natoms = int(atomic_numbers.size)
        composition = _composition(atomic_numbers)
        atom_bin = _atom_count_bin(natoms, bins)
        candidates.append(
            {
                "molecule_id": molecule_id,
                "natoms": natoms,
                "composition": composition,
                "atom_count_bin": atom_bin,
                "stratum": f"{atom_bin}:{composition}",
            }
        )
    selected = _round_robin_select(
        candidates, int(inventory["selected_parent_count"]), seed
    )

    manifest_parents = []
    source_tasks = []
    kind_role_counts: Counter[str] = Counter()
    for candidate in selected:
        molecule_id = str(candidate["molecule_id"])
        directions = sorted(
            stable[molecule_id],
            key=lambda row: int(row["direction_index"]),
        )
        heldout_index = min(
            directions,
            key=lambda row: _rank(
                seed,
                "direction",
                molecule_id,
                row["direction_index"],
                row["direction_kind"],
            ),
        )["direction_index"]
        direction_manifest = []
        for row in directions:
            role = (
                "heldout"
                if int(row["direction_index"]) == int(heldout_index)
                else "train"
            )
            kind_role_counts[f"{role}:{row['direction_kind']}"] += 1
            direction_manifest.append({**row, "role": role})

        qrow = q59_parents[molecule_id]
        feature = feature_by_id[molecule_id]
        replay = replay_by_id.get(molecule_id)
        if replay is None or replay.get("strict_converged") != "True":
            raise ValueError(f"missing PBE base label provenance: {molecule_id}")
        summary_path = Path(replay["summary_path"])
        if _sha256(summary_path) != str(replay["summary_sha256"]):
            raise ValueError(f"base summary hash drift: {molecule_id}")
        base_summary = json.loads(summary_path.read_text())
        label_path = Path(base_summary["label_path"])
        label_sha256 = str(base_summary["label_sha256"])
        manifest_parents.append(
            {
                **candidate,
                "directions": direction_manifest,
                "feature_jet": str(feature["artifact"]),
                "feature_jet_sha256": str(feature["artifact_sha256"]),
                "pbe_base_array": str(qrow["baseline_array"]),
                "pbe_base_array_sha256": str(qrow["baseline_array_sha256"]),
                "pbe_hessian": str(qrow["pbe_hessian"]),
                "pbe_hessian_sha256": str(qrow["pbe_hessian_sha256"]),
                "label_path": label_path.resolve().as_posix(),
                "label_sha256": label_sha256,
            }
        )
        source_tasks.append(
            {
                "molecule_id": molecule_id,
                "sample_id": 0,
                "label_path": label_path.resolve().as_posix(),
                "label_sha256": label_sha256,
            }
        )

    train_direction_count = sum(
        row["role"] == "train"
        for parent in manifest_parents
        for row in parent["directions"]
    )
    heldout_direction_count = sum(
        row["role"] == "heldout"
        for parent in manifest_parents
        for row in parent["directions"]
    )
    if heldout_direction_count != len(manifest_parents):
        raise ValueError("held-direction count drift")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_tasks_path = output_dir / "source_base_tasks.csv"
    _write_csv(source_tasks_path, source_tasks)
    manifest = {
        "definition": protocol["scope"],
        "protocol": protocol_path.resolve().as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "selection_seed": seed,
        "candidate_parent_count": len(candidates),
        "selected_parent_count": len(manifest_parents),
        "train_direction_count": train_direction_count,
        "heldout_direction_count": heldout_direction_count,
        "selected_stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in manifest_parents).items())
        ),
        "kind_role_counts": dict(sorted(kind_role_counts.items())),
        "source_checkpoint": str(inputs["source_checkpoint"]),
        "source_checkpoint_sha256": str(inputs["source_checkpoint_sha256"]),
        "source_run_spec": str(inputs["source_run_spec"]).strip(),
        "source_base_tasks": source_tasks_path.resolve().as_posix(),
        "source_base_tasks_sha256": _sha256(source_tasks_path),
        "parents": manifest_parents,
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
        if key not in {"parents", "definition"}
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
    prepare(args.protocol, args.output_dir)
