"""Build sidecar leakage-safe splits without modifying archived split files."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import yaml


PARTITIONS = ("train", "val", "test")
RANK = {partition: rank for rank, partition in enumerate(PARTITIONS)}
DATASETS = (
    "QM9_perturbed_fock",
    "QMUGSBin0_perturbed_fock",
    "QMUGSBin0QM9_perturbed_fock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _rows_and_assignment(split: dict) -> tuple[dict, dict]:
    rows = {}
    assignment = {}
    for partition in PARTITIONS:
        for source, filename, iterations in split[partition]:
            label = (str(source), str(filename))
            if label in assignment:
                raise ValueError(f"Duplicate archived split label: {label}")
            assignment[label] = partition
            rows[label] = [label[0], label[1], int(iterations)]
    return rows, assignment


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _make_group_safe(
    rows: dict,
    initial_assignment: dict,
    metadata: dict,
) -> tuple[dict, dict]:
    labels = sorted(rows)
    union_find = _UnionFind(labels)
    first_by_key = {}
    missing_metadata = []
    for label in labels:
        item = metadata.get(label)
        if item is None:
            missing_metadata.append(label)
            continue
        for key_name in ("canonical_smiles", "parent"):
            value = item.get(key_name)
            if value is None:
                continue
            key = (key_name, value)
            if key in first_by_key:
                union_find.union(label, first_by_key[key])
            else:
                first_by_key[key] = label
    if missing_metadata:
        raise ValueError(f"Missing metadata for {len(missing_metadata)} labels")

    groups = collections.defaultdict(list)
    for label in labels:
        groups[union_find.find(label)].append(label)

    safe_assignment = {}
    group_sizes = collections.Counter()
    for members in groups.values():
        target = max((initial_assignment[label] for label in members), key=RANK.get)
        group_sizes[str(len(members))] += 1
        for label in members:
            safe_assignment[label] = target

    split = {partition: [] for partition in PARTITIONS}
    transitions = collections.Counter()
    for label in labels:
        old = initial_assignment[label]
        new = safe_assignment[label]
        transitions[f"{old}->{new}"] += 1
        split[new].append(rows[label])
    split["sizes"] = {
        partition: sum(row[2] for row in split[partition]) for partition in PARTITIONS
    }

    # Independent postcondition: neither molecular identity key may span splits.
    key_partitions = collections.defaultdict(set)
    for label, partition in safe_assignment.items():
        item = metadata[label]
        for key_name in ("canonical_smiles", "parent"):
            if item.get(key_name) is not None:
                key_partitions[(key_name, item[key_name])].add(partition)
    cross_partition_keys = [
        {"key_type": key[0], "key": key[1], "partitions": sorted(partitions)}
        for key, partitions in key_partitions.items()
        if len(partitions) > 1
    ]
    if cross_partition_keys:
        raise AssertionError(
            f"Group-safe split postcondition failed for {len(cross_partition_keys)} keys"
        )

    report = {
        "cross_partition_identity_keys": 0,
        "group_size_distribution": dict(sorted(group_sizes.items(), key=lambda item: int(item[0]))),
        "partitions": {
            partition: {
                "entries": len(split[partition]),
                "samples": split["sizes"][partition],
                "sources": dict(
                    sorted(collections.Counter(row[0] for row in split[partition]).items())
                ),
            }
            for partition in PARTITIONS
        },
        "transitions": dict(sorted(transitions.items())),
    }
    return split, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-jsonl-gz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if str(data_root).startswith("/export/scratch/ialgroup"):
        raise RuntimeError(f"Refusing forbidden effective DFT_DATA path: {data_root}")

    metadata = {}
    with gzip.open(args.metadata_jsonl_gz, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            label = (item.pop("source"), item.pop("filename"))
            metadata[label] = item

    archived = {
        name: _load_split(data_root / name / "split.pkl") for name in DATASETS
    }
    archived_rows = {}
    archived_assignments = {}
    for name, split in archived.items():
        archived_rows[name], archived_assignments[name] = _rows_and_assignment(split)

    # The archived combined split moved historical QM9 validation into train and
    # historical QM9 test into validation. Reconstruct its initial assignment
    # from the two source-domain splits, retaining only the one extra QMUGS-large
    # label in its archived test. This is identity-based, never metric-based.
    combo = "QMUGSBin0QM9_perturbed_fock"
    combo_initial = {}
    for label in archived_rows[combo]:
        if label in archived_assignments["QM9_perturbed_fock"]:
            combo_initial[label] = archived_assignments["QM9_perturbed_fock"][label]
        elif label in archived_assignments["QMUGSBin0_perturbed_fock"]:
            combo_initial[label] = archived_assignments["QMUGSBin0_perturbed_fock"][label]
        else:
            combo_initial[label] = archived_assignments[combo][label]

    reports = {}
    for name in DATASETS:
        initial = combo_initial if name == combo else archived_assignments[name]
        safe_split, report = _make_group_safe(archived_rows[name], initial, metadata)
        output_dir = args.output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = output_dir / "split.yaml"
        pickle_path = output_dir / "split.pkl"
        yaml_path.write_text(yaml.safe_dump(safe_split, sort_keys=False))
        with pickle_path.open("wb") as handle:
            pickle.dump(safe_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
        report.update(
            {
                "archived_split_pickle": str((data_root / name / "split.pkl").resolve()),
                "policy": (
                    "reconstruct source-domain assignment, then promote connected "
                    "canonical-SMILES/parent groups train<val<test"
                    if name == combo
                    else "promote connected canonical-SMILES/parent groups train<val<test"
                ),
                "safe_split_pickle": str(pickle_path.resolve()),
                "safe_split_pickle_sha256": _sha256(pickle_path),
                "safe_split_yaml": str(yaml_path.resolve()),
                "safe_split_yaml_sha256": _sha256(yaml_path),
            }
        )
        reports[name] = report

    result = {
        "data_root": str(data_root),
        "datasets": reports,
        "metadata_entries": len(metadata),
        "metadata_jsonl_gz": str(args.metadata_jsonl_gz.resolve()),
        "status": "passed",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
