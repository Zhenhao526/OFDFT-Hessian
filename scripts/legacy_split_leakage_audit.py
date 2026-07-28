"""Audit label, parent, conformer, and molecular-structure leakage in legacy splits."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import yaml
from rdkit import Chem, RDLogger


PARTITIONS = ("train", "val", "test")
DATASETS = (
    "QM9_perturbed_fock",
    "QMUGSBin0_perturbed_fock",
    "QMUGSBin0QM9_perturbed_fock",
)
QM9_ID_RE = re.compile(r"(\d{6})\.xyz$")


def _canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _qmugs_metadata(data_root: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    metadata = {}
    failures = []
    sources = (
        (data_root / "QMUGS" / "QMUGSBin0.csv", "QMUGS_perturbed_fock"),
        (data_root / "QMUGS" / "QMUGSLargeBins.csv", "QMUGS"),
    )
    for csv_path, source in sources:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                filename = f"{int(row['id']):07d}.zarr.zip"
                canonical = _canonical_smiles(row["smiles"])
                if canonical is None:
                    failures.append(
                        {"filename": filename, "reason": "invalid_smiles", "source": source}
                    )
                metadata[(source, filename)] = {
                    "canonical_smiles": canonical,
                    "conformer": f"qmugs:{row['chembl_id']}:{row['conf_id']}",
                    "parent": f"qmugs:{row['chembl_id']}",
                    "reported_atoms": int(row["atoms"]),
                    "reported_heavy_atoms": int(row["heavy_atoms"]),
                }
    return metadata, failures


def _qm9_metadata(raw_root: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    metadata = {}
    failures = []
    for entry in os.scandir(raw_root):
        if not entry.is_file() or not entry.name.endswith(".xyz"):
            continue
        match = QM9_ID_RE.search(entry.name)
        if match is None:
            failures.append({"filename": entry.name, "reason": "unparsed_qm9_filename"})
            continue
        molecule_id = int(match.group(1))
        try:
            with open(entry.path) as handle:
                lines = handle.readlines()
            atoms = int(lines[0].strip())
            smiles_line = lines[atoms + 3].split()
            smiles = smiles_line[0]
            canonical = _canonical_smiles(smiles)
        except Exception as error:  # keep the complete anomaly instead of hiding a raw-file issue
            failures.append(
                {"filename": entry.name, "reason": "parse_error", "error": repr(error)}
            )
            continue
        filename = f"{molecule_id:07d}.zarr.zip"
        if canonical is None:
            failures.append(
                {"filename": filename, "reason": "invalid_smiles", "smiles": smiles}
            )
        metadata[("QM9_perturbed_fock", filename)] = {
            "canonical_smiles": canonical,
            "conformer": f"qm9:{molecule_id:07d}",
            "parent": f"qm9:{molecule_id:07d}",
            "reported_atoms": atoms,
            "reported_heavy_atoms": None,
        }
    return metadata, failures


def _key_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _audit_dataset(
    data_root: Path,
    dataset_name: str,
    metadata: dict[tuple[str, str], dict],
    detail_handle,
) -> tuple[dict, dict[tuple[str, str], str]]:
    with (data_root / dataset_name / "split.yaml").open() as handle:
        split = yaml.safe_load(handle)

    assignment = {}
    missing_metadata = []
    by_key = {
        name: collections.defaultdict(lambda: collections.defaultdict(list))
        for name in ("label", "parent", "conformer", "canonical_smiles")
    }
    partition_entries = {}
    for partition in PARTITIONS:
        partition_entries[partition] = len(split[partition])
        for source, filename, _ in split[partition]:
            label = (str(source), str(filename))
            assignment[label] = partition
            by_key["label"][f"{label[0]}/{label[1]}"][partition].append(label)
            item = metadata.get(label)
            if item is None:
                missing_metadata.append(label)
                continue
            for key_type in ("parent", "conformer", "canonical_smiles"):
                value = item.get(key_type)
                if value is not None:
                    by_key[key_type][value][partition].append(label)

    summaries = {}
    for key_type, values in by_key.items():
        overlap_values = 0
        affected_entries = collections.Counter()
        transition_counts = collections.Counter()
        for value, partitions in values.items():
            active = sorted(partition for partition, entries in partitions.items() if entries)
            if len(active) < 2:
                continue
            overlap_values += 1
            for partition in active:
                affected_entries[partition] += len(partitions[partition])
            for left_index, left in enumerate(active):
                for right in active[left_index + 1 :]:
                    transition_counts[f"{left}<->{right}"] += 1
            detail = {
                "dataset": dataset_name,
                "key": value,
                "key_sha256": _key_digest(value),
                "key_type": key_type,
                "partitions": {
                    partition: {
                        "count": len(partitions[partition]),
                        "examples": [
                            f"{source}/{filename}"
                            for source, filename in partitions[partition][:10]
                        ],
                    }
                    for partition in active
                },
            }
            detail_handle.write(json.dumps(detail, sort_keys=True) + "\n")
        summaries[key_type] = {
            "affected_entries_by_partition": dict(sorted(affected_entries.items())),
            "overlap_values": overlap_values,
            "partition_pair_values": dict(sorted(transition_counts.items())),
            "unique_values": len(values),
        }
    return (
        {
            "missing_metadata_count": len(missing_metadata),
            "missing_metadata_examples": [f"{x[0]}/{x[1]}" for x in missing_metadata[:100]],
            "partition_entries": partition_entries,
            "within_dataset_overlaps": summaries,
        },
        assignment,
    )


def _cross_dataset_transitions(assignments: dict[str, dict]) -> dict:
    dataset_pairs = {}
    names = sorted(assignments)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = set(assignments[left]) & set(assignments[right])
            transitions = collections.Counter(
                f"{assignments[left][key]}->{assignments[right][key]}" for key in shared
            )
            dataset_pairs[f"{left} -> {right}"] = {
                "shared_labels": len(shared),
                "transitions": dict(sorted(transitions.items())),
            }
    return dataset_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qm9-raw-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-jsonl-gz", type=Path)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    data_root = args.data_root.resolve()
    qmugs, qmugs_failures = _qmugs_metadata(data_root)
    qm9, qm9_failures = _qm9_metadata(args.qm9_raw_root.resolve())
    metadata = {**qmugs, **qm9}

    if args.metadata_jsonl_gz:
        args.metadata_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.metadata_jsonl_gz, "wt") as handle:
            for (source, filename), item in sorted(metadata.items()):
                handle.write(
                    json.dumps(
                        {"filename": filename, "source": source, **item}, sort_keys=True
                    )
                    + "\n"
                )

    args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    assignments = {}
    dataset_reports = {}
    with args.details_jsonl.open("w") as detail_handle:
        for dataset_name in DATASETS:
            dataset_reports[dataset_name], assignments[dataset_name] = _audit_dataset(
                data_root, dataset_name, metadata, detail_handle
            )

    report = {
        "cross_dataset_label_transitions": _cross_dataset_transitions(assignments),
        "datasets": dataset_reports,
        "details_jsonl": str(args.details_jsonl.resolve()),
        "metadata": {
            "entries": len(metadata),
            "qm9_entries": len(qm9),
            "qm9_failures": qm9_failures,
            "qmugs_entries": len(qmugs),
            "qmugs_failures": qmugs_failures,
            "metadata_jsonl_gz": (
                str(args.metadata_jsonl_gz.resolve()) if args.metadata_jsonl_gz else None
            ),
        },
        "status": "passed" if not (qm9_failures or qmugs_failures) else "completed_with_metadata_anomalies",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
