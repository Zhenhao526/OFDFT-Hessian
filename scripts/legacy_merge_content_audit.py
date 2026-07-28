"""Merge sharded legacy content audits and detect exact configuration leakage."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


PARTITIONS = ("train", "val", "test")
DATASETS = (
    "QM9_perturbed_fock",
    "QMUGSBin0_perturbed_fock",
    "QMUGSBin0QM9_perturbed_fock",
)
SOURCES = ("QM9_perturbed_fock", "QMUGS_perturbed_fock", "QMUGS")
HASH_FIELDS = (
    "geometry_sha256",
    "distance_invariant_sha256",
    "configuration_trajectory_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "quantiles": {
            str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
    }


def _assignments(data_root: Path) -> dict[str, dict[tuple[str, str], str]]:
    result = {}
    for name in DATASETS:
        with (data_root / name / "split.pkl").open("rb") as handle:
            split = pickle.load(handle)
        assignment = {}
        for partition in PARTITIONS:
            for source, filename, _ in split[partition]:
                label = (str(source), str(filename))
                if label in assignment:
                    raise ValueError(f"Duplicate split label in {name}: {label}")
                assignment[label] = partition
        result[name] = assignment
    return result


def _duplicate_memberships(members, assignments: dict) -> dict:
    result = {}
    member_labels = [member[0] if isinstance(member, tuple) and len(member) == 2 and isinstance(member[1], list) else member for member in members]
    for dataset, assignment in assignments.items():
        selected = [label for label in member_labels if label in assignment]
        if len(set(selected)) < 2:
            continue
        partitions = collections.Counter(assignment[label] for label in set(selected))
        result[dataset] = {
            "cross_partition": len(partitions) > 1,
            "partitions": dict(sorted(partitions.items())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--duplicates-jsonl-gz", type=Path, required=True)
    parser.add_argument("--anomalies-jsonl-gz", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    data_root = args.data_root.resolve()
    if str(data_root).startswith("/export/scratch/ialgroup"):
        raise RuntimeError(f"Refusing forbidden effective DFT_DATA path: {data_root}")
    audit_root = args.audit_root.resolve()
    assignments = _assignments(data_root)

    expected_inventory = {
        source: sum(
            1
            for entry in os.scandir(data_root / source / "labels")
            if entry.is_file() and entry.name.endswith(".zarr.zip")
        )
        for source in SOURCES
    }
    source_stats = {
        source: {
            "anomaly_counts": collections.Counter(),
            "archive_bytes": 0,
            "atom_counts": collections.Counter(),
            "composition_counts": collections.Counter(),
            "electron_absolute_final": [],
            "electron_absolute_max_labeled": [],
            "electron_relative_final": [],
            "electron_relative_max_labeled": [],
            "failures": [],
            "has_energy_steps": 0,
            "processed": 0,
            "schema_counts": collections.Counter(),
            "scf_steps": 0,
        }
        for source in SOURCES
    }
    labels_seen = set()
    label_by_id = []
    simple_hashes = {
        field: collections.defaultdict(list) for field in HASH_FIELDS
    }
    artifact_records = {}

    database_path = audit_root / "configuration_hashes.merge.sqlite3"
    if database_path.exists():
        database_path.unlink()
    database = sqlite3.connect(database_path)
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute("PRAGMA temp_store=FILE")
    database.execute("CREATE TABLE configurations (hash BLOB, label_id INTEGER, step INTEGER)")
    pending_configurations = []

    args.anomalies_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    partial_anomalies = args.anomalies_jsonl_gz.with_suffix(
        args.anomalies_jsonl_gz.suffix + ".partial"
    )
    with gzip.open(partial_anomalies, "wt") as anomaly_handle:
        for source in SOURCES:
            stats = source_stats[source]
            for shard in range(args.num_shards):
                summary_path = audit_root / "summaries" / f"{source}_{shard}.json"
                rows_path = audit_root / "rows" / f"{source}_{shard}.jsonl.gz"
                if not summary_path.is_file() or not rows_path.is_file():
                    raise FileNotFoundError(f"Missing shard artifacts for {source}/{shard}")
                summary = json.loads(summary_path.read_text())
                if summary["processed"] != summary["selected"]:
                    raise RuntimeError(f"Incomplete shard: {summary_path}")
                stats["failures"].extend(summary["failures"])
                stats["failures"].extend(summary["metadata_failures"])
                artifact_records[str(summary_path.relative_to(audit_root))] = {
                    "bytes": summary_path.stat().st_size,
                    "sha256": _sha256(summary_path),
                }
                artifact_records[str(rows_path.relative_to(audit_root))] = {
                    "bytes": rows_path.stat().st_size,
                    "sha256": _sha256(rows_path),
                }
                with gzip.open(rows_path, "rt") as handle:
                    for line in handle:
                        row = json.loads(line)
                        label = (source, row["filename"])
                        if label in labels_seen:
                            raise RuntimeError(f"Duplicate audited label: {label}")
                        labels_seen.add(label)
                        label_id = len(label_by_id)
                        label_by_id.append(label)
                        stats["processed"] += 1
                        stats["archive_bytes"] += row["archive_bytes"]
                        stats["scf_steps"] += row["scf_steps"]
                        stats["has_energy_steps"] += row["has_energy_steps"]
                        stats["atom_counts"][str(row["n_atoms"])] += 1
                        stats["composition_counts"][
                            json.dumps(row["composition"], sort_keys=True)
                        ] += 1
                        stats["schema_counts"][row["schema_sha256"]] += 1
                        stats["anomaly_counts"].update(row["anomalies"])
                        stats["electron_absolute_final"].append(
                            row["electron_count_error_final"]
                        )
                        stats["electron_absolute_max_labeled"].append(
                            row["electron_count_error_max_labeled"]
                        )
                        stats["electron_relative_final"].append(
                            row["electron_count_relative_error_final"]
                        )
                        stats["electron_relative_max_labeled"].append(
                            row["electron_count_relative_error_max_labeled"]
                        )
                        if row["anomalies"]:
                            anomaly_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        for field in HASH_FIELDS:
                            simple_hashes[field][row[field]].append(label)
                        pending_configurations.extend(
                            (bytes.fromhex(value), label_id, step)
                            for step, value in enumerate(row["configuration_step_sha256"])
                        )
                        if len(pending_configurations) >= 100_000:
                            database.executemany(
                                "INSERT INTO configurations VALUES (?, ?, ?)",
                                pending_configurations,
                            )
                            pending_configurations.clear()
                print(
                    json.dumps(
                        {
                            "elapsed_s": time.perf_counter() - started,
                            "merged": stats["processed"],
                            "shard": shard,
                            "source": source,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    partial_anomalies.replace(args.anomalies_jsonl_gz)
    if pending_configurations:
        database.executemany("INSERT INTO configurations VALUES (?, ?, ?)", pending_configurations)
    database.commit()
    # Cover label_id in the index so exact cross-label duplicate detection does
    # not perform millions of random table lookups after grouping by hash.
    database.execute(
        "CREATE INDEX configurations_hash ON configurations(hash, label_id)"
    )
    database.commit()

    duplicate_counts = {
        dataset: {
            field: {"cross_partition": 0, "within_partition": 0}
            for field in (*HASH_FIELDS, "configuration_step_sha256")
        }
        for dataset in DATASETS
    }
    args.duplicates_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    partial_duplicates = args.duplicates_jsonl_gz.with_suffix(
        args.duplicates_jsonl_gz.suffix + ".partial"
    )
    with gzip.open(partial_duplicates, "wt") as duplicate_handle:
        for field, values in simple_hashes.items():
            for value, members in values.items():
                unique_members = sorted(set(members))
                if len(unique_members) < 2:
                    continue
                memberships = _duplicate_memberships(unique_members, assignments)
                for dataset, membership in memberships.items():
                    kind = "cross_partition" if membership["cross_partition"] else "within_partition"
                    duplicate_counts[dataset][field][kind] += 1
                duplicate_handle.write(
                    json.dumps(
                        {
                            "hash": value,
                            "hash_type": field,
                            "members": [f"{x[0]}/{x[1]}" for x in unique_members],
                            "split_memberships": memberships,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        duplicate_hashes = database.execute(
            "SELECT hash FROM configurations GROUP BY hash HAVING MIN(label_id) != MAX(label_id)"
        )
        for (value,) in duplicate_hashes:
            member_rows = database.execute(
                "SELECT label_id, GROUP_CONCAT(step) FROM configurations "
                "WHERE hash=? GROUP BY label_id ORDER BY label_id",
                (value,),
            ).fetchall()
            members = [
                (label_by_id[label_id], [int(step) for step in steps.split(",")])
                for label_id, steps in member_rows
            ]
            memberships = _duplicate_memberships(members, assignments)
            for dataset, membership in memberships.items():
                kind = "cross_partition" if membership["cross_partition"] else "within_partition"
                duplicate_counts[dataset]["configuration_step_sha256"][kind] += 1
            duplicate_handle.write(
                json.dumps(
                    {
                        "hash": bytes(value).hex(),
                        "hash_type": "configuration_step_sha256",
                        "members": [
                            {"label": f"{label[0]}/{label[1]}", "steps": steps}
                            for label, steps in members
                        ],
                        "split_memberships": memberships,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    partial_duplicates.replace(args.duplicates_jsonl_gz)
    database.close()
    database_path.unlink()

    errors = []
    rendered_sources = {}
    for source, stats in source_stats.items():
        if stats["processed"] != expected_inventory[source]:
            errors.append(
                f"{source}: processed {stats['processed']} != inventory {expected_inventory[source]}"
            )
        if stats["failures"]:
            errors.append(f"{source}: {len(stats['failures'])} failures")
        if stats["anomaly_counts"]:
            errors.append(f"{source}: {sum(stats['anomaly_counts'].values())} anomalies")
        rendered_sources[source] = {
            "anomaly_counts": dict(sorted(stats["anomaly_counts"].items())),
            "archive_bytes": stats["archive_bytes"],
            "atom_counts": dict(sorted(stats["atom_counts"].items(), key=lambda x: int(x[0]))),
            "composition_counts": dict(sorted(stats["composition_counts"].items())),
            "electron_count_absolute_error_final": _summary(stats["electron_absolute_final"]),
            "electron_count_absolute_error_max_labeled": _summary(
                stats["electron_absolute_max_labeled"]
            ),
            "electron_count_relative_error_final": _summary(stats["electron_relative_final"]),
            "electron_count_relative_error_max_labeled": _summary(
                stats["electron_relative_max_labeled"]
            ),
            "expected_inventory": expected_inventory[source],
            "failures": stats["failures"],
            "has_energy_steps": stats["has_energy_steps"],
            "processed": stats["processed"],
            "schema_counts": dict(sorted(stats["schema_counts"].items())),
            "scf_steps": stats["scf_steps"],
        }

    report = {
        "anomalies_jsonl_gz": str(args.anomalies_jsonl_gz.resolve()),
        "artifacts": artifact_records,
        "audit_root": str(audit_root),
        "command": [sys.executable, *sys.argv],
        "data_root": str(data_root),
        "duplicate_counts": duplicate_counts,
        "duplicates_jsonl_gz": str(args.duplicates_jsonl_gz.resolve()),
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
        "sources": rendered_sources,
        "status": "passed" if not errors else "failed",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
