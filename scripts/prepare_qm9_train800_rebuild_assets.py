#!/usr/bin/env python3
"""Build and verify an isolated train-only QM9 dataset from a frozen label CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import zarr


TRANSFORM = "labels_local_frames_global_symmetric_natrep"
STATISTICS = (
    "dataset_statistics/"
    "dataset_statistics_labels_local_frames_global_symmetric_natrep_"
    "e_kin_plus_xc.zarr"
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"label_path", "label_sha256", "molecule_id", "sample_id"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Unexpected train CSV schema: {set(rows[0]) if rows else set()}")
    if len(rows) != 3200:
        raise ValueError(f"Expected 3200 train geometries, found {len(rows)}")
    parents = Counter(row["molecule_id"] for row in rows)
    samples = Counter(row["sample_id"] for row in rows)
    if len(parents) != 800 or set(parents.values()) != {4}:
        raise ValueError("Frozen train CSV must contain 800 parents with four geometries each")
    if samples != Counter({"0": 800, "1": 800, "2": 800, "3": 800}):
        raise ValueError(f"Unexpected sample-id counts: {samples}")
    return sorted(rows, key=lambda row: (row["molecule_id"], int(row["sample_id"])))


def write_files_from(args: argparse.Namespace) -> None:
    rows = _rows(args.source_csv)
    source_root = args.source_dataset_root.resolve()
    paths = ["dataset_info.yaml"]
    paths.extend(
        f"labels/{Path(row['label_path']).name}" for row in rows
    )
    paths.extend(
        f"{TRANSFORM}/{Path(row['label_path']).name}" for row in rows
    )
    statistics_root = source_root / STATISTICS
    paths.extend(
        path.relative_to(source_root).as_posix()
        for path in sorted(statistics_root.rglob("*"))
        if path.is_file()
    )
    missing = [relative for relative in paths if not (source_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source assets; first={missing[:3]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(paths) + "\n")
    print(json.dumps({"files": len(paths), "output": str(args.output)}, sort_keys=True))


def write_raw_manifest(args: argparse.Namespace) -> None:
    rows = _rows(args.source_csv)
    parent_ids = sorted({row["molecule_id"] for row in rows})
    raw_dir = args.raw_dir.resolve()
    expected = {f"dsgdb9nsd_{int(molecule_id):06d}.xyz" for molecule_id in parent_ids}
    actual = {path.name for path in raw_dir.glob("dsgdb9nsd_*.xyz")}
    if actual != expected:
        raise ValueError(
            f"Raw train-only subset drift: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )
    file_rows = [
        {
            "molecule_id": molecule_id,
            "path": str(raw_dir / f"dsgdb9nsd_{int(molecule_id):06d}.xyz"),
            "sha256": _sha256(raw_dir / f"dsgdb9nsd_{int(molecule_id):06d}.xyz"),
        }
        for molecule_id in parent_ids
    ]
    digest = hashlib.sha256()
    for row in file_rows:
        digest.update(
            f"{row['molecule_id']}\0{row['sha256']}\n".encode()
        )
    manifest = {
        "schema_version": 1,
        "source_csv": str(args.source_csv),
        "source_csv_sha256": _sha256(args.source_csv),
        "parent_count": len(parent_ids),
        "raw_dir": str(raw_dir),
        "raw_file_index_sha256": digest.hexdigest(),
        "validation_accessed": False,
        "test100_accessed": False,
        "files": file_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "manifest_sha256": _sha256(args.output),
                "parent_count": len(parent_ids),
                "raw_file_index_sha256": digest.hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_entries(
    rows: list[dict[str, str]],
    dataset_root: Path,
    dataset_name: str,
    *,
    label_hash_policy: str,
) -> tuple[list[tuple[str, str, int]], list[dict[str, str]], str, str]:
    entries = []
    label_records = []
    raw_hashes = hashlib.sha256()
    transformed_hashes = hashlib.sha256()
    for row in rows:
        name = Path(row["label_path"]).name
        expected_name = f"{row['molecule_id']}.{int(row['sample_id']):07d}.zarr.zip"
        if name != expected_name:
            raise ValueError(f"Filename identity drift: {name} != {expected_name}")
        raw_path = dataset_root / "labels" / name
        actual_raw_hash = _sha256(raw_path)
        source_hash_matches = actual_raw_hash == row["label_sha256"]
        if label_hash_policy == "exact" and not source_hash_matches:
            raise ValueError(
                f"Raw label hash mismatch for {name}: "
                f"{actual_raw_hash} != {row['label_sha256']}"
            )
        root = zarr.open(raw_path, mode="r")
        if "metadata/reference/source_molecule_id" in root:
            source_molecule_id = int(root["metadata/reference/source_molecule_id"][()])
            sample_id = int(root["metadata/reference/sample_id"][()])
            if source_molecule_id != int(row["molecule_id"]):
                raise ValueError(f"Molecule identity drift in {name}")
            if sample_id != int(row["sample_id"]):
                raise ValueError(f"Sample identity drift in {name}")
        n_scf_steps = int(root["of_labels/n_scf_steps"][()])
        if n_scf_steps < 2:
            raise ValueError(f"{name} has only {n_scf_steps} SCF steps")
        transformed_path = dataset_root / TRANSFORM / name
        transformed_hash = (
            _sha256(transformed_path) if transformed_path.is_file() else ""
        )
        raw_hashes.update(f"{name}\0{actual_raw_hash}\n".encode())
        if transformed_hash:
            transformed_hashes.update(f"{name}\0{transformed_hash}\n".encode())
        entries.append((dataset_name, name, n_scf_steps))
        label_records.append(
            {
                "molecule_id": row["molecule_id"],
                "sample_id": row["sample_id"],
                "filename": name,
                "source_label_sha256": row["label_sha256"],
                "new_label_sha256": actual_raw_hash,
                "source_hash_matches": str(source_hash_matches).lower(),
                "transformed_label_sha256": transformed_hash,
                "n_scf_steps": str(n_scf_steps),
            }
        )
    return (
        entries,
        label_records,
        raw_hashes.hexdigest(),
        transformed_hashes.hexdigest(),
    )


def _write_split(
    dataset_root: Path,
    dataset_name: str,
    entries: list[tuple[str, str, int]],
) -> tuple[Path, int]:
    usable_samples = sum(entry[2] - 1 for entry in entries)
    split = {
        "train": entries,
        "val": [],
        "test": [],
        "sizes": {"train": usable_samples, "val": 0, "test": 0},
        "train_only": True,
        "validation_access_allowed": False,
        "test_access_allowed": False,
    }
    split_path = dataset_root / "split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(split, handle, protocol=4)
    return split_path, usable_samples


def _apply_training_sample_budget(
    entries: list[tuple[str, str, int]],
    *,
    expected_samples: int | None,
    seed: str,
) -> tuple[list[tuple[str, str, int]], int, list[dict[str, int | str]]]:
    available_samples = sum(entry[2] - 1 for entry in entries)
    if expected_samples is None or expected_samples == available_samples:
        return list(entries), available_samples, []
    if expected_samples > available_samples:
        raise ValueError(
            f"Training sample budget {expected_samples} exceeds "
            f"{available_samples} available samples"
        )
    if expected_samples < len(entries):
        raise ValueError("Training sample budget would remove entire geometries")

    adjusted = list(entries)
    trim_count = available_samples - expected_samples
    candidates = [
        index for index, (_, _, n_scf_steps) in enumerate(adjusted)
        if n_scf_steps > 2
    ]
    if trim_count > len(candidates):
        raise ValueError(
            "Training sample budget requires more than one terminal trim per geometry"
        )
    ranked = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"{seed}\0{adjusted[index][1]}".encode()
        ).hexdigest(),
    )
    trimmed = []
    for index in ranked[:trim_count]:
        dataset_name, filename, n_scf_steps = adjusted[index]
        adjusted[index] = (dataset_name, filename, n_scf_steps - 1)
        trimmed.append(
            {
                "filename": filename,
                "physical_n_scf_steps": n_scf_steps,
                "registered_n_scf_steps": n_scf_steps - 1,
                "omitted_terminal_scf_iteration": n_scf_steps - 1,
            }
        )
    selected_samples = sum(entry[2] - 1 for entry in adjusted)
    if selected_samples != expected_samples:
        raise RuntimeError(
            f"Sample-budget implementation drift: {selected_samples} "
            f"!= {expected_samples}"
        )
    return adjusted, available_samples, sorted(
        trimmed, key=lambda row: str(row["filename"])
    )


def _load_and_validate_train_only_split(
    split_path: Path,
    entries: list[tuple[str, str, int]],
    usable_samples: int,
) -> dict:
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    expected_sizes = {"train": usable_samples, "val": 0, "test": 0}
    if split.get("train") != entries:
        raise ValueError("Registered train split does not match rebuilt entries")
    if split.get("val") != [] or split.get("test") != []:
        raise ValueError("Clean rebuild split must not expose validation or test entries")
    if split.get("sizes") != expected_sizes:
        raise ValueError(
            f"Unexpected train-only split sizes: {split.get('sizes')} != {expected_sizes}"
        )
    if split.get("train_only") is not True:
        raise ValueError("Clean rebuild split must be marked train_only")
    if split.get("validation_access_allowed") is not False:
        raise ValueError("Validation access must remain disabled")
    if split.get("test_access_allowed") is not False:
        raise ValueError("Test access must remain disabled")
    return split


def write_split(args: argparse.Namespace) -> None:
    rows = _rows(args.source_csv)
    dataset_root = args.dataset_root.resolve()
    entries, label_records, raw_index_hash, _ = _build_entries(
        rows,
        dataset_root,
        args.dataset_name,
        label_hash_policy=args.label_hash_policy,
    )
    entries, available_samples, trimmed = _apply_training_sample_budget(
        entries,
        expected_samples=getattr(args, "expected_usable_training_samples", None),
        seed=getattr(args, "sample_budget_seed", "train-only-sample-budget-v1"),
    )
    split_path, usable_samples = _write_split(
        dataset_root, args.dataset_name, entries
    )
    provenance = dataset_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    records_path = provenance / "recomputed_label_index.csv"
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_records[0]))
        writer.writeheader()
        writer.writerows(label_records)
    record = {
        "dataset_name": args.dataset_name,
        "label_hash_policy": args.label_hash_policy,
        "geometry_count": len(entries),
        "available_training_samples": available_samples,
        "usable_training_samples": usable_samples,
        "sample_budget_trim_count": len(trimmed),
        "split": str(split_path),
        "split_sha256": _sha256(split_path),
        "raw_label_index_sha256": raw_index_hash,
        "recomputed_label_index": str(records_path),
        "recomputed_label_index_sha256": _sha256(records_path),
        "source_hash_match_count": sum(
            row["source_hash_matches"] == "true" for row in label_records
        ),
        "validation_entries": 0,
        "test_entries": 0,
    }
    output = provenance / "split_registration.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**record, "registration": str(output)}, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    rows = _rows(args.source_csv)
    dataset_root = args.dataset_root.resolve()
    copied_csv = dataset_root / "provenance" / "train800_replay_labels.source.csv"
    copied_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_csv, copied_csv)

    entries, label_records, raw_index_hash, transformed_index_hash = _build_entries(
        rows,
        dataset_root,
        args.dataset_name,
        label_hash_policy=args.label_hash_policy,
    )
    if any(not row["transformed_label_sha256"] for row in label_records):
        raise FileNotFoundError("One or more transformed labels are missing")
    entries, available_samples, trimmed = _apply_training_sample_budget(
        entries,
        expected_samples=getattr(args, "expected_usable_training_samples", None),
        seed=getattr(args, "sample_budget_seed", "train-only-sample-budget-v1"),
    )
    split_path, usable_samples = _write_split(dataset_root, args.dataset_name, entries)
    split = _load_and_validate_train_only_split(
        split_path, entries, usable_samples
    )
    parent_ids = sorted({row["molecule_id"] for row in rows})
    parent_path = dataset_root / "provenance" / "train800_parent_ids.txt"
    parent_path.write_text("\n".join(parent_ids) + "\n")
    sample_budget_path = (
        dataset_root / "provenance" / "training_sample_budget.json"
    )
    sample_budget_path.write_text(
        json.dumps(
            {
                "policy": "hash_ranked_terminal_scf_iteration_trim",
                "seed": getattr(
                    args, "sample_budget_seed", "train-only-sample-budget-v1"
                ),
                "available_training_samples": available_samples,
                "registered_training_samples": usable_samples,
                "trim_count": len(trimmed),
                "trimmed": trimmed,
                "validation_accessed": False,
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    records_path = dataset_root / "provenance" / "recomputed_label_index.csv"
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_records[0]))
        writer.writeheader()
        writer.writerows(label_records)
    manifest = {
        "schema_version": 1,
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "source_csv": str(args.source_csv),
        "source_csv_sha256": _sha256(args.source_csv),
        "copied_source_csv": str(copied_csv),
        "copied_source_csv_sha256": _sha256(copied_csv),
        "parent_count": len(parent_ids),
        "geometry_count": len(entries),
        "sample_counts": dict(sorted(Counter(row["sample_id"] for row in rows).items())),
        "available_training_samples": available_samples,
        "usable_training_samples": usable_samples,
        "sample_budget_policy": "hash_ranked_terminal_scf_iteration_trim",
        "sample_budget_trim_count": len(trimmed),
        "sample_budget_sha256": _sha256(sample_budget_path),
        "train_entries": len(split["train"]),
        "validation_entries": len(split["val"]),
        "test_entries": len(split["test"]),
        "split_sha256": _sha256(split_path),
        "parent_ids_sha256": _sha256(parent_path),
        "label_hash_policy": args.label_hash_policy,
        "source_hash_match_count": sum(
            row["source_hash_matches"] == "true" for row in label_records
        ),
        "source_hash_mismatch_count": sum(
            row["source_hash_matches"] == "false" for row in label_records
        ),
        "raw_label_index_sha256": raw_index_hash,
        "transformed_label_index_sha256": transformed_index_hash,
        "recomputed_label_index_sha256": _sha256(records_path),
        "dataset_info_sha256": _sha256(dataset_root / "dataset_info.yaml"),
        "dataset_statistics_tree_sha256": _tree_hash(dataset_root / STATISTICS),
        "validation_accessed": False,
        "test_accessed": False,
    }
    manifest_path = dataset_root / "provenance" / "train_only_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    files = subparsers.add_parser("files-from")
    files.add_argument("--source-csv", type=Path, required=True)
    files.add_argument("--source-dataset-root", type=Path, required=True)
    files.add_argument("--output", type=Path, required=True)
    raw = subparsers.add_parser("raw-manifest")
    raw.add_argument("--source-csv", type=Path, required=True)
    raw.add_argument("--raw-dir", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    split = subparsers.add_parser("write-split")
    split.add_argument("--source-csv", type=Path, required=True)
    split.add_argument("--dataset-root", type=Path, required=True)
    split.add_argument("--dataset-name", required=True)
    split.add_argument(
        "--label-hash-policy", choices=("exact", "regenerated"), default="exact"
    )
    split.add_argument("--expected-usable-training-samples", type=int)
    split.add_argument(
        "--sample-budget-seed", default="train-only-sample-budget-v1"
    )
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--source-csv", type=Path, required=True)
    finish.add_argument("--dataset-root", type=Path, required=True)
    finish.add_argument("--dataset-name", required=True)
    finish.add_argument(
        "--label-hash-policy", choices=("exact", "regenerated"), default="exact"
    )
    finish.add_argument("--expected-usable-training-samples", type=int)
    finish.add_argument(
        "--sample-budget-seed", default="train-only-sample-budget-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "files-from":
        write_files_from(args)
    elif args.command == "raw-manifest":
        write_raw_manifest(args)
    elif args.command == "write-split":
        write_split(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
