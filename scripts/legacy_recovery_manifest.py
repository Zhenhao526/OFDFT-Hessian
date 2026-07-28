"""Create a strict, machine-readable manifest for the legacy DFT recovery.

This command never resolves or executes ``hparams_resolved.yaml``. Those files
are retained only as provenance and scanned for forbidden legacy absolute paths.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

import torch
import yaml


FORBIDDEN_PATH_RE = re.compile(r"/export/scratch/ialgroup[^\s'\"]*")
VIRTUAL_DATASETS = (
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


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "path": str(path),
        "sha256": _sha256(path),
    }


def _directory_content_record(path: Path) -> dict:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        file_sha = _sha256(child)
        size = child.stat().st_size
        digest.update(f"{relative}\t{size}\t{file_sha}\n".encode())
        files += 1
        total_bytes += size
    return {
        "bytes": total_bytes,
        "content_manifest_sha256": digest.hexdigest(),
        "files": files,
        "path": str(path),
    }


def _label_inventory(path: Path) -> dict:
    records = []
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith(".zarr.zip"):
            records.append((entry.name, entry.stat().st_size))
    records.sort()
    digest = hashlib.sha256()
    total_bytes = 0
    for name, size in records:
        digest.update(f"{name}\t{size}\n".encode())
        total_bytes += size
    numeric_ids = sorted(int(name.split(".", 1)[0]) for name, _ in records)
    return {
        "bytes": total_bytes,
        "count": len(records),
        "id_max": numeric_ids[-1] if numeric_ids else None,
        "id_min": numeric_ids[0] if numeric_ids else None,
        "path": str(path),
        "path_size_inventory_sha256": digest.hexdigest(),
    }


def _split_record(data_root: Path, dataset_name: str, errors: list[str]) -> dict:
    dataset_root = data_root / dataset_name
    split_yaml = dataset_root / "split.yaml"
    split_pickle = dataset_root / "split.pkl"
    dataset_info = dataset_root / "dataset_info.yaml"
    for required in (split_yaml, split_pickle, dataset_info):
        if not required.is_file():
            errors.append(f"Missing required file: {required}")

    with split_yaml.open() as handle:
        split = yaml.safe_load(handle)

    result = {
        "dataset_info": _file_record(dataset_info),
        "partitions": {},
        "split_pickle": _file_record(split_pickle),
        "split_yaml": _file_record(split_yaml),
        "statistics": [],
    }
    seen = set()
    split_manifest = hashlib.sha256()
    for partition in ("train", "val", "test"):
        rows = split.get(partition, [])
        sources = collections.Counter(str(row[0]) for row in rows)
        summed_samples = sum(int(row[2]) for row in rows)
        declared_samples = int(split["sizes"][partition])
        overlap = []
        missing = []
        duplicate_within = []
        local_seen = set()
        for source, filename, iterations in rows:
            key = (str(source), str(filename))
            split_manifest.update(
                f"{partition}\t{key[0]}\t{key[1]}\t{int(iterations)}\n".encode()
            )
            if key in local_seen:
                duplicate_within.append(key)
            if key in seen:
                overlap.append(key)
            local_seen.add(key)
            seen.add(key)
            label_path = data_root / key[0] / "labels" / key[1]
            if not label_path.is_file():
                missing.append(str(label_path))
        if declared_samples != summed_samples:
            errors.append(
                f"{dataset_name}/{partition}: declared samples {declared_samples} "
                f"!= summed samples {summed_samples}"
            )
        if missing:
            errors.append(f"{dataset_name}/{partition}: {len(missing)} missing labels")
        if overlap:
            errors.append(f"{dataset_name}/{partition}: {len(overlap)} prior-split overlaps")
        if duplicate_within:
            errors.append(
                f"{dataset_name}/{partition}: {len(duplicate_within)} duplicate label entries"
            )
        result["partitions"][partition] = {
            "declared_samples": declared_samples,
            "duplicate_entries": len(duplicate_within),
            "entries": len(rows),
            "missing_labels": len(missing),
            "prior_partition_overlaps": len(overlap),
            "sources": dict(sorted(sources.items())),
            "summed_samples": summed_samples,
        }
    result["split_entry_manifest_sha256"] = split_manifest.hexdigest()

    statistics_root = dataset_root / "dataset_statistics"
    if statistics_root.is_dir():
        result["statistics"] = [
            _directory_content_record(path)
            for path in sorted(statistics_root.glob("*.zarr"))
        ]
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_record(code_root: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=code_root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


def _run_records(models_root: Path, errors: list[str]) -> list[dict]:
    records = []
    for hparams in sorted((models_root / "train" / "runs").glob("*/hparams.yaml")):
        if hparams.name != "hparams.yaml":
            errors.append(f"Refusing non-portable model config: {hparams}")
            continue
        with hparams.open() as handle:
            config = yaml.safe_load(handle)
        checkpoint = hparams.parent / "checkpoints" / "last.ckpt"
        resolved = hparams.with_name("hparams_resolved.yaml")
        if not checkpoint.is_file():
            errors.append(f"Missing checkpoint: {checkpoint}")
        if not resolved.is_file():
            errors.append(f"Missing provenance config: {resolved}")
            forbidden_paths = []
        else:
            forbidden_paths = sorted(set(FORBIDDEN_PATH_RE.findall(resolved.read_text())))
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        records.append(
            {
                "checkpoint": _file_record(checkpoint),
                "dataset": data_cfg.get("dataset_name"),
                "effective_checkpoint": str(checkpoint.resolve()),
                "forbidden_resolved_paths": forbidden_paths,
                "hparams": _file_record(hparams),
                "hparams_resolved_provenance_only": _file_record(resolved),
                "model_target": model_cfg.get("_target_"),
                "net_target": model_cfg.get("net", {}).get("_target_"),
                "raw_parent_checkpoint_provenance": config.get("ckpt_path")
                or config.get("weight_ckpt_path"),
                "target_key": data_cfg.get("target_key"),
            }
        )
    if not records:
        errors.append(f"No raw hparams.yaml found below {models_root}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    models_root = args.models_root.resolve()
    restore_root = args.restore_root.resolve()
    errors: list[str] = []
    for name, path in (("DFT_DATA", data_root), ("DFT_MODELS", models_root)):
        if str(path).startswith("/export/scratch/ialgroup"):
            errors.append(f"Forbidden effective {name} path: {path}")
        os.environ[name] = str(path)

    datasets = {
        name: _split_record(data_root, name, errors) for name in VIRTUAL_DATASETS
    }
    label_sources = sorted(
        {
            source
            for dataset in datasets.values()
            for partition in dataset["partitions"].values()
            for source in partition["sources"]
        }
    )
    label_inventories = {}
    for source in label_sources:
        labels = data_root / source / "labels"
        if not labels.is_dir():
            errors.append(f"Missing label source directory: {labels}")
            continue
        label_inventories[source] = _label_inventory(labels)

    artifact_records = {}
    for path in sorted(item for item in restore_root.rglob("*") if item.is_file()):
        if path.resolve() == args.output_json.resolve():
            continue
        artifact_records[path.relative_to(restore_root).as_posix()] = _file_record(path)

    report = {
        "artifacts": artifact_records,
        "code": _git_record(code_root),
        "command": [sys.executable, *sys.argv],
        "datasets": datasets,
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "hostname": socket.gethostname(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "hydra-core",
                    "lightning",
                    "numpy",
                    "pyscf",
                    "tensorframes",
                    "torch",
                    "torch-geometric",
                    "zarr",
                )
            },
            "platform": platform.platform(),
            "python": sys.version,
        },
        "errors": errors,
        "label_inventories": label_inventories,
        "models": _run_records(models_root, errors),
        "path_policy": {
            "DFT_DATA": str(data_root),
            "DFT_MODELS": str(models_root),
            "effective_configs": "raw hparams.yaml plus explicit roots",
            "forbidden_for_execution": "hparams_resolved.yaml and /export/scratch/ialgroup paths",
        },
        "status": "failed" if errors else "passed",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
