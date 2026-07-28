"""Verify that original legacy data and checkpoints stayed unchanged during recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_content(path: Path) -> dict:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(f"{relative}\t{size}\t{_sha256(child)}\n".encode())
        files += 1
        total_bytes += size
    return {
        "bytes": total_bytes,
        "content_manifest_sha256": digest.hexdigest(),
        "files": files,
        "path": str(path.resolve()),
    }


def _label_inventory(path: Path, cutoff_ns: int) -> dict:
    records = []
    modified_after = []
    maximum_mtime_ns = 0
    for entry in os.scandir(path):
        if not entry.is_file() or not entry.name.endswith(".zarr.zip"):
            continue
        stat = entry.stat()
        records.append((entry.name, stat.st_size))
        maximum_mtime_ns = max(maximum_mtime_ns, stat.st_mtime_ns)
        if stat.st_mtime_ns > cutoff_ns:
            modified_after.append(
                {"filename": entry.name, "mtime_ns": stat.st_mtime_ns, "bytes": stat.st_size}
            )
    records.sort()
    digest = hashlib.sha256()
    total_bytes = 0
    for name, size in records:
        digest.update(f"{name}\t{size}\n".encode())
        total_bytes += size
    return {
        "bytes": total_bytes,
        "count": len(records),
        "maximum_mtime_ns": maximum_mtime_ns,
        "modified_after_initial_manifest": modified_after,
        "path": str(path.resolve()),
        "path_size_inventory_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    initial = json.loads(args.initial_manifest.read_text())
    cutoff_ns = args.initial_manifest.stat().st_mtime_ns
    errors = []
    files = []
    directories = []

    def compare_file(kind: str, record: dict) -> None:
        path = Path(record["path"])
        current = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "path": str(path.resolve()),
            "sha256": _sha256(path) if path.is_file() else None,
        }
        passed = (
            path.is_file()
            and current["bytes"] == record["bytes"]
            and current["sha256"] == record["sha256"]
        )
        files.append(
            {"kind": kind, "initial": record, "current": current, "passed": passed}
        )
        if not passed:
            errors.append(f"changed or missing {kind}: {path}")

    for dataset, record in initial["datasets"].items():
        for key in ("dataset_info", "split_pickle", "split_yaml"):
            compare_file(f"{dataset}/{key}", record[key])
        for statistics in record["statistics"]:
            current = _directory_content(Path(statistics["path"]))
            passed = all(
                current[key] == statistics[key]
                for key in ("bytes", "content_manifest_sha256", "files")
            )
            directories.append(
                {
                    "kind": f"{dataset}/statistics",
                    "initial": statistics,
                    "current": current,
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"changed statistics directory: {statistics['path']}")
    for model in initial["models"]:
        dataset = model["dataset"]
        for key in ("checkpoint", "hparams", "hparams_resolved_provenance_only"):
            compare_file(f"{dataset}/{key}", model[key])
    label_sources = {}
    for source, record in initial["label_inventories"].items():
        current = _label_inventory(Path(record["path"]), cutoff_ns)
        passed = (
            current["bytes"] == record["bytes"]
            and current["count"] == record["count"]
            and current["path_size_inventory_sha256"]
            == record["path_size_inventory_sha256"]
            and not current["modified_after_initial_manifest"]
        )
        label_sources[source] = {
            "initial": record,
            "current": current,
            "passed": passed,
        }
        if not passed:
            errors.append(f"changed label inventory or post-manifest mtime: {source}")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "definition": (
            "End-of-funnel comparison against the initial recovery manifest. Label archives use "
            "name/size inventory plus a no-post-manifest-mtime condition; the earlier full content "
            "audit independently read and hashed every numeric configuration. Static metadata, "
            "statistics, hparams and checkpoints are rehashed byte-for-byte."
        ),
        "initial_manifest": str(args.initial_manifest.resolve()),
        "initial_manifest_cutoff_mtime_ns": cutoff_ns,
        "files": files,
        "directories": directories,
        "label_sources": label_sources,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
