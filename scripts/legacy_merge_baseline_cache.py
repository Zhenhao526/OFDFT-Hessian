"""Verify all derived-cache shards and freeze their aggregate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors = []
    rows = []
    shard_artifacts = []
    for shard in range(args.num_shards):
        path = args.run_root / "summaries" / f"shard_{shard}.json"
        if not path.is_file():
            errors.append(f"missing shard summary: {path}")
            continue
        summary = json.loads(path.read_text())
        shard_artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        if summary["status"] != "passed":
            errors.append(f"failed cache shard: {shard}")
        rows.extend(summary["rows"])
    keys = [
        (row["source"], row["filename"], int(row.get("scf_selector", -1)))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        errors.append("duplicate cache keys across shards")
    if len(rows) != manifest["unique_labels"]:
        errors.append(f"cache row count {len(rows)} != manifest {manifest['unique_labels']}")
    if any(row.get("status") != "passed" for row in rows):
        errors.append("one or more cache rows failed")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "definition": (
            "Derived selected-density transformed cache. Raw labels are read-only; these cache "
            "files are preprocessing artifacts, not new scientific labels."
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "partition": manifest["partition"],
        "scf_selectors": sorted({key[2] for key in keys}),
        "rows": len(rows),
        "total_bytes": sum(row.get("cache_bytes", 0) for row in rows),
        "shard_artifacts": shard_artifacts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors, "rows": len(rows)}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
