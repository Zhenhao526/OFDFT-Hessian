#!/usr/bin/env python3
"""Materialize one frozen held-out partition for the residual random1000 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from prepare_qm9_residual_random1000_split import historical_parent_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--full-qm9-raw-dir", type=Path)
    source_group.add_argument("--qm9-archive", type=Path)
    parser.add_argument("--output-raw-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", choices=["val", "test"], required=True)
    args = parser.parse_args()

    split = historical_parent_split()
    held_parents = sorted(split[args.partition])
    args.output_raw_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"dsgdb9nsd_{parent_id:06d}.xyz" for parent_id in held_parents}
    unexpected = {
        path.name for path in args.output_raw_dir.glob("dsgdb9nsd_*.xyz")
    } - expected_names
    if unexpected:
        raise ValueError(
            f"Held-out raw directory contains unexpected files: {sorted(unexpected)[:5]}"
        )

    archive_members = {}
    archive = None
    if args.qm9_archive is not None:
        archive = tarfile.open(args.qm9_archive, mode="r:bz2")
        archive_members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile()
        }

    records = []
    for parent_id in held_parents:
        name = f"dsgdb9nsd_{parent_id:06d}.xyz"
        destination = args.output_raw_dir / name
        if args.full_qm9_raw_dir is not None:
            source = args.full_qm9_raw_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            if not destination.exists():
                shutil.copy2(source, destination)
            source_hash = _sha256(source)
        else:
            member = archive_members.get(name)
            if member is None:
                raise FileNotFoundError(f"{name} is absent from {args.qm9_archive}")
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"Cannot read {member.name} from {args.qm9_archive}")
            content = handle.read().decode().replace("*^", "e")
            encoded = content.encode()
            source_hash = hashlib.sha256(encoded).hexdigest()
            if not destination.exists():
                destination.write_bytes(encoded)
        destination_hash = _sha256(destination)
        if destination_hash != source_hash:
            raise ValueError(f"Raw geometry hash mismatch for {name}")
        records.append(
            {
                "parent_id": parent_id,
                "partition": args.partition,
                "filename": name,
                "sha256": destination_hash,
            }
        )

    if archive is not None:
        archive.close()
    source_path = (
        args.full_qm9_raw_dir
        if args.full_qm9_raw_dir is not None
        else args.qm9_archive
    )
    manifest = {
        "schema_version": 1,
        "protocol": "qm9_random1000_residual_graphformer_v1",
        "source": str(source_path.resolve()),
        "source_kind": "raw_directory" if args.full_qm9_raw_dir is not None else "tar_bz2",
        "output": str(args.output_raw_dir.resolve()),
        "partition": args.partition,
        "parent_count": len(held_parents),
        "test_labels_accessed": args.partition == "test",
        "files": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_sha256": _sha256(args.manifest),
                "partition": args.partition,
                "parent_count": len(held_parents),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
