#!/usr/bin/env python3
"""Create a reproducible random QM9 raw-file subset.

The subset preserves original QM9 file names so generated labels keep source QM9
molecule ids. Files are symlinked by default to avoid copying the raw dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def qm9_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw-dir", type=Path, required=True)
    parser.add_argument("--output-raw-dir", type=Path, required=True)
    parser.add_argument("--n-molecules", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory manifest and missing links.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_raw_dir = args.source_raw_dir
    output_raw_dir = args.output_raw_dir
    if not source_raw_dir.exists():
        raise SystemExit(f"source raw directory does not exist: {source_raw_dir}")
    all_files = sorted(source_raw_dir.glob("dsgdb9nsd_*.xyz"))
    if len(all_files) < args.n_molecules:
        raise SystemExit(
            f"requested {args.n_molecules} molecules, but only found {len(all_files)} raw files"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(all_files, args.n_molecules)
    selected_ids = {qm9_id(path) for path in selected}

    # QM9.get_ids has a fast path for apparently contiguous raw directories.
    # Avoid accidentally triggering that path for a random subset of size N.
    if 1 in selected_ids and args.n_molecules in selected_ids:
        unselected = [path for path in all_files if qm9_id(path) not in selected_ids]
        replacement = rng.choice(unselected)
        selected = [replacement if qm9_id(path) == args.n_molecules else path for path in selected]
        selected_ids = {qm9_id(path) for path in selected}

    selected = sorted(selected, key=qm9_id)
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)

    for source in selected:
        target = output_raw_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)

    linked = sorted(output_raw_dir.glob("dsgdb9nsd_*.xyz"))
    linked_ids = [qm9_id(path) for path in linked]
    expected_ids = [qm9_id(path) for path in selected]
    missing = sorted(set(expected_ids).difference(linked_ids))
    extra = sorted(set(linked_ids).difference(expected_ids))
    manifest = {
        "source_raw_dir": source_raw_dir.as_posix(),
        "output_raw_dir": output_raw_dir.as_posix(),
        "n_molecules": args.n_molecules,
        "seed": args.seed,
        "copy": bool(args.copy),
        "selected_ids": expected_ids,
        "linked_count": len(linked),
        "missing_ids": missing,
        "extra_ids": extra,
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_ids"}, indent=2))
    if missing:
        raise SystemExit(f"missing selected raw links: {missing[:20]}")


if __name__ == "__main__":
    main()
