"""Build a deduplicated, partition-scoped manifest for derived baseline samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    dataset_names = (
        "QM9_perturbed_fock",
        "QMUGSBin0_perturbed_fock",
        "QMUGSBin0QM9_perturbed_fock",
    )
    unique = {}
    inputs = {}
    memberships = {}
    for dataset in dataset_names:
        path = args.safe_root / dataset / "split.pkl"
        with path.open("rb") as handle:
            split = pickle.load(handle)
        inputs[dataset] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        memberships[dataset] = len(split[args.partition])
        for source, filename, scf_steps in split[args.partition]:
            key = (str(source), str(filename))
            previous = unique.get(key)
            if previous is not None and previous != int(scf_steps):
                raise RuntimeError(f"SCF count conflict for {key}: {previous} vs {scf_steps}")
            unique[key] = int(scf_steps)
    rows = []
    for source, filename in sorted(unique):
        numeric_id = int(filename.split(".", 1)[0])
        rows.append(
            {
                "filename": filename,
                "numeric_id": numeric_id,
                "scf_steps": unique[(source, filename)],
                "shard": numeric_id % args.num_shards,
                "source": source,
            }
        )
    report = {
        "definition": (
            "Deduplicated physical labels required by the three group-safe virtual datasets for "
            f"partition={args.partition}. A cache task chooses the archived SCF selector."
        ),
        "inputs": inputs,
        "memberships_before_deduplication": memberships,
        "num_shards": args.num_shards,
        "partition": args.partition,
        "rows": rows,
        "unique_labels": len(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"partition": args.partition, "unique_labels": len(rows), "memberships": memberships}, sort_keys=True))


if __name__ == "__main__":
    main()
