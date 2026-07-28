#!/usr/bin/env python3
"""Generate newly registered PBE Hessians for frozen train20 parents only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyscf import scf

from qm9_pbe_hessian_reference_set import _compute_reference


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _chk_path(dataset_dir: Path, molecule_id: str) -> Path:
    matches = sorted(
        (dataset_dir / "kohn_sham").glob(f"*_{molecule_id}.0000000.chk")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one sample-0 chk for {molecule_id}, found {len(matches)}"
        )
    return matches[0]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = yaml.safe_load(args.protocol.read_text())
    if (
        protocol.get("validation_access_allowed") is not False
        or protocol.get("test100_access_allowed") is not False
    ):
        raise ValueError("Protocol must prohibit validation and Test100 access")
    molecule_ids = [
        str(value) for value in protocol["parent_sets"]["train20"]["molecule_ids"]
    ]
    if len(molecule_ids) != 20 or len(set(molecule_ids)) != 20:
        raise ValueError("Frozen train20 must contain 20 unique parents")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tasks = []
    for molecule_id in molecule_ids:
        chk_path = _chk_path(args.dataset_dir, molecule_id)
        mol = scf.chkfile.load_mol(chk_path.as_posix())
        tasks.append(
            {
                "dataset_dir": args.dataset_dir.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "molecule_id": molecule_id,
                "sample_id": 0,
                "natoms": int(mol.natm),
                "backend": args.backend,
                "recompute": args.recompute,
            }
        )

    records_by_id = {}

    def register(record: dict[str, Any]) -> None:
        molecule_id = str(record["molecule_id"])
        chk_path = _chk_path(args.dataset_dir, molecule_id)
        record["chk_sha256"] = _sha256(chk_path)
        if record["success"]:
            cache_path = Path(record["cache_path"])
            record["pbe_hessian_sha256"] = _sha256(cache_path)
            with np.load(cache_path) as payload:
                hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
            if not np.isfinite(hessian).all():
                raise ValueError(f"Non-finite PBE Hessian for {molecule_id}")
        else:
            record["pbe_hessian_sha256"] = None
        records_by_id[molecule_id] = record
        print(json.dumps(record, sort_keys=True), flush=True)

    if args.workers == 1:
        for task in tasks:
            register(_compute_reference(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_compute_reference, task): task for task in tasks
            }
            for future in as_completed(futures):
                register(future.result())
    records = [records_by_id[molecule_id] for molecule_id in molecule_ids]

    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "definition": "PBE analytic nuclear Cartesian Hessian restored from a newly generated sample-0 SCF checkpoint.",
        "backend": args.backend,
        "workers": args.workers,
        "dataset_dir": args.dataset_dir.resolve().as_posix(),
        "parent_set": "train20",
        "parent_count": len(records),
        "success_count": sum(bool(row["success"]) for row in records),
        "failed_count": sum(not bool(row["success"]) for row in records),
        "wall_time_s": time.perf_counter() - started,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parents": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    registration = {
        "manifest": args.manifest.resolve().as_posix(),
        "manifest_sha256": _sha256(args.manifest),
        "success_count": manifest["success_count"],
        "failed_count": manifest["failed_count"],
        "wall_time_s": manifest["wall_time_s"],
        "workers": args.workers,
    }
    args.manifest.with_suffix(".registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    if manifest["failed_count"]:
        raise RuntimeError(f"{manifest['failed_count']} PBE Hessians failed")
    return registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu4pyscf"), default="gpu4pyscf")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.backend == "gpu4pyscf" and args.workers != 1:
        parser.error("gpu4pyscf requires --workers 1")
    return args


if __name__ == "__main__":
    print(json.dumps(generate(parse_args()), indent=2, sort_keys=True))
