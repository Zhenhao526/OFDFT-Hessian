#!/usr/bin/env python3
"""Build a small cached PBE Hessian reference set from existing QM9PBEForcePilot chk files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from pyscf import dft, scf
from pyscf.hessian import rks


def _load_split_candidates(dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    import pickle

    with (dataset_dir / "split.pkl").open("rb") as f:
        split = pickle.load(f)
    rows = []
    seen = set()
    if split_name not in split:
        raise KeyError(f"Split {split_name!r} not found in {dataset_dir / 'split.pkl'}")
    for _, label_name, n_scf_iter in split[split_name]:
        molecule_id, sample_text, *_ = label_name.removesuffix(".zarr.zip").split(".")
        if molecule_id in seen or int(sample_text) != 0:
            continue
        seen.add(molecule_id)
        label_path = dataset_dir / "labels" / label_name
        root = zarr.open(label_path, mode="r")
        natoms = int(root["geometry/atomic_numbers"].shape[0])
        rows.append(
            {
                "molecule_id": molecule_id,
                "sample_id": int(sample_text),
                "label_name": label_name,
                "natoms": natoms,
                "n_scf_iter": int(n_scf_iter),
            }
        )
    return sorted(rows, key=lambda row: (row["natoms"], row["molecule_id"]))


def _chk_path(dataset_dir: Path, molecule_id: str, sample_id: int) -> Path:
    kohn_sham_dir = dataset_dir / "kohn_sham"
    known_names = [
        f"qm9_pbe_force_pilot_{molecule_id}.{sample_id:07d}.chk",
        f"qm9_pbe_force_random1000_{molecule_id}.{sample_id:07d}.chk",
        f"qm9_pbe_force_full_{molecule_id}.{sample_id:07d}.chk",
    ]
    for name in known_names:
        chk_path = kohn_sham_dir / name
        if chk_path.exists():
            return chk_path
    matches = sorted(kohn_sham_dir.glob(f"*_{molecule_id}.{sample_id:07d}.chk"))
    if matches:
        return matches[0]
    return kohn_sham_dir / known_names[0]


def _maybe_to_numpy(value: Any) -> Any:
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        return value.get()
    return value


def _mf_from_chk(chk_path: Path) -> tuple[Any, Any]:
    mol, rec = scf.chkfile.load_scf(chk_path.as_posix())
    xc = scf.chkfile.load(chk_path.as_posix(), "Results/name_xc_functional")
    if isinstance(xc, bytes):
        xc = xc.decode()
    grid_level = int(scf.chkfile.load(chk_path.as_posix(), "Results/grid_level"))
    mf = dft.RKS(mol, xc=xc)
    # Hessian CPHF writes scf_f1ao/scf_mo1 when chkfile is set. The source SCF
    # checkpoint is a frozen input and must remain byte-for-byte unchanged.
    mf.chkfile = None
    mf.grids.level = grid_level
    mf.mo_coeff = rec["mo_coeff"]
    mf.mo_occ = rec["mo_occ"]
    mf.e_tot = rec["e_tot"]
    if "mo_energy" in rec:
        mf.mo_energy = rec["mo_energy"]
    return mol, mf


def _hessian_from_chk(chk_path: Path, backend: str) -> np.ndarray:
    mol, mf = _mf_from_chk(chk_path)
    if backend == "cpu":
        hessian = rks.Hessian(mf).kernel()
    elif backend == "gpu4pyscf":
        from gpu4pyscf.hessian import rks as gpu_rks

        hessian = gpu_rks.Hessian(mf.to_gpu()).kernel()
        hessian = _maybe_to_numpy(hessian)
    else:
        raise ValueError(f"Unknown Hessian backend: {backend}")
    return np.asarray(hessian).transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)


def _compute_reference(task: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(task["dataset_dir"])
    output_dir = Path(task["output_dir"])
    molecule_id = task["molecule_id"]
    sample_id = int(task["sample_id"])
    natoms = int(task["natoms"])
    cache_path = output_dir / f"pbe_hessian_{molecule_id}_{sample_id:07d}.npz"
    chk_path = _chk_path(dataset_dir, molecule_id, sample_id)

    record = {
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": natoms,
        "backend": task["backend"],
        "chk_path": chk_path.as_posix(),
        "cache_path": cache_path.as_posix(),
        "success": False,
        "cached": False,
        "elapsed_s": None,
        "shape": None,
        "finite": None,
        "symmetry_max_abs_error": None,
        "max_abs": None,
        "error": None,
    }

    t0 = time.time()
    try:
        chk_sha256_before = _file_sha256(chk_path)
        if cache_path.exists() and not task["recompute"]:
            hessian_matrix = np.load(cache_path)["pbe_hessian"]
            record["cached"] = True
        else:
            hessian_matrix = _hessian_from_chk(chk_path, task["backend"])
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, pbe_hessian=hessian_matrix)
        chk_sha256_after = _file_sha256(chk_path)
        if chk_sha256_after != chk_sha256_before:
            raise RuntimeError(
                f"Frozen checkpoint was modified: {chk_sha256_before} "
                f"!= {chk_sha256_after}"
            )

        record.update(
            {
                "success": True,
                "elapsed_s": time.time() - t0,
                "shape": [int(dim) for dim in hessian_matrix.shape],
                "finite": bool(np.isfinite(hessian_matrix).all()),
                "symmetry_max_abs_error": float(np.max(np.abs(hessian_matrix - hessian_matrix.T))),
                "max_abs": float(np.max(np.abs(hessian_matrix))),
                "chk_sha256_before": chk_sha256_before,
                "chk_sha256_after": chk_sha256_after,
                "chk_unchanged": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["elapsed_s"] = time.time() - t0
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    return record


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(records: list[dict[str, Any]], manifest_json: Path, manifest_csv: Path) -> None:
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "molecule_id",
        "sample_id",
        "natoms",
        "backend",
        "chk_path",
        "cache_path",
        "success",
        "cached",
        "elapsed_s",
        "shape",
        "finite",
        "symmetry_max_abs_error",
        "max_abs",
        "error",
    ]
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--molecules", default=None, help="Comma-separated molecule ids.")
    parser.add_argument("--max-molecules", type=int, default=10)
    parser.add_argument("--max-natoms", type=int, default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "gpu4pyscf"], default="cpu")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend != "cpu" and args.workers != 1:
        raise SystemExit(
            "GPU Hessian backend must be launched with --workers 1. "
            "Use external process sharding with one CUDA_VISIBLE_DEVICES value per process."
        )
    if args.molecules:
        requested = {item.strip().zfill(7) for item in args.molecules.split(",") if item.strip()}
        candidates = [
            row
            for row in _load_split_candidates(args.dataset_dir, args.split)
            if row["molecule_id"] in requested
        ]
        order = {molecule_id: idx for idx, molecule_id in enumerate(requested)}
        candidates = sorted(candidates, key=lambda row: order.get(row["molecule_id"], 0))
    else:
        candidates = _load_split_candidates(args.dataset_dir, args.split)
    candidates = [row for row in candidates if row["sample_id"] == args.sample_id]
    if args.max_natoms is not None:
        candidates = [row for row in candidates if row["natoms"] <= args.max_natoms]
    candidates = candidates[: args.max_molecules]

    tasks = [
        {
            **row,
            "dataset_dir": args.dataset_dir.as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "backend": args.backend,
            "recompute": args.recompute,
        }
        for row in candidates
    ]
    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            record = _compute_reference(task)
            records.append(record)
            print(json.dumps(record, sort_keys=True))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(_compute_reference, task): task for task in tasks}
            for future in as_completed(future_to_task):
                record = future.result()
                records.append(record)
                print(json.dumps(record, sort_keys=True))
    records = sorted(records, key=lambda row: (int(row["natoms"]), row["molecule_id"]))
    _write_manifest(records, args.manifest_json, args.manifest_csv)
    summary = {
        "workers": args.workers,
        "backend": args.backend,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "requested": len(tasks),
        "success": sum(1 for row in records if row["success"]),
        "failed": sum(1 for row in records if not row["success"]),
        "manifest_json": args.manifest_json.as_posix(),
        "manifest_csv": args.manifest_csv.as_posix(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
