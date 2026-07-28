#!/usr/bin/env python3
"""Compute or compare one registered train-only PBE analytic Hessian."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
import zarr
from pyscf import scf

from qm9_pbe_hessian_reference_set import _hessian_from_chk


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one match for {pattern} below {root}, found {len(matches)}"
        )
    return matches[0]


def _identity_audit(
    dataset_dir: Path, molecule_id: str, sample_id: int
) -> tuple[Path, Path, dict[str, Any]]:
    suffix = f"{molecule_id}.{sample_id:07d}"
    chk_path = _one_match(dataset_dir / "kohn_sham", f"*_{suffix}.chk")
    label_path = _one_match(dataset_dir / "labels", f"{suffix}.zarr.zip")
    molecule = scf.chkfile.load_mol(chk_path.as_posix())
    root = zarr.open(label_path, mode="r")
    label_numbers = np.asarray(
        root["geometry/atomic_numbers"][:], dtype=np.int64
    )
    chk_numbers = np.asarray(molecule.atom_charges(), dtype=np.int64)
    if not np.array_equal(label_numbers, chk_numbers):
        raise ValueError("Label and checkpoint atomic-number order differs")

    positions_unit = str(root["metadata/reference/positions_unit"][()])
    geometry_positions = np.asarray(
        root["geometry/atom_pos"][:], dtype=np.float64
    )
    geometry_coordinate_max_abs = float(
        np.max(
            np.abs(
                geometry_positions - molecule.atom_coords(unit="Bohr")
            )
        )
    )
    if geometry_coordinate_max_abs > 1e-8:
        raise ValueError(
            "geometry/atom_pos must match checkpoint coordinates in Bohr: "
            f"{geometry_coordinate_max_abs}"
        )
    sample_positions = np.asarray(
        root["metadata/reference/sample_atom_pos"][:], dtype=np.float64
    )
    if positions_unit.lower() in {"angstrom", "a"}:
        chk_positions = molecule.atom_coords(unit="Angstrom")
    elif positions_unit.lower() in {"bohr", "au"}:
        chk_positions = molecule.atom_coords(unit="Bohr")
    else:
        raise ValueError(f"Unsupported label positions unit: {positions_unit}")
    metadata_coordinate_max_abs = float(
        np.max(np.abs(sample_positions - chk_positions))
    )
    if metadata_coordinate_max_abs > 1e-8:
        raise ValueError(
            "metadata/reference/sample_atom_pos coordinate mismatch: "
            f"{metadata_coordinate_max_abs}"
        )
    if "metadata/reference/source_molecule_id" in root:
        registered_id = int(root["metadata/reference/source_molecule_id"][()])
        if registered_id != int(molecule_id):
            raise ValueError("Label source molecule ID differs from filename")
    if "metadata/reference/sample_id" in root:
        registered_sample = int(root["metadata/reference/sample_id"][()])
        if registered_sample != sample_id:
            raise ValueError("Label sample ID differs from filename")
    return chk_path, label_path, {
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": int(molecule.natm),
        "atom_symbols": [molecule.atom_symbol(i) for i in range(molecule.natm)],
        "atomic_numbers": label_numbers.tolist(),
        "geometry_positions_unit": "Bohr",
        "positions_unit": positions_unit,
        "geometry_coordinate_max_abs": geometry_coordinate_max_abs,
        "metadata_coordinate_max_abs": metadata_coordinate_max_abs,
        "atom_order_matches": True,
    }


def compute(args: argparse.Namespace) -> dict[str, Any]:
    molecule_id = args.molecule.zfill(7)
    chk_path, label_path, identity = _identity_audit(
        args.dataset_dir, molecule_id, args.sample_id
    )
    chk_sha256_before = _sha256(chk_path)
    started = time.perf_counter()
    hessian = _hessian_from_chk(chk_path, args.backend)
    elapsed = time.perf_counter() - started
    chk_sha256_after = _sha256(chk_path)
    if chk_sha256_after != chk_sha256_before:
        raise RuntimeError(
            f"Frozen checkpoint was modified: {chk_sha256_before} "
            f"!= {chk_sha256_after}"
        )
    finite = bool(np.isfinite(hessian).all())
    symmetry_max_abs = float(np.max(np.abs(hessian - hessian.T)))
    if not finite:
        raise ValueError("PBE Hessian contains NaN or Inf")
    if symmetry_max_abs > args.symmetry_tolerance:
        raise ValueError(
            f"PBE Hessian symmetry error {symmetry_max_abs} exceeds "
            f"{args.symmetry_tolerance}"
        )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        pbe_hessian=np.asarray(hessian, dtype=np.float64),
        atomic_numbers=np.asarray(identity["atomic_numbers"], dtype=np.int64),
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pyscf_version": pyscf.__version__,
        "backend": args.backend,
        "definition": "PBE analytic nuclear Cartesian Hessian",
        "output_unit": "Hartree/Bohr^2",
        "dataset_dir": args.dataset_dir.resolve().as_posix(),
        "chk_path": chk_path.resolve().as_posix(),
        "chk_sha256": chk_sha256_before,
        "chk_sha256_after": chk_sha256_after,
        "chk_unchanged": True,
        "label_path": label_path.resolve().as_posix(),
        "label_sha256": _sha256(label_path),
        "output_npz": args.output_npz.resolve().as_posix(),
        "output_npz_sha256": _sha256(args.output_npz),
        "identity": identity,
        "shape": list(hessian.shape),
        "finite": finite,
        "symmetry_max_abs": symmetry_max_abs,
        "max_abs": float(np.max(np.abs(hessian))),
        "wall_seconds": elapsed,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.gpu_npz) as payload:
        gpu = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        gpu_numbers = np.asarray(payload["atomic_numbers"], dtype=np.int64)
    with np.load(args.cpu_npz) as payload:
        cpu = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        cpu_numbers = np.asarray(payload["atomic_numbers"], dtype=np.int64)
    if gpu.shape != cpu.shape or not np.array_equal(gpu_numbers, cpu_numbers):
        raise ValueError("GPU and CPU Hessian identity differs")
    difference = gpu - cpu
    relative_frobenius = float(
        np.linalg.norm(difference) / max(np.linalg.norm(cpu), 1e-15)
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "definition": "GPU4PySCF versus CPU PySCF PBE analytic Hessian",
        "output_unit": "Hartree/Bohr^2",
        "shape": list(gpu.shape),
        "atomic_numbers_match": True,
        "gpu_finite": bool(np.isfinite(gpu).all()),
        "cpu_finite": bool(np.isfinite(cpu).all()),
        "gpu_symmetry_max_abs": float(np.max(np.abs(gpu - gpu.T))),
        "cpu_symmetry_max_abs": float(np.max(np.abs(cpu - cpu.T))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": relative_frobenius,
        "relative_frobenius_tolerance": args.relative_frobenius_tolerance,
        "gpu_npz_sha256": _sha256(args.gpu_npz),
        "cpu_npz_sha256": _sha256(args.cpu_npz),
        "validation_accessed": False,
        "test100_accessed": False,
    }
    if not result["gpu_finite"] or not result["cpu_finite"]:
        raise ValueError("GPU or CPU Hessian contains NaN/Inf")
    if relative_frobenius > args.relative_frobenius_tolerance:
        result["status"] = "fail"
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        raise ValueError(
            f"GPU/CPU relative Frobenius {relative_frobenius} exceeds "
            f"{args.relative_frobenius_tolerance}"
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("compute")
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--molecule", required=True)
    run.add_argument("--sample-id", type=int, default=0)
    run.add_argument("--backend", choices=("cpu", "gpu4pyscf"), required=True)
    run.add_argument("--output-npz", type=Path, required=True)
    run.add_argument("--output-json", type=Path, required=True)
    run.add_argument("--symmetry-tolerance", type=float, default=1e-8)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--gpu-npz", type=Path, required=True)
    comparison.add_argument("--cpu-npz", type=Path, required=True)
    comparison.add_argument("--output-json", type=Path, required=True)
    comparison.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1e-4
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute(args) if args.command == "compute" else compare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
