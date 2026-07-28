#!/usr/bin/env python3
"""Validate the isolated node02 GPU4PySCF runtime and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy
import gpu4pyscf
import numpy as np
import pyscf
from gpu4pyscf.hessian import rks
from pyscf import dft, gto


DISTRIBUTIONS = (
    "gpu4pyscf-cuda12x",
    "gpu4pyscf-libxc-cuda12x",
    "cupy-cuda12x",
    "pyscf",
    "numpy",
    "scipy",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _distribution_records() -> dict[str, dict[str, str | None]]:
    records = {}
    for name in DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        metadata = distribution.read_text("METADATA")
        record = distribution.read_text("RECORD")
        records[name] = {
            "version": distribution.version,
            "metadata_sha256": (
                _sha256_bytes(metadata.encode()) if metadata is not None else None
            ),
            "record_sha256": (
                _sha256_bytes(record.encode()) if record is not None else None
            ),
        }
    return records


def run_smoke() -> dict:
    started = time.perf_counter()
    cupy_value = float(cupy.arange(8, dtype=cupy.float64).sum().get())
    if cupy_value != 28.0:
        raise RuntimeError(f"Unexpected CuPy sum: {cupy_value}")

    molecule = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = dft.RKS(molecule, xc="pbe").density_fit()
    mean_field.grids.level = 1
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("H2 CPU initialization SCF did not converge")

    gpu_mean_field = mean_field.to_gpu()
    hessian_started = time.perf_counter()
    hessian = rks.Hessian(gpu_mean_field).kernel()
    cupy.cuda.Stream.null.synchronize()
    hessian_seconds = time.perf_counter() - hessian_started
    if isinstance(hessian, cupy.ndarray):
        hessian = cupy.asnumpy(hessian)
    hessian = np.asarray(hessian, dtype=np.float64)
    hessian_matrix = hessian.transpose(0, 2, 1, 3).reshape(
        molecule.natm * 3, molecule.natm * 3
    )
    finite = bool(np.isfinite(hessian_matrix).all())
    symmetry_max_abs = float(np.max(np.abs(hessian_matrix - hessian_matrix.T)))
    if not finite or symmetry_max_abs > 1e-10:
        raise RuntimeError(
            f"Invalid H2 Hessian: finite={finite}, symmetry={symmetry_max_abs}"
        )

    return {
        "schema_version": 1,
        "status": "pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda": {
            "visible_device_count": cupy.cuda.runtime.getDeviceCount(),
            "runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
            "driver_api_version": cupy.cuda.runtime.driverGetVersion(),
        },
        "packages": _distribution_records(),
        "imports": {
            "pyscf": pyscf.__version__,
            "gpu4pyscf": gpu4pyscf.__version__,
            "cupy": cupy.__version__,
        },
        "cupy_smoke": {"sum_expected": 28.0, "sum_observed": cupy_value},
        "h2_pbe_hessian_smoke": {
            "functional": "PBE",
            "basis": "sto-3g",
            "geometry_unit": "Angstrom",
            "atom_order": ["H", "H"],
            "output_unit": "Hartree/Bohr^2",
            "raw_shape": list(hessian.shape),
            "matrix_shape": list(hessian_matrix.shape),
            "finite": finite,
            "symmetry_max_abs": symmetry_max_abs,
            "electronic_energy_hartree": float(mean_field.e_tot),
            "gpu_hessian_wall_seconds": hessian_seconds,
        },
        "total_wall_seconds": time.perf_counter() - started,
        "validation_accessed": False,
        "test100_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
