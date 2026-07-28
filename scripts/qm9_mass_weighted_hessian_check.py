#!/usr/bin/env python3
"""Mass-weighted Hessian and frequency self-check for cached QM9 PBE Hessians."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from pyscf.data import elements
from scipy import constants


def _frequency_conversion_cm() -> float:
    hartree_j = constants.physical_constants["Hartree energy"][0]
    bohr_m = constants.physical_constants["Bohr radius"][0]
    amu_kg = constants.physical_constants["atomic mass constant"][0]
    omega_si = math.sqrt(hartree_j / (bohr_m * bohr_m * amu_kg))
    return omega_si / (2.0 * math.pi * constants.c * 100.0)


FREQUENCY_CONVERSION_CM = _frequency_conversion_cm()


def _load_geometry(dataset_dir: Path, molecule_id: str, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    root = zarr.open(label_path, mode="r")
    return (
        np.asarray(root["geometry/atomic_numbers"], dtype=np.int64),
        np.asarray(root["geometry/atom_pos"], dtype=np.float64),
    )


def _mass_vector(atomic_numbers: np.ndarray) -> np.ndarray:
    masses = np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    return np.repeat(masses, 3)


def _mass_weight_hessian(hessian: np.ndarray, atomic_numbers: np.ndarray) -> np.ndarray:
    mass_vec = _mass_vector(atomic_numbers)
    return hessian / np.sqrt(np.outer(mass_vec, mass_vec))


def _external_mode_basis(atomic_numbers: np.ndarray, positions_bohr: np.ndarray) -> np.ndarray:
    masses = np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    sqrt_masses = np.sqrt(masses)
    center_of_mass = np.sum(masses[:, None] * positions_bohr, axis=0) / np.sum(masses)
    centered = positions_bohr - center_of_mass

    vectors: list[np.ndarray] = []
    for axis in range(3):
        vec = np.zeros_like(positions_bohr)
        vec[:, axis] = sqrt_masses
        vectors.append(vec.reshape(-1))

    unit_axes = np.eye(3)
    for axis in unit_axes:
        vec = sqrt_masses[:, None] * np.cross(axis[None, :], centered)
        vectors.append(vec.reshape(-1))

    orthonormal: list[np.ndarray] = []
    for vec in vectors:
        work = vec.astype(np.float64, copy=True)
        for basis in orthonormal:
            work -= np.dot(basis, work) * basis
        norm = np.linalg.norm(work)
        if norm > 1e-10:
            orthonormal.append(work / norm)
    return np.stack(orthonormal, axis=1)


def _signed_frequencies_cm(eigvals: np.ndarray) -> np.ndarray:
    return np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * FREQUENCY_CONVERSION_CM


def _row_for_reference(ref: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    molecule_id = ref["molecule_id"]
    sample_id = int(ref["sample_id"])
    atomic_numbers, positions_bohr = _load_geometry(dataset_dir, molecule_id, sample_id)
    hessian = np.load(ref["cache_path"])["pbe_hessian"]
    hessian_sym = 0.5 * (hessian + hessian.T)
    mw_hessian = _mass_weight_hessian(hessian_sym, atomic_numbers)
    external_basis = _external_mode_basis(atomic_numbers, positions_bohr)
    projector = np.eye(mw_hessian.shape[0]) - external_basis @ external_basis.T
    projected = projector @ mw_hessian @ projector
    projected = 0.5 * (projected + projected.T)
    eigvals = np.linalg.eigvalsh(projected)
    freqs = _signed_frequencies_cm(eigvals)

    n_external = int(external_basis.shape[1])
    external_indices = np.argsort(np.abs(freqs))[:n_external]
    vib_mask = np.ones(freqs.shape[0], dtype=bool)
    vib_mask[external_indices] = False
    vib_freqs = freqs[vib_mask]
    external_freqs = freqs[external_indices]

    return {
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": int(ref["natoms"]),
        "n_external_modes": n_external,
        "n_vibrational_modes": int(vib_freqs.shape[0]),
        "finite": bool(np.isfinite(freqs).all()),
        "raw_symmetry_max_abs": float(np.max(np.abs(hessian - hessian.T))),
        "projected_symmetry_max_abs": float(np.max(np.abs(projected - projected.T))),
        "max_external_abs_cm-1": float(np.max(np.abs(external_freqs))) if external_freqs.size else None,
        "min_vibrational_cm-1": float(np.min(vib_freqs)) if vib_freqs.size else None,
        "max_vibrational_cm-1": float(np.max(vib_freqs)) if vib_freqs.size else None,
        "negative_vibrational_modes": int(np.sum(vib_freqs < -1.0)),
        "frequency_conversion_cm-1_per_sqrt_au_amu": FREQUENCY_CONVERSION_CM,
        "cache_path": ref["cache_path"],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest_json.read_text())
    references = [
        row for row in manifest if row.get("success") and Path(row["cache_path"]).exists()
    ][: args.max_molecules]
    rows = [_row_for_reference(ref, args.dataset_dir) for ref in references]
    summary = {
        "definition": "PBE analytic Hessian mass-weighting/frequency self-check",
        "assumed_hessian_units": "Hartree / Bohr^2",
        "mass_units": "atomic mass unit",
        "frequency_conversion_cm-1_per_sqrt_au_amu": FREQUENCY_CONVERSION_CM,
        "manifest_json": args.manifest_json.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "n_references": len(rows),
        "n_finite": sum(1 for row in rows if row["finite"]),
        "max_external_abs_cm-1": max(
            (row["max_external_abs_cm-1"] for row in rows if row["max_external_abs_cm-1"] is not None),
            default=None,
        ),
        "negative_vibrational_rows": sum(1 for row in rows if row["negative_vibrational_modes"] > 0),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "molecule_id",
            "sample_id",
            "natoms",
            "n_external_modes",
            "n_vibrational_modes",
            "finite",
            "raw_symmetry_max_abs",
            "projected_symmetry_max_abs",
            "max_external_abs_cm-1",
            "min_vibrational_cm-1",
            "max_vibrational_cm-1",
            "negative_vibrational_modes",
            "frequency_conversion_cm-1_per_sqrt_au_amu",
            "cache_path",
        ]
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
