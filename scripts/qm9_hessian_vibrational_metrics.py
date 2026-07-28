#!/usr/bin/env python3
"""Evaluate frequencies, imaginary modes, and mode overlaps from model Hessian results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from pyscf.data import elements
from scipy import constants
from scipy.optimize import linear_sum_assignment


def _frequency_conversion_cm() -> float:
    hartree_j = constants.physical_constants["Hartree energy"][0]
    bohr_m = constants.physical_constants["Bohr radius"][0]
    amu_kg = constants.physical_constants["atomic mass constant"][0]
    omega_si = math.sqrt(hartree_j / (bohr_m * bohr_m * amu_kg))
    return omega_si / (2.0 * math.pi * constants.c * 100.0)


FREQUENCY_CONVERSION_CM = _frequency_conversion_cm()


def _load_geometry(
    dataset_dir: Path, molecule_id: str, sample_id: int
) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open(
        dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip", mode="r"
    )
    return (
        np.asarray(root["geometry/atomic_numbers"], dtype=np.int64),
        np.asarray(root["geometry/atom_pos"], dtype=np.float64),
    )


def _external_basis(atomic_numbers: np.ndarray, positions_bohr: np.ndarray) -> np.ndarray:
    masses = np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    sqrt_masses = np.sqrt(masses)
    center = np.sum(masses[:, None] * positions_bohr, axis=0) / np.sum(masses)
    centered = positions_bohr - center
    vectors = []
    for axis in range(3):
        vector = np.zeros_like(positions_bohr)
        vector[:, axis] = sqrt_masses
        vectors.append(vector.reshape(-1))
    for axis in np.eye(3):
        vectors.append((sqrt_masses[:, None] * np.cross(axis[None, :], centered)).reshape(-1))
    matrix = np.stack(vectors, axis=1)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > max(matrix.shape) * np.finfo(float).eps * singular_values[0]))
    return u[:, :rank]


def _vibrational_eigensystem(
    hessian: np.ndarray,
    atomic_numbers: np.ndarray,
    external_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (hessian + hessian.T)
    masses = np.repeat(
        np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64), 3
    )
    mass_weighted = symmetric / np.sqrt(np.outer(masses, masses))
    u, _, _ = np.linalg.svd(external_basis, full_matrices=True)
    complement = u[:, external_basis.shape[1] :]
    reduced = complement.T @ mass_weighted @ complement
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_modes = np.linalg.eigh(reduced)
    frequencies = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * FREQUENCY_CONVERSION_CM
    )
    cartesian_mass_weighted_modes = complement @ reduced_modes
    return frequencies, cartesian_mass_weighted_modes


def _load_model_hessian(path: Path) -> np.ndarray:
    payload = np.load(path)
    for key in (
        "total_ofdft_hessian",
        "density_relaxed_hessian",
        "autograd_hessian",
        "finite_difference_hessian",
        "model_hessian",
        "predicted_hessian",
    ):
        if key in payload:
            return np.asarray(payload[key], dtype=np.float64)
    raise KeyError(f"No supported model Hessian key in {path}: {list(payload.keys())}")


def _parse_result(item: str) -> tuple[str, Path]:
    name, path = item.split("=", maxsplit=1)
    return name, Path(path).resolve()


def _result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rows", "metric_rows", "fixed_density_full_hessian"):
        if key in result:
            return result[key]
    return []


def _capacity_result_rows(result_name: str, run_dir: Path) -> list[dict[str, Any]]:
    """Adapt a train-only capacity directory to the ordinary result-row schema."""
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"{run_dir} does not certify frozen Test100")
    if int(summary.get("test100_evaluations_used", 0)) != 0:
        raise ValueError(f"{run_dir} records Test100 evaluations")
    parents = summary.get("per_parent", [])
    if not parents and summary.get("per_parent_metrics"):
        metrics_path = Path(summary["per_parent_metrics"])
        with metrics_path.open(newline="") as handle:
            parents = list(csv.DictReader(handle))
    rows = []
    for parent in parents:
        molecule_id = str(parent["molecule_id"])
        hessian_path = (run_dir / f"{molecule_id}_result.npz").resolve()
        if not hessian_path.is_file():
            raise FileNotFoundError(hessian_path)
        rows.append(
            {
                "run": result_name,
                "molecule_id": molecule_id,
                "sample_id": 0,
                "success": True,
                "hessian_npz": hessian_path.as_posix(),
            }
        )
    return rows


def _baseline_result_rows(result_name: str, manifest_path: Path) -> list[dict[str, Any]]:
    """Adapt a frozen complete-total baseline manifest to result rows."""
    payload = json.loads(manifest_path.read_text())
    if payload.get("test100_accessed") is not False:
        raise ValueError(f"{manifest_path} does not certify frozen Test100")
    if int(payload.get("test100_evaluations_used", 0)) != 0:
        raise ValueError(f"{manifest_path} records Test100 evaluations")
    rows = []
    for parent in payload.get("parents", []):
        hessian_path = Path(parent["capacity_array"]).resolve()
        if not hessian_path.is_file():
            raise FileNotFoundError(hessian_path)
        rows.append(
            {
                "run": result_name,
                "molecule_id": str(parent["molecule_id"]),
                "sample_id": int(parent.get("sample_id", 0)),
                "success": True,
                "hessian_npz": hessian_path.as_posix(),
            }
        )
    if not rows:
        raise ValueError(f"no baseline parents in {manifest_path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _reference_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "parents" in payload:
        rows = []
        for parent in payload["parents"]:
            rows.append(
                {
                    **parent,
                    "sample_id": int(parent.get("sample_id", 0)),
                    "cache_path": parent.get("cache_path")
                    or parent.get("capacity_array"),
                    "success": bool(parent.get("success", True)),
                }
            )
        return rows
    raise ValueError(f"Unsupported Hessian reference manifest schema: {path}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        (row["molecule_id"], int(row["sample_id"])): row
        for row in _reference_manifest_rows(args.manifest_json)
        if row.get("success") and Path(row["cache_path"]).exists()
    }
    rows: list[dict[str, Any]] = []
    result_groups = []
    for result_name, result_path in map(_parse_result, args.result_json or []):
        result_groups.append((result_name, _result_rows(json.loads(result_path.read_text()))))
    for result_name, run_dir in map(_parse_result, args.capacity_result or []):
        result_groups.append(
            (result_name, _capacity_result_rows(result_name, run_dir))
        )
    for result_name, manifest_path in map(_parse_result, args.baseline_result or []):
        result_groups.append(
            (result_name, _baseline_result_rows(result_name, manifest_path))
        )
    if not result_groups:
        raise ValueError("At least one --result-json or --capacity-result is required")
    for result_name, model_rows in result_groups:
        for model_row in model_rows:
            if not model_row.get("success"):
                continue
            molecule_id = model_row["molecule_id"]
            sample_id = int(model_row.get("sample_id", 0))
            ref = manifest.get((molecule_id, sample_id))
            if ref is None:
                continue
            hessian_path_text = model_row.get("hessian_npz") or model_row.get("artifact_npz")
            if not hessian_path_text:
                continue
            hessian_path = Path(hessian_path_text)
            model_hessian = _load_model_hessian(hessian_path)
            pbe_hessian = np.load(ref["cache_path"])["pbe_hessian"]
            atomic_numbers, positions_bohr = _load_geometry(
                args.dataset_dir, molecule_id, sample_id
            )
            external = _external_basis(atomic_numbers, positions_bohr)
            ref_freq, ref_modes = _vibrational_eigensystem(
                pbe_hessian, atomic_numbers, external
            )
            model_freq, model_modes = _vibrational_eigensystem(
                model_hessian, atomic_numbers, external
            )
            overlap = np.abs(ref_modes.T @ model_modes)
            ref_indices, model_indices = linear_sum_assignment(-overlap)
            matched_ref = ref_freq[ref_indices]
            matched_model = model_freq[model_indices]
            frequency_diff = matched_model - matched_ref
            matched_overlap = overlap[ref_indices, model_indices]
            rows.append(
                {
                    "result": result_name,
                    "run": model_row.get("run"),
                    "molecule_id": molecule_id,
                    "sample_id": sample_id,
                    "natoms": int(ref["natoms"]),
                    "n_external_modes": int(external.shape[1]),
                    "n_vibrational_modes": int(ref_freq.size),
                    "frequency_mae_cm-1": float(np.mean(np.abs(frequency_diff))),
                    "frequency_rmse_cm-1": float(
                        np.sqrt(np.mean(frequency_diff * frequency_diff))
                    ),
                    "frequency_max_abs_cm-1": float(np.max(np.abs(frequency_diff))),
                    "mean_mode_overlap": float(np.mean(matched_overlap)),
                    "median_mode_overlap": float(np.median(matched_overlap)),
                    "min_mode_overlap": float(np.min(matched_overlap)),
                    "pbe_imaginary_modes": int(np.sum(ref_freq < -args.imaginary_threshold_cm)),
                    "model_imaginary_modes": int(
                        np.sum(model_freq < -args.imaginary_threshold_cm)
                    ),
                    "imaginary_mode_count_error": int(
                        np.sum(model_freq < -args.imaginary_threshold_cm)
                        - np.sum(ref_freq < -args.imaginary_threshold_cm)
                    ),
                    "raw_symmetry_max_abs": float(
                        np.max(np.abs(model_hessian - model_hessian.T))
                    ),
                    "hessian_npz": hessian_path.as_posix(),
                    "matched_frequencies_npz": (
                        args.output_dir
                        / "modes"
                        / f"{result_name}_{model_row.get('run')}_{molecule_id}_{sample_id:07d}.npz"
                    ).as_posix(),
                }
            )
            mode_path = Path(rows[-1]["matched_frequencies_npz"])
            mode_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mode_path,
                pbe_frequencies_cm=ref_freq,
                model_frequencies_cm=model_freq,
                matched_pbe_frequencies_cm=matched_ref,
                matched_model_frequencies_cm=matched_model,
                matched_mode_overlap=matched_overlap,
                assignment_ref_indices=ref_indices,
                assignment_model_indices=model_indices,
            )

    summaries = []
    group_keys = sorted({(row["result"], row["run"]) for row in rows})
    for result_name, run in group_keys:
        group = [row for row in rows if row["result"] == result_name and row["run"] == run]
        summaries.append(
            {
                "result": result_name,
                "run": run,
                "n_molecules": len(group),
                "mean_frequency_mae_cm-1": float(
                    np.mean([row["frequency_mae_cm-1"] for row in group])
                ),
                "mean_frequency_rmse_cm-1": float(
                    np.mean([row["frequency_rmse_cm-1"] for row in group])
                ),
                "mean_mode_overlap": float(
                    np.mean([row["mean_mode_overlap"] for row in group])
                ),
                "total_pbe_imaginary_modes": sum(
                    row["pbe_imaginary_modes"] for row in group
                ),
                "total_model_imaginary_modes": sum(
                    row["model_imaginary_modes"] for row in group
                ),
            }
        )
    output = {
        "definition": (
            "Hessians are explicitly symmetrized, mass weighted with PySCF isotope-averaged "
            "masses, projected into the complement of translation/rotation modes, and matched "
            "to PBE modes by maximum absolute eigenvector overlap."
        ),
        "warning": (
            "Symmetrization is numerical postprocessing for vibrational diagnostics and is not "
            "evidence that an originally non-conservative force field is physically repaired."
        ),
        "assumed_hessian_units": "Hartree / Bohr^2",
        "frequency_conversion_cm-1_per_sqrt_au_amu": FREQUENCY_CONVERSION_CM,
        "imaginary_threshold_cm-1": args.imaginary_threshold_cm,
        "reference_manifest": args.manifest_json.resolve().as_posix(),
        "reference_manifest_sha256": hashlib.sha256(
            args.manifest_json.read_bytes()
        ).hexdigest(),
        "dataset_dir": args.dataset_dir.resolve().as_posix(),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "rows": rows,
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "per_molecule_vibrational_metrics.csv", rows)
    _write_csv(args.output_dir / "vibrational_summary.csv", summaries)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--result-json", action="append", default=[])
    parser.add_argument(
        "--capacity-result",
        action="append",
        default=[],
        help="NAME=DIR for a train-only capacity output containing summary.json and *_result.npz",
    )
    parser.add_argument(
        "--baseline-result",
        action="append",
        default=[],
        help="NAME=MANIFEST for a frozen baseline manifest containing capacity_array paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imaginary-threshold-cm", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
