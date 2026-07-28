"""Sharded full-content audit for legacy Zarr-ZIP density-training labels."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import zarr
from rdkit import Chem, RDLogger


BASIS_DIMENSIONS = {1: 20, 6: 109, 7: 116, 8: 116, 9: 116}
COMMON_ARRAYS = (
    "geometry/atom_pos",
    "geometry/atomic_numbers",
    "geometry/mol_id",
    "ks_labels/basis",
    "ks_labels/energies/e_electron",
    "ks_labels/energies/e_ext",
    "ks_labels/energies/e_hartree",
    "ks_labels/energies/e_kin",
    "ks_labels/energies/e_nuc_nuc",
    "ks_labels/energies/e_tot",
    "ks_labels/energies/e_xc",
    "ks_labels/energies/has_energy_label",
    "of_labels/basis",
    "of_labels/energies/e_electron",
    "of_labels/energies/e_ext",
    "of_labels/energies/e_hartree",
    "of_labels/energies/e_kin",
    "of_labels/energies/e_kin_minus_apbe",
    "of_labels/energies/e_kin_plus_xc",
    "of_labels/energies/e_kinapbe",
    "of_labels/energies/e_tot",
    "of_labels/energies/e_xc",
    "of_labels/n_scf_steps",
    "of_labels/spatial/coeffs",
    "of_labels/spatial/dual_basis_integrals",
    "of_labels/spatial/grad_kin",
    "of_labels/spatial/grad_kin_minus_apbe",
    "of_labels/spatial/grad_kin_plus_xc",
    "of_labels/spatial/grad_tot",
)


def _read_array(array: zarr.Array) -> np.ndarray:
    return np.asarray(array[()] if array.shape == () else array[:])


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _geometry_hash(atomic_numbers: np.ndarray, positions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atomic_numbers, dtype="<u2").tobytes())
    digest.update(np.asarray(np.round(positions, 10), dtype="<f8").tobytes())
    return digest.hexdigest()


def _distance_invariant_hash(atomic_numbers: np.ndarray, positions: np.ndarray) -> str:
    pairs = []
    for left in range(len(atomic_numbers)):
        for right in range(left + 1, len(atomic_numbers)):
            z_left = int(atomic_numbers[left])
            z_right = int(atomic_numbers[right])
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            pairs.append((min(z_left, z_right), max(z_left, z_right), round(distance, 7)))
    value = json.dumps(
        {
            "composition": sorted(collections.Counter(map(int, atomic_numbers)).items()),
            "pairs": sorted(pairs),
        },
        separators=(",", ":"),
    )
    return _sha256_text(value)


def _configuration_hashes(
    atomic_numbers: np.ndarray, positions: np.ndarray, coefficients: np.ndarray
) -> list[str]:
    """Hash every exact geometry+density-coefficient configuration.

    The atom and coefficient order are intentionally retained: a match is an
    exact stored-configuration duplicate, while parent/canonical-SMILES and
    distance-invariant leakage are audited separately.
    """
    prefix = hashlib.sha256()
    prefix.update(np.asarray(atomic_numbers, dtype="<u2").tobytes())
    prefix.update(np.asarray(positions, dtype="<f8").tobytes())
    hashes = []
    for row in np.asarray(coefficients, dtype="<f8"):
        digest = prefix.copy()
        digest.update(row.tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def _formal_charges(data_root: Path, source: str) -> tuple[dict[str, int], list[dict]]:
    if source == "QM9_perturbed_fock":
        return {}, []
    csv_name = "QMUGSBin0.csv" if source == "QMUGS_perturbed_fock" else "QMUGSLargeBins.csv"
    charges = {}
    failures = []
    with (data_root / "QMUGS" / csv_name).open(newline="") as handle:
        for row in csv.DictReader(handle):
            filename = f"{int(row['id']):07d}.zarr.zip"
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is None:
                failures.append({"filename": filename, "reason": "invalid_smiles"})
                continue
            charges[filename] = int(Chem.GetFormalCharge(molecule))
    return charges, failures


def _schema_record(root: zarr.Group) -> tuple[str, dict, int, dict[str, np.ndarray]]:
    # Molecular, SCF-step, and coefficient dimensions legitimately vary between
    # archives. The schema fingerprint therefore records ranks rather than
    # concrete extents; concrete extents are checked below and retained per row.
    schema = {}
    nonfinite = 0
    values = {}

    def visit(name: str, obj) -> None:
        nonlocal nonfinite
        if not isinstance(obj, zarr.Array):
            return
        value = _read_array(obj)
        dtype_name = (
            "unicode"
            if value.dtype.kind == "U"
            else "bytes"
            if value.dtype.kind == "S"
            else str(value.dtype)
        )
        schema[name] = {"dtype": dtype_name, "ndim": value.ndim}
        if name in COMMON_ARRAYS:
            values[name] = value
        if np.issubdtype(value.dtype, np.number):
            nonfinite += int((~np.isfinite(value)).sum())

    root.visititems(visit)
    rendered = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return _sha256_text(rendered), schema, nonfinite, values


def _audit_label(path: Path, source: str, formal_charge: int) -> tuple[dict, list[str], dict]:
    anomalies = []
    store = zarr.ZipStore(path, mode="r")
    try:
        root = zarr.group(store=store)
        names = set()

        def collect(name: str, obj) -> None:
            if isinstance(obj, zarr.Array):
                names.add(name)

        root.visititems(collect)
        missing_arrays = sorted(set(COMMON_ARRAYS) - names)
        if missing_arrays:
            anomalies.append("missing_required_arrays")

        schema_hash, schema, nonfinite, values = _schema_record(root)
        if nonfinite:
            anomalies.append("nonfinite_values")

        positions = np.asarray(values["geometry/atom_pos"], dtype=np.float64)
        atomic_numbers = np.asarray(values["geometry/atomic_numbers"], dtype=np.int64)
        molecule_id = str(values["geometry/mol_id"].item())
        coeffs = np.asarray(values["of_labels/spatial/coeffs"], dtype=np.float64)
        integrals = np.asarray(
            values["of_labels/spatial/dual_basis_integrals"], dtype=np.float64
        )
        has_energy = np.asarray(
            values["ks_labels/energies/has_energy_label"], dtype=bool
        )
        steps = int(values["of_labels/n_scf_steps"].item())
        ks_basis = str(values["ks_labels/basis"].item())
        of_basis = str(values["of_labels/basis"].item())

        if positions.shape != (len(atomic_numbers), 3):
            anomalies.append("geometry_shape_mismatch")
        if coeffs.ndim != 2 or integrals.shape != coeffs.shape:
            anomalies.append("coefficient_integral_shape_mismatch")
        if coeffs.shape[0] != steps or len(has_energy) != steps:
            anomalies.append("scf_step_shape_mismatch")
        energy_shape_mismatches = sorted(
            name
            for name, value in values.items()
            if "/energies/" in name and value.shape != (steps,)
        )
        spatial_shape_mismatches = sorted(
            name
            for name, value in values.items()
            if name.startswith("of_labels/spatial/") and value.shape != coeffs.shape
        )
        if energy_shape_mismatches:
            anomalies.append("energy_array_shape_mismatch")
        if spatial_shape_mismatches:
            anomalies.append("spatial_array_shape_mismatch")
        unsupported_elements = sorted(set(map(int, atomic_numbers)) - set(BASIS_DIMENSIONS))
        expected_coefficients = (
            sum(BASIS_DIMENSIONS[int(z)] for z in atomic_numbers)
            if not unsupported_elements
            else None
        )
        if expected_coefficients is not None and coeffs.shape[1] != expected_coefficients:
            anomalies.append("basis_dimension_mismatch")
        if ks_basis != "6-31G(2df,p)":
            anomalies.append("unexpected_ks_basis")
        if not of_basis.endswith("basis_info.npz"):
            anomalies.append("unexpected_of_basis_metadata")

        expected_electrons = int(atomic_numbers.sum()) - formal_charge
        electron_counts = np.einsum("ij,ij->i", coeffs, integrals)
        labeled_electron_errors = np.abs(electron_counts[has_energy] - expected_electrons)
        electron_error_max = (
            float(labeled_electron_errors.max()) if labeled_electron_errors.size else math.nan
        )
        electron_error_final = float(abs(electron_counts[-1] - expected_electrons))
        electron_relative_error_max = electron_error_max / expected_electrons
        electron_relative_error_final = electron_error_final / expected_electrons
        if not math.isfinite(electron_error_max):
            anomalies.append("no_energy_labeled_steps")
        # This is the warning threshold used by the historical label generator
        # (mldft/datagen/methods/label_generation.py), not an invented stricter
        # absolute tolerance.
        elif electron_relative_error_max > 1.0e-3:
            anomalies.append("electron_count_relative_error_gt_1e-3")

        composition = dict(sorted(collections.Counter(map(int, atomic_numbers)).items()))
        configuration_hashes = _configuration_hashes(atomic_numbers, positions, coeffs)
        trajectory_digest = hashlib.sha256()
        for value in configuration_hashes:
            trajectory_digest.update(value.encode())
        row = {
            "anomalies": anomalies,
            "archive_bytes": path.stat().st_size,
            "archive_members": len(store.zf.infolist()),
            "atomic_order_sha256": hashlib.sha256(
                np.asarray(atomic_numbers, dtype="<u2").tobytes()
            ).hexdigest(),
            "coefficients": int(coeffs.shape[1]),
            "composition": composition,
            "configuration_step_sha256": configuration_hashes,
            "configuration_trajectory_sha256": trajectory_digest.hexdigest(),
            "distance_invariant_sha256": _distance_invariant_hash(atomic_numbers, positions),
            "electron_count_error_final": electron_error_final,
            "electron_count_error_max_labeled": electron_error_max,
            "electron_count_relative_error_final": electron_relative_error_final,
            "electron_count_relative_error_max_labeled": electron_relative_error_max,
            "energy_shape_mismatches": energy_shape_mismatches,
            "expected_coefficients": expected_coefficients,
            "expected_electrons": expected_electrons,
            "filename": path.name,
            "formal_charge": formal_charge,
            "geometry_sha256": _geometry_hash(atomic_numbers, positions),
            "has_energy_steps": int(has_energy.sum()),
            "ks_basis": ks_basis,
            "molecule_id": molecule_id,
            "n_atoms": int(len(atomic_numbers)),
            "nonfinite_values": nonfinite,
            "of_basis_recorded_path": of_basis,
            "schema_sha256": schema_hash,
            "scf_steps": steps,
            "source": source,
            "spatial_shape_mismatches": spatial_shape_mismatches,
            "unsupported_elements": unsupported_elements,
        }
        return row, anomalies, schema
    finally:
        store.close()


def _selected_files(labels_root: Path, shard_index: int, num_shards: int) -> list[Path]:
    selected = []
    for entry in os.scandir(labels_root):
        if not entry.is_file() or not entry.name.endswith(".zarr.zip"):
            continue
        numeric_id = int(entry.name.split(".", 1)[0])
        if numeric_id % num_shards == shard_index:
            selected.append(Path(entry.path))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("QM9_perturbed_fock", "QMUGS_perturbed_fock", "QMUGS"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-jsonl-gz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    RDLogger.DisableLog("rdApp.*")
    data_root = args.data_root.resolve()
    labels_root = data_root / args.source / "labels"
    formal_charges, metadata_failures = _formal_charges(data_root, args.source)
    files = _selected_files(labels_root, args.shard_index, args.num_shards)

    args.output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output_jsonl_gz.with_suffix(args.output_jsonl_gz.suffix + ".partial")
    anomaly_counts = collections.Counter()
    schema_counts = collections.Counter()
    schemas = {}
    atom_counts = collections.Counter()
    composition_counts = collections.Counter()
    failures = []
    processed = 0
    started = time.perf_counter()
    with gzip.open(partial_output, "wt") as output:
        for path in files:
            try:
                formal_charge = formal_charges.get(path.name, 0)
                row, anomalies, schema = _audit_label(path, args.source, formal_charge)
                output.write(json.dumps(row, sort_keys=True) + "\n")
                anomaly_counts.update(anomalies)
                schema_counts[row["schema_sha256"]] += 1
                schemas.setdefault(row["schema_sha256"], schema)
                atom_counts[str(row["n_atoms"])] += 1
                composition_counts[json.dumps(row["composition"], sort_keys=True)] += 1
            except Exception as error:
                failures.append(
                    {"filename": path.name, "error_type": type(error).__name__, "error": repr(error)}
                )
            processed += 1
            if args.progress_interval and processed % args.progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "elapsed_s": time.perf_counter() - started,
                            "failures": len(failures),
                            "processed": processed,
                            "selected": len(files),
                            "shard": args.shard_index,
                            "source": args.source,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    partial_output.replace(args.output_jsonl_gz)

    summary = {
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "atom_count_distribution": dict(sorted(atom_counts.items(), key=lambda item: int(item[0]))),
        "composition_counts": dict(sorted(composition_counts.items())),
        "elapsed_s": time.perf_counter() - started,
        "failures": failures,
        "metadata_failures": metadata_failures,
        "num_shards": args.num_shards,
        "output_jsonl_gz": str(args.output_jsonl_gz.resolve()),
        "processed": processed,
        "schema_counts": dict(sorted(schema_counts.items())),
        "schemas": schemas,
        "selected": len(files),
        "shard_index": args.shard_index,
        "source": args.source,
        "status": "passed" if not failures and not anomaly_counts else "completed_with_anomalies",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
