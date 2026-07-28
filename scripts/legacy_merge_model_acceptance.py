"""Compare CPU/GPU recovery acceptance and freeze CPU-float64 golden outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ("QM9_perturbed_fock", "QMUGSBin0_perturbed_fock")
MODES = ("cpu_float64", "gpu_float32", "gpu_float64")
TOLERANCES = {
    "gpu_float64": {
        "energy": {"max_abs": 1.0e-8, "relative_l2": 1.0e-8},
        "coefficient_outputs": {"max_abs": 1.0e-8, "relative_l2": 1.0e-8},
    },
    "gpu_float32": {
        # Energy is extensive.  Gate it by a scale-free relative error and a
        # per-electron absolute error, rather than a molecule-size-dependent
        # fixed absolute threshold.
        "energy": {"max_abs_per_electron": 5.0e-6, "relative_l2": 5.0e-6},
        "coefficient_outputs": {"max_abs": 5.0e-4, "relative_l2": 5.0e-4},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: dict) -> str:
    return (
        f"{row['partition']}:{row['source']}/{row['filename']}:"
        f"{row['scf_iteration']}"
    )


def _comparison(reference, candidate) -> dict:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_match": False, "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
    difference = right - left
    denominator = np.linalg.norm(left.ravel())
    return {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "relative_l2": float(
            np.linalg.norm(difference.ravel()) / max(denominator, np.finfo(np.float64).tiny)
        ),
        "shape_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "datasets": {},
        "golden_policy": (
            "CPU float64 output is frozen only after its determinism/batch checks and both GPU "
            "precision comparisons pass preregistered tolerances"
        ),
        "tolerances": TOLERANCES,
    }
    global_errors = []
    for dataset in DATASETS:
        mode_reports = {}
        artifact_records = {}
        for mode in MODES:
            path = args.acceptance_root / f"{dataset}__{mode}.json.gz"
            summary_path = args.acceptance_root / f"{dataset}__{mode}.summary.json"
            with gzip.open(path, "rt") as handle:
                mode_reports[mode] = json.load(handle)
            summary = json.loads(summary_path.read_text())
            artifact_records[mode] = {
                "report": str(path.resolve()),
                "report_sha256": _sha256(path),
                "summary": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
                "status": summary["status"],
            }

        errors = []
        if any(item["status"] != "passed" for item in artifact_records.values()):
            errors.append("one or more mode-level acceptance runs failed")
        checkpoint_hashes = {
            value["checkpoint_sha256"] for value in mode_reports.values()
        }
        if len(checkpoint_hashes) != 1:
            errors.append("checkpoint SHA differs across modes")

        indexed = {
            mode: {_key(row): row for row in value["results"]}
            for mode, value in mode_reports.items()
        }
        golden_keys = set(indexed["cpu_float64"])
        if any(set(rows) != golden_keys for rows in indexed.values()):
            errors.append("sample identity differs across modes")
        comparisons = []
        maxima = {
            mode: {
                tensor: {
                    "max_abs": 0.0,
                    "relative_l2": 0.0,
                    **({"max_abs_per_electron": 0.0} if tensor == "energy" else {}),
                }
                for tensor in ("energy", "density_gradient", "difference")
            }
            for mode in ("gpu_float32", "gpu_float64")
        }
        order_metadata_match = True
        electron_diagnostics = []
        for key in sorted(golden_keys):
            reference = indexed["cpu_float64"][key]
            reference_metadata = (
                reference["input"]["atomic_numbers"],
                reference["input"]["n_basis_per_atom"],
                reference["input"]["coefficients"],
            )
            for mode in ("gpu_float32", "gpu_float64"):
                candidate = indexed[mode][key]
                candidate_metadata = (
                    candidate["input"]["atomic_numbers"],
                    candidate["input"]["n_basis_per_atom"],
                    candidate["input"]["coefficients"],
                )
                if candidate_metadata != reference_metadata:
                    order_metadata_match = False
                tensor_comparisons = {}
                for tensor in ("energy", "density_gradient", "difference"):
                    value = _comparison(
                        reference["first"]["outputs"][tensor],
                        candidate["first"]["outputs"][tensor],
                    )
                    tensor_comparisons[tensor] = value
                    if value.get("shape_match"):
                        for metric in ("max_abs", "relative_l2"):
                            maxima[mode][tensor][metric] = max(
                                maxima[mode][tensor][metric], value[metric]
                            )
                        if tensor == "energy":
                            expected_electrons = float(
                                reference["first"]["expected_electrons"]
                            )
                            value["max_abs_per_electron"] = (
                                value["max_abs"] / expected_electrons
                            )
                            maxima[mode][tensor]["max_abs_per_electron"] = max(
                                maxima[mode][tensor]["max_abs_per_electron"],
                                value["max_abs_per_electron"],
                            )
                comparisons.append(
                    {"key": key, "mode": mode, "tensors": tensor_comparisons}
                )
            electron_diagnostics.append(
                {
                    "key": key,
                    "predicted_ground_state_electron_error": {
                        mode: indexed[mode][key]["first"][
                            "predicted_ground_state_electron_error"
                        ]
                        for mode in MODES
                    },
                }
            )
        if not order_metadata_match:
            errors.append("atomic/coefficient ordering metadata differs across modes")
        for mode, tensor_values in maxima.items():
            for tensor, metrics in tensor_values.items():
                tolerance = TOLERANCES[mode][
                    "energy" if tensor == "energy" else "coefficient_outputs"
                ]
                if any(metrics[metric] > limit for metric, limit in tolerance.items()):
                    errors.append(
                        f"{mode}/{tensor} exceeds tolerance: {metrics}"
                    )

        dataset_info = args.data_root / dataset / "dataset_info.yaml"
        golden_path = args.acceptance_root / f"{dataset}__cpu_float64.json.gz"
        dataset_report = {
            "artifacts": artifact_records,
            "basis_metadata": {
                "dataset_info": str(dataset_info.resolve()),
                "dataset_info_sha256": _sha256(dataset_info),
                "order_metadata_match": order_metadata_match,
            },
            "checkpoint_sha256": sorted(checkpoint_hashes),
            "comparisons": comparisons,
            "electron_number_diagnostics": electron_diagnostics,
            "errors": errors,
            "golden_report": str(golden_path.resolve()),
            "golden_report_sha256": _sha256(golden_path),
            "maxima": maxima,
            "samples": len(golden_keys),
            "status": "passed" if not errors else "failed",
        }
        report["datasets"][dataset] = dataset_report
        global_errors.extend(f"{dataset}: {error}" for error in errors)

    report["errors"] = global_errors
    report["status"] = "passed" if not global_errors else "failed"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": global_errors}, sort_keys=True))
    if global_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
