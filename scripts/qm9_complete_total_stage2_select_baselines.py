#!/usr/bin/env python3
"""Select the first frozen, numerically stable Stage-2 complete-total baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_manifest_parent(row: dict[str, Any]) -> dict[str, Any]:
    asym_over_pbe = row.get("baseline_asym_over_pbe_frobenius")
    if asym_over_pbe is None:
        with np.load(row["capacity_array"]) as payload:
            predicted = np.asarray(payload["predicted_hessian"], dtype=np.float64)
            reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        asym_over_pbe = float(
            np.linalg.norm(0.5 * (predicted - predicted.T))
            / max(np.linalg.norm(reference), np.finfo(float).tiny)
        )
    return {
        **row,
        "baseline_asym_over_pbe_frobenius": float(asym_over_pbe),
        "baseline_max_density_gradient_norm": float(
            row["baseline_max_density_gradient_norm"]
        ),
        "full_hessian_complete": True,
    }


def _run_parent(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    metric_path = run_dir / "full_hessian_metrics.csv"
    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one Hessian row in {metric_path}, got {len(rows)}")
    metric = rows[0]
    molecule_id = str(metric["molecule_id"])
    if molecule_id != str(candidate["molecule_id"]):
        raise ValueError(f"run/candidate mismatch: {molecule_id} != {candidate['molecule_id']}")
    step = int(metric["step"])
    capacity_array = (
        run_dir / "hessian_arrays" / f"step_{step:07d}_{molecule_id}.npz"
    ).resolve()
    with np.load(capacity_array) as payload:
        predicted = np.asarray(payload["predicted_hessian"], dtype=np.float64)
        reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
    antisymmetric = 0.5 * (predicted - predicted.T)
    asym_over_pbe = float(
        np.linalg.norm(antisymmetric)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )
    return {
        "molecule_id": molecule_id,
        "natoms": int(metric["natoms"]),
        "label_path": candidate["label_path"],
        "label_sha256": candidate["label_sha256"],
        "pbe_hessian_path": candidate["pbe_hessian_path"],
        "pbe_hessian_sha256": candidate["pbe_hessian_sha256"],
        "baseline_run_dir": run_dir.resolve().as_posix(),
        "baseline_step": step,
        "baseline_total_energy_hartree": float(metric["total_energy_hartree"]),
        "baseline_energy_abs_error_hartree": float(
            metric["total_energy_abs_error_hartree"]
        ),
        "baseline_force_mae_hartree_per_bohr": float(
            metric["complete_total_force_mae_hartree_per_bohr"]
        ),
        "baseline_hessian_relative_frobenius": float(metric["relative_frobenius"]),
        "baseline_max_density_gradient_norm": float(
            metric["max_cached_density_gradient_norm"]
        ),
        "baseline_asym_over_pbe_frobenius": asym_over_pbe,
        "full_hessian_complete": metric["full_hessian_complete"].lower() == "true",
        "capacity_array": capacity_array.as_posix(),
        "capacity_array_sha256": _sha256(capacity_array),
    }


def _load_provenance_audit(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("test100_accessed") is not False:
        raise ValueError("checkpoint provenance audit does not certify frozen Test100")
    return {
        str(Path(row["baseline_run_dir"]).resolve()): row
        for row in payload.get("records", [])
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    frozen = json.loads(args.candidate_manifest.read_text())
    if frozen.get("test100_accessed") is not False:
        raise ValueError("candidate manifest does not certify frozen Test100")
    candidates = {str(row["molecule_id"]): row for row in frozen["candidates"]}
    available: dict[str, dict[str, Any]] = {}
    for path in args.known_baseline_manifest:
        manifest = json.loads(path.read_text())
        if manifest.get("test100_accessed") is not False:
            raise ValueError(f"{path} does not certify frozen Test100")
        for row in manifest["parents"]:
            molecule_id = str(row["molecule_id"])
            if molecule_id in candidates:
                available.setdefault(molecule_id, _normalize_manifest_parent(row))
    run_dirs = list(args.run_dir)
    for root in args.run_root:
        run_dirs.extend(path for path in sorted(root.glob("*_job*")) if path.is_dir())
    for run_dir in run_dirs:
        metric_path = run_dir / "full_hessian_metrics.csv"
        if not metric_path.is_file():
            continue
        with metric_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            continue
        molecule_id = str(rows[0]["molecule_id"])
        if molecule_id not in candidates:
            raise ValueError(f"non-frozen candidate found in {run_dir}: {molecule_id}")
        available[molecule_id] = _run_parent(run_dir, candidates[molecule_id])

    curl_gate = float(
        frozen["numerical_gate"]["baseline_asym_over_pbe_frobenius_max"]
    )
    density_gate = float(frozen["numerical_gate"]["strict_density_gradient_max"])
    target_count = int(frozen["final_parent_count"])
    selected = []
    audit_rows = []
    direction_lookup = {
        str(row["molecule_id"]): row for row in frozen["directions"]
    }
    provenance_path = getattr(args, "provenance_audit", None)
    provenance = (
        _load_provenance_audit(provenance_path) if provenance_path is not None else None
    )
    for candidate in frozen["candidates"]:
        molecule_id = str(candidate["molecule_id"])
        baseline = available.get(molecule_id)
        if baseline is None:
            audit_rows.append(
                {
                    "candidate_order": candidate["candidate_order"],
                    "molecule_id": molecule_id,
                    "natoms": candidate["natoms"],
                    "status": "missing",
                    "selected": False,
                }
            )
            continue
        reasons = []
        if not baseline["full_hessian_complete"]:
            reasons.append("incomplete_hessian")
        if baseline["baseline_max_density_gradient_norm"] > density_gate:
            reasons.append("density_gradient")
        if baseline["baseline_asym_over_pbe_frobenius"] > curl_gate:
            reasons.append("pbe_normalized_curl")
        if provenance is not None:
            run_dir = str(Path(baseline.get("baseline_run_dir", "")).resolve())
            provenance_row = provenance.get(run_dir)
            if provenance_row is None:
                reasons.append("missing_checkpoint_provenance")
            elif not (
                provenance_row.get("status") == "pass"
                and provenance_row.get("base_state_exact_match") is True
            ):
                reasons.append("checkpoint_provenance")
        passes = not reasons
        take = passes and len(selected) < target_count
        audit_rows.append(
            {
                "candidate_order": candidate["candidate_order"],
                "molecule_id": molecule_id,
                "natoms": candidate["natoms"],
                "status": "pass" if passes else "fail",
                "failure_reasons": ";".join(reasons),
                "selected": take,
                "baseline_asym_over_pbe_frobenius": baseline[
                    "baseline_asym_over_pbe_frobenius"
                ],
                "baseline_max_density_gradient_norm": baseline[
                    "baseline_max_density_gradient_norm"
                ],
                "baseline_hessian_relative_frobenius": baseline[
                    "baseline_hessian_relative_frobenius"
                ],
            }
        )
        if take:
            selected.append(
                {
                    **baseline,
                    "candidate_order": int(candidate["candidate_order"]),
                    "direction_path": direction_lookup[molecule_id]["direction_path"],
                    "direction_sha256": direction_lookup[molecule_id][
                        "direction_sha256"
                    ],
                }
            )
    if len(selected) != target_count:
        raise ValueError(
            f"only {len(selected)} stable baselines available; expected {target_count}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in audit_rows for key in row})
    with (args.output_dir / "selection_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    result = {
        "definition": "first-20 frozen train-parent strict complete-total baseline selection",
        "source_candidate_manifest": args.candidate_manifest.resolve().as_posix(),
        "source_candidate_manifest_sha256": _sha256(args.candidate_manifest),
        "source_split_sha256": frozen["source_split_sha256"],
        "protocol_id": frozen["protocol_id"],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "baseline_asym_over_pbe_gate": curl_gate,
        "baseline_density_gradient_gate": density_gate,
        "checkpoint_provenance_audit": (
            provenance_path.resolve().as_posix() if provenance_path is not None else None
        ),
        "checkpoint_provenance_audit_sha256": (
            _sha256(provenance_path) if provenance_path is not None else None
        ),
        "parent_count": len(selected),
        "parents": selected,
        "audit_rows": audit_rows,
    }
    output = args.output_dir / "selected_baseline_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_parent_ids": [row["molecule_id"] for row in selected],
                "manifest": output.as_posix(),
                "test100_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--known-baseline-manifest", type=Path, action="append", default=[])
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--provenance-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    select(parse_args())
