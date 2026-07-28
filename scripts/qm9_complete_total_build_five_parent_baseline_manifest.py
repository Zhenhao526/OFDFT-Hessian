#!/usr/bin/env python3
"""Build the frozen five-parent strict baseline manifest for capacity fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    stage1_manifest_path: Path,
    run_dirs: list[Path],
    output_path: Path,
    max_baseline_asym_over_pbe: float | None = None,
) -> dict[str, object]:
    stage1 = json.loads(stage1_manifest_path.read_text())
    if stage1.get("test100_accessed") is not False:
        raise ValueError("Stage1 manifest does not certify frozen Test100")
    frozen = {str(row["molecule_id"]): row for row in stage1["parents"]}
    discovered: dict[str, dict[str, object]] = {}
    for run_dir in run_dirs:
        metric_path = run_dir / "full_hessian_metrics.csv"
        with metric_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected one metric row in {metric_path}, got {len(rows)}")
        metric = rows[0]
        molecule_id = str(metric["molecule_id"])
        if molecule_id not in frozen:
            raise ValueError(f"run contains non-frozen parent {molecule_id}")
        if molecule_id in discovered:
            raise ValueError(f"duplicate baseline run for {molecule_id}")
        if metric["full_hessian_complete"].strip().lower() != "true":
            raise ValueError(f"incomplete Hessian for {molecule_id}")
        step = int(metric["step"])
        capacity_array = (
            run_dir / "hessian_arrays" / f"step_{step:07d}_{molecule_id}.npz"
        ).resolve()
        if not capacity_array.is_file():
            raise FileNotFoundError(capacity_array)
        with np.load(capacity_array) as payload:
            predicted_hessian = np.asarray(payload["predicted_hessian"], dtype=float)
            pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=float)
        antisymmetric = 0.5 * (predicted_hessian - predicted_hessian.T)
        baseline_asym_over_pbe = float(
            np.linalg.norm(antisymmetric)
            / max(
                np.linalg.norm(pbe_hessian),
                np.finfo(float).tiny,
            )
        )
        if (
            max_baseline_asym_over_pbe is not None
            and baseline_asym_over_pbe > max_baseline_asym_over_pbe
        ):
            raise ValueError(
                f"baseline curl gate failed for {molecule_id}: "
                f"{baseline_asym_over_pbe} > {max_baseline_asym_over_pbe}"
            )
        source = frozen[molecule_id]
        discovered[molecule_id] = {
            "molecule_id": molecule_id,
            "natoms": int(metric["natoms"]),
            "label_path": str(Path(source["label_path"]).resolve()),
            "label_sha256": source["label_sha256"],
            "pbe_hessian_path": source["pbe_hessian_path"],
            "pbe_hessian_sha256": source["pbe_hessian_sha256"],
            "baseline_run_dir": str(run_dir.resolve()),
            "baseline_step": step,
            "baseline_total_energy_hartree": float(metric["total_energy_hartree"]),
            "baseline_energy_abs_error_hartree": float(
                metric["total_energy_abs_error_hartree"]
            ),
            "baseline_force_mae_hartree_per_bohr": float(
                metric["complete_total_force_mae_hartree_per_bohr"]
            ),
            "baseline_hessian_relative_frobenius": float(
                metric["relative_frobenius"]
            ),
            "baseline_max_density_gradient_norm": float(
                metric["max_cached_density_gradient_norm"]
            ),
            "baseline_asym_over_pbe_frobenius": baseline_asym_over_pbe,
            "capacity_array": str(capacity_array),
            "capacity_array_sha256": _sha256(capacity_array),
        }
    missing = sorted(set(frozen) - set(discovered))
    if missing:
        raise ValueError(f"missing frozen baseline parents: {missing}")
    result = {
        "definition": "strict complete-total density-relaxed five-parent baseline",
        "source_stage1_manifest": str(stage1_manifest_path.resolve()),
        "source_stage1_manifest_sha256": _sha256(stage1_manifest_path),
        "source_split_sha256": stage1["split_sha256"],
        "protocol_sha256": stage1["protocol_sha256"],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "baseline_asym_over_pbe_gate": max_baseline_asym_over_pbe,
        "parents": [discovered[molecule_id] for molecule_id in frozen],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-baseline-asym-over-pbe", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_manifest(
                args.stage1_manifest,
                args.run_dir,
                args.output,
                args.max_baseline_asym_over_pbe,
            ),
            indent=2,
            sort_keys=True,
        )
    )
