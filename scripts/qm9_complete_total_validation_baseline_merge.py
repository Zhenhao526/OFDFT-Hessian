#!/usr/bin/env python3
"""Validate and merge independent validation-parent complete-total baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.qm9_complete_total_stage2_select_baselines import (
        _load_provenance_audit,
        _run_parent,
        _sha256,
    )
except ModuleNotFoundError:
    from qm9_complete_total_stage2_select_baselines import (
        _load_provenance_audit,
        _run_parent,
        _sha256,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    frozen = json.loads(args.capacity_manifest.read_text())
    if frozen.get("test100_accessed") is not False:
        raise ValueError("validation capacity manifest does not freeze Test100")
    candidates = {
        str(row["molecule_id"]): row for row in frozen.get("parents", [])
    }
    if len(candidates) != int(frozen.get("parent_count", -1)):
        raise ValueError("validation capacity manifest parent count is inconsistent")

    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("scope", {}).get("test100_accessed") is not False:
        raise ValueError("Stage-3 protocol does not freeze Test100")
    density_gate = float(protocol["stability"]["strict_density_gradient_max"])
    curl_gate = float(
        protocol["stability"]["baseline_asym_over_pbe_frobenius_max"]
    )
    provenance = _load_provenance_audit(args.provenance_audit)

    available: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(args.run_root.glob("*_job*")):
        metric_path = run_dir / "full_hessian_metrics.csv"
        summary_path = run_dir / "summary.json"
        if not metric_path.is_file() or not summary_path.is_file():
            continue
        with metric_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            continue
        molecule_id = str(rows[0]["molecule_id"])
        if molecule_id not in candidates:
            raise ValueError(f"non-frozen validation parent in {run_dir}: {molecule_id}")
        available[molecule_id] = _run_parent(run_dir, candidates[molecule_id])

    parents = []
    audit_rows = []
    for order, candidate in enumerate(frozen["parents"]):
        molecule_id = str(candidate["molecule_id"])
        baseline = available.get(molecule_id)
        reasons = []
        if baseline is None:
            reasons.append("missing_complete_total_baseline")
        else:
            if not baseline["full_hessian_complete"]:
                reasons.append("incomplete_hessian")
            if baseline["baseline_max_density_gradient_norm"] > density_gate:
                reasons.append("density_gradient")
            if baseline["baseline_asym_over_pbe_frobenius"] > curl_gate:
                reasons.append("pbe_normalized_curl")
            run_dir = str(Path(baseline["baseline_run_dir"]).resolve())
            provenance_row = provenance.get(run_dir)
            if provenance_row is None:
                reasons.append("missing_checkpoint_provenance")
            elif not (
                provenance_row.get("status") == "pass"
                and provenance_row.get("base_state_exact_match") is True
            ):
                reasons.append("checkpoint_provenance")
        audit_rows.append(
            {
                "candidate_order": order,
                "molecule_id": molecule_id,
                "natoms": candidate["natoms"],
                "status": "pass" if not reasons else "failed",
                "failure_reasons": ";".join(reasons),
                "baseline_max_density_gradient_norm": (
                    None
                    if baseline is None
                    else baseline["baseline_max_density_gradient_norm"]
                ),
                "baseline_asym_over_pbe_frobenius": (
                    None
                    if baseline is None
                    else baseline["baseline_asym_over_pbe_frobenius"]
                ),
                "baseline_hessian_relative_frobenius": (
                    None
                    if baseline is None
                    else baseline["baseline_hessian_relative_frobenius"]
                ),
            }
        )
        if not reasons:
            parents.append({**baseline, "candidate_order": order})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "validation_baseline_audit.csv"
    _write_csv(audit_path, audit_rows)
    result = {
        "definition": (
            "Independent validation-parent original-A strict complete-total "
            "density-relaxed full-Hessian baselines."
        ),
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_capacity_manifest": args.capacity_manifest.resolve().as_posix(),
        "source_capacity_manifest_sha256": _sha256(args.capacity_manifest),
        "source_split": args.source_split.resolve().as_posix(),
        "source_split_sha256": _sha256(args.source_split),
        "checkpoint_provenance_audit": args.provenance_audit.resolve().as_posix(),
        "checkpoint_provenance_audit_sha256": _sha256(args.provenance_audit),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "strict_density_gradient_max": density_gate,
        "baseline_asym_over_pbe_frobenius_max": curl_gate,
        "expected_parent_count": len(candidates),
        "parent_count": len(parents),
        "complete": len(parents) == len(candidates),
        "parents": parents,
        "audit_rows": audit_rows,
        "audit_csv": audit_path.resolve().as_posix(),
        "audit_csv_sha256": _sha256(audit_path),
    }
    output = args.output_dir / "validation_baseline_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"parents", "audit_rows"}}, indent=2, sort_keys=True))
    if args.require_complete and not result["complete"]:
        raise RuntimeError(
            f"only {len(parents)}/{len(candidates)} validation baselines passed"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provenance-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
