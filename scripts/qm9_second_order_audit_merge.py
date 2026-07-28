#!/usr/bin/env python3
"""Merge independently evaluated fixed-density Hessian/HVP audit shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LIST_FIELDS = (
    "fixed_density_full_hessian",
    "hvp",
    "drop_self_edges_diagnostic",
    "unrolled_density_prototype",
)
CONSISTENT_FIELDS = (
    "compare_pbe_force_secant",
    "definition",
    "device",
    "directional_sample_ids",
    "fd_displacement",
    "hvp_eps",
    "limitations",
    "model_dtype",
    "reference_dir",
    "scf_iteration",
    "split",
)


def _ranking_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("success"):
            by_molecule[str(row["molecule_id"])].append(row)

    result = []
    for molecule_id, molecule_rows in sorted(by_molecule.items()):
        autograd_rows = [
            row for row in molecule_rows if row.get("autograd_vs_pbe", {}).get("mae") is not None
        ]
        fd_rows = [row for row in molecule_rows if row.get("fd_vs_pbe", {}).get("mae") is not None]
        if not autograd_rows or not fd_rows:
            continue
        autograd_best = min(
            autograd_rows, key=lambda row: float(row["autograd_vs_pbe"]["mae"])
        )["run"]
        fd_best = min(fd_rows, key=lambda row: float(row["fd_vs_pbe"]["mae"]))["run"]
        result.append(
            {
                "molecule_id": molecule_id,
                "autograd_best_by_pbe_mae": autograd_best,
                "fd_best_by_pbe_mae": fd_best,
                "consistent": autograd_best == fd_best,
            }
        )
    return result


def merge(input_paths: list[Path]) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one audit summary is required")
    payloads = [json.loads(path.read_text()) for path in input_paths]
    first = payloads[0]
    for path, payload in zip(input_paths[1:], payloads[1:]):
        for field in CONSISTENT_FIELDS:
            if payload.get(field) != first.get(field):
                raise ValueError(f"{path}: inconsistent {field}")

    result = {field: first.get(field) for field in CONSISTENT_FIELDS}
    for field in LIST_FIELDS:
        result[field] = [row for payload in payloads for row in payload.get(field, [])]

    full_rows = result["fixed_density_full_hessian"]
    identities = [
        (row.get("run"), row.get("molecule_id"), row.get("sample_id")) for row in full_rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate model/molecule/sample Hessian rows across audit shards")

    result["molecules"] = sorted(
        {str(molecule) for payload in payloads for molecule in payload.get("molecules", [])}
    )
    worker_rss = [int(payload.get("max_rss_kb") or 0) for payload in payloads]
    result["max_rss_kb"] = max(worker_rss, default=0)
    result["parallel_merge"] = {
        "workers": len(payloads),
        "worker_max_rss_kb": worker_rss,
        "sum_worker_max_rss_kb": sum(worker_rss),
        "source_summaries": [str(path) for path in input_paths],
    }
    result["ranking_consistency"] = _ranking_consistency(full_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.input_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "workers": result["parallel_merge"]["workers"],
                "molecules": len(result["molecules"]),
                "full_hessians": len(result["fixed_density_full_hessian"]),
                "hvp": len(result["hvp"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
