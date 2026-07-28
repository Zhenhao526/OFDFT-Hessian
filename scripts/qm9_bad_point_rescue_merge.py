#!/usr/bin/env python3
"""Merge sharded bad-point density-optimization rescue results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: list[Any]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--expected-variants", type=int, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in sorted(args.shard_dir.glob("result_*.json"))]
    if not payloads or any(payload.get("partial", True) for payload in payloads):
        raise SystemExit("Missing or partial shard result")
    rows = [row for payload in payloads for row in payload.get("rows", [])]
    key_fields = ("run", "molecule_id", "sample_id", "coord_idx", "side", "variant")
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate rescue run key across shards")
    expected_rows = args.expected_points * args.expected_variants
    if len(rows) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, found {len(rows)}")
    missing_curves = [row.get("curve_file") for row in rows if not row.get("curve_file") or not Path(row["curve_file"]).exists()]
    if missing_curves:
        raise SystemExit(f"Missing {len(missing_curves)} curve files")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["variant"])].append(row)
    summaries = []
    for (run, variant), group in sorted(groups.items()):
        cycles = [row.get("cycles_run") for row in group]
        elapsed = [row.get("elapsed_s") for row in group]
        gradients = [row.get("final_projected_gradient_norm") for row in group]
        summaries.append(
            {
                "run": run,
                "variant": variant,
                "n_points": len(group),
                "n_converged": sum(bool(row.get("converged")) for row in group),
                "convergence_rate": sum(bool(row.get("converged")) for row in group) / len(group),
                "n_errors": sum(row.get("error") is not None for row in group),
                "mean_cycles": _finite_mean(cycles),
                "median_cycles": float(np.median([float(x) for x in cycles if x is not None])),
                "max_cycles": max(int(x) for x in cycles if x is not None),
                "mean_final_projected_gradient_norm": _finite_mean(gradients),
                "max_final_projected_gradient_norm": max(float(x) for x in gradients if x is not None),
                "total_elapsed_s": float(sum(float(x) for x in elapsed if x is not None)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rescue_points.csv", rows)
    _write_csv(args.output_dir / "rescue_summary.csv", summaries)
    result = {
        "definition": "Merged sharded bad-point density-optimization rescue",
        "n_shards": len(payloads),
        "n_points": args.expected_points,
        "n_variant_runs": len(rows),
        "all_curve_files_present": True,
        "summary_rows": summaries,
        "rows": rows,
    }
    (args.output_dir / "rescue_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
