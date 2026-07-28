"""Merge sharded historical-model forward errors into scientific baselines."""

from __future__ import annotations

import argparse
import array
import collections
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np


class _Accumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.coefficients = 0
        self.energy_abs = 0.0
        self.energy_sq = 0.0
        self.energy_signed = 0.0
        self.gradient_abs = 0.0
        self.gradient_sq = 0.0
        self.gradient_l1 = 0.0
        self.gradient_l1_sq = 0.0
        self.difference_abs = 0.0
        self.difference_sq = 0.0
        self.difference_l1 = 0.0
        self.difference_l1_sq = 0.0

    def update(self, row: dict) -> None:
        energy = float(row["energy_error"])
        coefficients = int(row["coefficients"])
        gradient_l1 = float(row["gradient_abs_sum"])
        difference_l1 = float(row["difference_abs_sum"])
        self.samples += 1
        self.coefficients += coefficients
        self.energy_abs += abs(energy)
        self.energy_sq += energy * energy
        self.energy_signed += energy
        self.gradient_abs += gradient_l1
        self.gradient_sq += float(row["gradient_mse_per_coefficient"]) * coefficients
        self.gradient_l1 += gradient_l1
        self.gradient_l1_sq += gradient_l1 * gradient_l1
        self.difference_abs += difference_l1
        self.difference_sq += float(row["difference_mse_per_coefficient"]) * coefficients
        self.difference_l1 += difference_l1
        self.difference_l1_sq += difference_l1 * difference_l1

    def render(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        return {
            "coefficients": self.coefficients,
            "difference": {
                "legacy_per_molecule_l1_mean": self.difference_l1 / self.samples,
                "legacy_per_molecule_l1_rms": math.sqrt(
                    self.difference_l1_sq / self.samples
                ),
                "pooled_coefficient_mae": self.difference_abs / self.coefficients,
                "pooled_coefficient_rmse": math.sqrt(
                    self.difference_sq / self.coefficients
                ),
            },
            "energy": {
                "mae": self.energy_abs / self.samples,
                "mean_signed_error": self.energy_signed / self.samples,
                "rmse": math.sqrt(self.energy_sq / self.samples),
            },
            "projected_density_gradient": {
                "legacy_per_molecule_l1_mean": self.gradient_l1 / self.samples,
                "legacy_per_molecule_l1_rms": math.sqrt(
                    self.gradient_l1_sq / self.samples
                ),
                "pooled_coefficient_mae": self.gradient_abs / self.coefficients,
                "pooled_coefficient_rmse": math.sqrt(
                    self.gradient_sq / self.coefficients
                ),
            },
            "samples": self.samples,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: array.array) -> dict:
    data = np.frombuffer(values, dtype=np.float64)
    return {
        "count": int(len(data)),
        "max": float(data.max()),
        "quantiles": {
            str(q): float(np.quantile(data, q))
            for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--outliers-jsonl-gz", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    summaries = []
    artifacts = {}
    overall = _Accumulator()
    groups = {
        "source": collections.defaultdict(_Accumulator),
        "density_role": collections.defaultdict(_Accumulator),
        "n_atoms": collections.defaultdict(_Accumulator),
        "composition": collections.defaultdict(_Accumulator),
    }
    quantile_fields = (
        "absolute_energy_error",
        "gradient_l1",
        "gradient_mae_per_coefficient",
        "difference_l1",
        "difference_mae_per_coefficient",
        "predicted_ground_state_electron_error",
    )
    quantiles = {
        "overall": {field: array.array("d") for field in quantile_fields}
    }
    quantiles_by_source = collections.defaultdict(
        lambda: {field: array.array("d") for field in quantile_fields}
    )
    outlier_heaps = {field: [] for field in quantile_fields}
    sequence = 0
    rows = 0
    nonfinite = 0
    unlabeled = 0

    for shard in range(args.num_shards):
        summary_path = args.run_root / "summaries" / f"shard_{shard}.json"
        rows_path = args.run_root / "rows" / f"shard_{shard}.jsonl.gz"
        if not summary_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(f"Missing baseline shard {shard}")
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        artifacts[str(summary_path.relative_to(args.run_root))] = {
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256(summary_path),
        }
        artifacts[str(rows_path.relative_to(args.run_root))] = {
            "bytes": rows_path.stat().st_size,
            "sha256": _sha256(rows_path),
        }
        with gzip.open(rows_path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                if not row["finite"]:
                    nonfinite += 1
                if not row["has_energy_label"]:
                    unlabeled += 1
                overall.update(row)
                composition = json.dumps(row["composition"], sort_keys=True)
                groups["source"][row["source"]].update(row)
                groups["density_role"][row["density_role"]].update(row)
                groups["n_atoms"][str(row["n_atoms"])].update(row)
                groups["composition"][composition].update(row)
                values = {
                    "absolute_energy_error": abs(float(row["energy_error"])),
                    "gradient_l1": float(row["gradient_abs_sum"]),
                    "gradient_mae_per_coefficient": float(
                        row["gradient_mae_per_coefficient"]
                    ),
                    "difference_l1": float(row["difference_abs_sum"]),
                    "difference_mae_per_coefficient": float(
                        row["difference_mae_per_coefficient"]
                    ),
                    "predicted_ground_state_electron_error": float(
                        row["predicted_ground_state_electron_error"]
                    ),
                }
                outlier_record = {
                    key: row[key]
                    for key in (
                        "composition",
                        "density_role",
                        "filename",
                        "n_atoms",
                        "partition",
                        "scf_iteration",
                        "source",
                    )
                }
                for field, value in values.items():
                    quantiles["overall"][field].append(value)
                    quantiles_by_source[row["source"]][field].append(value)
                    entry = (value, sequence, outlier_record)
                    sequence += 1
                    heap = outlier_heaps[field]
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, entry)
                    elif value > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    errors = []
    if any(summary["status"] != "passed" for summary in summaries):
        errors.append("one or more shard summaries failed")
    selected = sum(summary["selected"] for summary in summaries)
    processed = sum(summary["processed"] for summary in summaries)
    selected_labels = sum(summary["selected_labels"] for summary in summaries)
    if rows != processed or processed != selected:
        errors.append(f"row/processed/selected mismatch: {rows}/{processed}/{selected}")
    if nonfinite:
        errors.append(f"{nonfinite} nonfinite samples")
    if unlabeled:
        errors.append(f"{unlabeled} samples without energy labels")
    scf_iteration_sets = sorted(
        {tuple(summary["scf_iterations"]) for summary in summaries}
    )
    density_coverage = (
        "all molecules in the fixed split at the archived final ground-state density (-1); "
        "perturbed-density coverage is supplied by the separate multi-sample recovery acceptance"
        if scf_iteration_sets == [(-1,)]
        else (
            "all molecules in the fixed split at the explicitly recorded SCF selectors; "
            "not a claim of evaluating every archived SCF trajectory point"
        )
    )

    args.outliers_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.outliers_jsonl_gz, "wt") as handle:
        for field, heap in outlier_heaps.items():
            for value, _, row in sorted(heap, reverse=True):
                handle.write(
                    json.dumps(
                        {"metric": field, "value": value, **row}, sort_keys=True
                    )
                    + "\n"
                )

    report = {
        "artifacts": artifacts,
        "checkpoint_sha256": sorted(
            set(summary["checkpoint_sha256"] for summary in summaries)
        ),
        "cost": {
            "allocated_device_seconds_sum": sum(
                summary["elapsed_s"] for summary in summaries
            ),
            "batches": sum(summary["batches"] for summary in summaries),
            "parallel_wall_seconds_estimate": max(
                summary["elapsed_s"] for summary in summaries
            ),
        },
        "dataset": sorted(set(summary["dataset"] for summary in summaries)),
        "definitions": {
            "difference": "predicted (current-ground) coefficient difference error",
            "energy": "forward error at the archived label density; not optimized-density error",
            "density_coverage": density_coverage,
            "projected_density_gradient": (
                "electron-number tangent projection of predicted-minus-label coefficient gradient"
            ),
        },
        "errors": errors,
        "groups": {
            group_name: {
                key: accumulator.render()
                for key, accumulator in sorted(values.items())
            }
            for group_name, values in groups.items()
        },
        "metrics": overall.render(),
        "outliers_jsonl_gz": str(args.outliers_jsonl_gz.resolve()),
        "partition": sorted(set(summary["partition"] for summary in summaries)),
        "quantiles": {
            "overall": {
                field: _quantiles(values)
                for field, values in quantiles["overall"].items()
            },
            "source": {
                source: {
                    field: _quantiles(values) for field, values in fields.items()
                }
                for source, fields in sorted(quantiles_by_source.items())
            },
        },
        "rows": rows,
        "run_root": str(args.run_root.resolve()),
        "selected_labels": selected_labels,
        "scf_iterations": scf_iteration_sets,
        "split_file_sha256": sorted(
            set(summary["split_file_sha256"] for summary in summaries)
        ),
        "status": "passed" if not errors else "failed",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
