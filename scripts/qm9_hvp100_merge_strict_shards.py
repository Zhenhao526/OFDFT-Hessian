#!/usr/bin/env python3
"""Merge successful strict run-by-molecule shards into one evaluator directory per run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-tsv", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with args.task_tsv.open() as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    by_run: dict[str, list[tuple[dict, Path, dict]]] = defaultdict(list)
    for task in tasks:
        task_dir = args.task_root / (
            f"task_{int(task['task_index']):03d}_{task['run_name']}_{task['molecule_id']}"
        )
        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        by_run[task["run_name"]].append((task, task_dir, json.loads(summary_path.read_text())))
    args.output_root.mkdir(parents=True, exist_ok=True)
    merge_manifest = []
    for run, shards in sorted(by_run.items()):
        out = args.output_root / run
        out.mkdir(parents=True, exist_ok=True)
        summaries, metrics, points, curvatures = [], [], [], []
        wall_time = 0.0; max_rss = 0.0; peak_gpu = 0.0
        for task, task_dir, result in shards:
            summaries.extend(result["summaries"])
            with (task_dir / "metrics.csv").open() as handle:
                metrics.extend(csv.DictReader(handle))
            with (task_dir / "points.csv").open() as handle:
                points.extend(csv.DictReader(handle))
            with (task_dir / "curvature_scan.csv").open() as handle:
                curvatures.extend(csv.DictReader(handle))
            wall_time += float(result["wall_time_s"])
            max_rss = max(max_rss, float(result["max_rss_mb"]))
            peak_gpu = max(peak_gpu, float(result["peak_gpu_memory_mb"]))
            for artifact in task_dir.glob("*_hvp_arrays.npz"):
                shutil.copy2(artifact, out / artifact.name)
        if len(summaries) != len(shards):
            raise RuntimeError(f"{run}: {len(summaries)} summaries for {len(shards)} shards")
        merged = {
            "definition": "Merged validation-only strict density-relaxed complete-total HVP shards.",
            "run": run,
            "molecules": [item[0]["molecule_id"] for item in shards],
            "summaries": summaries,
            "wall_time_s": wall_time,
            "max_rss_mb": max_rss,
            "peak_gpu_memory_mb": peak_gpu,
            "test_accessed": False,
        }
        (out / "summary.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        _write_csv(out / "metrics.csv", metrics)
        _write_csv(out / "points.csv", points)
        _write_csv(out / "curvature_scan.csv", curvatures)
        merge_manifest.append({"run": run, "molecules": len(summaries),
                               "output": str(out), "summed_wall_time_s": wall_time})
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(merge_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(merge_manifest, indent=2))


if __name__ == "__main__":
    main()
