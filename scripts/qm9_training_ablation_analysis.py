#!/usr/bin/env python3
"""Extract comparable training curves and throughput for random1000 model ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    "train_loss/total",
    "train_loss/energy_loss",
    "train_loss/gradient_loss",
    "train_loss/force_loss",
    "val_loss/total",
    "val_loss/energy_loss",
    "val_loss/force_loss",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    summaries = []
    curve_rows = []
    for model in manifest:
        name, run_dir = model["name"], Path(model["run_dir"])
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        summary = {**model}
        wall_times = []
        for tag in TAGS:
            if tag not in available:
                continue
            events = accumulator.Scalars(tag)
            values = [float(event.value) for event in events]
            wall_times.extend(float(event.wall_time) for event in events)
            key = tag.replace("/", "_")
            summary[f"{key}_final"] = values[-1]
            summary[f"{key}_min"] = min(values)
            for event in events:
                curve_rows.append(
                    {
                        "name": name,
                        "tag": tag,
                        "step": int(event.step),
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                    }
                )
        summary["event_wall_span_s"] = max(wall_times) - min(wall_times) if wall_times else None
        cfg = OmegaConf.load(run_dir / "hparams.yaml")
        summary["force_weight"] = OmegaConf.select(
            cfg, "model.loss_function.force_loss.weight", default=0.0
        )
        summary["seed"] = OmegaConf.select(cfg, "seed")
        throughput_path = run_dir / "throughput" / "throughput.csv"
        if throughput_path.exists():
            throughput = list(csv.DictReader(throughput_path.open()))
            # Exclude the first two intervals where dataloader/model warmup dominates.
            stable = throughput[2:] if len(throughput) > 2 else throughput
            for field in (
                "samples_per_sec",
                "steps_per_sec",
                "data_loading_s_per_batch",
                "net_forward_s_per_batch",
                "density_gradient_autograd_s_per_batch",
                "force_autograd_s_per_batch",
                "backward_s_per_batch",
                "optimizer_s_per_batch",
                "estimated_epoch_seconds",
            ):
                summary[f"throughput_mean_{field}"] = _finite_mean(
                    [row.get(field) for row in stable]
                )
            summary["peak_gpu_memory_mb"] = max(
                float(row["peak_gpu_memory_mb"]) for row in throughput
            )
            summary["global_steps"] = max(int(row["global_step"]) for row in throughput)
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "training_ablation_summary.csv", summaries)
    _write_csv(output_dir / "training_curves.csv", curve_rows)
    plots = []
    try:
        import matplotlib.pyplot as plt

        plot_tags = (
            "val_loss/energy_loss",
            "val_loss/force_loss",
            "train_loss/gradient_loss",
            "train_loss/total",
        )
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
        for axis, tag in zip(axes.flat, plot_tags):
            for model in manifest:
                rows = [
                    row for row in curve_rows if row["name"] == model["name"] and row["tag"] == tag
                ]
                if rows:
                    axis.plot([row["step"] for row in rows], [row["value"] for row in rows], label=model["name"])
            axis.set_title(tag)
            axis.set_xlabel("global step")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        path = output_dir / "training_loss_curves.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path))
    except ImportError:
        pass

    result = {
        "definition": "TensorBoard scalar and throughput extraction for compute-comparable runs.",
        "model_manifest": str(manifest_path.resolve()),
        "models": summaries,
        "curve_points": len(curve_rows),
        "plots": plots,
    }
    (output_dir / "training_ablation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.model_manifest, args.output_dir)


if __name__ == "__main__":
    main()
