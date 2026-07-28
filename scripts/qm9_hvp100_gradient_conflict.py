#!/usr/bin/env python3
"""Measure pairwise parameter-gradient cosines on HVP-active train batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.mldft_module import MLDFTLitModule


def _parse_run(item: str) -> tuple[str, Path, Path]:
    name, run_dir, checkpoint = item.split("=", maxsplit=2)
    return name, Path(run_dir), Path(checkpoint)


def _dot(first: list[torch.Tensor | None], second: list[torch.Tensor | None]) -> float:
    value = torch.zeros((), dtype=torch.float64, device=next(x for x in first if x is not None).device)
    for left, right in zip(first, second, strict=True):
        if left is not None and right is not None:
            value = value + torch.sum(left.detach() * right.detach())
    return float(value.cpu())


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for name, run_dir, checkpoint in map(_parse_run, args.run):
        cfg = OmegaConf.load(run_dir / "hparams.yaml")
        with open_dict(cfg):
            cfg.data.datamodule.data_dir = args.data_dir.as_posix()
            cfg.data.datamodule.split_file = args.split_file.as_posix()
            cfg.data.datamodule.batch_size = args.batch_size
            cfg.data.datamodule.num_workers = 0
            cfg.data.datamodule.shuffle_train = True
            cfg.data.datamodule.pair_grouped_train_batches = True
            cfg.data.datamodule.dataset_kwargs.load_hvp_label = True
            cfg.data.datamodule.dataset_kwargs.hvp_label_dir = args.sidecar_dir.as_posix()
            cfg.data.datamodule.dataset_kwargs.hvp_reference_scale_floor = 1e-2
        datamodule = hydra.utils.instantiate(cfg.data.datamodule)
        datamodule.setup("fit")
        model = MLDFTLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
        model.to(device=device, dtype=torch.float64)
        model.net.train()
        parameters = [parameter for parameter in model.net.parameters() if parameter.requires_grad]
        used = 0
        for loader_index, batch in enumerate(datamodule.train_dataloader()):
            if used >= args.batches:
                break
            if not bool(batch.hvp_label_mask.any()):
                continue
            batch = batch.to(device)
            pred_energy, pred_gradients, pred_diff, pred_forces = model.forward_predictions(batch)
            direction = batch.hvp_direction.to(pred_forces)
            pred_hvp = -torch.autograd.grad(
                torch.sum(pred_forces * direction),
                batch.pos,
                create_graph=True,
                retain_graph=True,
            )[0]
            projected = project_gradient_difference(pred_gradients, batch)
            weights, losses = model.loss_function(
                batch,
                pred_energy=pred_energy,
                projected_gradient_difference=projected,
                pred_diff=pred_diff,
                pred_gradients=pred_gradients,
                pred_forces=pred_forces,
                pred_hvp=pred_hvp,
                hvp_active_mask=batch.hvp_label_mask.reshape(-1).bool(),
            )
            gradients: dict[str, list[torch.Tensor | None]] = {}
            norms: dict[str, float] = {}
            for loss_name, loss in losses.items():
                if float(weights[loss_name]) == 0.0:
                    continue
                gradient = list(torch.autograd.grad(
                    float(weights[loss_name]) * loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                ))
                gradients[loss_name] = gradient
                norms[loss_name] = float(np.sqrt(max(_dot(gradient, gradient), 0.0)))
            source_ids = sorted({int(value) for value in batch.source_molecule_id.reshape(-1)})
            base = {
                "run": name,
                "batch": used,
                "loader_index": loader_index,
                "source_molecule_ids": ",".join(map(str, source_ids)),
                "hvp_labelled_graphs": int(batch.hvp_label_mask.sum()),
            }
            for loss_name, norm in norms.items():
                rows.append({**base, "kind": "norm", "first": loss_name,
                             "second": loss_name, "value": norm})
            names = sorted(gradients)
            for first_index, first in enumerate(names):
                for second in names[first_index + 1 :]:
                    denominator = norms[first] * norms[second]
                    cosine = _dot(gradients[first], gradients[second]) / denominator if denominator else None
                    rows.append({**base, "kind": "cosine", "first": first,
                                 "second": second, "value": cosine})
            used += 1
        if used != args.batches:
            raise RuntimeError(f"{name}: found {used}/{args.batches} HVP-active batches")

    cosines = [row for row in rows if row["kind"] == "cosine"]
    groups = []
    for key in sorted({(row["run"], row["first"], row["second"]) for row in cosines}):
        values = [float(row["value"]) for row in cosines
                  if (row["run"], row["first"], row["second"]) == key]
        groups.append({"run": key[0], "first": key[1], "second": key[2],
                       "batches": len(values), "mean_cosine": float(np.mean(values)),
                       "median_cosine": float(np.median(values)),
                       "negative_fraction": float(np.mean(np.asarray(values) < 0))})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "gradient_conflict_rows.csv", rows)
    _write_csv(args.output_dir / "gradient_conflict_groups.csv", groups)
    result = {
        "definition": "Validation-independent parameter-gradient audit on frozen train100 batches.",
        "test_accessed": False,
        "rows": rows,
        "groups": groups,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    (args.output_dir / "gradient_conflict_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"groups": groups, "peak_gpu_memory_mb": result["peak_gpu_memory_mb"]}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
