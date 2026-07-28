#!/usr/bin/env python3
"""Measure train-parent force and paired-direction HVP difficulty without touching Test100."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _composition(atomic_numbers: np.ndarray) -> str:
    return ";".join(
        f"{int(z)}:{int(np.sum(atomic_numbers == z))}"
        for z in sorted(set(atomic_numbers.tolist()))
    )


def _load_datamodule(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.split_file = args.split_file.as_posix()
        cfg.data.datamodule.data_dir = args.data_dir.as_posix()
        cfg.data.datamodule.batch_size = args.batch_size
        cfg.data.datamodule.num_workers = args.num_workers
        cfg.data.datamodule.shuffle_train = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
        cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = True
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("fit")
    return datamodule


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    datamodule = _load_datamodule(args)
    model = MLDFTLitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.force_supervision = True
    model.to(device=device, dtype=torch.float64).eval()

    parent_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for batch_index, batch in enumerate(datamodule.train_dataloader()):
        batch = batch.to(device)
        with torch.enable_grad():
            _, _, _, pred_forces = model.forward_predictions(
                batch, compute_density_gradients=False, compute_forces=True
            )
        if pred_forces is None:
            raise RuntimeError("Model did not return scalar-derived forces")
        atom_batch = batch.atomic_numbers_batch
        for graph_index in range(int(batch.batch_size)):
            atom_mask = atom_batch == graph_index
            atomic_numbers = batch.atomic_numbers[atom_mask].detach().cpu().numpy()
            parent_id = int(batch.source_molecule_id.reshape(-1)[graph_index])
            row = {
                "parent_id": parent_id,
                "geometry_sample_id": int(
                    batch.geometry_sample_id.reshape(-1)[graph_index]
                ),
                "paired": bool(batch.paired_perturbations.reshape(-1)[graph_index]),
                "pair_id": int(batch.perturbation_pair_id.reshape(-1)[graph_index]),
                "pair_sign": int(batch.perturbation_pair_sign.reshape(-1)[graph_index]),
                "natoms": int(atomic_numbers.size),
                "composition": _composition(atomic_numbers),
                "positions": batch.pos[atom_mask].detach().cpu().numpy(),
                "reference_force": batch.force_label[atom_mask].detach().cpu().numpy(),
                "predicted_force": pred_forces[atom_mask].detach().cpu().numpy(),
            }
            parent_rows[parent_id].append(row)
        if (batch_index + 1) % 100 == 0:
            print(f"batches={batch_index + 1}", flush=True)

    rows: list[dict[str, Any]] = []
    for parent_id, samples in sorted(parent_rows.items()):
        ordinary = [sample for sample in samples if not sample["paired"]]
        pairs: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for sample in samples:
            if sample["paired"]:
                pairs[sample["pair_id"]][sample["pair_sign"]] = sample
        force_errors = np.concatenate(
            [
                (sample["predicted_force"] - sample["reference_force"]).reshape(-1)
                for sample in ordinary
            ]
        )
        pair_metrics = []
        for pair_id, signs in sorted(pairs.items()):
            if set(signs) != {-1, 1}:
                raise ValueError(f"Incomplete pair for parent={parent_id}, pair={pair_id}")
            minus, plus = signs[-1], signs[1]
            delta = plus["positions"] - minus["positions"]
            separation = float(np.linalg.norm(delta))
            if separation <= 0:
                raise ValueError(f"Zero pair separation for parent={parent_id}")
            reference_hvp = -(
                plus["reference_force"] - minus["reference_force"]
            ) / separation
            predicted_hvp = -(
                plus["predicted_force"] - minus["predicted_force"]
            ) / separation
            difference = predicted_hvp - reference_hvp
            pair_metrics.append(
                {
                    "pair_id": pair_id,
                    "pair_separation_bohr": separation,
                    "reference_hvp_rms": float(np.sqrt(np.mean(reference_hvp**2))),
                    "hvp_component_mae": float(np.mean(np.abs(difference))),
                    "hvp_component_rmse": float(np.sqrt(np.mean(difference**2))),
                }
            )
        if not ordinary or not pair_metrics:
            raise ValueError(f"Parent {parent_id} lacks ordinary samples or paired directions")
        rows.append(
            {
                "parent_id": f"{parent_id:07d}",
                "natoms": ordinary[0]["natoms"],
                "composition": ordinary[0]["composition"],
                "ordinary_geometries": len(ordinary),
                "paired_directions": len(pair_metrics),
                "force_component_mae": float(np.mean(np.abs(force_errors))),
                "force_component_rmse": float(np.sqrt(np.mean(force_errors**2))),
                "paired_hvp_component_mae": float(
                    np.mean([metric["hvp_component_mae"] for metric in pair_metrics])
                ),
                "paired_hvp_component_rmse": float(
                    np.mean([metric["hvp_component_rmse"] for metric in pair_metrics])
                ),
                "paired_reference_hvp_rms": float(
                    np.mean([metric["reference_hvp_rms"] for metric in pair_metrics])
                ),
                "pair_separation_bohr": float(
                    np.mean([metric["pair_separation_bohr"] for metric in pair_metrics])
                ),
            }
        )

    output = {
        "definition": (
            "Ground-state scalar-derived model forces on the 800 train parents. HVP difficulty "
            "is the centered PBE-force secant along the pre-existing train-parent R-/R+ pair."
        ),
        "run_dir": args.run_dir.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "split_file": args.split_file.as_posix(),
        "parents": len(rows),
        "test_accessed": False,
        "mean_force_component_mae": float(
            np.mean([row["force_component_mae"] for row in rows])
        ),
        "mean_paired_hvp_component_mae": float(
            np.mean([row["paired_hvp_component_mae"] for row in rows])
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
