#!/usr/bin/env python3
"""Evaluate autograd-derived OFDFT forces against stored PBE force labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.mldft_module import MLDFTLitModule


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _split_dataset(datamodule: Any, split: str) -> Any:
    if split == "train":
        return datamodule.train_set
    if split == "val":
        return datamodule.val_set
    return datamodule.test_set


def _selected_indices_and_keys(
    dataset: Any,
    parent_ids: set[str] | None,
    base_geometry_only: bool,
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    keys: list[str] = []
    for path_index, (path, scf_iterations) in enumerate(
        zip(dataset.paths, dataset.scf_iterations_per_path)
    ):
        parts = path.name.removesuffix(".zarr.zip").split(".")
        parent_id = parts[0].zfill(7)
        sample_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        if parent_ids is not None and parent_id not in parent_ids:
            continue
        if base_geometry_only and sample_id != 0:
            continue
        path_start = int(dataset.path_indices[path_index])
        for local_index, scf_iteration in enumerate(scf_iterations):
            indices.append(path_start + local_index)
            keys.append(f"{path.name}:scf={int(scf_iteration)}")
    return indices, keys


def _load_cfg(
    run_dir: Path,
    batch_size: int,
    num_workers: int,
    ground_state_only: bool,
) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = batch_size
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
        cfg.data.datamodule.shuffle_val = False
        cfg.data.datamodule.shuffle_train = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = False
        cfg.data.datamodule.dataset_kwargs.load_hvp_label = False
        if ground_state_only:
            cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
            cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
    return cfg


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt.resolve()
    cfg = _load_cfg(
        run_dir,
        args.batch_size,
        args.num_workers,
        args.ground_state_only,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    stage = "fit" if args.split == "train" else ("validate" if args.split == "val" else "test")
    datamodule.setup(stage)
    dataset = _split_dataset(datamodule, args.split)
    parent_ids = None
    if args.parent_ids is not None:
        parent_ids = {
            token.strip().zfill(7)
            for token in args.parent_ids.read_text().replace(",", "\n").splitlines()
            if token.strip()
        }
    selected_indices, sample_keys = _selected_indices_and_keys(
        dataset, parent_ids, args.base_geometry_only
    )
    if not selected_indices:
        raise RuntimeError("No samples matched the requested split/parent/geometry filters")

    model = MLDFTLitModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.force_supervision = True
    model.to(device)
    model.eval()

    energy_abs_sum = 0.0
    energy_sq_sum = 0.0
    energy_count = 0
    energy_max_abs_error = 0.0
    force_abs_sum = 0.0
    force_sq_sum = 0.0
    component_count = 0
    atom_vector_error_sum = 0.0
    atom_count = 0
    pred_abs_sum = 0.0
    ref_abs_sum = 0.0
    max_component_abs_error = 0.0
    failures: list[dict[str, Any]] = []
    worst_samples: list[dict[str, Any]] = []
    sample_offset = 0
    n_batches = 0

    loader = OFLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=datamodule.pin_memory,
        worker_init_fn=datamodule.worker_init_fn,
        **datamodule.dataloader_kwargs,
    )
    for batch_idx, batch in enumerate(loader):
        n_batches += 1
        batch = batch.to(device)
        try:
            with torch.enable_grad():
                pred_energy, _, _, pred_forces = model.forward_predictions(
                    batch, compute_density_gradients=False, compute_forces=True
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"batch_idx": batch_idx, "error": repr(exc)})
            sample_offset += int(batch.batch_size)
            continue

        if hasattr(batch, "energy_label"):
            energy_error = pred_energy.reshape(-1) - batch.energy_label.reshape(-1)
            if torch.isfinite(energy_error).all():
                energy_abs = energy_error.abs()
                energy_abs_sum += _as_float(energy_abs.sum())
                energy_sq_sum += _as_float((energy_error * energy_error).sum())
                energy_count += int(energy_error.numel())
                energy_max_abs_error = max(energy_max_abs_error, _as_float(energy_abs.max()))
            else:
                failures.append({"batch_idx": batch_idx, "error": "non-finite energy error"})

        if pred_forces is None:
            failures.append({"batch_idx": batch_idx, "error": "pred_forces is None"})
            sample_offset += int(batch.batch_size)
            continue
        if not hasattr(batch, "force_label"):
            failures.append({"batch_idx": batch_idx, "error": "batch has no force_label"})
            sample_offset += int(batch.batch_size)
            continue

        ref_forces = batch.force_label
        error = pred_forces - ref_forces
        if not torch.isfinite(error).all():
            failures.append({"batch_idx": batch_idx, "error": "non-finite force error"})
            sample_offset += int(batch.batch_size)
            continue

        abs_error = error.abs()
        force_abs_sum += _as_float(abs_error.sum())
        force_sq_sum += _as_float((error * error).sum())
        component_count += int(error.numel())
        atom_errors = torch.linalg.vector_norm(error, dim=1)
        atom_vector_error_sum += _as_float(atom_errors.sum())
        atom_count += int(atom_errors.numel())
        pred_abs_sum += _as_float(pred_forces.abs().sum())
        ref_abs_sum += _as_float(ref_forces.abs().sum())
        max_component_abs_error = max(max_component_abs_error, _as_float(abs_error.max()))

        atom_batch = getattr(batch, "atomic_numbers_batch", None)
        if atom_batch is None:
            atom_batch = getattr(batch, "batch", None)
        if atom_batch is None:
            failures.append({"batch_idx": batch_idx, "error": "missing atom batch indices"})
            sample_offset += int(batch.batch_size)
            continue

        sample_count = int(batch.batch_size)
        sample_sums = torch.zeros(sample_count, dtype=atom_errors.dtype, device=device)
        sample_sums.scatter_add_(0, atom_batch, atom_errors)
        sample_counts = torch.bincount(atom_batch, minlength=sample_count).clamp_min(1)
        sample_mae = sample_sums / sample_counts
        for local_idx, value in enumerate(sample_mae.detach().cpu().tolist()):
            key_idx = sample_offset + local_idx
            sample_key = sample_keys[key_idx] if key_idx < len(sample_keys) else f"sample={key_idx}"
            worst_samples.append(
                {
                    "sample_index": key_idx,
                    "sample_key": sample_key,
                    "force_vector_mae": float(value),
                }
            )

        sample_offset += sample_count

    worst_samples = sorted(
        worst_samples, key=lambda item: item["force_vector_mae"], reverse=True
    )[: args.top_k]
    energy_mae = energy_abs_sum / energy_count if energy_count else None
    energy_rmse = (energy_sq_sum / energy_count) ** 0.5 if energy_count else None
    force_component_mae = force_abs_sum / component_count if component_count else None
    force_component_rmse = (force_sq_sum / component_count) ** 0.5 if component_count else None
    force_vector_mae = atom_vector_error_sum / atom_count if atom_count else None
    pred_abs_mean = pred_abs_sum / component_count if component_count else None
    ref_abs_mean = ref_abs_sum / component_count if component_count else None

    return {
        "run_dir": run_dir.as_posix(),
        "ckpt": ckpt_path.as_posix(),
        "device": str(device),
        "split": args.split,
        "ground_state_only": args.ground_state_only,
        "base_geometry_only": args.base_geometry_only,
        "parent_ids_file": None if args.parent_ids is None else args.parent_ids.as_posix(),
        "parent_count_filter": None if parent_ids is None else len(parent_ids),
        "batch_size": args.batch_size,
        "batches": n_batches,
        "samples_expected": len(sample_keys),
        "samples_evaluated": sample_offset,
        "energy_samples_evaluated": energy_count,
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "energy_max_abs_error": energy_max_abs_error,
        "atoms_evaluated": atom_count,
        "force_component_mae": force_component_mae,
        "force_component_rmse": force_component_rmse,
        "force_vector_mae": force_vector_mae,
        "force_pred_component_abs_mean": pred_abs_mean,
        "force_ref_component_abs_mean": ref_abs_mean,
        "max_component_abs_error": max_component_abs_error,
        "failures": failures,
        "worst_samples": worst_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--worst-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--parent-ids", type=Path)
    parser.add_argument(
        "--base-geometry-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ground-state-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate only the final SCF density state for each geometry.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.worst_csv is not None:
        args.worst_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.worst_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sample_index", "sample_key", "force_vector_mae"]
            )
            writer.writeheader()
            writer.writerows(summary["worst_samples"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
