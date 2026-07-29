#!/usr/bin/env python3
"""Evaluate direct and APBE-residual Graphformers on a common physical target.

For a residual model, errors in ``Delta Ts = Ts - T_APBEK`` are exactly the
errors of the reconstructed ``Ts + Exc`` contribution because APBEK and PBE XC
are added deterministically.  This makes the reported energy and projected
density-gradient errors directly comparable with a model trained on
``Ts + Exc``.  ``--zero-model`` evaluates parameter-free APBEK as the residual
baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset
from torch_geometric.nn import global_add_pool

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.mldft_module import MLDFTLitModule


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
        cfg.data.datamodule.shuffle_train = False
        cfg.data.datamodule.shuffle_val = False
        cfg.data.datamodule.shuffle_test = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = False
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = False
        cfg.data.datamodule.dataset_kwargs.load_hvp_label = False
        if ground_state_only:
            cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
            cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
    return cfg


def _dataset_for_split(datamodule: Any, split: str) -> Any:
    if split == "train":
        return datamodule.train_set
    if split == "val":
        return datamodule.val_set
    return datamodule.test_set


def _base_geometry_indices(dataset: Any) -> list[int]:
    selected = []
    for path_index, path in enumerate(dataset.paths):
        parts = path.name.removesuffix(".zarr.zip").split(".")
        sample_id = int(parts[1]) if len(parts) == 2 else -1
        if sample_id != 0:
            continue
        start = int(dataset.path_indices[path_index])
        stop = int(dataset.path_indices[path_index + 1])
        selected.extend(range(start, stop))
    return selected


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_cfg(
        args.run_dir.resolve(),
        args.batch_size,
        args.num_workers,
        args.ground_state_only,
    )
    target_key = str(cfg.data.target_key)
    if target_key not in {"kin_plus_xc", "kin_minus_apbe"}:
        raise ValueError(f"Unsupported comparison target {target_key!r}")
    if args.zero_model and target_key != "kin_minus_apbe":
        raise ValueError("--zero-model is the APBEK baseline and requires kin_minus_apbe")

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    stage = "fit" if args.split == "train" else (
        "validate" if args.split == "val" else "test"
    )
    datamodule.setup(stage)
    dataset = _dataset_for_split(datamodule, args.split)
    if args.base_geometry_only:
        indices = _base_geometry_indices(dataset)
        if not indices:
            raise RuntimeError("No sample-0 geometries found")
        dataset = Subset(dataset, indices)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = None
    if not args.zero_model:
        if args.ckpt is None:
            raise ValueError("--ckpt is required unless --zero-model is used")
        model = MLDFTLitModule.load_from_checkpoint(args.ckpt.resolve(), map_location="cpu")
        if model.target_key != target_key:
            raise ValueError(
                f"Checkpoint target {model.target_key!r} != config target {target_key!r}"
            )
        model.force_supervision = False
        model.to(device)
        model.eval()

    loader = OFLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=datamodule.pin_memory,
        worker_init_fn=datamodule.worker_init_fn,
        **datamodule.dataloader_kwargs,
    )

    energy_abs_sum = 0.0
    energy_sq_sum = 0.0
    energy_per_electron_abs_sum = 0.0
    gradient_abs_sum = 0.0
    gradient_sq_sum = 0.0
    gradient_l2_sum = 0.0
    samples = 0
    gradient_components = 0
    failures: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(loader):
        batch = batch.to(device)
        try:
            if model is None:
                pred_energy = torch.zeros_like(batch.energy_label)
                pred_gradients = torch.zeros_like(batch.gradient_label)
            else:
                with torch.enable_grad():
                    pred_energy, pred_gradients, _, _ = model.forward_predictions(
                        batch,
                        compute_density_gradients=True,
                        compute_forces=False,
                    )
            energy_error = pred_energy.reshape(-1) - batch.energy_label.reshape(-1)
            gradient_error = project_gradient_difference(pred_gradients, batch)
            if not torch.isfinite(energy_error).all():
                raise FloatingPointError("non-finite energy error")
            if not torch.isfinite(gradient_error).all():
                raise FloatingPointError("non-finite projected gradient error")
        except Exception as exc:  # noqa: BLE001
            failures.append({"batch_index": batch_index, "error": repr(exc)})
            continue

        weights = batch.has_energy_label.reshape(-1).to(dtype=energy_error.dtype)
        atom_electrons = global_add_pool(
            batch.atomic_numbers.to(dtype=energy_error.dtype),
            batch.atomic_numbers_batch,
        )
        energy_abs_sum += _as_float((energy_error.abs() * weights).sum())
        energy_sq_sum += _as_float((energy_error.square() * weights).sum())
        energy_per_electron_abs_sum += _as_float(
            (energy_error.abs() / atom_electrons * weights).sum()
        )

        coefficient_weights = weights[batch.coeffs_batch]
        weighted_gradient = gradient_error * coefficient_weights
        gradient_abs_sum += _as_float(weighted_gradient.abs().sum())
        gradient_sq_sum += _as_float((gradient_error.square() * coefficient_weights).sum())
        per_sample_gradient_l2 = torch.sqrt(
            global_add_pool(gradient_error.square(), batch.coeffs_batch)
        )
        gradient_l2_sum += _as_float((per_sample_gradient_l2 * weights).sum())
        samples += int(weights.sum().item())
        gradient_components += int(coefficient_weights.sum().item())

    if samples == 0:
        raise RuntimeError(f"No labelled samples evaluated; failures={failures[:3]}")
    model_kind = "apbek_zero_residual" if model is None else target_key
    return {
        "protocol": "qm9_random1000_residual_graphformer_v1",
        "model_kind": model_kind,
        "learned_target": target_key,
        "physical_reconstruction_target": "kin_plus_xc",
        "residual_error_identity": (
            "error(DeltaTs) == error(DeltaTs + APBEK + PBE_XC)"
            if target_key == "kin_minus_apbe"
            else None
        ),
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": None if args.ckpt is None else str(args.ckpt.resolve()),
        "split": args.split,
        "ground_state_only": args.ground_state_only,
        "base_geometry_only": args.base_geometry_only,
        "samples": samples,
        "gradient_components": gradient_components,
        "energy_mae_hartree": energy_abs_sum / samples,
        "energy_rmse_hartree": (energy_sq_sum / samples) ** 0.5,
        "energy_mae_hartree_per_electron": energy_per_electron_abs_sum / samples,
        "projected_gradient_l1_per_sample": gradient_abs_sum / samples,
        "projected_gradient_l2_per_sample": gradient_l2_sum / samples,
        "projected_gradient_component_rmse": (
            gradient_sq_sum / gradient_components
        ) ** 0.5,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--zero-model", action="store_true")
    parser.add_argument(
        "--ground-state-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--base-geometry-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
