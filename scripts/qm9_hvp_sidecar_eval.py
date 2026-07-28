#!/usr/bin/env python3
"""Evaluate trainer-path and corrected fixed-density HVPs on frozen sidecar directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.mldft_module import MLDFTLitModule


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_frobenius": float(
            np.linalg.norm(difference) / max(reference_norm, np.finfo(float).tiny)
        ),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_samples(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.split_file = args.split_file.as_posix()
        cfg.data.datamodule.data_dir = args.data_dir.as_posix()
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = 0
        cfg.data.datamodule.shuffle_val = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
        cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = True
        cfg.data.datamodule.dataset_kwargs.load_hvp_label = True
        cfg.data.datamodule.dataset_kwargs.hvp_label_dir = args.sidecar_dir.as_posix()
        cfg.data.datamodule.dataset_kwargs.hvp_reference_scale_floor = 1e-2
        cfg.data.datamodule.dataset_kwargs.hvp_target_key = "model_hvp_target"
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("validate")
    samples = {}
    dataset = datamodule.val_set
    for path_index, path in enumerate(dataset.paths):
        name = path.name.removesuffix(".zarr.zip").split(".")
        molecule_id, sample_id = name[0].zfill(7), int(name[1])
        if sample_id != 0 or not (args.sidecar_dir / f"{molecule_id}.0000000.npz").exists():
            continue
        item = int(dataset.path_indices[path_index])
        samples.setdefault(molecule_id, dataset[item])
    return samples


def _trainer_hvp(model: MLDFTLitModule, sample: Any, direction: np.ndarray, device: torch.device) -> np.ndarray:
    batch = next(
        iter(
            OFLoader(
                [sample],
                batch_size=1,
                follow_batch=["coeffs", "atomic_numbers"],
                list_keys=["overlap_matrix"],
            )
        )
    ).to(device)
    batch.pos.requires_grad_(True)
    energy, _ = model.net(batch)
    gradient = torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
    vector = torch.as_tensor(direction, dtype=gradient.dtype, device=gradient.device)
    hvp = torch.autograd.grad(torch.sum(gradient * vector), batch.pos)[0]
    return hvp.detach().cpu().numpy()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    samples = _load_samples(args, args.run_dir)
    model = MLDFTLitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device=device, dtype=torch.float64).eval()
    rows = []
    for molecule_id, sample in sorted(samples.items()):
        payload = np.load(args.sidecar_dir / f"{molecule_id}.0000000.npz")
        arrays = {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in (
                "direction",
                "model_hvp_target",
                "fixed_density_correction_hvp",
                "complete_total_reference_hvp",
            )
        }
        for key, value in arrays.items():
            if value.ndim == 2:
                arrays[key] = value[None, ...]
        direction_count = arrays["direction"].shape[0]
        if any(value.shape[0] != direction_count for value in arrays.values()):
            raise ValueError(f"Direction count mismatch for {molecule_id}")
        stability_mask = np.asarray(
            payload.get("stability_mask", np.ones(direction_count, dtype=np.bool_)),
            dtype=np.bool_,
        ).reshape(-1)
        if stability_mask.size != direction_count:
            raise ValueError(f"Stability mask length mismatch for {molecule_id}")
        direction_kinds = np.asarray(
            payload.get("direction_kind", "unknown")
        ).reshape(-1)
        for direction_index in np.flatnonzero(stability_mask):
            direction = arrays["direction"][direction_index]
            target = arrays["model_hvp_target"][direction_index]
            correction = arrays["fixed_density_correction_hvp"][direction_index]
            reference = arrays["complete_total_reference_hvp"][direction_index]
            predicted = _trainer_hvp(model, sample, direction, device)
            corrected = predicted + correction
            target_metrics = _metrics(predicted, target)
            corrected_metrics = _metrics(corrected, reference)
            reference_rms = float(np.sqrt(np.mean(reference**2)))
            kind_index = 0 if direction_kinds.size == 1 else direction_index
            rows.append(
                {
                    "run": args.run_name,
                    "molecule_id": molecule_id,
                    "natoms": int(direction.shape[0]),
                    "direction_index": int(direction_index),
                    "direction_kind": str(direction_kinds[kind_index]),
                    "reference_hvp_rms": reference_rms,
                    "small_reference": reference_rms < args.small_reference_rms,
                    **{
                        f"trainer_vs_target_{key}": value
                        for key, value in target_metrics.items()
                    },
                    **{
                        f"corrected_vs_pbe_{key}": value
                        for key, value in corrected_metrics.items()
                    },
                }
            )
    if not rows:
        raise RuntimeError("No validation sidecars were evaluated")
    summary = {
        "definition": (
            "Fixed-density screening: trainer-path learned scalar-energy HVP compared with PBE "
            "analytic total H.v. This is a deliberately named surrogate, not density-relaxed "
            "complete-total OFDFT HVP."
        ),
        "run": args.run_name,
        "run_dir": args.run_dir.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "molecules": len({row["molecule_id"] for row in rows}),
        "directions": len(rows),
        "test_accessed": False,
        "mean_mae": float(np.mean([row["corrected_vs_pbe_mae"] for row in rows])),
        "median_mae": float(np.median([row["corrected_vs_pbe_mae"] for row in rows])),
        "mean_rmse": float(np.mean([row["corrected_vs_pbe_rmse"] for row in rows])),
        "mean_relative_frobenius": float(
            np.mean([row["corrected_vs_pbe_relative_frobenius"] for row in rows])
        ),
        "median_relative_frobenius": float(
            np.median([row["corrected_vs_pbe_relative_frobenius"] for row in rows])
        ),
        "small_reference_count": sum(bool(row["small_reference"]) for row in rows),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output_dir / "per_direction.csv", rows)
    per_molecule = []
    for molecule_id in sorted({row["molecule_id"] for row in rows}):
        molecule_rows = [row for row in rows if row["molecule_id"] == molecule_id]
        per_molecule.append(
            {
                "run": args.run_name,
                "molecule_id": molecule_id,
                "natoms": molecule_rows[0]["natoms"],
                "direction_count": len(molecule_rows),
                "small_reference_count": sum(
                    bool(row["small_reference"]) for row in molecule_rows
                ),
                "mean_mae": float(
                    np.mean([row["corrected_vs_pbe_mae"] for row in molecule_rows])
                ),
                "mean_rmse": float(
                    np.mean([row["corrected_vs_pbe_rmse"] for row in molecule_rows])
                ),
                "mean_relative_frobenius": float(
                    np.mean(
                        [
                            row["corrected_vs_pbe_relative_frobenius"]
                            for row in molecule_rows
                        ]
                    )
                ),
            }
        )
    _write_csv(args.output_dir / "per_molecule.csv", per_molecule)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--small-reference-rms", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
