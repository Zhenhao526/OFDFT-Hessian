#!/usr/bin/env python3
"""Probe model Hessians from scalar energy on held-out QM9 samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import torch
import zarr
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


def _load_cfg(run_dir: Path, num_workers: int) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
    return cfg


def _sample_keys(datamodule: Any) -> list[tuple[str, Path, int]]:
    keys: list[tuple[str, Path, int]] = []
    for path, scf_iterations in zip(
        datamodule.test_set.paths, datamodule.test_set.scf_iterations_per_path
    ):
        for scf_iteration in scf_iterations:
            keys.append((f"{path.name}:scf={int(scf_iteration)}", path, int(scf_iteration)))
    return keys


def _reference_hessian(path: Path) -> tuple[bool, tuple[int, ...] | None]:
    root = zarr.open(path, mode="r")
    key = "metadata/pbe_derivatives/hessian_matrix"
    if key not in root:
        return False, None
    return True, tuple(int(dim) for dim in root[key].shape)


def _energy_hessian(model: MLDFTLitModule, batch: Any) -> torch.Tensor:
    batch.pos.requires_grad_(True)
    pred_energy, _ = model.net(batch)
    pred_forces = -torch.autograd.grad(
        pred_energy.sum(),
        batch.pos,
        create_graph=True,
        retain_graph=True,
    )[0]
    flat_forces = pred_forces.reshape(-1)
    rows = []
    for component in flat_forces:
        dforce_dpos = torch.autograd.grad(
            component,
            batch.pos,
            retain_graph=True,
            create_graph=False,
        )[0]
        rows.append((-dforce_dpos).reshape(-1).detach().cpu())
    return torch.stack(rows, dim=0)


def _pred_forces(model: MLDFTLitModule, batch: Any, positions: torch.Tensor) -> torch.Tensor:
    batch.pos = positions.detach().clone().requires_grad_(True)
    pred_energy, _ = model.net(batch)
    pred_forces = -torch.autograd.grad(
        pred_energy.sum(),
        batch.pos,
        create_graph=False,
        retain_graph=False,
    )[0]
    return pred_forces.detach()


def _finite_difference_hessian(
    model: MLDFTLitModule, batch: Any, displacement: float
) -> torch.Tensor:
    base_pos = batch.pos.detach().clone()
    n_coords = int(base_pos.numel())
    columns = []
    for coord_idx in range(n_coords):
        pos_plus = base_pos.clone().reshape(-1)
        pos_minus = base_pos.clone().reshape(-1)
        pos_plus[coord_idx] += displacement
        pos_minus[coord_idx] -= displacement
        pos_plus = pos_plus.reshape_as(base_pos)
        pos_minus = pos_minus.reshape_as(base_pos)
        f_plus = _pred_forces(model, batch, pos_plus).reshape(-1)
        f_minus = _pred_forces(model, batch, pos_minus).reshape(-1)
        dforce_dcoord = (f_plus - f_minus) / (2.0 * displacement)
        columns.append((-dforce_dcoord).cpu())
    batch.pos = base_pos
    return torch.stack(columns, dim=1)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt.resolve()
    cfg = _load_cfg(run_dir, args.num_workers)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    sample_keys = _sample_keys(datamodule)

    model = MLDFTLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_molecule_ids: set[str] = set()

    loader = datamodule.test_dataloader()
    for sample_index, batch in enumerate(loader):
        if len(records) >= args.max_molecules:
            break
        sample_key, path, scf_iteration = sample_keys[sample_index]
        label_name = sample_key.split(":")[0]
        molecule_id = label_name.split(".")[0]
        if molecule_id in seen_molecule_ids:
            continue
        if scf_iteration != args.scf_iteration:
            continue

        seen_molecule_ids.add(molecule_id)
        batch = batch.to(device)
        natoms = int(batch.pos.shape[0])
        try:
            with torch.enable_grad():
                if args.method == "autograd":
                    hessian = _energy_hessian(model, batch)
                elif args.method == "finite_difference":
                    hessian = _finite_difference_hessian(model, batch, args.displacement)
                else:
                    raise ValueError(f"Unknown method: {args.method}")
            finite = bool(torch.isfinite(hessian).all())
            finite_mask = torch.isfinite(hessian)
            nan_count = int(torch.isnan(hessian).sum())
            inf_count = int(torch.isinf(hessian).sum())
            if finite:
                symmetry_error = float((hessian - hessian.T).abs().max())
                max_abs = float(hessian.abs().max())
            else:
                symmetry_error = None
                max_abs = None
            ref_available, ref_shape = _reference_hessian(path)
            records.append(
                {
                    "sample_index": sample_index,
                    "sample_key": sample_key,
                    "molecule_id": molecule_id,
                    "natoms": natoms,
                    "hessian_shape": list(hessian.shape),
                    "finite": finite,
                    "finite_entries": int(finite_mask.sum()),
                    "total_entries": int(hessian.numel()),
                    "nan_entries": nan_count,
                    "inf_entries": inf_count,
                    "symmetry_max_abs_error": symmetry_error,
                    "hessian_max_abs": max_abs,
                    "reference_hessian_available": ref_available,
                    "reference_hessian_shape": list(ref_shape) if ref_shape else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "sample_index": sample_index,
                    "sample_key": sample_key,
                    "molecule_id": molecule_id,
                    "error": repr(exc),
                }
            )

    return {
        "run_dir": run_dir.as_posix(),
        "ckpt": ckpt_path.as_posix(),
        "device": str(device),
        "method": args.method,
        "displacement": args.displacement if args.method == "finite_difference" else None,
        "requested_molecules": args.max_molecules,
        "completed_molecules": len(records),
        "failures": failures,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=5)
    parser.add_argument("--scf-iteration", type=int, default=1)
    parser.add_argument(
        "--method",
        choices=["autograd", "finite_difference"],
        default="autograd",
    )
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = probe(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
