#!/usr/bin/env python3
"""Compare model Hessians with an on-demand PySCF PBE Hessian reference."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from pyscf import dft, scf
from pyscf.hessian import rks

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


def _load_cfg(run_dir: Path, num_workers: int) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
    return cfg


def _sample_keys(datamodule: Any) -> list[tuple[str, str, int, int]]:
    keys: list[tuple[str, str, int, int]] = []
    for path, scf_iterations in zip(
        datamodule.test_set.paths, datamodule.test_set.scf_iterations_per_path
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_id_text = label_name.split(".")
        for scf_iteration in scf_iterations:
            keys.append(
                (
                    f"{path.name}:scf={int(scf_iteration)}",
                    molecule_id,
                    int(sample_id_text),
                    int(scf_iteration),
                )
            )
    return keys


def _chk_path(dataset_dir: Path, molecule_id: str, sample_id: int) -> Path:
    return (
        dataset_dir
        / "kohn_sham"
        / f"qm9_pbe_force_pilot_{molecule_id}.{sample_id:07d}.chk"
    )


def _pbe_hessian_from_chk(chk_path: Path) -> np.ndarray:
    mol, rec = scf.chkfile.load_scf(chk_path.as_posix())
    xc = scf.chkfile.load(chk_path.as_posix(), "Results/name_xc_functional")
    if isinstance(xc, bytes):
        xc = xc.decode()
    grid_level = int(scf.chkfile.load(chk_path.as_posix(), "Results/grid_level"))

    mf = dft.RKS(mol, xc=xc)
    mf.chkfile = chk_path.as_posix()
    mf.grids.level = grid_level
    mf.mo_coeff = rec["mo_coeff"]
    mf.mo_occ = rec["mo_occ"]
    mf.e_tot = rec["e_tot"]
    if "mo_energy" in rec:
        mf.mo_energy = rec["mo_energy"]

    hessian = rks.Hessian(mf).kernel()
    natoms = mol.natm
    return hessian.transpose(0, 2, 1, 3).reshape(3 * natoms, 3 * natoms)


def _get_batch(
    run_dir: Path,
    molecule_id: str,
    sample_id: int,
    scf_iteration: int,
    num_workers: int,
    device: torch.device,
) -> Any:
    cfg = _load_cfg(run_dir, num_workers)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    sample_keys = _sample_keys(datamodule)
    loader = datamodule.test_dataloader()
    for sample_index, batch in enumerate(loader):
        _, key_molecule_id, key_sample_id, key_scf_iteration = sample_keys[sample_index]
        if (
            key_molecule_id == molecule_id
            and key_sample_id == sample_id
            and key_scf_iteration == scf_iteration
        ):
            return batch.to(device)
    raise ValueError(
        f"Sample not found in test split: {molecule_id}.{sample_id:07d}:scf={scf_iteration}"
    )


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


def _compare(model_hessian: np.ndarray, ref_hessian: np.ndarray) -> dict[str, float | bool | list[int]]:
    diff = model_hessian - ref_hessian
    ref_norm = float(np.linalg.norm(ref_hessian))
    return {
        "shape": [int(dim) for dim in model_hessian.shape],
        "finite": bool(np.isfinite(model_hessian).all()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_fro_error": float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else None,
        "model_symmetry_max_abs_error": float(np.max(np.abs(model_hessian - model_hessian.T))),
        "model_hessian_max_abs": float(np.max(np.abs(model_hessian))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    molecule_id = args.molecule_id.zfill(7)
    dataset_dir = args.dataset_dir.resolve()
    chk_path = _chk_path(dataset_dir, molecule_id, args.sample_id)
    if not chk_path.exists():
        raise FileNotFoundError(chk_path)

    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_npz.exists() and not args.recompute_reference:
        ref_hessian = np.load(args.cache_npz)["pbe_hessian"]
        reference_source = "cache"
    else:
        ref_hessian = _pbe_hessian_from_chk(chk_path)
        np.savez_compressed(args.cache_npz, pbe_hessian=ref_hessian)
        reference_source = "pyscf"

    run_results = {}
    for item in args.run:
        name, run_dir, ckpt = item.split("=", maxsplit=2)
        spec = RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve())
        batch = _get_batch(
            spec.run_dir,
            molecule_id=molecule_id,
            sample_id=args.sample_id,
            scf_iteration=args.scf_iteration,
            num_workers=args.num_workers,
            device=device,
        )
        model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location=device)
        model.to(device)
        model.eval()
        with torch.enable_grad():
            model_hessian = _finite_difference_hessian(model, batch, args.displacement).numpy()
        run_results[name] = {
            "run_dir": spec.run_dir.as_posix(),
            "ckpt": spec.ckpt.as_posix(),
            **_compare(model_hessian, ref_hessian),
        }

    summary = {
        "molecule_id": molecule_id,
        "sample_id": args.sample_id,
        "scf_iteration": args.scf_iteration,
        "device": str(device),
        "dataset_dir": dataset_dir.as_posix(),
        "chk_path": chk_path.as_posix(),
        "reference_source": reference_source,
        "reference_cache": args.cache_npz.as_posix(),
        "reference_shape": [int(dim) for dim in ref_hessian.shape],
        "reference_finite": bool(np.isfinite(ref_hessian).all()),
        "reference_symmetry_max_abs_error": float(np.max(np.abs(ref_hessian - ref_hessian.T))),
        "reference_hessian_max_abs": float(np.max(np.abs(ref_hessian))),
        "model_displacement": args.displacement,
        "runs": run_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--scf-iteration", type=int, default=1)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--cache-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--recompute-reference", action="store_true")
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
