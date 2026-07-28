#!/usr/bin/env python3
"""Second-order autograd Hessian feasibility audit for QM9 P1-410.

This script is deliberately separate from the production Hessian evaluators. It
tests whether model Hessians/HVPs can be obtained directly from the scalar model
energy with PyTorch autograd, first at fixed density. The unrolled density
optimization section is a prototype only: it unrolls coefficient updates for the
model energy with differentiable torch operations and does not replace the
current OFDFT density optimizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ml.models.mldft_module import MLDFTLitModule


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


def _parse_run(item: str) -> RunSpec:
    name, run_dir, ckpt = item.split("=", maxsplit=2)
    return RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve())


def _load_cfg(
    run_dir: Path,
    num_workers: int,
    load_force_labels: bool,
    scf_iteration: int,
) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
        cfg.data.datamodule.shuffle_val = False
        if load_force_labels:
            cfg.data.datamodule.dataset_kwargs.load_force_label = True
        if scf_iteration < 0:
            cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [scf_iteration]
            cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
    return cfg


def _sample_keys(datamodule: Any, split: str) -> list[tuple[str, str, int, int]]:
    keys: list[tuple[str, str, int, int]] = []
    dataset = datamodule.val_set if split == "val" else datamodule.test_set
    for path, scf_iterations in zip(
        dataset.paths, dataset.scf_iterations_per_path
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_text = label_name.split(".")
        for scf_iteration in scf_iterations:
            keys.append(
                (
                    f"{path.name}:scf={int(scf_iteration)}",
                    molecule_id,
                    int(sample_text),
                    int(scf_iteration),
                )
            )
    return keys


def _collect_batches(
    run_dir: Path,
    targets: set[tuple[str, int, int]],
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    split: str,
    load_force_labels: bool,
) -> dict[tuple[str, int, int], Any]:
    target_scf_iterations = {target[2] for target in targets}
    if len(target_scf_iterations) != 1:
        raise ValueError(f"Expected one target SCF iteration, got {target_scf_iterations}")
    cfg = _load_cfg(
        run_dir,
        num_workers,
        load_force_labels,
        next(iter(target_scf_iterations)),
    )
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("validate" if split == "val" else "test")
    keys = _sample_keys(datamodule, split)
    batches: dict[tuple[str, int, int], Any] = {}
    loader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()
    for sample_index, batch in enumerate(loader):
        _, molecule_id, sample_id, scf_iteration = keys[sample_index]
        key = (molecule_id, sample_id, scf_iteration)
        if key in targets:
            batches[key] = to_torch(batch, device=device, float_dtype=dtype)
            if len(batches) == len(targets):
                break
    return batches


def _load_model(
    spec: RunSpec, device: torch.device, dtype: torch.dtype
) -> MLDFTLitModule:
    # Some PyTorch/Lightning combinations hit a thread-local map_location cleanup
    # bug when loading directly onto CUDA. Load on CPU first, then move normally.
    original_thread_local_state = torch._utils._thread_local_state

    class _SafeThreadLocalState:
        def __init__(self, source: Any) -> None:
            self.__dict__.update(getattr(source, "__dict__", {}))

        def __delattr__(self, name: str) -> None:
            self.__dict__.pop(name, None)

    torch._utils._thread_local_state = _SafeThreadLocalState(original_thread_local_state)
    try:
        model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location="cpu")
    finally:
        torch._utils._thread_local_state = original_thread_local_state
    model.eval()
    model.to(device)
    model.to(dtype)
    return model


def _cuda_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _measure(device: torch.device, fn):
    _reset_cuda_peak(device)
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0
    return out, elapsed_s, _cuda_peak_mb(device)


def _set_fixed_state(batch: Any, pos: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    batch.pos = pos.detach().clone().requires_grad_(True)
    batch.coeffs = coeffs.detach().clone()
    return batch.pos


def _energy(model: MLDFTLitModule, batch: Any) -> torch.Tensor:
    pred_energy, _ = model.net(batch)
    return pred_energy.sum()


def _force_at(
    model: MLDFTLitModule,
    batch: Any,
    pos: torch.Tensor,
    coeffs: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    pos_req = _set_fixed_state(batch, pos, coeffs)
    energy = _energy(model, batch)
    grad_pos = torch.autograd.grad(
        energy,
        pos_req,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return -grad_pos


def _full_hessian_autograd(
    model: MLDFTLitModule,
    batch: Any,
    base_pos: torch.Tensor,
    base_coeffs: torch.Tensor,
) -> np.ndarray:
    pos_req = _set_fixed_state(batch, base_pos, base_coeffs)
    energy = _energy(model, batch)
    grad_pos = torch.autograd.grad(
        energy,
        pos_req,
        create_graph=True,
        retain_graph=True,
    )[0]
    rows = []
    for component in grad_pos.reshape(-1):
        row = torch.autograd.grad(
            component,
            pos_req,
            retain_graph=True,
            create_graph=False,
        )[0]
        rows.append(row.reshape(-1).detach().cpu())
    return torch.stack(rows, dim=0).numpy()


def _full_hessian_fd(
    model: MLDFTLitModule,
    batch: Any,
    base_pos: torch.Tensor,
    base_coeffs: torch.Tensor,
    displacement: float,
) -> np.ndarray:
    n_coords = int(base_pos.numel())
    flat_base = base_pos.detach().clone().reshape(-1)
    columns = []
    for coord_idx in range(n_coords):
        pos_plus = flat_base.clone()
        pos_minus = flat_base.clone()
        pos_plus[coord_idx] += displacement
        pos_minus[coord_idx] -= displacement
        f_plus = _force_at(model, batch, pos_plus.reshape_as(base_pos), base_coeffs).reshape(-1)
        f_minus = _force_at(model, batch, pos_minus.reshape_as(base_pos), base_coeffs).reshape(-1)
        dforce = (f_plus - f_minus) / (2.0 * displacement)
        columns.append((-dforce).detach().cpu())
    batch.pos = base_pos
    batch.coeffs = base_coeffs
    return torch.stack(columns, dim=1).numpy()


def _hvp_autograd(
    model: MLDFTLitModule,
    batch: Any,
    base_pos: torch.Tensor,
    base_coeffs: torch.Tensor,
    direction: torch.Tensor,
) -> np.ndarray:
    pos_req = _set_fixed_state(batch, base_pos, base_coeffs)
    energy = _energy(model, batch)
    grad_pos = torch.autograd.grad(
        energy,
        pos_req,
        create_graph=True,
        retain_graph=True,
    )[0].reshape(-1)
    scalar = torch.dot(grad_pos, direction.reshape(-1))
    hvp = torch.autograd.grad(scalar, pos_req, create_graph=False, retain_graph=False)[0]
    return hvp.reshape(-1).detach().cpu().numpy()


def _hvp_fd(
    model: MLDFTLitModule,
    batch: Any,
    base_pos: torch.Tensor,
    base_coeffs: torch.Tensor,
    direction: torch.Tensor,
    eps: float,
) -> np.ndarray:
    pos_plus = base_pos + eps * direction.reshape_as(base_pos)
    pos_minus = base_pos - eps * direction.reshape_as(base_pos)
    f_plus = _force_at(model, batch, pos_plus, base_coeffs).reshape(-1)
    f_minus = _force_at(model, batch, pos_minus, base_coeffs).reshape(-1)
    return (-(f_plus - f_minus) / (2.0 * eps)).detach().cpu().numpy()


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = a - b
    ref_norm = float(np.linalg.norm(b))
    return {
        "finite": bool(np.isfinite(a).all()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_fro_error": float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else None,
    }


def _matrix_stats(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in matrix.shape],
        "finite": bool(np.isfinite(matrix).all()),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "max_abs": float(np.max(np.abs(matrix))),
    }


def _vector_stats(vector: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in vector.shape],
        "finite": bool(np.isfinite(vector).all()),
        "nan_count": int(np.isnan(vector).sum()),
        "inf_count": int(np.isinf(vector).sum()),
        "max_abs": float(np.max(np.abs(vector))),
    }


def _reference_path(reference_dir: Path, molecule_id: str, sample_id: int) -> Path:
    return reference_dir / f"pbe_hessian_{molecule_id}_{sample_id:07d}.npz"


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.as_posix()


def _unit_random_like(base_pos: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=base_pos.device)
    gen.manual_seed(seed)
    direction = torch.randn(
        base_pos.numel(),
        dtype=base_pos.dtype,
        device=base_pos.device,
        generator=gen,
    )
    direction = direction / torch.linalg.vector_norm(direction)
    return direction.reshape_as(base_pos)


def _unit_perturb_direction(
    base_batch: Any,
    perturbed_batch: Any | None,
) -> tuple[torch.Tensor | None, float | None]:
    if perturbed_batch is None:
        return None, None
    delta = (perturbed_batch.pos.detach() - base_batch.pos.detach()).reshape(-1)
    norm = torch.linalg.vector_norm(delta)
    if norm <= 0:
        return None, None
    return (delta / norm).reshape_as(base_batch.pos), float(norm.detach().cpu())


def _copy_without_self_edges(batch: Any) -> tuple[Any, dict[str, int]]:
    out = copy.copy(batch)
    edge_index = batch.edge_index
    keep = edge_index[0] != edge_index[1]
    out.edge_index = edge_index[:, keep].contiguous()
    return out, {
        "original_edges": int(edge_index.shape[1]),
        "kept_edges": int(keep.sum().detach().cpu()),
        "dropped_self_edges": int((~keep).sum().detach().cpu()),
    }


def _evaluate_fixed_density(
    spec: RunSpec,
    model: MLDFTLitModule,
    batches: dict[tuple[str, int, int], Any],
    molecule_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    records = []
    for molecule_id in molecule_ids:
        key = (molecule_id, 0, args.scf_iteration)
        if key not in batches:
            records.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "success": False,
                    "error": f"missing base batch {key}",
                }
            )
            continue
        batch = copy.copy(batches[key])
        base_pos = batch.pos.detach().clone()
        base_coeffs = batch.coeffs.detach().clone()
        natoms = int(base_pos.shape[0])

        try:
            (autograd_hessian, autograd_elapsed, autograd_peak_mb) = _measure(
                device,
                lambda: _full_hessian_autograd(model, batch, base_pos, base_coeffs),
            )
            (fd_hessian, fd_elapsed, fd_peak_mb) = _measure(
                device,
                lambda: _full_hessian_fd(
                    model,
                    batch,
                    base_pos,
                    base_coeffs,
                    args.fd_displacement,
                ),
            )
            artifact = args.output_dir / "hessians" / f"{spec.name}_{molecule_id}_fixed_density.npz"
            artifact_path = _save_npz(
                artifact,
                autograd_hessian=autograd_hessian,
                finite_difference_hessian=fd_hessian,
            )
            row: dict[str, Any] = {
                "run": spec.name,
                "molecule_id": molecule_id,
                "sample_id": 0,
                "scf_iteration": args.scf_iteration,
                "natoms": natoms,
                "success": True,
                "artifact_npz": artifact_path,
                "autograd_elapsed_s": autograd_elapsed,
                "autograd_peak_cuda_mb": autograd_peak_mb,
                "fd_elapsed_s": fd_elapsed,
                "fd_peak_cuda_mb": fd_peak_mb,
                "autograd_stats": _matrix_stats(autograd_hessian),
                "fd_stats": _matrix_stats(fd_hessian),
                "autograd_vs_fd": _compare(autograd_hessian, fd_hessian),
            }
            ref_path = _reference_path(args.reference_dir, molecule_id, 0)
            if ref_path.exists():
                ref = np.load(ref_path)["pbe_hessian"]
                row["reference_npz"] = ref_path.as_posix()
                row["autograd_vs_pbe"] = _compare(autograd_hessian, ref)
                row["fd_vs_pbe"] = _compare(fd_hessian, ref)
            records.append(row)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "natoms": natoms,
                    "success": False,
                    "error": repr(exc),
                }
            )
    return records


def _evaluate_drop_self_edges_diagnostic(
    spec: RunSpec,
    model: MLDFTLitModule,
    batches: dict[tuple[str, int, int], Any],
    molecule_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    records = []
    diagnostic_ids = [
        item.strip().zfill(7)
        for item in args.self_edge_diagnostic_molecules.split(",")
        if item.strip()
    ]
    for molecule_id in diagnostic_ids:
        if molecule_id not in molecule_ids:
            continue
        key = (molecule_id, 0, args.scf_iteration)
        if key not in batches:
            records.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "success": False,
                    "error": f"missing base batch {key}",
                }
            )
            continue
        batch, edge_counts = _copy_without_self_edges(batches[key])
        base_pos = batch.pos.detach().clone()
        base_coeffs = batch.coeffs.detach().clone()
        natoms = int(base_pos.shape[0])

        try:
            (autograd_hessian, autograd_elapsed, autograd_peak_mb) = _measure(
                device,
                lambda: _full_hessian_autograd(model, batch, base_pos, base_coeffs),
            )
            (fd_hessian, fd_elapsed, fd_peak_mb) = _measure(
                device,
                lambda: _full_hessian_fd(
                    model,
                    batch,
                    base_pos,
                    base_coeffs,
                    args.fd_displacement,
                ),
            )
            artifact = (
                args.output_dir
                / "hessians"
                / f"{spec.name}_{molecule_id}_drop_self_edges_diagnostic.npz"
            )
            artifact_path = _save_npz(
                artifact,
                autograd_hessian=autograd_hessian,
                finite_difference_hessian=fd_hessian,
            )
            records.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "natoms": natoms,
                    "success": True,
                    "diagnostic_only": True,
                    "definition": "Same scalar model energy, but self-loop edges are removed before forward.",
                    "edge_counts": edge_counts,
                    "artifact_npz": artifact_path,
                    "autograd_elapsed_s": autograd_elapsed,
                    "autograd_peak_cuda_mb": autograd_peak_mb,
                    "fd_elapsed_s": fd_elapsed,
                    "fd_peak_cuda_mb": fd_peak_mb,
                    "autograd_stats": _matrix_stats(autograd_hessian),
                    "fd_stats": _matrix_stats(fd_hessian),
                    "autograd_vs_fd": _compare(autograd_hessian, fd_hessian),
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "natoms": natoms,
                    "success": False,
                    "diagnostic_only": True,
                    "edge_counts": edge_counts,
                    "error": repr(exc),
                }
            )
    return records


def _evaluate_hvp(
    spec: RunSpec,
    model: MLDFTLitModule,
    batches: dict[tuple[str, int, int], Any],
    molecule_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for molecule_index, molecule_id in enumerate(molecule_ids):
        key = (molecule_id, 0, args.scf_iteration)
        if key not in batches:
            continue
        base_batch = copy.copy(batches[key])
        base_pos = base_batch.pos.detach().clone()
        base_coeffs = base_batch.coeffs.detach().clone()
        directions: list[tuple[str, torch.Tensor, float | None, Any | None]] = [
            (
                "random_unit",
                _unit_random_like(base_pos, args.seed + molecule_index),
                None,
                None,
            )
        ]
        for perturb_sample_id in args.directional_sample_ids:
            pert_batch = batches.get(
                (molecule_id, perturb_sample_id, args.scf_iteration)
            )
            perturb_direction, perturb_norm = _unit_perturb_direction(base_batch, pert_batch)
            if perturb_direction is not None:
                directions.append(
                    (
                        f"perturb_unit_s{perturb_sample_id}",
                        perturb_direction,
                        perturb_norm,
                        pert_batch,
                    )
                )

        for direction_name, direction, perturb_delta_norm, pert_batch in directions:
            try:
                (hvp, hvp_elapsed, hvp_peak_mb) = _measure(
                    device,
                    lambda direction=direction: _hvp_autograd(
                        model,
                        base_batch,
                        base_pos,
                        base_coeffs,
                        direction,
                    ),
                )
                (fd_hvp, fd_elapsed, fd_peak_mb) = _measure(
                    device,
                    lambda direction=direction: _hvp_fd(
                        model,
                        base_batch,
                        base_pos,
                        base_coeffs,
                        direction,
                        args.hvp_eps,
                    ),
                )
                row: dict[str, Any] = {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "direction": direction_name,
                    "direction_norm": float(torch.linalg.vector_norm(direction).detach().cpu()),
                    "perturb_delta_norm": perturb_delta_norm,
                    "success": True,
                    "hvp_elapsed_s": hvp_elapsed,
                    "hvp_peak_cuda_mb": hvp_peak_mb,
                    "fd_directional_elapsed_s": fd_elapsed,
                    "fd_directional_peak_cuda_mb": fd_peak_mb,
                    "hvp_stats": _vector_stats(hvp),
                    "hvp_vs_fd_directional": _compare(hvp, fd_hvp),
                }
                if pert_batch is not None and perturb_delta_norm:
                    f_base = _force_at(model, base_batch, base_pos, base_coeffs).reshape(-1)
                    f_pert = _force_at(
                        model,
                        base_batch,
                        pert_batch.pos.detach().clone(),
                        base_coeffs,
                    ).reshape(-1)
                    secant = (-(f_pert - f_base)).detach().cpu().numpy()
                    row["hvp_scaled_vs_existing_perturb_secant"] = _compare(
                        hvp * perturb_delta_norm,
                        secant,
                    )
                    if args.compare_pbe_force_secant:
                        if not hasattr(base_batch, "force_label") or not hasattr(
                            pert_batch, "force_label"
                        ):
                            raise AttributeError(
                                "PBE force secant comparison requires force labels on both geometries"
                            )
                        pbe_secant = -(
                            pert_batch.force_label.detach().reshape(-1)
                            - base_batch.force_label.detach().reshape(-1)
                        ).cpu().numpy()
                        row["hvp_scaled_vs_pbe_force_secant"] = _compare(
                            hvp * perturb_delta_norm,
                            pbe_secant,
                        )
                        row["hvp_vs_pbe_force_secant_per_bohr"] = _compare(
                            hvp,
                            pbe_secant / perturb_delta_norm,
                        )
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "run": spec.name,
                        "molecule_id": molecule_id,
                        "sample_id": 0,
                        "scf_iteration": args.scf_iteration,
                        "direction": direction_name,
                        "success": False,
                        "error": repr(exc),
                    }
                )
    return rows


def _unrolled_hvp(
    model: MLDFTLitModule,
    batch: Any,
    base_pos: torch.Tensor,
    base_coeffs: torch.Tensor,
    direction: torch.Tensor,
    steps: int,
    lr: float,
    use_projection: bool,
) -> tuple[np.ndarray, float]:
    pos = base_pos.detach().clone().requires_grad_(True)
    coeffs = base_coeffs.detach().clone().requires_grad_(True)
    final_energy = math.nan
    for _ in range(steps):
        batch.pos = pos
        batch.coeffs = coeffs
        energy = _energy(model, batch)
        grad_coeffs = torch.autograd.grad(
            energy,
            coeffs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if use_projection:
            grad_coeffs = project_gradient(grad_coeffs, batch)
        coeffs = coeffs - lr * grad_coeffs
        final_energy = float(energy.detach().cpu())

    batch.pos = pos
    batch.coeffs = coeffs
    energy = _energy(model, batch)
    final_energy = float(energy.detach().cpu())
    grad_pos = torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
    scalar = torch.dot(grad_pos.reshape(-1), direction.reshape(-1))
    hvp = torch.autograd.grad(scalar, pos, create_graph=False, retain_graph=False)[0]
    return hvp.reshape(-1).detach().cpu().numpy(), final_energy


def _evaluate_unrolled(
    spec: RunSpec,
    model: MLDFTLitModule,
    batches: dict[tuple[str, int, int], Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    molecule_id = args.unroll_molecule.zfill(7)
    key = (molecule_id, 0, args.scf_iteration)
    if key not in batches:
        return [{"run": spec.name, "molecule_id": molecule_id, "success": False, "error": "missing base batch"}]
    base_batch = copy.copy(batches[key])
    base_pos = base_batch.pos.detach().clone()
    base_coeffs = base_batch.coeffs.detach().clone()
    direction = _unit_random_like(base_pos, args.seed + 991)
    rows = []
    for steps in args.unroll_steps:
        try:
            def _run():
                return _unrolled_hvp(
                    model,
                    base_batch,
                    base_pos,
                    base_coeffs,
                    direction,
                    steps=steps,
                    lr=args.unroll_lr,
                    use_projection=args.unroll_project_gradient,
                )

            ((hvp, final_energy), elapsed, peak_mb) = _measure(device, _run)
            rows.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "steps": steps,
                    "lr": args.unroll_lr,
                    "project_gradient": args.unroll_project_gradient,
                    "success": True,
                    "elapsed_s": elapsed,
                    "peak_cuda_mb": peak_mb,
                    "final_energy": final_energy,
                    "hvp_stats": _vector_stats(hvp),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "run": spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": 0,
                    "scf_iteration": args.scf_iteration,
                    "steps": steps,
                    "lr": args.unroll_lr,
                    "project_gradient": args.unroll_project_gradient,
                    "success": False,
                    "error": repr(exc),
                }
            )
    return rows


def _ranking_consistency(fixed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mol: dict[str, list[dict[str, Any]]] = {}
    for row in fixed_rows:
        if (
            row.get("success")
            and "autograd_vs_pbe" in row
            and "fd_vs_pbe" in row
            and np.isfinite(row["autograd_vs_pbe"]["mae"])
            and np.isfinite(row["fd_vs_pbe"]["mae"])
        ):
            by_mol.setdefault(row["molecule_id"], []).append(row)
    out = []
    for molecule_id, rows in sorted(by_mol.items()):
        if len(rows) < 2:
            continue
        ag_sorted = sorted(rows, key=lambda row: row["autograd_vs_pbe"]["mae"])
        fd_sorted = sorted(rows, key=lambda row: row["fd_vs_pbe"]["mae"])
        out.append(
            {
                "molecule_id": molecule_id,
                "autograd_best_by_pbe_mae": ag_sorted[0]["run"],
                "fd_best_by_pbe_mae": fd_sorted[0]["run"],
                "consistent": ag_sorted[0]["run"] == fd_sorted[0]["run"],
            }
        )
    return out


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.float64 if args.model_dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    molecule_ids = [item.strip().zfill(7) for item in args.molecules.split(",") if item.strip()]
    targets = {
        (molecule_id, sample_id, args.scf_iteration)
        for molecule_id in molecule_ids
        for sample_id in (0, *args.directional_sample_ids)
    }
    if args.unroll_molecule:
        targets.add((args.unroll_molecule.zfill(7), 0, args.scf_iteration))

    fixed_rows: list[dict[str, Any]] = []
    drop_self_edge_rows: list[dict[str, Any]] = []
    hvp_rows: list[dict[str, Any]] = []
    unroll_rows: list[dict[str, Any]] = []

    for item in args.run:
        spec = _parse_run(item)
        model = _load_model(spec, device, dtype)
        batches = _collect_batches(
            spec.run_dir,
            targets,
            args.num_workers,
            device,
            dtype,
            args.split,
            args.compare_pbe_force_secant,
        )
        run_rows = _evaluate_fixed_density(spec, model, batches, molecule_ids, args, device)
        fixed_rows.extend(run_rows)
        if args.run_self_edge_diagnostic:
            drop_self_edge_rows.extend(
                _evaluate_drop_self_edges_diagnostic(
                    spec,
                    model,
                    batches,
                    molecule_ids,
                    args,
                    device,
                )
            )
        if args.run_unrolled and spec.name == args.unroll_run:
            unroll_rows.extend(_evaluate_unrolled(spec, model, batches, args, device))

        stable_fixed_density = all(
            row.get("success") and row.get("autograd_stats", {}).get("finite")
            for row in run_rows
        )
        if args.run_hvp and stable_fixed_density:
            hvp_rows.extend(_evaluate_hvp(spec, model, batches, molecule_ids, args, device))
        elif args.run_hvp:
            hvp_rows.append(
                {
                    "run": spec.name,
                    "success": False,
                    "skipped": True,
                    "reason": "fixed-density full-edge autograd Hessian was not finite",
                }
            )

    result = {
        "definition": (
            "Fixed-density second-order autograd audit. Hessian is d2E_model/dR2 "
            "from scalar model energy at fixed coefficients. HVP is d/dR[(dE/dR) dot v]."
        ),
        "limitations": [
            "Fixed-density sections do not optimize density and do not include implicit density response.",
            "Unrolled section is a differentiable prototype over model-energy coefficient updates, not the production OFDFT optimizer.",
            "Classical OFDFT integral/nuclear terms are not differentiated in this audit.",
        ],
        "device": str(device),
        "model_dtype": str(dtype),
        "split": args.split,
        "molecules": molecule_ids,
        "scf_iteration": args.scf_iteration,
        "fd_displacement": args.fd_displacement,
        "hvp_eps": args.hvp_eps,
        "directional_sample_ids": args.directional_sample_ids,
        "compare_pbe_force_secant": args.compare_pbe_force_secant,
        "reference_dir": args.reference_dir.as_posix(),
        "fixed_density_full_hessian": fixed_rows,
        "drop_self_edges_diagnostic": drop_self_edge_rows,
        "hvp": hvp_rows,
        "unrolled_density_prototype": unroll_rows,
        "ranking_consistency": _ranking_consistency(fixed_rows),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--molecules", default="0000010,0000062")
    parser.add_argument("--scf-iteration", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("_runtime/qm9_p1_models/eval/qm9_p1_410_hessian_ref_benchmark/pbe_hessians"),
    )
    parser.add_argument("--fd-displacement", type=float, default=1e-3)
    parser.add_argument("--hvp-eps", type=float, default=1e-3)
    parser.add_argument("--directional-sample-ids", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--compare-pbe-force-secant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-hvp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-self-edge-diagnostic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-edge-diagnostic-molecules", default="0000010")
    parser.add_argument("--run-unrolled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unroll-run", default="EGF_lam1_s3000")
    parser.add_argument("--unroll-molecule", default="0000010")
    parser.add_argument("--unroll-steps", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--unroll-lr", type=float, default=1e-4)
    parser.add_argument("--unroll-project-gradient", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
