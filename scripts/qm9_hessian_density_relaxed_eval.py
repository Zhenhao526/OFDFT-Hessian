#!/usr/bin/env python3
"""Density-relaxed derived-force finite-difference Hessian audit for QM9 P1-410.

This is intentionally not named a full OFDFT Hessian evaluator.  Each displaced
geometry is first passed through the normal OFDFT density optimization machinery,
then the force is derived from the model scalar energy as ``F = -dE_model/dR`` at
the optimized density.  Classical OFDFT energy terms are used during density
optimization, but their nuclear derivatives are not included in the reported
force because the current PySCF-built integral path is not differentiable with
respect to ``sample.pos``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from pyscf import gto

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.optimizer import TorchOptimizer
from mldft.ofdft.run_density_optimization import SampleGenerator, density_optimization


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


@dataclass
class OptimizationTrace:
    total_energies: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)

    def __call__(self, minimization_locals: dict[str, Any]) -> None:
        self.total_energies.append(float(minimization_locals["energy"].total_energy))
        self.gradient_norms.append(float(minimization_locals["gradient_norm"]))


def _save_optimization_trace(
    trace_dir: Path | None,
    context: "RunContext",
    molecule_id: str,
    sample_id: int,
    coord_idx: int | None,
    side: str,
    trace: OptimizationTrace,
    metadata: dict[str, Any],
) -> str | None:
    """Persist a complete optimization curve without inflating the summary JSON."""
    if trace_dir is None:
        return None
    trace_dir.mkdir(parents=True, exist_ok=True)
    coord = side if coord_idx is None else f"coord_{coord_idx:03d}_{side}"
    path = trace_dir / (
        f"{context.spec.name}_{molecule_id}_{sample_id:07d}_{coord}_optimization_trace.npz"
    )
    np.savez_compressed(
        path,
        cycle=np.arange(len(trace.gradient_norms), dtype=np.int64),
        total_energy=np.asarray(trace.total_energies, dtype=np.float64),
        projected_gradient_norm=np.asarray(trace.gradient_norms, dtype=np.float64),
        first_stage_cycles=np.asarray(metadata["first_stage_cycles"], dtype=np.int64),
        fallback_cycles=np.asarray(metadata["fallback_cycles"], dtype=np.int64),
    )
    return path.as_posix()


@dataclass
class RunContext:
    spec: RunSpec
    model: MLDFTLitModule
    sample_generator: SampleGenerator
    functional_factory: FunctionalFactory
    optimizer: TorchOptimizer
    prepared_geometry_signature: str


@dataclass
class PreparedGeometryCache:
    """CPU cache for model-independent transformed samples at exact geometries."""

    sample_generator: SampleGenerator
    entries: dict[str, Any] = field(default_factory=dict)
    build_count: int = 0
    hit_count: int = 0

    @staticmethod
    def _key(atomic_numbers: np.ndarray, positions_bohr: np.ndarray, charge: int) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(atomic_numbers, dtype=np.int64).tobytes())
        digest.update(np.asarray(positions_bohr, dtype=np.float64).tobytes())
        digest.update(int(charge).to_bytes(4, byteorder="little", signed=True))
        return digest.hexdigest()

    def get(
        self,
        atomic_numbers: np.ndarray,
        positions_bohr: np.ndarray,
        charge: int,
        device: torch.device,
    ) -> Any:
        key = self._key(atomic_numbers, positions_bohr, charge)
        if key not in self.entries:
            mol = _make_mol(atomic_numbers, positions_bohr, charge)
            sample = self.sample_generator.get_sample_from_mol(mol)
            self.entries[key] = sample.to("cpu")
            self.build_count += 1
        else:
            self.hit_count += 1
        return self.entries[key].clone().to(device)


def _prepared_sample(
    context: RunContext,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    charge: int,
    cache: PreparedGeometryCache | None,
) -> Any:
    if cache is not None:
        return cache.get(atomic_numbers, positions_bohr, charge, context.model.device)
    mol = _make_mol(atomic_numbers, positions_bohr, charge)
    return context.sample_generator.get_sample_from_mol(mol)


def _parse_run(item: str) -> RunSpec:
    name, run_dir, ckpt = item.split("=", maxsplit=2)
    return RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve())


def _build_optimizer(args: argparse.Namespace) -> TorchOptimizer:
    if args.optimizer == "sgd":
        return TorchOptimizer(
            torch.optim.SGD,
            lr=args.lr,
            momentum=args.momentum,
            max_cycle=args.max_cycle,
            convergence_tolerance=args.convergence_tolerance,
        )
    if args.optimizer == "adam":
        return TorchOptimizer(
            torch.optim.Adam,
            lr=args.lr,
            max_cycle=args.max_cycle,
            convergence_tolerance=args.convergence_tolerance,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _build_optimizer_from_settings(
    optimizer_name: str,
    lr: float,
    max_cycle: int,
    convergence_tolerance: float,
    momentum: float,
) -> TorchOptimizer:
    if optimizer_name == "sgd":
        return TorchOptimizer(
            torch.optim.SGD,
            lr=lr,
            momentum=momentum,
            max_cycle=max_cycle,
            convergence_tolerance=convergence_tolerance,
        )
    if optimizer_name == "adam":
        return TorchOptimizer(
            torch.optim.Adam,
            lr=lr,
            max_cycle=max_cycle,
            convergence_tolerance=convergence_tolerance,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _load_context(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> RunContext:
    cfg = OmegaConf.load(spec.run_dir / "hparams.yaml")
    signature_payload = OmegaConf.to_yaml(
        OmegaConf.create(
            {
                "basis_info": OmegaConf.to_container(cfg.data.basis_info, resolve=True),
                "transforms": OmegaConf.to_container(cfg.data.transforms, resolve=True),
                "target_key": str(cfg.data.target_key),
            }
        ),
        resolve=True,
        sort_keys=True,
    )
    model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location=device)
    model.eval()
    model.to(device)
    model.to(torch.float64)
    sample_generator = SampleGenerator(
        cfg,
        model,
        negative_integrated_density_penalty_weight=args.negative_integrated_density_penalty_weight,
        transform_device=args.transform_device,
    )
    functional_factory = FunctionalFactory.from_module(
        model,
        negative_integrated_density_penalty_weight=args.negative_integrated_density_penalty_weight,
    )
    optimizer = _build_optimizer(args)
    return RunContext(
        spec=spec,
        model=model,
        sample_generator=sample_generator,
        functional_factory=functional_factory,
        optimizer=optimizer,
        prepared_geometry_signature=hashlib.sha256(signature_payload.encode()).hexdigest(),
    )


def _load_geometry(dataset_dir: Path, molecule_id: str, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    root = zarr.open(label_path, mode="r")
    return (
        np.asarray(root["geometry/atomic_numbers"], dtype=np.int64),
        np.asarray(root["geometry/atom_pos"], dtype=np.float64),
    )


def _make_mol(atomic_numbers: np.ndarray, positions_bohr: np.ndarray, charge: int) -> gto.Mole:
    atoms = [
        (int(atomic_number), tuple(float(x) for x in position))
        for atomic_number, position in zip(atomic_numbers, positions_bohr)
    ]
    nelectron = int(np.sum(atomic_numbers) - charge)
    spin = nelectron % 2
    return gto.M(
        atom=atoms,
        unit="Bohr",
        charge=charge,
        spin=spin,
        basis="sto-3g",
        verbose=0,
    )


def _optimize_density(
    context: RunContext,
    sample: Any,
    args: argparse.Namespace,
    initialization: str | torch.Tensor,
    initialization_mode: str,
) -> tuple[Any, torch.Tensor, dict[str, Any], OptimizationTrace]:
    trace = OptimizationTrace()
    t0 = time.time()
    fast_modes = {"parameter_response_prediction", "checkpoint_density_warm_start"}
    fast_path_attempted = bool(
        getattr(args, "response_predictor_fast_refine", False)
        and initialization_mode in fast_modes
        and isinstance(initialization, torch.Tensor)
    )
    fast_path_used = False
    fast_path_fallback_reason = None
    fast_path_initial_gradient_norm = None
    if fast_path_attempted:
        sample.coeffs = initialization.detach().clone()
        normalization = sample.dual_basis_integrals.detach().to(sample.coeffs)
        target = torch.as_tensor(
            sample.mol.nelectron,
            dtype=sample.coeffs.dtype,
            device=sample.coeffs.device,
        )
        sample.coeffs = sample.coeffs + normalization * (
            (target - torch.dot(normalization, sample.coeffs))
            / torch.dot(normalization, normalization)
        )
        variable = sample.coeffs.detach().clone().requires_grad_(True)
        sample.coeffs = variable
        tensor_energies = context.functional_factory.evaluate_tensor_functional(
            sample,
            sample.coulomb_matrix,
            sample.nuclear_attraction_vector,
        )
        gradient = torch.autograd.grad(tensor_energies.total_energy, variable)[0]
        projected = gradient - normalization * (
            torch.dot(normalization, gradient)
            / torch.dot(normalization, normalization)
        )
        fast_path_initial_gradient_norm = float(
            torch.linalg.vector_norm(projected).detach().cpu()
        )
        sample.coeffs = variable.detach()
        fast_threshold = float(
            getattr(args, "response_predictor_fast_refine_threshold", 1.0e-5)
        )
        if (
            math.isfinite(fast_path_initial_gradient_norm)
            and fast_path_initial_gradient_norm < fast_threshold
        ):
            fast_path_used = True
            final_coeffs = transform_tensor_with_sample(
                sample, sample.coeffs, Representation.VECTOR, invert=True
            ).detach()
            metadata = {
                "converged": fast_path_initial_gradient_norm
                < args.convergence_tolerance,
                "cycles": 0,
                "elapsed_s": time.time() - t0,
                "final_total_energy": float(
                    tensor_energies.total_energy.detach().cpu()
                ),
                "initialization_mode": initialization_mode,
                "initial_gradient_norm": fast_path_initial_gradient_norm,
                "final_gradient_norm": fast_path_initial_gradient_norm,
                "first_stage_converged": False,
                "first_stage_cycles": 0,
                "first_stage_final_gradient_norm": None,
                "used_fallback": False,
                "fallback_converged": None,
                "fallback_cycles": 0,
                "fallback_final_gradient_norm": None,
                "response_predictor_fast_path_attempted": True,
                "response_predictor_fast_path_used": True,
                "response_predictor_fast_path_succeeded": None,
                "response_predictor_fast_path_fallback_reason": None,
                "response_predictor_fast_path_initial_gradient_norm": (
                    fast_path_initial_gradient_norm
                ),
            }
            return tensor_energies.detached(), final_coeffs, metadata, trace
        fast_path_fallback_reason = (
            "initial_projected_gradient_not_within_fast_threshold: "
            f"{fast_path_initial_gradient_norm:.6e} >= {fast_threshold:.6e}"
        )

    energies, final_coeffs, first_stage_converged, _ = density_optimization(
        sample,
        sample.mol,
        context.optimizer,
        context.functional_factory,
        callback=trace,
        initialization=initialization,
        max_xc_memory=args.max_xc_memory,
        normalize_initial_guess=args.normalize_initial_guess,
        ks_basis=args.ks_basis,
        disable_printing=True,
        disable_pbar=True,
    )
    first_stage_cycles = len(trace.gradient_norms)
    first_stage_final_gradient_norm = trace.gradient_norms[-1] if trace.gradient_norms else None
    used_fallback = False
    fallback_cycles = 0
    fallback_final_gradient_norm = None
    fallback_converged = None

    converged = bool(first_stage_converged)
    should_fallback = (
        args.fallback_optimizer is not None
        and ((not converged) or args.fallback_always)
        and (
            first_stage_final_gradient_norm is None
            or first_stage_final_gradient_norm >= args.fallback_convergence_tolerance
        )
    )
    if should_fallback:
        used_fallback = True
        fallback_trace = OptimizationTrace()
        fallback_optimizer = _build_optimizer_from_settings(
            args.fallback_optimizer,
            args.fallback_lr,
            args.fallback_max_cycle,
            args.fallback_convergence_tolerance,
            args.momentum,
        )
        energies, final_coeffs, fallback_converged, _ = density_optimization(
            sample,
            sample.mol,
            fallback_optimizer,
            context.functional_factory,
            callback=fallback_trace,
            initialization=sample.coeffs.detach().clone(),
            max_xc_memory=args.max_xc_memory,
            normalize_initial_guess=args.normalize_initial_guess,
            ks_basis=args.ks_basis,
            disable_printing=True,
            disable_pbar=True,
        )
        trace.total_energies.extend(fallback_trace.total_energies)
        trace.gradient_norms.extend(fallback_trace.gradient_norms)
        fallback_cycles = len(fallback_trace.gradient_norms)
        fallback_final_gradient_norm = (
            fallback_trace.gradient_norms[-1] if fallback_trace.gradient_norms else None
        )
        converged = bool(fallback_converged)
    elapsed_s = time.time() - t0

    metadata = {
        "converged": bool(converged),
        "cycles": len(trace.gradient_norms),
        "elapsed_s": elapsed_s,
        "final_total_energy": float(energies.total_energy),
        "initialization_mode": initialization_mode,
        "initial_gradient_norm": trace.gradient_norms[0] if trace.gradient_norms else None,
        "final_gradient_norm": trace.gradient_norms[-1] if trace.gradient_norms else None,
        "first_stage_converged": bool(first_stage_converged),
        "first_stage_cycles": first_stage_cycles,
        "first_stage_final_gradient_norm": first_stage_final_gradient_norm,
        "used_fallback": used_fallback,
        "fallback_converged": fallback_converged,
        "fallback_cycles": fallback_cycles,
        "fallback_final_gradient_norm": fallback_final_gradient_norm,
        "response_predictor_fast_path_attempted": fast_path_attempted,
        "response_predictor_fast_path_used": fast_path_used,
        "response_predictor_fast_path_succeeded": False,
        "response_predictor_fast_path_fallback_reason": fast_path_fallback_reason,
        "response_predictor_fast_path_initial_gradient_norm": (
            fast_path_initial_gradient_norm
        ),
    }
    return energies, final_coeffs, metadata, trace


def _density_relaxed_force(
    context: RunContext,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
    base_coeffs_untransformed: torch.Tensor | None = None,
    initialization_mode_override: str | None = None,
    return_final_coeffs: bool = False,
    prepared_geometry_cache: PreparedGeometryCache | None = None,
) -> (
    tuple[np.ndarray, dict[str, Any], OptimizationTrace]
    | tuple[np.ndarray, dict[str, Any], OptimizationTrace, torch.Tensor]
):
    point_t0 = time.perf_counter()
    sample_build_t0 = time.perf_counter()
    sample = _prepared_sample(
        context,
        atomic_numbers,
        positions_bohr,
        args.charge,
        prepared_geometry_cache,
    )
    sample_build_elapsed_s = time.perf_counter() - sample_build_t0
    if base_coeffs_untransformed is None:
        initialization: str | torch.Tensor = args.initialization
        initialization_mode = args.initialization
    else:
        coeffs = base_coeffs_untransformed.to(
            device=sample.coeffs.device,
            dtype=sample.coeffs.dtype,
        )
        initialization = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
        initialization_mode = initialization_mode_override or "base_density_warm_start"
    _, final_coeffs, metadata, trace = _optimize_density(
        context,
        sample,
        args,
        initialization,
        initialization_mode,
    )

    sample.pos = sample.pos.detach().clone().requires_grad_(True)
    sample.coeffs = sample.coeffs.detach().clone()
    force_t0 = time.perf_counter()
    with torch.enable_grad():
        _, _, _, pred_forces = context.model.forward_predictions(
            sample,
            compute_density_gradients=False,
            compute_forces=True,
        )
    if pred_forces is None:
        raise RuntimeError("model.forward_predictions returned pred_forces=None")
    force_autograd_elapsed_s = time.perf_counter() - force_t0
    force = pred_forces.detach().cpu().numpy()

    metadata.update(
        {
            "sample_build_elapsed_s": sample_build_elapsed_s,
            "density_optimization_elapsed_s": metadata["elapsed_s"],
            "force_autograd_elapsed_s": force_autograd_elapsed_s,
            "total_point_elapsed_s": time.perf_counter() - point_t0,
            "finite_force": bool(np.isfinite(force).all()),
        }
    )
    if return_final_coeffs:
        return force, metadata, trace, final_coeffs.detach().clone()
    return force, metadata, trace


def _density_relaxed_base_coeffs(
    context: RunContext,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
    prepared_geometry_cache: PreparedGeometryCache | None = None,
) -> tuple[torch.Tensor, dict[str, Any], OptimizationTrace]:
    point_t0 = time.perf_counter()
    sample_build_t0 = time.perf_counter()
    sample = _prepared_sample(
        context,
        atomic_numbers,
        positions_bohr,
        args.charge,
        prepared_geometry_cache,
    )
    sample_build_elapsed_s = time.perf_counter() - sample_build_t0
    _, final_coeffs, metadata, trace = _optimize_density(
        context,
        sample,
        args,
        args.initialization,
        args.initialization,
    )
    metadata.update(
        {
            "sample_build_elapsed_s": sample_build_elapsed_s,
            "density_optimization_elapsed_s": metadata["elapsed_s"],
            "force_autograd_elapsed_s": 0.0,
            "total_point_elapsed_s": time.perf_counter() - point_t0,
        }
    )
    return final_coeffs.detach().clone(), metadata, trace


def _finite_difference_hessian_density_relaxed(
    context: RunContext,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
    molecule_id: str,
    sample_id: int,
    prepared_geometry_cache: PreparedGeometryCache | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any] | None]:
    n_coords = int(positions_bohr.size)
    columns: list[np.ndarray] = []
    optimization_rows: list[dict[str, Any]] = []
    base_coeffs_untransformed = None
    base_optimization_row = None
    if args.base_density_warm_start:
        base_coeffs_untransformed, base_metadata, base_trace = _density_relaxed_base_coeffs(
            context,
            atomic_numbers,
            positions_bohr,
            args,
            prepared_geometry_cache,
        )
        base_optimization_row = {
            "coord_idx": None,
            "side": "base",
            **base_metadata,
        }
        base_optimization_row["trace_file"] = _save_optimization_trace(
            args.optimization_trace_dir,
            context,
            molecule_id,
            sample_id,
            None,
            "base",
            base_trace,
            base_metadata,
        )
    flat_base = positions_bohr.reshape(-1)
    for coord_idx in range(n_coords):
        forces = {}
        plus_coeffs_untransformed = None
        for sign, label in [(1.0, "plus"), (-1.0, "minus")]:
            flat = flat_base.copy()
            flat[coord_idx] += sign * args.displacement
            initial_coeffs = base_coeffs_untransformed
            initialization_mode_override = None
            if (
                label == "minus"
                and args.pair_response_extrapolation
                and base_coeffs_untransformed is not None
                and plus_coeffs_untransformed is not None
            ):
                initial_coeffs = 2.0 * base_coeffs_untransformed - plus_coeffs_untransformed
                initialization_mode_override = "pair_response_extrapolation"
            result = _density_relaxed_force(
                context,
                atomic_numbers,
                flat.reshape(positions_bohr.shape),
                args,
                base_coeffs_untransformed=initial_coeffs,
                initialization_mode_override=initialization_mode_override,
                return_final_coeffs=(label == "plus" and args.pair_response_extrapolation),
                prepared_geometry_cache=prepared_geometry_cache,
            )
            if label == "plus" and args.pair_response_extrapolation:
                force, metadata, trace, plus_coeffs_untransformed = result
            else:
                force, metadata, trace = result
            forces[label] = force.reshape(-1)
            metadata["trace_file"] = _save_optimization_trace(
                args.optimization_trace_dir,
                context,
                molecule_id,
                sample_id,
                coord_idx,
                label,
                trace,
                metadata,
            )
            optimization_rows.append(
                {
                    "coord_idx": coord_idx,
                    "side": label,
                    **metadata,
                }
            )
        dforce_dcoord = (forces["plus"] - forces["minus"]) / (2.0 * args.displacement)
        columns.append(-dforce_dcoord)
    return np.stack(columns, axis=1), optimization_rows, base_optimization_row


def _compare(model_hessian: np.ndarray, ref_hessian: np.ndarray) -> dict[str, Any]:
    diff = model_hessian - ref_hessian
    ref_norm = float(np.linalg.norm(ref_hessian))
    model_sym = 0.5 * (model_hessian + model_hessian.T)
    model_asym = 0.5 * (model_hessian - model_hessian.T)
    ref_sym = 0.5 * (ref_hessian + ref_hessian.T)
    sym_diff = model_sym - ref_sym
    ref_sym_norm = float(np.linalg.norm(ref_sym))
    model_sym_norm = float(np.linalg.norm(model_sym))
    raw_abs_error = np.abs(diff).reshape(-1)
    sym_abs_error = np.abs(sym_diff).reshape(-1)
    asym_abs = np.abs(model_hessian - model_hessian.T).reshape(-1)

    def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_q50": float(np.quantile(values, 0.50)),
            f"{prefix}_q90": float(np.quantile(values, 0.90)),
            f"{prefix}_q95": float(np.quantile(values, 0.95)),
            f"{prefix}_q99": float(np.quantile(values, 0.99)),
            f"{prefix}_q100": float(np.max(values)),
        }

    return {
        "finite": bool(np.isfinite(model_hessian).all()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_fro_error": float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else None,
        "model_symmetry_max_abs_error": float(np.max(np.abs(model_hessian - model_hessian.T))),
        "model_hessian_max_abs": float(np.max(np.abs(model_hessian))),
        "symmetrized_mae": float(np.mean(sym_abs_error)),
        "symmetrized_rmse": float(np.sqrt(np.mean(sym_diff * sym_diff))),
        "symmetrized_relative_fro_error": (
            float(np.linalg.norm(sym_diff) / ref_sym_norm) if ref_sym_norm > 0 else None
        ),
        "model_symmetric_fro_norm": model_sym_norm,
        "model_antisymmetric_fro_norm": float(np.linalg.norm(model_asym)),
        "antisymmetric_over_symmetric_fro": (
            float(np.linalg.norm(model_asym) / model_sym_norm) if model_sym_norm > 0 else None
        ),
        "model_antisymmetric_max_abs": float(np.max(np.abs(model_asym))),
        **quantiles(raw_abs_error, "raw_abs_error"),
        **quantiles(sym_abs_error, "sym_abs_error"),
        **quantiles(asym_abs, "symmetry_abs"),
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run",
        "molecule_id",
        "sample_id",
        "natoms",
        "success",
        "mae",
        "rmse",
        "relative_fro_error",
        "max_abs_error",
        "model_symmetry_max_abs_error",
        "model_hessian_max_abs",
        "symmetrized_mae",
        "symmetrized_rmse",
        "symmetrized_relative_fro_error",
        "model_symmetric_fro_norm",
        "model_antisymmetric_fro_norm",
        "antisymmetric_over_symmetric_fro",
        "model_antisymmetric_max_abs",
        "raw_abs_error_q50",
        "raw_abs_error_q90",
        "raw_abs_error_q95",
        "raw_abs_error_q99",
        "raw_abs_error_q100",
        "sym_abs_error_q50",
        "sym_abs_error_q90",
        "sym_abs_error_q95",
        "sym_abs_error_q99",
        "sym_abs_error_q100",
        "symmetry_abs_q50",
        "symmetry_abs_q90",
        "symmetry_abs_q95",
        "symmetry_abs_q99",
        "symmetry_abs_q100",
        "n_optimizations",
        "n_converged",
        "mean_opt_cycles",
        "mean_final_gradient_norm",
        "elapsed_s",
        "reference_cache",
        "base_warm_start_converged",
        "base_warm_start_cycles",
        "base_warm_start_final_gradient_norm",
        "base_warm_start_elapsed_s",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.pair_response_extrapolation and not args.base_density_warm_start:
        raise ValueError("--pair-response-extrapolation requires --base-density-warm-start")
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    manifest = json.loads(args.manifest_json.read_text())
    references = [
        row for row in manifest if row.get("success") and Path(row["cache_path"]).exists()
    ][: args.max_molecules]

    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    prepared_geometry_cache = None
    if args.share_prepared_geometry_across_runs:
        reference_basis = contexts[0].sample_generator.basis_info
        for context in contexts[1:]:
            candidate_basis = context.sample_generator.basis_info
            if (
                not np.array_equal(
                    np.asarray(candidate_basis.atomic_numbers),
                    np.asarray(reference_basis.atomic_numbers),
                )
                or not np.array_equal(
                    np.asarray(candidate_basis.basis_dim_per_atom),
                    np.asarray(reference_basis.basis_dim_per_atom),
                )
                or context.model.target_key != contexts[0].model.target_key
                or context.prepared_geometry_signature
                != contexts[0].prepared_geometry_signature
            ):
                raise ValueError(
                    "Prepared-geometry sharing requires identical basis metadata, transforms, "
                    "and target_key"
                )
        prepared_geometry_cache = PreparedGeometryCache(contexts[0].sample_generator)
    rows: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    base_optimization_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for context in contexts:
        for ref in references:
            base_row = {
                "run": context.spec.name,
                "molecule_id": ref["molecule_id"],
                "sample_id": int(ref["sample_id"]),
                "natoms": int(ref["natoms"]),
                "reference_cache": ref["cache_path"],
                "success": False,
                "error": None,
            }
            t0 = time.time()
            try:
                atomic_numbers, positions_bohr = _load_geometry(
                    args.dataset_dir,
                    ref["molecule_id"],
                    int(ref["sample_id"]),
                )
                ref_hessian = np.load(ref["cache_path"])["pbe_hessian"]
                model_hessian, opt_rows, base_opt_row = _finite_difference_hessian_density_relaxed(
                    context,
                    atomic_numbers,
                    positions_bohr,
                    args,
                    ref["molecule_id"],
                    int(ref["sample_id"]),
                    prepared_geometry_cache,
                )
                hessian_npz = None
                if args.hessian_npz_dir is not None:
                    args.hessian_npz_dir.mkdir(parents=True, exist_ok=True)
                    hessian_npz = (
                        args.hessian_npz_dir
                        / f"{context.spec.name}_{ref['molecule_id']}_{int(ref['sample_id']):07d}_density_relaxed_hessian.npz"
                    )
                    np.savez_compressed(
                        hessian_npz,
                        density_relaxed_hessian=model_hessian,
                        pbe_hessian=ref_hessian,
                    )
                if base_opt_row is not None:
                    base_optimization_rows.append(
                        {
                            "run": context.spec.name,
                            "molecule_id": ref["molecule_id"],
                            "sample_id": int(ref["sample_id"]),
                            **base_opt_row,
                        }
                    )
                for opt_row in opt_rows:
                    optimization_rows.append(
                        {
                            "run": context.spec.name,
                            "molecule_id": ref["molecule_id"],
                            "sample_id": int(ref["sample_id"]),
                            **opt_row,
                        }
                    )
                n_optimizations = len(opt_rows)
                n_converged = sum(1 for row in opt_rows if row["converged"])
                final_grad_norms = [
                    row["final_gradient_norm"]
                    for row in opt_rows
                    if row["final_gradient_norm"] is not None
                ]
                row = {
                    **base_row,
                    "success": True,
                    "n_optimizations": n_optimizations,
                    "n_converged": n_converged,
                    "mean_opt_cycles": _mean([float(row["cycles"]) for row in opt_rows]),
                    "mean_final_gradient_norm": _mean(final_grad_norms),
                    "base_warm_start_converged": (
                        bool(base_opt_row["converged"]) if base_opt_row is not None else None
                    ),
                    "base_warm_start_cycles": (
                        int(base_opt_row["cycles"]) if base_opt_row is not None else None
                    ),
                    "base_warm_start_final_gradient_norm": (
                        base_opt_row["final_gradient_norm"] if base_opt_row is not None else None
                    ),
                    "base_warm_start_elapsed_s": (
                        base_opt_row["elapsed_s"] if base_opt_row is not None else None
                    ),
                    "elapsed_s": time.time() - t0,
                    "hessian_npz": hessian_npz.as_posix() if hessian_npz is not None else None,
                    **_compare(model_hessian, ref_hessian),
                }
                rows.append(row)
                print(
                    "completed",
                    context.spec.name,
                    ref["molecule_id"],
                    f"converged={n_converged}/{n_optimizations}",
                    f"mae={row['mae']:.6g}",
                    f"elapsed_s={row['elapsed_s']:.1f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({**base_row, "elapsed_s": time.time() - t0, "error": repr(exc)})
                print(
                    "failed",
                    context.spec.name,
                    ref["molecule_id"],
                    repr(exc),
                    flush=True,
                )

        run_rows = [row for row in rows if row["run"] == context.spec.name and row["success"]]
        failed_rows = [row for row in rows if row["run"] == context.spec.name and not row["success"]]
        run_optimization_rows = [
            row for row in optimization_rows if row["run"] == context.spec.name
        ]
        summaries[context.spec.name] = {
            "run_dir": context.spec.run_dir.as_posix(),
            "ckpt": context.spec.ckpt.as_posix(),
            "n_success": len(run_rows),
            "n_failed": len(failed_rows),
            "mean_mae": _mean([row["mae"] for row in run_rows]),
            "mean_rmse": _mean([row["rmse"] for row in run_rows]),
            "mean_relative_fro_error": _mean([row["relative_fro_error"] for row in run_rows]),
            "mean_symmetrized_mae": _mean([row["symmetrized_mae"] for row in run_rows]),
            "mean_symmetrized_rmse": _mean([row["symmetrized_rmse"] for row in run_rows]),
            "mean_symmetrized_relative_fro_error": _mean(
                [row["symmetrized_relative_fro_error"] for row in run_rows]
            ),
            "mean_antisymmetric_over_symmetric_fro": _mean(
                [row["antisymmetric_over_symmetric_fro"] for row in run_rows]
            ),
            "mean_model_symmetry_max_abs_error": _mean(
                [row["model_symmetry_max_abs_error"] for row in run_rows]
            ),
            "mean_opt_cycles": _mean([row["mean_opt_cycles"] for row in run_rows]),
            "mean_final_gradient_norm": _mean(
                [row["mean_final_gradient_norm"] for row in run_rows]
            ),
            "mean_elapsed_s": _mean([row["elapsed_s"] for row in run_rows]),
            "total_elapsed_s": float(sum(row["elapsed_s"] for row in run_rows)),
            "timing_breakdown_per_displacement_point_s": {
                "sample_build_mean": _mean(
                    [row.get("sample_build_elapsed_s") for row in run_optimization_rows]
                ),
                "density_optimization_mean": _mean(
                    [
                        row.get("density_optimization_elapsed_s")
                        for row in run_optimization_rows
                    ]
                ),
                "force_autograd_mean": _mean(
                    [row.get("force_autograd_elapsed_s") for row in run_optimization_rows]
                ),
                "total_mean": _mean(
                    [row.get("total_point_elapsed_s") for row in run_optimization_rows]
                ),
            },
        }

    result = {
        "definition": (
            "density-relaxed derived-force proxy: each displaced geometry is density-optimized, "
            "then forces are autograd derivatives of the model scalar energy at the optimized density"
        ),
        "limitations": [
            "Not a full total OFDFT Hessian because classical integral/nuclear terms are not differentiated with respect to nuclear coordinates.",
            "Density optimization is truncated by max_cycle and may be unconverged.",
        ],
        "manifest_json": args.manifest_json.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "device": str(device),
        "displacement": args.displacement,
        "max_molecules": args.max_molecules,
        "initialization": args.initialization,
        "base_density_warm_start": args.base_density_warm_start,
        "pair_response_extrapolation": args.pair_response_extrapolation,
        "share_prepared_geometry_across_runs": args.share_prepared_geometry_across_runs,
        "prepared_geometry_cache": (
            {
                "entries": len(prepared_geometry_cache.entries),
                "build_count": prepared_geometry_cache.build_count,
                "hit_count": prepared_geometry_cache.hit_count,
                "storage_device": "cpu",
                "signature": contexts[0].prepared_geometry_signature,
            }
            if prepared_geometry_cache is not None
            else None
        ),
        "optimization_trace_dir": (
            args.optimization_trace_dir.as_posix()
            if args.optimization_trace_dir is not None
            else None
        ),
        "optimizer": {
            "name": f"torch.optim.{args.optimizer.upper() if args.optimizer == 'sgd' else 'Adam'}",
            "lr": args.lr,
            "momentum": args.momentum,
            "max_cycle": args.max_cycle,
            "convergence_tolerance": args.convergence_tolerance,
        },
        "fallback_optimizer": (
            {
                "name": (
                    f"torch.optim.{args.fallback_optimizer.upper()}"
                    if args.fallback_optimizer == "sgd"
                    else "torch.optim.Adam"
                ),
                "lr": args.fallback_lr,
                "max_cycle": args.fallback_max_cycle,
                "convergence_tolerance": args.fallback_convergence_tolerance,
                "fallback_always": args.fallback_always,
                "trigger": (
                    "if first stage does not converge, or whenever --fallback-always is set; "
                    "skipped if the first-stage final gradient is already below fallback tolerance"
                ),
            }
            if args.fallback_optimizer is not None
            else None
        ),
        "n_references": len(references),
        "runs": summaries,
        "rows": rows,
        "optimization_rows": optimization_rows,
        "base_optimization_rows": base_optimization_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
    if args.optimization_csv is not None:
        args.optimization_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.optimization_csv.open("w", newline="") as f:
            fieldnames = [
                "run",
                "molecule_id",
                "sample_id",
                "coord_idx",
                "side",
                "converged",
                "cycles",
                "elapsed_s",
                "final_total_energy",
                "initialization_mode",
                "initial_gradient_norm",
                "final_gradient_norm",
                "first_stage_converged",
                "first_stage_cycles",
                "first_stage_final_gradient_norm",
                "used_fallback",
                "fallback_converged",
                "fallback_cycles",
                "fallback_final_gradient_norm",
                "finite_force",
                "sample_build_elapsed_s",
                "density_optimization_elapsed_s",
                "force_autograd_elapsed_s",
                "total_point_elapsed_s",
                "trace_file",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(optimization_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--optimization-csv", type=Path, default=None)
    parser.add_argument("--optimization-trace-dir", type=Path, default=None)
    parser.add_argument("--hessian-npz-dir", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=5)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--max-cycle", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fallback-optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--fallback-max-cycle", type=int, default=10000)
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1e-4)
    parser.add_argument("--fallback-lr", type=float, default=3e-4)
    parser.add_argument("--fallback-always", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-density-warm-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pair-response-extrapolation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize each minus displacement from 2*rho(base)-rho(plus). "
            "Requires --base-density-warm-start."
        ),
    )
    parser.add_argument(
        "--share-prepared-geometry-across-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache transformed geometry/integral samples on CPU and reuse them across model "
            "runs. Only valid when runs share basis metadata, transforms, and target_key."
        ),
    )
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
