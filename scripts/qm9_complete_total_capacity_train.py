#!/usr/bin/env python3
"""Block-coordinate capacity fit with complete-total relaxed scalar-derived force secants."""

from __future__ import annotations

import argparse
import gc
import hashlib
import csv
import json
import math
import os
import resource
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr
from pyscf.data.elements import MASSES

from mldft.ml.models.components.three_body_geometry_residual import (
    ThreeBodyGeometryResidual,
)
from mldft.ofdft.complete_total_training import (
    assign_two_task_pcgrad,
    alternating_update_kind,
    assemble_hessian_columns,
    central_energy_directional_curvature,
    central_force_secant_hvp,
    consume_implicit_response_diagnostics,
    differentiable_constrained_density_unroll,
    hessian_error_metrics,
    hutchinson_internal_frobenius_squared_loss,
    implicit_stationary_density_parameter_response,
    internal_coordinate_projector,
    low_mode_curvature_loss,
    mixed_absolute_relative_l1,
    mixed_absolute_relative_rmse,
    normalized_energy_l1,
    parameter_gradient_diagnostics,
    stationary_density_parameter_step_prediction,
)
from mldft.ofdft.conservative_force import (
    evaluate_total_ofdft_force,
    prepare_differentiable_geometry,
    prepare_fixed_geometry_scalar_energy,
)
from mldft.ofdft.geometry_integrals import FiniteDifferencePySCFIntegralProvider
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.implicit_response import ConstrainedResponseSystem
from mldft.ofdft.internal_directions import (
    InternalDirectionBank,
    sample_internal_rademacher_direction,
)

from qm9_hessian_density_relaxed_eval import _load_context, _parse_run
from qm9_total_ofdft_force_audit import _evaluate_point


@dataclass(frozen=True)
class Direction:
    index: int
    kind: str
    vector: np.ndarray
    target_hvp: np.ndarray
    role: str = "train"


@dataclass
class RelaxedPoint:
    positions_bohr: np.ndarray
    coefficients: torch.Tensor
    final_gradient_norm: float
    cycles: int
    total_energy: float


@dataclass
class MoleculeState:
    molecule_id: str
    atomic_numbers: np.ndarray
    positions_bohr: np.ndarray
    pbe_total_energy: float
    pbe_force: np.ndarray
    pbe_hessian: np.ndarray
    label_coefficients: torch.Tensor
    directions: list[Direction]
    direction_basis_definition: str = "cartesian"
    external_basis: np.ndarray | None = None
    expected_internal_dimension: int | None = None
    loaded_direction_role: str = "all"
    base: RelaxedPoint | None = None
    displaced: dict[tuple[int, str], RelaxedPoint] | None = None

    def __post_init__(self) -> None:
        if self.displaced is None:
            self.displaced = {}


class IntegralBundleCache:
    """Small LRU because one all-coordinate derivative bundle can occupy hundreds of MiB."""

    def __init__(self, context: Any, args: argparse.Namespace):
        self.context = context
        self.args = args
        self.entries: OrderedDict[bytes, Any] = OrderedDict()
        self.build_count = 0
        self.hit_count = 0

    def get(self, atomic_numbers: np.ndarray, positions_bohr: np.ndarray) -> Any:
        key = (
            np.asarray(atomic_numbers, dtype=np.int64).tobytes()
            + np.asarray(positions_bohr, dtype=np.float64).tobytes()
        )
        if key in self.entries:
            self.hit_count += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        provider = FiniteDifferencePySCFIntegralProvider(
            atomic_numbers=atomic_numbers,
            basis=self.context.sample_generator.basis_info.basis_dict,
            charge=self.args.charge,
            derivative_step_bohr=self.args.integral_derivative_step,
            derivative_workers=self.args.integral_derivative_workers,
        )
        bundle = provider.evaluate_with_derivatives(positions_bohr)
        self.entries[key] = bundle
        self.build_count += 1
        while len(self.entries) > self.args.integral_cache_entries:
            self.entries.popitem(last=False)
        return bundle

    def get_directional(
        self,
        atomic_numbers: np.ndarray,
        positions_bohr: np.ndarray,
        direction: np.ndarray,
    ) -> Any:
        direction = np.asarray(direction, dtype=np.float64)
        key = (
            b"directional:"
            + np.asarray(atomic_numbers, dtype=np.int64).tobytes()
            + np.asarray(positions_bohr, dtype=np.float64).tobytes()
            + direction.tobytes()
            + np.asarray(
                [self.args.integral_directional_second_step],
                dtype=np.float64,
            ).tobytes()
        )
        if key in self.entries:
            self.hit_count += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        provider = FiniteDifferencePySCFIntegralProvider(
            atomic_numbers=atomic_numbers,
            basis=self.context.sample_generator.basis_info.basis_dict,
            charge=self.args.charge,
            derivative_step_bohr=self.args.integral_derivative_step,
            derivative_workers=self.args.integral_derivative_workers,
        )
        bundle = provider.evaluate_with_directional_second_derivatives(
            positions_bohr,
            direction,
            directional_step_bohr=self.args.integral_directional_second_step,
        )
        self.entries[key] = bundle
        self.build_count += 1
        while len(self.entries) > self.args.integral_cache_entries:
            self.entries.popitem(last=False)
        return bundle


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_molecule(
    manifest_entry: dict[str, Any],
    low_mode_count: int,
    direction_entry: dict[str, Any] | None = None,
    direction_role: str = "all",
) -> MoleculeState:
    molecule_id = str(manifest_entry["molecule_id"])
    label_path = Path(manifest_entry["label_path"])
    root = zarr.open(label_path, mode="r")
    atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
    positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    pbe_force = np.asarray(root["metadata/pbe_derivatives/forces"], dtype=np.float64)
    label_coefficients = torch.as_tensor(
        np.asarray(root["of_labels/spatial/coeffs"][-1], dtype=np.float64)
    )
    energy_trace = np.asarray(root["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        root["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    with np.load(manifest_entry["pbe_hessian_path"]) as payload:
        pbe_hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
    direction_basis_definition = "cartesian"
    external_basis = None
    expected_internal_dimension = None
    if direction_entry is None:
        directions = _directions(
            atomic_numbers,
            positions,
            pbe_hessian,
            low_mode_count=low_mode_count,
        )
    else:
        direction_path = Path(direction_entry["direction_path"])
        expected_hash = str(direction_entry["direction_sha256"])
        actual_hash = _sha256(direction_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"direction artifact hash mismatch for {molecule_id}: "
                f"{actual_hash} != {expected_hash}"
            )
        with np.load(direction_path) as payload:
            direction_vectors = np.asarray(payload["directions"], dtype=np.float64)
            direction_kinds = np.asarray(payload["kinds"]).astype(str)
            partial_roles = np.asarray(payload["partial_roles"]).astype(str)
            target_hvps = np.asarray(payload["pbe_hvp"], dtype=np.float64)
            artifact_positions = np.asarray(payload["positions_bohr"], dtype=np.float64)
            artifact_atomic_numbers = np.asarray(
                payload["atomic_numbers"], dtype=np.int64
            )
            external_basis = np.asarray(payload["external_basis"], dtype=np.float64)
            expected_internal_dimension = int(
                np.asarray(payload["internal_dimension"]).item()
            )
        if not np.array_equal(artifact_atomic_numbers, atomic_numbers):
            raise ValueError(f"direction atomic-number drift for {molecule_id}")
        if not np.allclose(artifact_positions, positions, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"direction geometry drift for {molecule_id}")
        expected_shape = (expected_internal_dimension, positions.size)
        if direction_vectors.shape != expected_shape:
            raise ValueError(
                f"direction shape drift for {molecule_id}: "
                f"{direction_vectors.shape} != {expected_shape}"
            )
        if (
            direction_kinds.shape != (expected_internal_dimension,)
            or partial_roles.shape != (expected_internal_dimension,)
            or target_hvps.shape != direction_vectors.shape
        ):
            raise ValueError(f"direction metadata shape drift for {molecule_id}")
        if direction_role not in {"all", "train", "heldout"}:
            raise ValueError(f"unsupported direction role {direction_role}")
        selection = (
            np.ones(expected_internal_dimension, dtype=np.bool_)
            if direction_role == "all"
            else partial_roles == direction_role
        )
        if not np.any(selection):
            raise ValueError(
                f"no {direction_role} directions selected for {molecule_id}"
            )
        expected_targets = np.einsum(
            "ij,dj->di", pbe_hessian, direction_vectors
        )
        if not np.allclose(
            target_hvps, expected_targets, atol=1.0e-11, rtol=1.0e-11
        ):
            raise ValueError(f"PBE HVP target drift for {molecule_id}")
        directions = [
            Direction(
                index=int(index),
                kind=str(direction_kinds[index]),
                vector=direction_vectors[index].reshape(positions.shape),
                target_hvp=target_hvps[index].reshape(positions.shape),
                role=str(partial_roles[index]),
            )
            for index in np.flatnonzero(selection)
        ]
        direction_basis_definition = "structured_internal_orthonormal"
    return MoleculeState(
        molecule_id=molecule_id,
        atomic_numbers=atomic_numbers,
        positions_bohr=positions,
        pbe_total_energy=float(energy_trace[has_energy][-1]),
        pbe_force=pbe_force,
        pbe_hessian=pbe_hessian,
        label_coefficients=label_coefficients,
        directions=directions,
        direction_basis_definition=direction_basis_definition,
        external_basis=external_basis,
        expected_internal_dimension=expected_internal_dimension,
        loaded_direction_role=direction_role,
    )


def _directions(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    pbe_hessian: np.ndarray,
    *,
    low_mode_count: int,
) -> list[Direction]:
    ncoordinate = positions_bohr.size
    directions = []
    identity = np.eye(ncoordinate, dtype=np.float64)
    for coordinate in range(ncoordinate):
        vector = identity[:, coordinate].reshape(positions_bohr.shape)
        directions.append(
            Direction(
                index=len(directions),
                kind="cartesian",
                vector=vector,
                target_hvp=(pbe_hessian @ vector.reshape(-1)).reshape(vector.shape),
            )
        )
    if low_mode_count <= 0:
        return directions

    masses = np.asarray([MASSES[int(number)] for number in atomic_numbers], dtype=np.float64)
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    mass_weighted = (
        inverse_sqrt_mass[:, None] * pbe_hessian * inverse_sqrt_mass[None, :]
    )
    projector = internal_coordinate_projector(
        torch.as_tensor(positions_bohr, dtype=torch.float64),
        torch.as_tensor(masses, dtype=torch.float64),
    ).numpy()
    projector_eigenvalues, projector_eigenvectors = np.linalg.eigh(projector)
    internal_basis = projector_eigenvectors[:, projector_eigenvalues > 0.5]
    reduced = internal_basis.T @ (0.5 * (mass_weighted + mass_weighted.T)) @ internal_basis
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    order = np.argsort(np.abs(eigenvalues))[: min(low_mode_count, eigenvalues.size)]
    for local_index in order:
        mass_weighted_mode = internal_basis @ eigenvectors[:, local_index]
        cartesian = inverse_sqrt_mass * mass_weighted_mode
        cartesian /= max(np.linalg.norm(cartesian), np.finfo(float).tiny)
        vector = cartesian.reshape(positions_bohr.shape)
        directions.append(
            Direction(
                index=len(directions),
                kind="low_mode",
                vector=vector,
                target_hvp=(pbe_hessian @ cartesian).reshape(vector.shape),
            )
        )
    return directions


def _density_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """The shared strict evaluator intentionally consumes one explicit settings namespace."""
    values = vars(args).copy()
    values.update(
        {
            "initialization": args.base_initialization,
            "optimizer": "adam",
            "lr": args.density_lr,
            "max_cycle": args.density_max_cycles,
            "convergence_tolerance": args.density_first_threshold,
            "momentum": 0.9,
            "fallback_optimizer": "adam",
            "fallback_lr": args.density_fallback_lr,
            "fallback_max_cycle": args.density_fallback_max_cycles,
            "fallback_convergence_tolerance": args.density_fallback_threshold,
            "fallback_always": True,
            "lbfgs_refine": bool(getattr(args, "lbfgs_refine", True)),
            "lbfgs_tolerance": args.density_strict_threshold,
            "lbfgs_max_iterations": args.lbfgs_max_iterations,
            "lbfgs_history_size": 50,
            "newton_refine": bool(getattr(args, "newton_refine", True)),
            "newton_tolerance": args.density_strict_threshold,
            "newton_max_iterations": args.newton_max_iterations,
            "newton_max_krylov_iterations": 200,
            "newton_krylov_tolerance": 1.0e-10,
            "newton_diagonal_probes": 8,
            "newton_damping": 1.0e-8,
            "negative_integrated_density_penalty_weight": 0.0,
            "max_xc_memory": 4000,
            "normalize_initial_guess": True,
            "ks_basis": "sto-3g",
            "model_geometry_derivative": "autograd",
            "model_geometry_fd_step": None,
            "model_geometry_fd_richardson": False,
            "optimization_trace_dir": args.output_dir / "density_traces",
            "coefficients_dir": args.output_dir / "density_coefficients",
        }
    )
    return argparse.Namespace(**values)


def _relax(
    context: Any,
    molecule: MoleculeState,
    positions_bohr: np.ndarray,
    density_args: argparse.Namespace,
    *,
    warm_start: torch.Tensor | None,
    explicit_start: torch.Tensor | None = None,
) -> tuple[RelaxedPoint, dict[str, Any]]:
    point = _evaluate_point(
        context,
        molecule.atomic_numbers,
        positions_bohr,
        density_args.charge,
        density_args,
        base_coeffs=warm_start,
        need_force=False,
        initial_coeffs=explicit_start,
        initialization_mode_override=(
            "previous_refresh" if explicit_start is not None else None
        ),
    )
    metadata = point["optimization_metadata"]
    final_norm = float(metadata["final_gradient_norm"])
    if not bool(metadata["converged"]) or final_norm >= density_args.density_strict_threshold:
        failure_root = Path(density_args.output_dir) / "failed_density_points"
        failure_root.mkdir(parents=True, exist_ok=True)
        positions_bytes = np.asarray(
            positions_bohr, dtype=np.float64
        ).tobytes()
        positions_sha256 = hashlib.sha256(positions_bytes).hexdigest()
        failure_stem = f"{molecule.molecule_id}_{positions_sha256[:16]}"
        coefficient_path = failure_root / f"{failure_stem}_coefficients.pt"
        trace_path = failure_root / f"{failure_stem}_trace.npz"
        metadata_path = failure_root / f"{failure_stem}_metadata.json"
        torch.save(point["final_coeffs"].detach().cpu(), coefficient_path)
        trace = point.get("trace")
        if trace is not None:
            np.savez_compressed(
                trace_path,
                total_energy=np.asarray(
                    trace.total_energies, dtype=np.float64
                ),
                projected_gradient_norm=np.asarray(
                    trace.gradient_norms, dtype=np.float64
                ),
            )
        serializable_metadata = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (bool, int, float, str, type(None), list))
        }
        serializable_metadata.update(
            {
                "molecule_id": molecule.molecule_id,
                "positions_sha256": positions_sha256,
                "coefficients": coefficient_path.as_posix(),
                "coefficients_sha256": _sha256(coefficient_path),
                "trace": trace_path.as_posix() if trace is not None else None,
                "trace_sha256": (
                    _sha256(trace_path) if trace is not None else None
                ),
            }
        )
        metadata_path.write_text(
            json.dumps(serializable_metadata, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            f"Density did not reach {density_args.density_strict_threshold:g}: "
            f"converged={metadata['converged']} norm={final_norm:.3e} "
            f"cycles={metadata.get('cycles')} "
            f"pre_lbfgs={metadata.get('pre_lbfgs_gradient_norm')} "
            f"lbfgs_converged={metadata.get('lbfgs_converged')} "
            f"lbfgs_final={metadata.get('lbfgs_final_gradient_norm')} "
            f"pre_newton={metadata.get('pre_newton_gradient_norm')} "
            f"newton_converged={metadata.get('newton_converged')} "
            f"newton_final={metadata.get('newton_final_gradient_norm')}"
        )
    return (
        RelaxedPoint(
            positions_bohr=np.asarray(positions_bohr, dtype=np.float64),
            coefficients=point["final_coeffs"].detach().cpu(),
            final_gradient_norm=final_norm,
            cycles=int(metadata["cycles"]),
            total_energy=float(metadata["final_total_energy"]),
        ),
        metadata,
    )


def _refresh_densities(
    context: Any,
    molecules: list[MoleculeState],
    density_args: argparse.Namespace,
    refresh_index: int,
) -> list[dict[str, Any]]:
    context.model.eval()
    rows = []
    for molecule in molecules:
        previous_base = (
            molecule.base.coefficients
            if molecule.base is not None
            else molecule.label_coefficients
        )
        molecule.base, metadata = _relax(
            context,
            molecule,
            molecule.positions_bohr,
            density_args,
            warm_start=None,
            explicit_start=previous_base,
        )
        rows.append(
            {
                "refresh": refresh_index,
                "molecule_id": molecule.molecule_id,
                "direction_index": -1,
                "kind": "base",
                "side": "base",
                "cycles": molecule.base.cycles,
                "final_gradient_norm": molecule.base.final_gradient_norm,
                "total_energy": molecule.base.total_energy,
            }
        )
        for direction in molecule.directions:
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                key = (direction.index, side)
                previous = molecule.displaced.get(key)
                positions = (
                    molecule.positions_bohr
                    + sign * density_args.displacement * direction.vector
                )
                relaxed, metadata = _relax(
                    context,
                    molecule,
                    positions,
                    density_args,
                    warm_start=molecule.base.coefficients,
                    explicit_start=(previous.coefficients if previous is not None else None),
                )
                molecule.displaced[key] = relaxed
                rows.append(
                    {
                        "refresh": refresh_index,
                        "molecule_id": molecule.molecule_id,
                        "direction_index": direction.index,
                        "kind": direction.kind,
                        "side": side,
                        "cycles": relaxed.cycles,
                        "final_gradient_norm": relaxed.final_gradient_norm,
                        "total_energy": relaxed.total_energy,
                    }
                )
    return rows


def _refresh_base_densities(
    context: Any,
    molecules: list[MoleculeState],
    density_args: argparse.Namespace,
    refresh_index: int,
    parameter_step: int | None = None,
) -> list[dict[str, Any]]:
    """Strictly relax one center density per molecule."""
    context.model.eval()
    rows = []
    for molecule in molecules:
        previous_base = (
            molecule.base.coefficients
            if molecule.base is not None
            else molecule.label_coefficients
        )
        molecule.base, _ = _relax(
            context,
            molecule,
            molecule.positions_bohr,
            density_args,
            warm_start=None,
            explicit_start=previous_base,
        )
        rows.append(
            {
                "refresh": refresh_index,
                "parameter_step": parameter_step,
                "refresh_scope": "strict_center_only",
                "molecule_id": molecule.molecule_id,
                "direction_index": -1,
                "kind": "base",
                "side": "base",
                "cycles": molecule.base.cycles,
                "final_gradient_norm": molecule.base.final_gradient_norm,
                "total_energy": molecule.base.total_energy,
            }
        )
    return rows


def _predict_next_parameter_step_densities(
    context: Any,
    molecules: list[MoleculeState],
    parameters: list[torch.Tensor],
    old_parameter_values: list[torch.Tensor],
    new_parameter_values: list[torch.Tensor],
    *,
    charge: int,
    damping: float,
) -> list[float]:
    """Replace center-density warm starts by exact linear-response predictions."""
    parameter_steps = [
        new - old
        for old, new in zip(
            old_parameter_values, new_parameter_values, strict=True
        )
    ]
    with torch.no_grad():
        for parameter, old in zip(
            parameters, old_parameter_values, strict=True
        ):
            parameter.copy_(old)
    correction_norms: list[float] = []
    try:
        for molecule in molecules:
            if molecule.base is None:
                raise RuntimeError("density predictor requires a center density")
            fixed_energy = prepare_fixed_geometry_scalar_energy(
                context.sample_generator,
                context.functional_factory,
                molecule.atomic_numbers,
                molecule.positions_bohr,
                charge=charge,
            )
            coefficients = molecule.base.coefficients.to(
                device=fixed_energy.normalization_untransformed.device,
                dtype=fixed_energy.normalization_untransformed.dtype,
            )
            correction = stationary_density_parameter_step_prediction(
                coefficients,
                fixed_energy.normalization_untransformed,
                fixed_energy,
                parameters,
                parameter_steps,
                damping=damping,
            )
            predicted = coefficients + correction.detach()
            normalization = fixed_energy.normalization_untransformed
            target = torch.as_tensor(
                int(np.sum(molecule.atomic_numbers) - charge),
                dtype=predicted.dtype,
                device=predicted.device,
            )
            residual = torch.dot(normalization, predicted) - target
            predicted = predicted - normalization * (
                residual / torch.dot(normalization, normalization)
            )
            molecule.base.coefficients = predicted.detach().cpu()
            correction_norms.append(
                float(torch.linalg.vector_norm(correction).detach().cpu())
            )
    finally:
        with torch.no_grad():
            for parameter, new in zip(
                parameters, new_parameter_values, strict=True
            ):
                parameter.copy_(new)
    return correction_norms


def _refresh_active_densities(
    context: Any,
    molecules: list[MoleculeState],
    selected_by_molecule: dict[str, list[Direction]],
    density_args: argparse.Namespace,
    refresh_index: int,
    parameter_step: int,
) -> list[dict[str, Any]]:
    """Strictly relax only the base and directions used by the next update."""
    context.model.eval()
    rows = []
    for molecule in molecules:
        previous_base = (
            molecule.base.coefficients
            if molecule.base is not None
            else molecule.label_coefficients
        )
        molecule.base, _ = _relax(
            context,
            molecule,
            molecule.positions_bohr,
            density_args,
            warm_start=None,
            explicit_start=previous_base,
        )
        rows.append(
            {
                "refresh": refresh_index,
                "parameter_step": parameter_step,
                "refresh_scope": "strict_active",
                "molecule_id": molecule.molecule_id,
                "direction_index": -1,
                "kind": "base",
                "side": "base",
                "cycles": molecule.base.cycles,
                "final_gradient_norm": molecule.base.final_gradient_norm,
                "total_energy": molecule.base.total_energy,
            }
        )
        for direction in selected_by_molecule[molecule.molecule_id]:
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                key = (direction.index, side)
                previous = molecule.displaced.get(key)
                positions = (
                    molecule.positions_bohr
                    + sign * density_args.displacement * direction.vector
                )
                relaxed, _ = _relax(
                    context,
                    molecule,
                    positions,
                    density_args,
                    warm_start=molecule.base.coefficients,
                    explicit_start=(
                        previous.coefficients if previous is not None else None
                    ),
                )
                molecule.displaced[key] = relaxed
                rows.append(
                    {
                        "refresh": refresh_index,
                        "parameter_step": parameter_step,
                        "refresh_scope": "strict_active",
                        "molecule_id": molecule.molecule_id,
                        "direction_index": direction.index,
                        "kind": direction.kind,
                        "side": side,
                        "cycles": relaxed.cycles,
                        "final_gradient_norm": relaxed.final_gradient_norm,
                        "total_energy": relaxed.total_energy,
                    }
                )
    return rows


def _density_refresh_scope(
    *,
    strict_active: bool,
    current_parameter_step: int,
    density_parameter_step: int,
    refresh_needed: bool,
    refresh_interval: int,
) -> str | None:
    if strict_active:
        return (
            "active"
            if current_parameter_step != density_parameter_step
            else None
        )
    if refresh_needed or (
        current_parameter_step - density_parameter_step >= refresh_interval
    ):
        return "all"
    return None


def _assert_training_density_stationarity(
    maximum_projected_gradient_norm: float,
    strict_threshold: float,
    *,
    enabled: bool,
) -> None:
    if enabled and (
        not math.isfinite(maximum_projected_gradient_norm)
        or maximum_projected_gradient_norm >= strict_threshold
    ):
        raise RuntimeError(
            "Training graph used a non-stationary density: "
            f"projected_gradient_norm={maximum_projected_gradient_norm:.3e} "
            f"threshold={strict_threshold:.3e}"
        )


def _training_density_stationarity_threshold(
    args: argparse.Namespace,
) -> float:
    threshold = args.training_density_stationarity_threshold
    return (
        float(args.density_strict_threshold)
        if threshold is None
        else float(threshold)
    )


def _evaluate_point_graph(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    point: RelaxedPoint,
    *,
    create_graph: bool,
    attach_density_parameter_response: bool = True,
    directional_second_direction: np.ndarray | None = None,
) -> Any:
    coefficients = point.coefficients
    detach_coefficients = True
    if create_graph and attach_density_parameter_response and (
        cache.args.implicit_density_parameter_response
        or cache.args.density_response_unroll_steps > 0
    ):
        fixed_energy = prepare_fixed_geometry_scalar_energy(
            context.sample_generator,
            context.functional_factory,
            molecule.atomic_numbers,
            point.positions_bohr,
            charge=cache.args.charge,
        )
        initial = coefficients.to(
            device=fixed_energy.normalization_untransformed.device,
            dtype=fixed_energy.normalization_untransformed.dtype,
        ).detach().clone().requires_grad_(True)
        n_electron = int(np.sum(molecule.atomic_numbers) - cache.args.charge)
        if cache.args.implicit_density_parameter_response:
            coefficients = implicit_stationary_density_parameter_response(
                initial,
                fixed_energy.normalization_untransformed,
                n_electron,
                fixed_energy,
                (
                    parameter
                    for parameter in context.model.net.parameters()
                    if parameter.requires_grad
                ),
                tolerance=cache.args.implicit_response_tolerance,
                max_iterations=cache.args.implicit_response_max_iterations,
                damping=cache.args.implicit_response_damping,
                diagonal_probes=cache.args.implicit_response_diagonal_probes,
                solver=cache.args.implicit_response_solver,
                warm_start_key=(
                    (
                        f"{molecule.molecule_id}:"
                        + hashlib.sha256(
                            np.asarray(
                                point.positions_bohr, dtype=np.float64
                            ).tobytes()
                        ).hexdigest()
                    )
                    if cache.args.implicit_response_warm_start
                    else None
                ),
            )
        else:
            coefficients = differentiable_constrained_density_unroll(
                initial,
                fixed_energy.normalization_untransformed,
                n_electron,
                fixed_energy,
                steps=cache.args.density_response_unroll_steps,
                learning_rate=cache.args.density_response_unroll_lr,
            )
        detach_coefficients = False
    bundle = (
        cache.get(molecule.atomic_numbers, point.positions_bohr)
        if directional_second_direction is None
        else cache.get_directional(
            molecule.atomic_numbers,
            point.positions_bohr,
            directional_second_direction,
        )
    )
    geometry = prepare_differentiable_geometry(
        context.sample_generator,
        molecule.atomic_numbers,
        point.positions_bohr,
        coefficients,
        integral_bundle=bundle,
        charge=cache.args.charge,
        detach_coefficients=detach_coefficients,
    )
    return evaluate_total_ofdft_force(
        context.functional_factory,
        geometry,
        n_electron=int(np.sum(molecule.atomic_numbers) - cache.args.charge),
        create_graph=create_graph,
        sample_generator=context.sample_generator,
        charge=cache.args.charge,
        detach_lagrange_multiplier=not (
            create_graph
            and (
                cache.args.connect_lagrange_multiplier_response
                or cache.args.implicit_density_parameter_response
            )
        ),
    )


def _label_density_replay_point(molecule: MoleculeState) -> RelaxedPoint:
    """Return the fixed PBE/KS-density point used only by the E/G/F replay."""
    return RelaxedPoint(
        positions_bohr=np.asarray(molecule.positions_bohr, dtype=np.float64),
        coefficients=molecule.label_coefficients,
        final_gradient_norm=math.nan,
        cycles=0,
        total_energy=molecule.pbe_total_energy,
    )


def _response_cancellation_diagnostics(
    partial_hvp: torch.Tensor,
    response_correction: torch.Tensor,
) -> dict[str, float]:
    """Summarize whether the relaxed HVP is a fragile cancellation."""
    partial = partial_hvp.detach()
    correction = response_correction.detach()
    relaxed = partial + correction
    tiny = torch.finfo(partial.dtype).tiny
    partial_norm = torch.linalg.vector_norm(partial)
    correction_norm = torch.linalg.vector_norm(correction)
    relaxed_norm = torch.linalg.vector_norm(relaxed)
    cosine = torch.sum(partial * correction) / (
        partial_norm * correction_norm
    ).clamp_min(tiny)
    return {
        "partial_hvp_norm": float(partial_norm.cpu()),
        "response_correction_norm": float(correction_norm.cpu()),
        "relaxed_hvp_norm": float(relaxed_norm.cpu()),
        "response_correction_fraction_of_relaxed_norm": float(
            (correction_norm / relaxed_norm.clamp_min(tiny)).cpu()
        ),
        "partial_response_cosine": float(cosine.cpu()),
        "cancellation_index": float(
            ((partial_norm + correction_norm) / relaxed_norm.clamp_min(tiny)).cpu()
        ),
    }


def _analytic_center_response(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    direction: Direction,
    *,
    create_graph: bool,
) -> tuple[
    Any,
    torch.Tensor,
    ConstrainedResponseSystem,
    Any,
    dict[str, float | int | str | bool],
]:
    """Build and solve one center-geometry constrained density response."""
    if molecule.base is None:
        raise RuntimeError("analytic relaxed HVP requires a strict center density")
    parameter_device = next(context.model.parameters()).device
    if parameter_device.type == "cuda":
        torch.cuda.synchronize(parameter_device)
    started = time.perf_counter()
    center = _evaluate_point_graph(
        context,
        cache,
        molecule,
        molecule.base,
        create_graph=True,
        attach_density_parameter_response=create_graph,
        directional_second_direction=direction.vector,
    )
    _synchronize(center.geometry.coeffs)
    graph_seconds = time.perf_counter() - started
    if center.geometry.bundle.directional_second_derivatives is None:
        raise RuntimeError(
            "analytic relaxed HVP requires directional second integral derivatives"
        )
    normalization_first_max = float(
        np.max(np.abs(center.geometry.bundle.derivatives.normalization))
    )
    normalization_second_max = float(
        np.max(
            np.abs(
                center.geometry.bundle.directional_second_derivatives.normalization
            )
        )
    )
    if max(normalization_first_max, normalization_second_max) > 1.0e-12:
        raise RuntimeError(
            "analytic relaxed HVP currently requires a geometry-independent "
            "electron-number constraint; normalization derivatives were nonzero"
        )
    position_direction = torch.as_tensor(
        direction.vector,
        dtype=center.geometry.positions.dtype,
        device=center.geometry.positions.device,
    )
    response_system = ConstrainedResponseSystem(
        total_energy=center.energies.total_energy,
        coeffs=center.geometry.coeffs,
        positions=center.geometry.positions,
        normalization=torch.as_tensor(
            center.geometry.bundle.values.normalization,
            dtype=center.geometry.coeffs.dtype,
            device=center.geometry.coeffs.device,
        ),
        n_electron=int(np.sum(molecule.atomic_numbers) - cache.args.charge),
        multiplier=center.lagrange_multiplier,
    )
    solve_started = time.perf_counter()
    response = response_system.solve_tangent_direct_implicit(
        position_direction,
        damping=cache.args.analytic_response_damping,
        create_graph=create_graph,
    )
    _synchronize(response.density_response)
    solve_seconds = time.perf_counter() - solve_started
    if not response.krylov.converged:
        raise RuntimeError("analytic density-response solve produced non-finite values")
    response_limit = cache.args.analytic_response_residual_tolerance
    if (
        response.stationarity_direction_residual >= response_limit
        or response.constraint_direction_residual >= response_limit
    ):
        raise RuntimeError(
            "analytic density-response residual exceeds fail-closed threshold: "
            f"stationarity={response.stationarity_direction_residual:.3e} "
            f"constraint={response.constraint_direction_residual:.3e} "
            f"threshold={response_limit:.3e}"
        )
    return (
        center,
        position_direction,
        response_system,
        response,
        {
            "response_method": response.krylov.method,
            "response_iterations": response.krylov.iterations,
            "response_stationarity_residual": (
                response.stationarity_direction_residual
            ),
            "response_constraint_residual": (
                response.constraint_direction_residual
            ),
            "center_graph_seconds": graph_seconds,
            "kkt_solve_seconds": solve_seconds,
        },
    )


def _analytic_relaxed_direction_prediction(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    direction: Direction,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float | int | str | bool]]:
    """Evaluate one complete-total relaxed HVP from a single center density."""
    (
        center,
        position_direction,
        response_system,
        response,
        diagnostics,
    ) = _analytic_center_response(
        context,
        cache,
        molecule,
        direction,
        create_graph=create_graph,
    )
    hvp_started = time.perf_counter()
    partial_hvp = response_system.partial_position_hvp(
        position_direction, create_graph=create_graph
    )
    response_correction = response_system.response_correction(
        response.density_response,
        response.multiplier_response,
        create_graph=create_graph,
    )
    hvp = partial_hvp + response_correction
    _synchronize(hvp)
    hvp_seconds = time.perf_counter() - hvp_started
    if not bool(torch.isfinite(hvp).all()):
        raise RuntimeError("analytic complete-total relaxed HVP is non-finite")
    diagnostics["relaxed_hvp_seconds"] = hvp_seconds
    diagnostics["density_response_norm"] = float(
        torch.linalg.vector_norm(response.density_response.detach()).cpu()
    )
    diagnostics.update(
        _response_cancellation_diagnostics(partial_hvp, response_correction)
    )
    return (
        hvp,
        [center.projected_density_gradient_norm],
        diagnostics,
    )


def _direction_prediction(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    direction: Direction,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    plus = _evaluate_point_graph(
        context,
        cache,
        molecule,
        molecule.displaced[(direction.index, "plus")],
        create_graph=create_graph,
    )
    minus = _evaluate_point_graph(
        context,
        cache,
        molecule,
        molecule.displaced[(direction.index, "minus")],
        create_graph=create_graph,
    )
    hvp = central_force_secant_hvp(
        plus.force, minus.force, cache.args.displacement
    )
    return hvp, [plus.projected_density_gradient_norm, minus.projected_density_gradient_norm]


def _direction_prediction_with_energies(
    context: Any,
    cache: IntegralBundleCache,
    molecule: MoleculeState,
    direction: Direction,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    plus = _evaluate_point_graph(
        context,
        cache,
        molecule,
        molecule.displaced[(direction.index, "plus")],
        create_graph=create_graph,
    )
    minus = _evaluate_point_graph(
        context,
        cache,
        molecule,
        molecule.displaced[(direction.index, "minus")],
        create_graph=create_graph,
    )
    hvp = central_force_secant_hvp(
        plus.force, minus.force, cache.args.displacement
    )
    return (
        hvp,
        [plus.projected_density_gradient_norm, minus.projected_density_gradient_norm],
        plus.energies.total_energy,
        minus.energies.total_energy,
    )


def _full_hessian_metrics(
    context: Any,
    cache: IntegralBundleCache,
    molecules: list[MoleculeState],
    step: int,
) -> list[dict[str, Any]]:
    context.model.eval()
    rows = []
    for molecule in molecules:
        base = _evaluate_point_graph(
            context, cache, molecule, molecule.base, create_graph=False
        )
        pbe_force = torch.as_tensor(molecule.pbe_force, dtype=base.force.dtype)
        force_difference = base.force.detach().cpu() - pbe_force
        columns = []
        energy_curvatures = []
        force_curvatures = []
        reference_curvatures = []
        direction_relative_errors = []
        direction_roles = []
        euler_norms = [float(base.projected_density_gradient_norm.detach().cpu())]
        for direction in molecule.directions:
            hvp, norms, plus_energy, minus_energy = _direction_prediction_with_energies(
                context, cache, molecule, direction, create_graph=False
            )
            columns.append(hvp.detach().cpu())
            target_hvp = torch.as_tensor(direction.target_hvp, dtype=hvp.dtype)
            direction_error = torch.linalg.vector_norm(
                hvp - target_hvp
            ) / torch.linalg.vector_norm(target_hvp).clamp_min(
                1.0e-2 * math.sqrt(float(target_hvp.numel()))
            )
            direction_relative_errors.append(
                float(direction_error.detach().cpu())
            )
            direction_roles.append(direction.role)
            direction_tensor = torch.as_tensor(direction.vector, dtype=hvp.dtype)
            force_curvatures.append(
                float(torch.sum(direction_tensor * hvp).detach().cpu())
            )
            energy_curvatures.append(
                float(
                    central_energy_directional_curvature(
                        base.energies.total_energy,
                        plus_energy,
                        minus_energy,
                        cache.args.displacement,
                    )
                    .detach()
                    .cpu()
                )
            )
            reference_curvatures.append(
                float(
                    torch.sum(
                        direction_tensor
                        * torch.as_tensor(direction.target_hvp, dtype=hvp.dtype)
                    )
                )
            )
            euler_norms.extend(float(norm.detach().cpu()) for norm in norms)
        coordinate_count = molecule.positions_bohr.size
        direction_matrix = torch.as_tensor(
            np.stack([item.vector.reshape(-1) for item in molecule.directions]),
            dtype=columns[0].dtype,
        )
        column_matrix = torch.stack(
            [column.reshape(-1) for column in columns], dim=1
        )
        cartesian_complete = (
            direction_matrix.shape == (coordinate_count, coordinate_count)
            and bool(
                torch.allclose(
                    direction_matrix,
                    torch.eye(coordinate_count, dtype=direction_matrix.dtype),
                    atol=1.0e-12,
                    rtol=0.0,
                )
            )
        )
        internal_complete = (
            molecule.direction_basis_definition
            == "structured_internal_orthonormal"
            and molecule.loaded_direction_role == "all"
            and molecule.expected_internal_dimension is not None
            and len(columns) == molecule.expected_internal_dimension
        )
        full_hessian_complete = cartesian_complete or internal_complete
        if cartesian_complete:
            hessian = assemble_hessian_columns(columns)
            reference = torch.as_tensor(molecule.pbe_hessian, dtype=hessian.dtype)
            hessian_definition = "cartesian_force_difference"
        elif internal_complete:
            projector = direction_matrix.T @ direction_matrix
            raw_hessian = column_matrix @ direction_matrix
            hessian = projector @ raw_hessian @ projector
            pbe_hessian = torch.as_tensor(
                molecule.pbe_hessian, dtype=hessian.dtype
            )
            reference = projector @ pbe_hessian @ projector
            hessian_definition = (
                "internal_projected_force_difference_from_complete_orthonormal_basis"
            )
        else:
            hessian = column_matrix
            reference = torch.stack(
                [
                    torch.as_tensor(item.target_hvp, dtype=hessian.dtype).reshape(-1)
                    for item in molecule.directions
                ],
                dim=1,
            )
            hessian_definition = (
                f"directional_columns_{molecule.loaded_direction_role}"
            )
        if full_hessian_complete:
            metrics = _to_float_metrics(hessian_error_metrics(hessian, reference))
        else:
            difference = hessian - reference
            reference_norm = torch.linalg.matrix_norm(reference).clamp_min(
                torch.finfo(hessian.dtype).tiny
            )
            metrics = {
                "mae": float(torch.mean(torch.abs(difference))),
                "rmse": float(torch.sqrt(torch.mean(difference * difference))),
                "relative_frobenius": float(
                    torch.linalg.matrix_norm(difference) / reference_norm
                ),
                "symmetric_mae": math.nan,
                "symmetric_rmse": math.nan,
                "symmetric_relative_frobenius": math.nan,
                "antisymmetric_over_symmetric_frobenius": math.nan,
                "symmetry_max_abs": math.nan,
            }
        role_metrics = {}
        for role in ("train", "heldout"):
            values = np.asarray(
                [
                    value
                    for value, direction_role in zip(
                        direction_relative_errors, direction_roles, strict=True
                    )
                    if direction_role == role
                ],
                dtype=np.float64,
            )
            role_metrics[f"{role}_direction_count"] = int(values.size)
            role_metrics[f"{role}_hvp_relative_median"] = (
                float(np.median(values)) if values.size else math.nan
            )
            role_metrics[f"{role}_hvp_relative_p90"] = (
                float(np.quantile(values, 0.9)) if values.size else math.nan
            )
            role_metrics[f"{role}_hvp_relative_max"] = (
                float(np.max(values)) if values.size else math.nan
            )
        rows.append(
            {
                "step": step,
                "molecule_id": molecule.molecule_id,
                "natoms": int(molecule.atomic_numbers.size),
                "full_hessian_complete": full_hessian_complete,
                "hessian_definition": hessian_definition,
                "direction_basis_definition": molecule.direction_basis_definition,
                "loaded_direction_role": molecule.loaded_direction_role,
                "direction_count": len(columns),
                "expected_internal_dimension": molecule.expected_internal_dimension,
                "total_energy_hartree": float(base.energies.total_energy.detach().cpu()),
                "pbe_total_energy_hartree": molecule.pbe_total_energy,
                "total_energy_abs_error_hartree": abs(
                    float(base.energies.total_energy.detach().cpu())
                    - molecule.pbe_total_energy
                ),
                "complete_total_force_mae_hartree_per_bohr": float(
                    torch.mean(torch.abs(force_difference))
                ),
                "complete_total_force_rmse_hartree_per_bohr": float(
                    torch.sqrt(torch.mean(force_difference * force_difference))
                ),
                "base_density_gradient_norm": euler_norms[0],
                "max_cached_density_gradient_norm": max(euler_norms),
                "energy_curvature_mae": float(
                    np.mean(
                        np.abs(
                            np.asarray(energy_curvatures)
                            - np.asarray(reference_curvatures)
                        )
                    )
                ),
                "force_curvature_mae": float(
                    np.mean(
                        np.abs(
                            np.asarray(force_curvatures)
                            - np.asarray(reference_curvatures)
                        )
                    )
                ),
                "energy_vs_force_curvature_mae": float(
                    np.mean(
                        np.abs(
                            np.asarray(energy_curvatures)
                            - np.asarray(force_curvatures)
                        )
                    )
                ),
                **role_metrics,
                **metrics,
            }
        )
        array_dir = cache.args.output_dir / "hessian_arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            array_dir / f"step_{step:07d}_{molecule.molecule_id}.npz",
            predicted_hessian=hessian.numpy(),
            comparison_reference_hessian=reference.numpy(),
            pbe_hessian=molecule.pbe_hessian,
            direction_matrix=direction_matrix.numpy(),
            direction_roles=np.asarray(direction_roles),
            direction_relative_errors=np.asarray(direction_relative_errors),
            predicted_base_force=base.force.detach().cpu().numpy(),
            pbe_base_force=molecule.pbe_force,
        )
    return rows


def _analytic_full_hessian_metrics(
    context: Any,
    cache: IntegralBundleCache,
    molecules: list[MoleculeState],
    step: int,
) -> list[dict[str, Any]]:
    """Evaluate every internal basis direction with one strict center density."""
    context.model.eval()
    rows = []
    for molecule in molecules:
        if (
            molecule.direction_basis_definition
            != "structured_internal_orthonormal"
            or molecule.loaded_direction_role != "all"
            or molecule.expected_internal_dimension is None
            or len(molecule.directions) != molecule.expected_internal_dimension
        ):
            raise ValueError(
                "analytic full-Hessian evaluation requires the complete internal basis"
            )
        started = time.perf_counter()
        base = _evaluate_point_graph(
            context,
            cache,
            molecule,
            molecule.base,
            create_graph=False,
        )
        force_difference = (
            base.force.detach().cpu()
            - torch.as_tensor(molecule.pbe_force, dtype=base.force.dtype)
        )
        columns = []
        direction_relative_errors = []
        response_seconds = 0.0
        integral_and_graph_seconds = 0.0
        maximum_response_residual = 0.0
        for direction in molecule.directions:
            hvp, _, diagnostics = _analytic_relaxed_direction_prediction(
                context,
                cache,
                molecule,
                direction,
                create_graph=False,
            )
            columns.append(hvp.detach().cpu().reshape(-1))
            target = torch.as_tensor(
                direction.target_hvp, dtype=hvp.dtype, device=hvp.device
            )
            relative = torch.linalg.vector_norm(
                hvp - target
            ) / torch.linalg.vector_norm(target).clamp_min(
                1.0e-2 * math.sqrt(float(target.numel()))
            )
            direction_relative_errors.append(float(relative.detach().cpu()))
            response_seconds += float(diagnostics["kkt_solve_seconds"])
            integral_and_graph_seconds += float(
                diagnostics["center_graph_seconds"]
            )
            maximum_response_residual = max(
                maximum_response_residual,
                float(diagnostics["response_stationarity_residual"]),
                float(diagnostics["response_constraint_residual"]),
            )

        direction_matrix = torch.as_tensor(
            np.stack(
                [item.vector.reshape(-1) for item in molecule.directions]
            ),
            dtype=columns[0].dtype,
        )
        projector = direction_matrix.T @ direction_matrix
        column_matrix = torch.stack(columns, dim=1)
        hessian = projector @ (column_matrix @ direction_matrix) @ projector
        pbe_hessian = torch.as_tensor(
            molecule.pbe_hessian, dtype=hessian.dtype
        )
        reference = projector @ pbe_hessian @ projector
        metrics = _to_float_metrics(hessian_error_metrics(hessian, reference))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "step": step,
                "molecule_id": molecule.molecule_id,
                "natoms": int(molecule.atomic_numbers.size),
                "full_hessian_complete": True,
                "hessian_definition": (
                    "analytic_complete_total_relaxed_internal_projected_"
                    "directional_second_integrals"
                ),
                "direction_basis_definition": molecule.direction_basis_definition,
                "loaded_direction_role": molecule.loaded_direction_role,
                "direction_count": len(columns),
                "expected_internal_dimension": molecule.expected_internal_dimension,
                "total_energy_hartree": float(
                    base.energies.total_energy.detach().cpu()
                ),
                "pbe_total_energy_hartree": molecule.pbe_total_energy,
                "total_energy_abs_error_hartree": abs(
                    float(base.energies.total_energy.detach().cpu())
                    - molecule.pbe_total_energy
                ),
                "complete_total_force_mae_hartree_per_bohr": float(
                    torch.mean(torch.abs(force_difference))
                ),
                "complete_total_force_rmse_hartree_per_bohr": float(
                    torch.sqrt(torch.mean(force_difference.square()))
                ),
                "base_density_gradient_norm": float(
                    base.projected_density_gradient_norm.detach().cpu()
                ),
                "max_cached_density_gradient_norm": float(
                    base.projected_density_gradient_norm.detach().cpu()
                ),
                "energy_curvature_mae": math.nan,
                "force_curvature_mae": math.nan,
                "energy_vs_force_curvature_mae": math.nan,
                "train_direction_count": len(columns),
                "train_hvp_relative_median": float(
                    np.median(direction_relative_errors)
                ),
                "train_hvp_relative_p90": float(
                    np.quantile(direction_relative_errors, 0.9)
                ),
                "train_hvp_relative_max": float(
                    np.max(direction_relative_errors)
                ),
                "heldout_direction_count": 0,
                "heldout_hvp_relative_median": math.nan,
                "heldout_hvp_relative_p90": math.nan,
                "heldout_hvp_relative_max": math.nan,
                "wall_seconds": elapsed,
                "integral_and_center_graph_seconds": integral_and_graph_seconds,
                "kkt_solve_seconds": response_seconds,
                "maximum_response_residual": maximum_response_residual,
                **metrics,
            }
        )
        array_dir = cache.args.output_dir / "hessian_arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            array_dir / f"step_{step:07d}_{molecule.molecule_id}.npz",
            predicted_hessian=hessian.numpy(),
            comparison_reference_hessian=reference.numpy(),
            pbe_hessian=molecule.pbe_hessian,
            direction_matrix=direction_matrix.numpy(),
            direction_roles=np.asarray(
                [item.role for item in molecule.directions]
            ),
            direction_relative_errors=np.asarray(direction_relative_errors),
            predicted_base_force=base.force.detach().cpu().numpy(),
            pbe_base_force=molecule.pbe_force,
        )
    return rows


def _save_checkpoint(
    source_checkpoint: Path,
    context: Any,
    optimizer: torch.optim.Optimizer,
    hvp_optimizer: torch.optim.Optimizer | None,
    output: Path,
    step: int,
    geometry_residual: ThreeBodyGeometryResidual | None = None,
    *,
    definition: str = (
        "block-coordinate complete-total relaxed-force secant capacity fit"
    ),
    provenance_update: dict[str, Any] | None = None,
) -> None:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    payload["state_dict"] = {
        key: value.detach().cpu() for key, value in context.model.state_dict().items()
    }
    previous_capacity = payload.get("complete_total_capacity")
    provenance = (
        dict(previous_capacity) if isinstance(previous_capacity, dict) else {}
    )
    if provenance_update is not None:
        provenance.update(provenance_update)
    provenance.update(
        {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "hvp_optimizer_state_dict": (
            hvp_optimizer.state_dict() if hvp_optimizer is not None else None
        ),
        "definition": definition,
        }
    )
    payload["complete_total_capacity"] = provenance
    if geometry_residual is not None:
        payload["complete_total_capacity"]["geometry_residual"] = {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in geometry_residual.state_dict().items()
            },
            "center_min_bohr": geometry_residual.center_min_bohr,
            "center_max_bohr": geometry_residual.center_max_bohr,
            "center_count": geometry_residual.center_count,
            "sigma_bohr": geometry_residual.sigma_bohr,
            "angular_order": geometry_residual.angular_order,
            "elements": list(geometry_residual.elements),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def _selected_directions(
    molecule: MoleculeState,
    step: int,
    count: int,
    repeat_steps: int,
    seed: int,
) -> list[Direction]:
    if molecule.direction_basis_definition == "structured_internal_orthonormal":
        if count <= 0:
            raise ValueError("directions-per-step must be positive")
        block = step // max(1, repeat_steps)
        molecule_seed = int(hashlib.sha256(molecule.molecule_id.encode()).hexdigest()[:8], 16)
        generator = np.random.default_rng(seed + molecule_seed + block)
        grouped = {
            kind: [item for item in molecule.directions if item.kind == kind]
            for kind in ("bond", "angle", "torsion", "random_internal")
        }
        for values in grouped.values():
            generator.shuffle(values)
        selected: list[Direction] = []
        selected_indices: set[int] = set()
        for offset in range(max(len(values) for values in grouped.values())):
            for kind in ("bond", "angle", "torsion", "random_internal"):
                values = grouped[kind]
                if offset >= len(values):
                    continue
                candidate = values[(block + offset) % len(values)]
                if candidate.index not in selected_indices:
                    selected.append(candidate)
                    selected_indices.add(candidate.index)
                if len(selected) >= min(count, len(molecule.directions)):
                    return selected
        return selected

    cartesian = [item for item in molecule.directions if item.kind == "cartesian"]
    low_modes = [item for item in molecule.directions if item.kind == "low_mode"]
    block = step // max(1, repeat_steps)
    generator = np.random.default_rng(seed + block)
    generator.shuffle(cartesian)
    generator.shuffle(low_modes)
    selected = []
    if cartesian:
        selected.append(cartesian[block % len(cartesian)])
    if low_modes and len(selected) < count:
        selected.append(low_modes[block % len(low_modes)])
    pool = cartesian + low_modes
    cursor = 0
    selected_indices = {candidate.index for candidate in selected}
    while pool and len(selected) < count:
        candidate = pool[(block + cursor) % len(pool)]
        if candidate.index not in selected_indices:
            selected.append(candidate)
            selected_indices.add(candidate.index)
        cursor += 1
    return selected


def _hutchinson_direction(
    molecule: MoleculeState,
    *,
    step: int,
    seed: int,
) -> Direction:
    """Sample one fresh unnormalized Rademacher probe in the full internal basis."""
    if molecule.direction_basis_definition != "structured_internal_orthonormal":
        raise ValueError(
            "internal Hutchinson training requires a structured internal basis"
        )
    if molecule.external_basis is None or molecule.expected_internal_dimension is None:
        raise ValueError("internal direction metadata are incomplete")
    ordered = sorted(molecule.directions, key=lambda item: item.index)
    internal_dimension = int(molecule.expected_internal_dimension)
    if len(ordered) != internal_dimension:
        raise ValueError(
            "internal Hutchinson training requires every internal basis direction: "
            f"loaded={len(ordered)} expected={internal_dimension}"
        )
    basis = np.stack(
        [np.asarray(item.vector, dtype=np.float64).reshape(-1) for item in ordered]
    )
    molecule_seed = int(
        hashlib.sha256(molecule.molecule_id.encode()).hexdigest()[:8], 16
    )
    sample = sample_internal_rademacher_direction(
        InternalDirectionBank(
            directions=basis,
            kinds=np.asarray([item.kind for item in ordered]),
            partial_roles=np.asarray([item.role for item in ordered]),
            external_basis=np.asarray(molecule.external_basis, dtype=np.float64),
            projector=basis.T @ basis,
            external_rank=int(np.asarray(molecule.external_basis).shape[1]),
            internal_dimension=internal_dimension,
            orthonormality_max_abs=float(
                np.max(np.abs(basis @ basis.T - np.eye(internal_dimension)))
            ),
            external_overlap_max_abs=float(
                np.max(
                    np.abs(
                        basis @ np.asarray(molecule.external_basis, dtype=np.float64)
                    )
                )
            ),
            projector_idempotence_max_abs=float(
                np.max(np.abs((basis.T @ basis) @ (basis.T @ basis) - basis.T @ basis))
            ),
        ),
        seed=int(seed + molecule_seed + step),
    )
    signs = sample.signs
    target_hvp = np.sum(
        np.stack(
            [
                sign * np.asarray(item.target_hvp, dtype=np.float64)
                for sign, item in zip(signs, ordered)
            ]
        ),
        axis=0,
    )
    return Direction(
        index=-(step + 1),
        kind="hutchinson_internal",
        vector=sample.vector.reshape(molecule.positions_bohr.shape),
        target_hvp=target_hvp,
        role="train",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    symmetric_matrix_power_mode = getattr(
        args, "symmetric_matrix_power_mode", "stable_first_order"
    )
    if (
        args.analytic_relaxed_hvp
        and symmetric_matrix_power_mode != "eigh_second_order_audit"
    ):
        raise ValueError(
            "analytic relaxed-HVP training requires an explicitly "
            "second-order-connected symmetric matrix power"
        )
    os.environ["MLDFT_SYMMETRIC_MATRIX_POWER_MODE"] = (
        symmetric_matrix_power_mode
    )
    if args.hvp_update_period <= 0:
        raise ValueError("--hvp-update-period must be positive")
    if args.replay_update_period is not None and args.replay_update_period <= 1:
        raise ValueError("--replay-update-period must be greater than one")
    if args.loss_balance_mode == "pcgrad" and args.alternating_hvp_updates:
        raise ValueError("PCGrad and alternating HVP updates are mutually exclusive")
    if (
        args.evaluation_mode == "none"
        and args.learning_rate != 0.0
        and not args.two_step_failure_reproduction
    ):
        raise ValueError(
            "--evaluation-mode=none is restricted to zero-learning-rate smokes"
        )
    if args.two_step_failure_reproduction and (
        args.evaluation_mode != "none"
        or args.max_steps > 2
        or args.checkpoint_interval != 1
    ):
        raise ValueError(
            "--two-step-failure-reproduction requires evaluation-mode=none, "
            "max-steps<=2, and checkpoint-interval=1"
        )
    if args.response_correction_fraction_max <= 0:
        raise ValueError("--response-correction-fraction-max must be positive")
    if args.cancellation_index_max < 1:
        raise ValueError("--cancellation-index-max must be at least one")
    if args.analytic_relaxed_hvp and not args.require_complete_total_relaxed_hvp:
        raise ValueError(
            "--analytic-relaxed-hvp requires the fail-closed "
            "--require-complete-total-relaxed-hvp preflight"
        )
    if (
        args.density_parameter_response_predictor
        and not args.analytic_relaxed_hvp
    ):
        raise ValueError(
            "--density-parameter-response-predictor requires "
            "--analytic-relaxed-hvp"
        )
    if args.integral_directional_second_step <= 0:
        raise ValueError("--integral-directional-second-step must be positive")
    if args.analytic_response_damping < 0:
        raise ValueError("--analytic-response-damping must be nonnegative")
    if args.analytic_response_residual_tolerance <= 0:
        raise ValueError(
            "--analytic-response-residual-tolerance must be positive"
        )
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocol = yaml.safe_load(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    protocol_sha256 = _sha256(args.protocol)
    manifest_sha256 = _sha256(args.manifest)
    if (
        protocol.get("method_name")
        != "解析密度/KKT响应的hybrid relaxed-HVP"
        or protocol.get("fully_analytic_libcint_second_integrals") is not False
        or protocol.get("validation_access_allowed") is not False
        or protocol.get("test100_access_allowed") is not False
        or protocol["identity"].get("old_original_a_identity_allowed") is not False
        or protocol["identity"].get(
            "old_checkpoint_or_manifest_hash_reuse_allowed"
        )
        is not False
    ):
        raise ValueError("Hybrid rebuild protocol identity/access boundary failed")
    training_direction = protocol["directions"]["training_direction"]
    if (
        int(training_direction["count_per_step"]) != 1
        or training_direction["distribution"] != "internal_space_rademacher"
        or training_direction["resample_each_step"] is not True
        or training_direction["normalization"]
        != "unnormalized_pm1_in_orthonormal_internal_coordinates"
        or training_direction["estimator"]
        != "unbiased_internal_hessian_frobenius_squared"
    ):
        raise ValueError("Hybrid rebuild Rademacher estimator protocol drift")
    frozen_asset_reuse = protocol.get("frozen_train_only_asset_reuse")
    if frozen_asset_reuse is not None:
        parent_binding_valid = (
            frozen_asset_reuse.get("allowed") is True
            and frozen_asset_reuse.get("parent_manifest_sha256")
            == manifest_sha256
            and frozen_asset_reuse.get("source_protocol_id")
            == manifest.get("protocol_id")
            and frozen_asset_reuse.get("source_protocol_sha256")
            == manifest.get("protocol_sha256")
        )
    else:
        parent_binding_valid = (
            manifest.get("protocol_id") == protocol["protocol_id"]
            and manifest.get("protocol_sha256") == protocol_sha256
        )
    if (
        manifest.get("validation_accessed") is not False
        or manifest.get("test100_accessed") is not False
        or manifest.get("old_original_a_identity_used") is not False
        or not parent_binding_valid
    ):
        raise ValueError(
            "Capacity manifest does not certify the new frozen train-only branch"
        )
    direction_manifest = None
    direction_entries: dict[str, dict[str, Any]] = {}
    direction_manifest_hash = None
    if args.direction_manifest is not None:
        direction_manifest_hash = _sha256(args.direction_manifest)
        if (
            args.direction_manifest_sha256 is not None
            and direction_manifest_hash != args.direction_manifest_sha256
        ):
            raise ValueError(
                "direction manifest hash mismatch: "
                f"{direction_manifest_hash} != {args.direction_manifest_sha256}"
            )
        direction_manifest = json.loads(args.direction_manifest.read_text())
        if frozen_asset_reuse is not None:
            direction_binding_valid = (
                frozen_asset_reuse.get("direction_manifest_sha256")
                == direction_manifest_hash
                and frozen_asset_reuse.get("source_protocol_id")
                == direction_manifest.get("protocol_id")
                and frozen_asset_reuse.get("source_protocol_sha256")
                == direction_manifest.get("protocol_sha256")
            )
        else:
            direction_binding_valid = (
                direction_manifest.get("protocol_id")
                == protocol["protocol_id"]
                and direction_manifest.get("protocol_sha256")
                == protocol_sha256
            )
        if (
            direction_manifest.get("validation_accessed") is not False
            or direction_manifest.get("test100_accessed") is not False
            or direction_manifest.get("old_original_a_identity_used") is not False
            or not direction_binding_valid
            or direction_manifest.get("parent_manifest_sha256")
            != manifest_sha256
        ):
            raise ValueError(
                "direction manifest is not bound to the new frozen parent manifest"
            )
        direction_entries = {
            str(row["molecule_id"]): row
            for row in direction_manifest["parents"]
        }
    expected_label_density_replay = (
        protocol.get("training_semantics", {}).get("egf_branch")
        == "fixed_pbe_ks_density_complete_total_replay"
    )
    if bool(args.egf_label_density_replay) != expected_label_density_replay:
        raise ValueError(
            "E/G/F replay branch does not match the frozen protocol: "
            f"cli={args.egf_label_density_replay} "
            f"protocol={expected_label_density_replay}"
        )
    expected_alternating = protocol.get("formal_training", {}).get(
        "alternating_hvp_updates"
    )
    if (
        expected_alternating is not None
        and bool(args.alternating_hvp_updates) != bool(expected_alternating)
    ):
        raise ValueError(
            "Alternating-update mode does not match the frozen protocol: "
            f"cli={args.alternating_hvp_updates} "
            f"protocol={expected_alternating}"
        )
    requested_ids = set(args.molecules.split(",")) if args.molecules else None
    entries = [
        row
        for row in manifest["parents"]
        if requested_ids is None or row["molecule_id"] in requested_ids
    ]
    if not entries:
        raise ValueError("No Stage-1 molecules selected")
    if requested_ids is not None and {row["molecule_id"] for row in entries} != requested_ids:
        raise ValueError("Requested molecule is absent from the frozen Stage-1 manifest")
    if args.direction_limit is not None or direction_manifest is not None:
        low_mode_count = 0
    else:
        low_mode_count = int(protocol["loss"]["low_mode"]["mode_count"])
    if direction_manifest is not None:
        missing_directions = sorted(
            str(row["molecule_id"])
            for row in entries
            if str(row["molecule_id"]) not in direction_entries
        )
        if missing_directions:
            raise ValueError(
                f"direction manifest is missing parents {missing_directions}"
            )
    molecules = [
        _load_molecule(
            row,
            low_mode_count,
            (
                direction_entries[str(row["molecule_id"])]
                if direction_manifest is not None
                else None
            ),
            args.direction_role,
        )
        for row in entries
    ]
    if args.direction_limit is not None:
        for molecule in molecules:
            molecule.directions = molecule.directions[: args.direction_limit]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    density_args = _density_namespace(args)
    run_spec = _parse_run(args.run)
    source_checkpoint_sha256 = _sha256(run_spec.ckpt)
    if (
        args.source_checkpoint_sha256 is not None
        and source_checkpoint_sha256 != args.source_checkpoint_sha256
    ):
        raise ValueError(
            "source checkpoint hash mismatch: "
            f"{source_checkpoint_sha256} != {args.source_checkpoint_sha256}"
        )
    context = _load_context(run_spec, density_args, device)
    context.model.to(torch.float64)
    source_payload = torch.load(run_spec.ckpt, map_location="cpu", weights_only=False)
    capacity_state = source_payload.get("complete_total_capacity")
    if capacity_state is None:
        if run_spec.name != protocol["identity"]["baseline_name"]:
            raise ValueError("Source run name is not the registered rebuild baseline")
        if int(source_payload.get("global_step", -1)) != int(
            protocol["baseline"]["max_steps"]
        ):
            raise ValueError("Source checkpoint is not the fixed-step rebuild baseline")
    loss_config = protocol["loss"]
    weights = dict(loss_config["initial_weights"])
    weights.setdefault("lambda_Q", 0.0)
    for key, override in (
        ("lambda_E", args.lambda_e),
        ("lambda_F", args.lambda_f),
        ("lambda_rho", args.lambda_rho),
        ("lambda_H", args.lambda_h),
        ("lambda_Q", args.lambda_q),
        ("lambda_spec", args.lambda_spec),
    ):
        if override is not None:
            weights[key] = float(override)
    if args.require_complete_total_relaxed_hvp:
        failures = []
        training_density_threshold = _training_density_stationarity_threshold(args)
        if direction_manifest is None:
            failures.append("a frozen structured direction manifest is required")
        if not args.implicit_density_parameter_response:
            failures.append("implicit density parameter response is required")
        if not args.strict_active_density_refresh:
            failures.append(
                "strict active-density refresh before every parameter-step graph "
                "is required"
            )
        if args.density_response_unroll_steps != 0:
            failures.append("density-response unrolling is forbidden")
        if args.density_strict_threshold > 1.0e-8:
            failures.append("density strict threshold must be <=1e-8")
        if training_density_threshold > 1.0e-8:
            failures.append(
                "training density stationarity threshold must be <=1e-8"
            )
        if training_density_threshold < args.density_strict_threshold:
            failures.append(
                "training density stationarity threshold must be >= the "
                "density solver target"
            )
        if args.direction_limit is not None:
            failures.append("direction truncation is forbidden")
        if args.analytic_relaxed_hvp:
            if args.directions_per_step != 1:
                failures.append(
                    "analytic Hutchinson training requires directions-per-step=1"
                )
            if args.direction_role != "all":
                failures.append(
                    "analytic Hutchinson training requires the complete internal basis"
                )
            if float(weights["lambda_Q"]) != 0.0:
                failures.append(
                    "finite-difference energy curvature is forbidden in analytic mode"
                )
            if float(weights["lambda_spec"]) != 0.0:
                failures.append(
                    "single-probe analytic mode cannot enable low-mode spectrum loss"
                )
        if capacity_state is not None:
            if not args.resume_capacity_optimizer:
                failures.append(
                    "capacity checkpoint requires --resume-capacity-optimizer"
                )
            elif not isinstance(capacity_state, dict):
                failures.append("capacity checkpoint metadata is malformed")
            else:
                if (
                    capacity_state.get("root_source_checkpoint_sha256")
                    != args.root_source_checkpoint_sha256
                ):
                    failures.append("root rebuild-baseline checkpoint provenance mismatch")
                if capacity_state.get("protocol_id") != protocol["protocol_id"]:
                    failures.append("capacity checkpoint protocol mismatch")
                if (
                    capacity_state.get("direction_manifest_sha256")
                    != direction_manifest_hash
                ):
                    failures.append("capacity checkpoint direction-manifest mismatch")
                if (
                    capacity_state.get("molecules")
                    != [molecule.molecule_id for molecule in molecules]
                ):
                    failures.append("capacity checkpoint molecule set/order mismatch")
                if capacity_state.get("direction_role") != args.direction_role:
                    failures.append("capacity checkpoint direction role mismatch")
                if (
                    bool(capacity_state.get("analytic_relaxed_hvp", False))
                    != args.analytic_relaxed_hvp
                ):
                    failures.append("capacity checkpoint analytic-HVP mode mismatch")
                if capacity_state.get("strict_active_density_refresh") is not True:
                    failures.append(
                        "capacity checkpoint lacks strict active-density provenance"
                    )
                if (
                    capacity_state.get("implicit_density_parameter_response")
                    is not True
                ):
                    failures.append(
                        "capacity checkpoint lacks implicit-response provenance"
                    )
                if capacity_state.get("proxy_hvp_fallback_allowed") is not False:
                    failures.append("capacity checkpoint permits proxy HVP fallback")
                if (
                    capacity_state.get("validation_accessed") is not False
                    or capacity_state.get("test100_accessed") is not False
                ):
                    failures.append(
                        "capacity checkpoint lacks frozen validation/Test100 provenance"
                    )
                if not capacity_state.get("root_initial_full_hessian_metrics"):
                    failures.append(
                        "capacity checkpoint lacks rebuild-baseline E/F gate baseline"
                    )
                if capacity_state.get("loss_weights") != weights:
                    failures.append("capacity checkpoint loss-weight mismatch")
                if (
                    bool(capacity_state.get("egf_label_density_replay", False))
                    != args.egf_label_density_replay
                ):
                    failures.append("capacity checkpoint E/G/F replay-branch mismatch")
        elif args.resume_capacity_optimizer:
            failures.append(
                "--resume-capacity-optimizer requires a bound capacity checkpoint"
            )
        elif (
            args.root_source_checkpoint_sha256 is not None
            and args.root_source_checkpoint_sha256 != source_checkpoint_sha256
        ):
            failures.append("root source hash does not match untouched checkpoint")
        if args.skip_resume_initial_full_hessian:
            if not isinstance(capacity_state, dict) or not args.resume_capacity_optimizer:
                failures.append(
                    "skipping the resume initial Hessian requires a bound "
                    "capacity checkpoint and optimizer resume"
                )
            elif not capacity_state.get("root_initial_full_hessian_metrics"):
                failures.append(
                    "skipping the resume initial Hessian requires frozen root metrics"
                )
            if args.evaluation_mode != "full":
                failures.append(
                    "skipping the resume initial Hessian requires final full evaluation"
                )
        if args.three_body_geometry_residual or args.freeze_base_model:
            failures.append("the Graphformer-only branch forbids auxiliary residual heads")
        if context.model.net.__class__.__name__ != "Graphformer":
            failures.append(
                f"expected Graphformer, got {context.model.net.__class__.__name__}"
            )
        if failures:
            raise ValueError(
                "complete-total relaxed HVP preflight failed: "
                + "; ".join(failures)
            )
    source_capacity_step = (
        int(capacity_state["step"]) if isinstance(capacity_state, dict) else 0
    )
    root_source_checkpoint_sha256 = (
        str(capacity_state["root_source_checkpoint_sha256"])
        if isinstance(capacity_state, dict)
        and capacity_state.get("root_source_checkpoint_sha256") is not None
        else source_checkpoint_sha256
    )
    geometry_residual = None
    if args.three_body_geometry_residual:
        geometry_residual = ThreeBodyGeometryResidual(
            center_min_bohr=args.three_body_center_min,
            center_max_bohr=args.three_body_center_max,
            center_count=args.three_body_center_count,
            sigma_bohr=args.three_body_sigma,
            angular_order=args.three_body_angular_order,
        ).to(device=device, dtype=torch.float64)
        if args.three_body_initial_coefficients is not None:
            with np.load(args.three_body_initial_coefficients) as payload:
                geometry_residual.load_sparse_coefficients(
                    payload["feature_keys"], payload["coefficients"]
                )
        residual_state = (
            capacity_state.get("geometry_residual")
            if isinstance(capacity_state, dict)
            else None
        )
        if args.resume_capacity_optimizer and residual_state is not None:
            geometry_residual.load_state_dict(residual_state["state_dict"])
        context.functional_factory = FunctionalFactory(
            context.model,
            geometry_residual,
            "hartree",
            "nuclear_attraction",
        )

    if args.freeze_base_model:
        if geometry_residual is None:
            raise ValueError("--freeze-base-model requires --three-body-geometry-residual")
        for parameter in context.model.net.parameters():
            parameter.requires_grad_(False)

    named_parameters = [
        (name, parameter)
        for name, parameter in context.model.net.named_parameters()
        if parameter.requires_grad
    ]
    if geometry_residual is not None:
        named_parameters.extend(
            (f"geometry_residual.{name}", parameter)
            for name, parameter in geometry_residual.named_parameters()
            if parameter.requires_grad
        )
    parameters = [parameter for _, parameter in named_parameters]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    hvp_optimizer = None
    if args.alternating_hvp_updates:
        hvp_optimizer = torch.optim.AdamW(
            parameters,
            lr=(
                args.hvp_learning_rate
                if args.hvp_learning_rate is not None
                else args.learning_rate
            ),
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
    if args.resume_capacity_optimizer:
        if not isinstance(capacity_state, dict):
            raise ValueError(
                "--resume-capacity-optimizer requires a capacity-fit checkpoint"
            )
        optimizer.load_state_dict(capacity_state["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["betas"] = (args.adam_beta1, args.adam_beta2)
            parameter_group["weight_decay"] = args.weight_decay
        if hvp_optimizer is not None:
            hvp_state = capacity_state.get("hvp_optimizer_state_dict")
            if hvp_state is None:
                raise ValueError(
                    "alternating optimizer resume requires hvp_optimizer_state_dict"
                )
            hvp_optimizer.load_state_dict(hvp_state)
            for parameter_group in hvp_optimizer.param_groups:
                parameter_group["lr"] = (
                    args.hvp_learning_rate
                    if args.hvp_learning_rate is not None
                    else args.learning_rate
                )
                parameter_group["betas"] = (args.adam_beta1, args.adam_beta2)
                parameter_group["weight_decay"] = args.weight_decay
    bundle_cache = IntegralBundleCache(context, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_definition = (
        "complete-total KKT-relaxed HVP internal-Hutchinson fit with "
        "directional numerical PySCF integral response"
        if args.analytic_relaxed_hvp
        else "block-coordinate complete-total relaxed-force secant capacity fit"
    )
    if args.egf_label_density_replay:
        checkpoint_definition += (
            " and fixed-PBE-density complete-total E/G/F replay"
        )

    density_rows = (
        _refresh_base_densities(
            context,
            molecules,
            density_args,
            refresh_index=0,
            parameter_step=source_capacity_step,
        )
        if args.analytic_relaxed_hvp
        else _refresh_densities(
            context, molecules, density_args, refresh_index=0
        )
    )
    _write_csv(args.output_dir / "density_refresh_points.csv", density_rows)
    hessian_rows = []
    if (
        args.evaluation_mode == "full"
        and not args.skip_resume_initial_full_hessian
    ):
        hessian_rows = (
            _analytic_full_hessian_metrics(
                context, bundle_cache, molecules, step=source_capacity_step
            )
            if args.analytic_relaxed_hvp
            else _full_hessian_metrics(
                context, bundle_cache, molecules, step=source_capacity_step
            )
        )
    _write_csv(args.output_dir / "full_hessian_metrics.csv", hessian_rows)
    initial_metrics_for_checkpoint = [
        row for row in hessian_rows if row["step"] == source_capacity_step
    ]
    root_initial_metrics_for_checkpoint = (
        capacity_state["root_initial_full_hessian_metrics"]
        if isinstance(capacity_state, dict)
        and capacity_state.get("root_initial_full_hessian_metrics")
        else initial_metrics_for_checkpoint
    )
    checkpoint_provenance = {
        "root_source_checkpoint_sha256": root_source_checkpoint_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "direction_manifest_sha256": direction_manifest_hash,
        "molecules": [molecule.molecule_id for molecule in molecules],
        "direction_role": args.direction_role,
        "strict_active_density_refresh": args.strict_active_density_refresh,
        "density_solver_target": args.density_strict_threshold,
        "training_density_stationarity_threshold": (
            _training_density_stationarity_threshold(args)
        ),
        "implicit_density_parameter_response": (
            args.implicit_density_parameter_response
        ),
        "analytic_relaxed_hvp": args.analytic_relaxed_hvp,
        "symmetric_matrix_power_mode": symmetric_matrix_power_mode,
        "density_parameter_response_predictor": (
            args.density_parameter_response_predictor
        ),
        "egf_label_density_replay": args.egf_label_density_replay,
        "proxy_hvp_fallback_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "loss_weights": weights,
        "root_initial_full_hessian_metrics": (
            root_initial_metrics_for_checkpoint
        ),
    }
    if args.max_steps == 0:
        _save_checkpoint(
            run_spec.ckpt,
            context,
            optimizer,
            hvp_optimizer,
            args.output_dir
            / "checkpoints"
            / f"step_{source_capacity_step:07d}.ckpt",
            source_capacity_step,
            geometry_residual,
            definition=checkpoint_definition,
            provenance_update=checkpoint_provenance,
        )
    loss_rows = []
    refresh_index = 0
    density_parameter_step = source_capacity_step
    refresh_needed = False
    stopped_early = False

    for step in range(1, args.max_steps + 1):
        step_started = time.perf_counter()
        density_refresh_seconds = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        reported_step = source_capacity_step + step
        current_parameter_step = reported_step - 1
        selected_by_molecule = {
            molecule.molecule_id: (
                [
                    _hutchinson_direction(
                        molecule,
                        step=reported_step + molecule_offset,
                        seed=args.seed,
                    )
                ]
                if args.analytic_relaxed_hvp
                else _selected_directions(
                    molecule,
                    reported_step + molecule_offset,
                    args.directions_per_step,
                    args.direction_repeat_steps,
                    args.seed,
                )
            )
            for molecule_offset, molecule in enumerate(molecules)
        }
        refresh_scope = _density_refresh_scope(
            strict_active=args.strict_active_density_refresh,
            current_parameter_step=current_parameter_step,
            density_parameter_step=density_parameter_step,
            refresh_needed=refresh_needed,
            refresh_interval=args.density_refresh_interval,
        )
        if refresh_scope is not None:
            refresh_started = time.perf_counter()
            refresh_index += 1
            if args.analytic_relaxed_hvp:
                refreshed = _refresh_base_densities(
                    context,
                    molecules,
                    density_args,
                    refresh_index=refresh_index,
                    parameter_step=current_parameter_step,
                )
            elif refresh_scope == "active":
                refreshed = _refresh_active_densities(
                    context,
                    molecules,
                    selected_by_molecule,
                    density_args,
                    refresh_index=refresh_index,
                    parameter_step=current_parameter_step,
                )
            else:
                refreshed = _refresh_densities(
                    context, molecules, density_args, refresh_index=refresh_index
                )
            density_rows.extend(refreshed)
            _write_csv(args.output_dir / "density_refresh_points.csv", density_rows)
            density_parameter_step = current_parameter_step
            refresh_needed = False
            density_refresh_seconds = time.perf_counter() - refresh_started

        # Density refreshes and parameter updates must use the same deterministic model mode.
        # Evaluation mode does not disable parameter gradients.
        context.model.eval()
        optimizer.zero_grad(set_to_none=True)
        if hvp_optimizer is not None:
            hvp_optimizer.zero_grad(set_to_none=True)
        consume_implicit_response_diagnostics()
        component_values: dict[str, list[torch.Tensor]] = {
            "energy": [],
            "force": [],
            "density": [],
            "hvp": [],
            "curvature": [],
            "spectrum": [],
        }
        active_directions = []
        analytic_response_rows: list[dict[str, float | int | str | bool]] = []
        max_density_norm = 0.0
        max_label_density_gradient_norm = 0.0
        response_enabled = (
            args.implicit_density_parameter_response
            or args.density_response_unroll_steps > 0
        )
        needs_envelope_energy = bool(
            float(weights["lambda_E"]) != 0.0
            or float(weights["lambda_Q"]) != 0.0
        )
        for molecule_offset, molecule in enumerate(molecules):
            if args.egf_label_density_replay:
                replay = _evaluate_point_graph(
                    context,
                    bundle_cache,
                    molecule,
                    _label_density_replay_point(molecule),
                    create_graph=True,
                    attach_density_parameter_response=False,
                )
                replay_envelope = replay
                if molecule.base is None:
                    raise RuntimeError("implicit HVP training requires a center density")
                max_density_norm = max(
                    max_density_norm, float(molecule.base.final_gradient_norm)
                )
                max_label_density_gradient_norm = max(
                    max_label_density_gradient_norm,
                    float(replay.projected_density_gradient_norm.detach().cpu()),
                )
            else:
                replay = _evaluate_point_graph(
                    context, bundle_cache, molecule, molecule.base, create_graph=True
                )
                replay_envelope = replay
                if response_enabled and needs_envelope_energy:
                    replay_envelope = _evaluate_point_graph(
                        context,
                        bundle_cache,
                        molecule,
                        molecule.base,
                        create_graph=True,
                        attach_density_parameter_response=False,
                    )
                max_density_norm = max(
                    max_density_norm,
                    float(
                        replay.projected_density_gradient_norm.detach().cpu()
                    ),
                )
            component_values["energy"].append(
                normalized_energy_l1(
                    replay_envelope.energies.total_energy,
                    molecule.pbe_total_energy,
                    absolute_scale_hartree=float(
                        loss_config["energy"]["absolute_scale_hartree"]
                    ),
                )
            )
            component_values["force"].append(
                mixed_absolute_relative_l1(
                    replay.force,
                    torch.as_tensor(
                        molecule.pbe_force, dtype=replay.force.dtype
                    ),
                    absolute_scale=float(
                        loss_config["force"]["absolute_scale_hartree_per_bohr"]
                    ),
                    relative_floor=float(
                        loss_config["force"]["relative_rms_floor_hartree_per_bohr"]
                    ),
                    relative_fraction=float(
                        loss_config["force"]["absolute_relative_mix"]
                    ),
                )
            )
            component_values["density"].append(
                (
                    replay.projected_density_gradient_norm
                    / args.density_loss_scale
                )
                ** 2
            )
            selected = selected_by_molecule[molecule.molecule_id]
            for direction in selected:
                if args.analytic_relaxed_hvp:
                    hvp, density_norms, analytic_diagnostics = (
                        _analytic_relaxed_direction_prediction(
                            context,
                            bundle_cache,
                            molecule,
                            direction,
                            create_graph=True,
                        )
                    )
                    analytic_response_rows.append(analytic_diagnostics)
                    plus_energy = None
                    minus_energy = None
                else:
                    hvp, density_norms, plus_energy, minus_energy = (
                        _direction_prediction_with_energies(
                            context,
                            bundle_cache,
                            molecule,
                            direction,
                            create_graph=True,
                        )
                    )
                if (
                    not args.analytic_relaxed_hvp
                    and response_enabled
                    and float(weights["lambda_Q"]) != 0.0
                ):
                    plus_energy = _evaluate_point_graph(
                        context,
                        bundle_cache,
                        molecule,
                        molecule.displaced[(direction.index, "plus")],
                        create_graph=True,
                        attach_density_parameter_response=False,
                    ).energies.total_energy
                    minus_energy = _evaluate_point_graph(
                        context,
                        bundle_cache,
                        molecule,
                        molecule.displaced[(direction.index, "minus")],
                        create_graph=True,
                        attach_density_parameter_response=False,
                    ).energies.total_energy
                if not args.analytic_relaxed_hvp:
                    assert plus_energy is not None and minus_energy is not None
                    predicted_curvature = central_energy_directional_curvature(
                        base_envelope.energies.total_energy,
                        plus_energy,
                        minus_energy,
                        args.displacement,
                    )
                    target_curvature = torch.sum(
                        torch.as_tensor(direction.vector, dtype=hvp.dtype)
                        * torch.as_tensor(direction.target_hvp, dtype=hvp.dtype)
                    )
                    component_values["curvature"].append(
                        mixed_absolute_relative_l1(
                            predicted_curvature.reshape(1),
                            target_curvature.reshape(1),
                            absolute_scale=float(
                                loss_config["hvp"][
                                    "absolute_scale_hartree_per_bohr2"
                                ]
                            ),
                            relative_floor=float(
                                loss_config["hvp"][
                                    "relative_rms_floor_hartree_per_bohr2"
                                ]
                            ),
                            relative_fraction=float(
                                loss_config["hvp"]["absolute_relative_mix"]
                            ),
                        )
                    )
                target_hvp = torch.as_tensor(
                    direction.target_hvp, dtype=hvp.dtype, device=hvp.device
                )
                if args.analytic_relaxed_hvp:
                    internal_dimension = int(
                        molecule.expected_internal_dimension
                    )
                    absolute_scale = float(
                        loss_config["hvp"][
                            "absolute_scale_hartree_per_bohr2"
                        ]
                    )
                    component_values["hvp"].append(
                        hutchinson_internal_frobenius_squared_loss(
                            hvp,
                            target_hvp,
                            internal_dimension=internal_dimension,
                            reduction="mean_internal_matrix",
                        )
                        / absolute_scale**2
                    )
                else:
                    hvp_loss_function = (
                        mixed_absolute_relative_rmse
                        if args.hvp_loss_norm == "l2"
                        else mixed_absolute_relative_l1
                    )
                    component_values["hvp"].append(
                        hvp_loss_function(
                            hvp,
                            target_hvp,
                            absolute_scale=float(
                                loss_config["hvp"][
                                    "absolute_scale_hartree_per_bohr2"
                                ]
                            ),
                            relative_floor=float(
                                loss_config["hvp"][
                                    "relative_rms_floor_hartree_per_bohr2"
                                ]
                            ),
                            relative_fraction=float(
                                loss_config["hvp"]["absolute_relative_mix"]
                            ),
                        )
                    )
                component_values["density"].extend(
                    (norm / args.density_loss_scale) ** 2 for norm in density_norms
                )
                if direction.kind == "low_mode":
                    component_values["spectrum"].append(
                        low_mode_curvature_loss(
                            hvp,
                            target_hvp,
                            torch.as_tensor(direction.vector, dtype=hvp.dtype),
                            curvature_floor=float(
                                loss_config["hvp"][
                                    "relative_rms_floor_hartree_per_bohr2"
                                ]
                            ),
                            wrong_curvature_multiplier=float(
                                loss_config["low_mode"]["wrong_curvature_multiplier"]
                            ),
                        )
                    )
                max_density_norm = max(
                    max_density_norm,
                    *(float(norm.detach().cpu()) for norm in density_norms),
                )
                active_directions.append(
                    f"{molecule.molecule_id}:{direction.kind}:{direction.index}"
                )

        maximum_response_correction_fraction = max(
            (
                float(item["response_correction_fraction_of_relaxed_norm"])
                for item in analytic_response_rows
            ),
            default=0.0,
        )
        maximum_cancellation_index = max(
            (
                float(item["cancellation_index"])
                for item in analytic_response_rows
            ),
            default=0.0,
        )
        if (
            maximum_response_correction_fraction
            > args.response_correction_fraction_max
        ):
            raise RuntimeError(
                "Density-response correction fraction exceeds fail-closed "
                f"threshold: {maximum_response_correction_fraction:.6g} > "
                f"{args.response_correction_fraction_max:.6g}"
            )
        if maximum_cancellation_index > args.cancellation_index_max:
            raise RuntimeError(
                "Partial/response cancellation index exceeds fail-closed "
                f"threshold: {maximum_cancellation_index:.6g} > "
                f"{args.cancellation_index_max:.6g}"
            )
        _assert_training_density_stationarity(
            max_density_norm,
            _training_density_stationarity_threshold(args),
            enabled=args.strict_active_density_refresh,
        )
        components = {}
        loss_device = parameters[0].device
        for name, values in component_values.items():
            if values:
                components[name] = torch.stack(
                    [value.reshape(()).to(loss_device) for value in values]
                ).mean()
            else:
                components[name] = (
                    next(iter(component_values["energy"])) * 0.0
                ).to(loss_device)
        weighted = {
            "energy": float(weights["lambda_E"]) * components["energy"],
            "force": float(weights["lambda_F"]) * components["force"],
            "density": float(weights["lambda_rho"]) * components["density"],
            "hvp": float(weights["lambda_H"]) * components["hvp"],
            "curvature": float(weights["lambda_Q"]) * components["curvature"],
            "spectrum": float(weights["lambda_spec"]) * components["spectrum"],
        }
        update_kind = "joint"
        backward_terms = weighted
        if hvp_optimizer is not None:
            update_kind = alternating_update_kind(
                reported_step,
                hvp_update_period=args.hvp_update_period,
                replay_update_period=args.replay_update_period,
            )
            if update_kind == "hvp":
                backward_terms = {"hvp": weighted["hvp"]}
            else:
                backward_terms = {
                    name: value for name, value in weighted.items() if name != "hvp"
                }
        total_loss = torch.stack(list(backward_terms.values())).sum()
        diagnostics = {}
        if args.gradient_diagnostics_interval > 0 and (
            step == 1 or step % args.gradient_diagnostics_interval == 0
        ):
            diagnostic_terms = {
                name: value
                for name, value in weighted.items()
                if float(weights[
                    {
                        "energy": "lambda_E",
                        "force": "lambda_F",
                        "density": "lambda_rho",
                        "hvp": "lambda_H",
                        "curvature": "lambda_Q",
                        "spectrum": "lambda_spec",
                    }[name]
                ])
                != 0.0
            }
            diagnostics = parameter_gradient_diagnostics(
                diagnostic_terms,
                parameters,
                parameter_names=[name for name, _ in named_parameters],
            )
        balance_diagnostics = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_started = time.perf_counter()
        if args.loss_balance_mode == "pcgrad":
            replay_loss = torch.stack(
                [weighted[name] for name in ("energy", "force", "density")]
            ).sum()
            curvature_loss = torch.stack(
                [weighted[name] for name in ("hvp", "curvature", "spectrum")]
            ).sum()
            balance_diagnostics = assign_two_task_pcgrad(
                replay_loss,
                curvature_loss,
                parameters,
            )
        else:
            total_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        implicit_diagnostics = consume_implicit_response_diagnostics()
        iterative_implicit_diagnostics = [
            item
            for item in implicit_diagnostics
            if int(item["iterations"]) > 0
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
        old_parameter_values = (
            [parameter.detach().clone() for parameter in parameters]
            if args.density_parameter_response_predictor
            else []
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_started = time.perf_counter()
        if update_kind == "hvp":
            assert hvp_optimizer is not None
            hvp_optimizer.step()
        else:
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        density_predictor_seconds = 0.0
        density_predictor_correction_norms: list[float] = []
        if args.density_parameter_response_predictor:
            predictor_started = time.perf_counter()
            density_predictor_correction_norms = (
                _predict_next_parameter_step_densities(
                    context,
                    molecules,
                    parameters,
                    old_parameter_values,
                    [parameter.detach().clone() for parameter in parameters],
                    charge=args.charge,
                    damping=args.analytic_response_damping,
                )
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            density_predictor_seconds = time.perf_counter() - predictor_started
        row = {
            "step": reported_step,
            "total_loss": float(total_loss.detach().cpu()),
            "parameter_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "max_cached_density_gradient_norm": max_density_norm,
            "max_label_density_projected_gradient_norm": (
                max_label_density_gradient_norm
            ),
            "density_parameter_step": density_parameter_step,
            "strict_active_density_refresh": args.strict_active_density_refresh,
            "active_directions": ";".join(active_directions),
            "update_kind": update_kind,
            "implicit_response_solve_count": len(implicit_diagnostics),
            "implicit_response_total_iterations": sum(
                int(item["iterations"]) for item in implicit_diagnostics
            ),
            "implicit_response_warm_start_count": sum(
                bool(item.get("warm_start_used", False))
                for item in implicit_diagnostics
            ),
            "implicit_response_max_iterations": max(
                (int(item["iterations"]) for item in implicit_diagnostics), default=0
            ),
            "implicit_response_max_absolute_residual": max(
                (
                    float(item["residual_norm"])
                    for item in iterative_implicit_diagnostics
                ),
                default=0.0,
            ),
            "implicit_response_max_scaled_residual": max(
                (
                    float(item["scaled_residual"])
                    for item in iterative_implicit_diagnostics
                ),
                default=0.0,
            ),
            "implicit_response_max_relative_residual": max(
                (
                    float(item["relative_residual"])
                    for item in iterative_implicit_diagnostics
                ),
                default=0.0,
            ),
            "implicit_response_zero_iteration_count": (
                len(implicit_diagnostics) - len(iterative_implicit_diagnostics)
            ),
            "density_refresh_seconds": density_refresh_seconds,
            "analytic_center_graph_seconds": sum(
                float(item["center_graph_seconds"])
                for item in analytic_response_rows
            ),
            "analytic_kkt_solve_seconds": sum(
                float(item["kkt_solve_seconds"])
                for item in analytic_response_rows
            ),
            "analytic_hvp_assembly_seconds": sum(
                float(item["relaxed_hvp_seconds"])
                for item in analytic_response_rows
            ),
            "analytic_response_max_residual": max(
                (
                    max(
                        float(item["response_stationarity_residual"]),
                        float(item["response_constraint_residual"]),
                    )
                    for item in analytic_response_rows
                ),
                default=0.0,
            ),
            "analytic_density_response_norm_mean": (
                float(
                    np.mean(
                        [
                            float(item["density_response_norm"])
                            for item in analytic_response_rows
                        ]
                    )
                )
                if analytic_response_rows
                else 0.0
            ),
            "analytic_response_correction_fraction_max": (
                maximum_response_correction_fraction
            ),
            "analytic_cancellation_index_max": maximum_cancellation_index,
            "analytic_partial_response_cosine_mean": (
                float(
                    np.mean(
                        [
                            float(item["partial_response_cosine"])
                            for item in analytic_response_rows
                        ]
                    )
                )
                if analytic_response_rows
                else 0.0
            ),
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "density_parameter_predictor_seconds": density_predictor_seconds,
            "density_parameter_predictor_mean_correction_norm": (
                float(np.mean(density_predictor_correction_norms))
                if density_predictor_correction_norms
                else 0.0
            ),
            "density_parameter_predictor_max_correction_norm": (
                max(density_predictor_correction_norms)
                if density_predictor_correction_norms
                else 0.0
            ),
            "step_seconds": time.perf_counter() - step_started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024.0**2
                if device.type == "cuda"
                else 0.0
            ),
            **{f"loss/{name}": float(value.detach().cpu()) for name, value in components.items()},
            **diagnostics,
            **balance_diagnostics,
        }
        loss_rows.append(row)
        _write_csv(args.output_dir / "training_curve.csv", loss_rows)

        # Higher-order autograd tensors otherwise survive in loop-local containers
        # until the next iteration (or final full-Hessian evaluation) has already
        # constructed another E_cc graph. Drop them before either operation.
        component_values.clear()
        components.clear()
        weighted.clear()
        backward_terms = {}
        diagnostic_terms = {}
        replay = None
        replay_envelope = None
        base_envelope = None
        hvp = None
        target_hvp = None
        total_loss = None
        old_parameter_values = []
        optimizer.zero_grad(set_to_none=True)
        if hvp_optimizer is not None:
            hvp_optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        evaluate_now = step == args.max_steps or step % args.eval_interval == 0
        checkpoint_now = (
            args.checkpoint_interval > 0
            and reported_step % args.checkpoint_interval == 0
        )
        if checkpoint_now and not evaluate_now:
            _save_checkpoint(
                run_spec.ckpt,
                context,
                optimizer,
                hvp_optimizer,
                args.output_dir
                / "checkpoints"
                / f"step_{reported_step:07d}.ckpt",
                reported_step,
                geometry_residual,
                definition=checkpoint_definition,
                provenance_update=checkpoint_provenance,
            )
        if evaluate_now:
            refresh_index += 1
            refreshed = (
                _refresh_base_densities(
                    context,
                    molecules,
                    density_args,
                    refresh_index=refresh_index,
                    parameter_step=reported_step,
                )
                if args.analytic_relaxed_hvp
                else _refresh_densities(
                    context,
                    molecules,
                    density_args,
                    refresh_index=refresh_index,
                )
            )
            density_rows.extend(refreshed)
            _write_csv(args.output_dir / "density_refresh_points.csv", density_rows)
            density_parameter_step = reported_step
            new_metrics = []
            if args.evaluation_mode == "full":
                new_metrics = (
                    _analytic_full_hessian_metrics(
                        context, bundle_cache, molecules, step=reported_step
                    )
                    if args.analytic_relaxed_hvp
                    else _full_hessian_metrics(
                        context, bundle_cache, molecules, step=reported_step
                    )
                )
            hessian_rows.extend(new_metrics)
            _write_csv(args.output_dir / "full_hessian_metrics.csv", hessian_rows)
            _save_checkpoint(
                run_spec.ckpt,
                context,
                optimizer,
                hvp_optimizer,
                args.output_dir / "checkpoints" / f"step_{reported_step:07d}.ckpt",
                reported_step,
                geometry_residual,
                definition=checkpoint_definition,
                provenance_update=checkpoint_provenance,
            )
            if new_metrics:
                median_relative = float(
                    np.median([row["relative_frobenius"] for row in new_metrics])
                )
                all_below_gate = all(
                    row["relative_frobenius"]
                    <= float(
                        protocol["stage1"]["gate"][
                            "training_all_parent_relative_frobenius_max"
                        ]
                    )
                    for row in new_metrics
                )
                if (
                    median_relative <= args.early_stop_relative_frobenius
                    and all_below_gate
                ):
                    stopped_early = True
                    break
        elif (
            not args.strict_active_density_refresh
            and max_density_norm > args.density_drift_trigger
        ):
            refresh_needed = True

    final_step = loss_rows[-1]["step"] if loss_rows else source_capacity_step
    final_metrics = [row for row in hessian_rows if row["step"] == final_step]
    initial_metrics = [
        row for row in hessian_rows if row["step"] == source_capacity_step
    ]
    root_initial_metrics = (
        capacity_state["root_initial_full_hessian_metrics"]
        if isinstance(capacity_state, dict)
        and capacity_state.get("root_initial_full_hessian_metrics")
        else initial_metrics
    )
    initial_by_molecule = {
        row["molecule_id"]: row for row in root_initial_metrics
    }
    gate_config = protocol["stage1"]["gate"]
    energy_force_regression_limit = float(
        gate_config.get("energy_force_relative_regression_max", math.inf)
    )
    energy_floor = float(gate_config.get("energy_error_floor_hartree", 1.0e-8))
    force_floor = float(
        gate_config.get("force_error_floor_hartree_per_bohr", 1.0e-8)
    )
    energy_force_gate_passed = bool(final_metrics) and all(
        row["total_energy_abs_error_hartree"]
        <= energy_force_regression_limit
        * max(
            initial_by_molecule[row["molecule_id"]][
                "total_energy_abs_error_hartree"
            ],
            energy_floor,
        )
        and row["complete_total_force_mae_hartree_per_bohr"]
        <= energy_force_regression_limit
        * max(
            initial_by_molecule[row["molecule_id"]][
                "complete_total_force_mae_hartree_per_bohr"
            ],
            force_floor,
        )
        for row in final_metrics
    )
    summary = {
        "definition": checkpoint_definition,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_run_name": run_spec.name,
        "source_run_dir": str(run_spec.run_dir),
        "source_checkpoint": str(run_spec.ckpt),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "root_source_checkpoint_sha256": root_source_checkpoint_sha256,
        "source_capacity_step": source_capacity_step,
        "source_is_untouched_original_checkpoint": capacity_state is None,
        "direction_manifest": (
            str(args.direction_manifest)
            if args.direction_manifest is not None
            else None
        ),
        "direction_manifest_sha256": direction_manifest_hash,
        "direction_role": args.direction_role,
        "complete_total_relaxed_hvp_graph_required": (
            args.require_complete_total_relaxed_hvp
        ),
        "proxy_hvp_fallback_allowed": False,
        "additional_training_steps": args.max_steps,
        "evaluation_mode": args.evaluation_mode,
        "skipped_resume_initial_full_hessian": (
            args.skip_resume_initial_full_hessian
        ),
        "two_step_failure_reproduction": args.two_step_failure_reproduction,
        "checkpoint_interval": args.checkpoint_interval,
        "resumed_capacity_optimizer": args.resume_capacity_optimizer,
        "alternating_hvp_updates": args.alternating_hvp_updates,
        "hvp_update_period": args.hvp_update_period,
        "replay_update_period": args.replay_update_period,
        "hvp_learning_rate": (
            args.hvp_learning_rate
            if args.hvp_learning_rate is not None
            else args.learning_rate
        ),
        "hvp_loss_norm": args.hvp_loss_norm,
        "analytic_relaxed_hvp": args.analytic_relaxed_hvp,
        "symmetric_matrix_power_mode": symmetric_matrix_power_mode,
        "analytic_response_solver": (
            "dense_direct_implicit_adjoint"
            if args.analytic_relaxed_hvp
            else None
        ),
        "analytic_response_damping": args.analytic_response_damping,
        "analytic_response_residual_tolerance": (
            args.analytic_response_residual_tolerance
        ),
        "integral_directional_second_step": (
            args.integral_directional_second_step
        ),
        "training_direction_estimator": (
            "one_fresh_internal_rademacher_hutchinson_probe_per_step"
            if args.analytic_relaxed_hvp
            else "explicit_direction_secant"
        ),
        "density_parameter_response_predictor": (
            args.density_parameter_response_predictor
        ),
        "egf_label_density_replay": args.egf_label_density_replay,
        "response_correction_fraction_max": (
            args.response_correction_fraction_max
        ),
        "cancellation_index_max": args.cancellation_index_max,
        "loss_balance_mode": args.loss_balance_mode,
        "three_body_geometry_residual": args.three_body_geometry_residual,
        "three_body_initial_coefficients": (
            str(args.three_body_initial_coefficients)
            if args.three_body_initial_coefficients is not None
            else None
        ),
        "freeze_base_model": args.freeze_base_model,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "validation_accessed": False,
        "molecules": [molecule.molecule_id for molecule in molecules],
        "final_step": final_step,
        "stopped_early": stopped_early,
        "density_refresh_count": refresh_index + 1,
        "density_response_unroll_steps": args.density_response_unroll_steps,
        "density_response_unroll_lr": args.density_response_unroll_lr,
        "connect_lagrange_multiplier_response": (
            args.connect_lagrange_multiplier_response
        ),
        "implicit_density_parameter_response": (
            args.implicit_density_parameter_response
        ),
        "implicit_response_tolerance": args.implicit_response_tolerance,
        "implicit_response_max_iterations": args.implicit_response_max_iterations,
        "implicit_response_damping": args.implicit_response_damping,
        "implicit_response_diagonal_probes": (
            args.implicit_response_diagonal_probes
        ),
        "implicit_response_solver": args.implicit_response_solver,
        "implicit_response_warm_start": args.implicit_response_warm_start,
        "strict_active_density_refresh": args.strict_active_density_refresh,
        "density_solver_target": args.density_strict_threshold,
        "training_density_stationarity_threshold": (
            _training_density_stationarity_threshold(args)
        ),
        "training_density_stationarity_gate_passed": bool(loss_rows)
        and all(
            row["max_cached_density_gradient_norm"]
            < _training_density_stationarity_threshold(args)
            for row in loss_rows
        ),
        "loss_weights": weights,
        "initial_full_hessian_metrics": initial_metrics,
        "root_initial_full_hessian_metrics": root_initial_metrics,
        "final_full_hessian_metrics": final_metrics,
        "energy_force_regression_gate_passed": energy_force_gate_passed,
        "energy_force_relative_regression_max": energy_force_regression_limit,
        "stage1_gate_passed": bool(final_metrics)
        and energy_force_gate_passed
        and (
            not args.strict_active_density_refresh
            or all(
                row["max_cached_density_gradient_norm"]
                < _training_density_stationarity_threshold(args)
                for row in loss_rows
            )
        )
        and all(
            row["full_hessian_complete"]
            and row["relative_frobenius"]
            <= float(
                protocol["stage1"]["gate"][
                    "training_all_parent_relative_frobenius_max"
                ]
            )
            and row["antisymmetric_over_symmetric_frobenius"]
            <= float(
                protocol["stage1"]["gate"][
                    "antisymmetric_over_symmetric_fro_max"
                ]
            )
            for row in final_metrics
        ),
        "integral_bundle_build_count": bundle_cache.build_count,
        "integral_bundle_cache_hit_count": bundle_cache.hit_count,
        "wall_time_s": time.perf_counter() - started,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, default=None)
    parser.add_argument("--direction-manifest-sha256", default=None)
    parser.add_argument(
        "--direction-role",
        choices=("all", "train", "heldout"),
        default="all",
    )
    parser.add_argument("--run", required=True, help="name=run_dir=checkpoint")
    parser.add_argument("--source-checkpoint-sha256", default=None)
    parser.add_argument("--root-source-checkpoint-sha256", default=None)
    parser.add_argument(
        "--require-complete-total-relaxed-hvp",
        action="store_true",
        help=(
            "Fail unless an untouched Graphformer source, strict density relaxation, "
            "and implicit density parameter response own every HVP training graph."
        ),
    )
    parser.add_argument(
        "--analytic-relaxed-hvp",
        action="store_true",
        help=(
            "Use one center density, directional second integral derivatives, "
            "and an analytic constrained KKT response. No secant fallback is allowed."
        ),
    )
    parser.add_argument(
        "--symmetric-matrix-power-mode",
        choices=("stable_first_order", "eigh_second_order_audit"),
        default="stable_first_order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecules", default=None, help="Comma-separated frozen Stage-1 IDs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--three-body-geometry-residual",
        action="store_true",
        help="Add a conservative density-independent RBF-Legendre scalar residual.",
    )
    parser.add_argument("--three-body-center-min", type=float, default=0.5)
    parser.add_argument("--three-body-center-max", type=float, default=8.0)
    parser.add_argument("--three-body-center-count", type=int, default=4)
    parser.add_argument("--three-body-sigma", type=float, default=0.5)
    parser.add_argument("--three-body-angular-order", type=int, default=3)
    parser.add_argument("--three-body-initial-coefficients", type=Path, default=None)
    parser.add_argument(
        "--freeze-base-model",
        action="store_true",
        help="Optimize only the optional scalar geometry residual.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument(
        "--resume-capacity-optimizer",
        action="store_true",
        help="Restore AdamW state and cumulative step from a capacity-fit checkpoint.",
    )
    parser.add_argument(
        "--skip-resume-initial-full-hessian",
        action="store_true",
        help=(
            "On a provenance-bound optimizer resume, reuse the frozen root "
            "initial metrics and avoid recomputing an intermediate full Hessian."
        ),
    )
    parser.add_argument(
        "--alternating-hvp-updates",
        action="store_true",
        help="Use independent AdamW states for replay and direct-HVP update steps.",
    )
    parser.add_argument("--hvp-learning-rate", type=float, default=None)
    parser.add_argument("--hvp-update-period", type=int, default=2)
    parser.add_argument(
        "--replay-update-period",
        type=int,
        default=None,
        help=(
            "When alternating, run replay every Nth cumulative step and HVP otherwise. "
            "This takes precedence over --hvp-update-period."
        ),
    )
    parser.add_argument(
        "--hvp-loss-norm",
        choices=("l1", "l2"),
        default="l1",
        help="Norm for direct HVP fitting; l2 aligns with the Frobenius acceptance metric.",
    )
    parser.add_argument(
        "--loss-balance-mode",
        choices=("none", "pcgrad"),
        default="none",
        help=(
            "Optional two-task gradient balancing between E/F/Euler replay and "
            "HVP/curvature/spectrum losses."
        ),
    )
    parser.add_argument(
        "--egf-label-density-replay",
        action="store_true",
        help=(
            "Evaluate E/G/F at the fixed PBE/KS label density while keeping "
            "the analytic HVP on the model-self-consistent density branch."
        ),
    )
    parser.add_argument(
        "--response-correction-fraction-max",
        type=float,
        default=math.inf,
        help="Fail before an update if ||response correction||/||relaxed HVP|| exceeds this.",
    )
    parser.add_argument(
        "--cancellation-index-max",
        type=float,
        default=math.inf,
        help=(
            "Fail before an update if "
            "(||partial HVP||+||response correction||)/||relaxed HVP|| "
            "exceeds this."
        ),
    )
    parser.add_argument("--directions-per-step", type=int, default=2)
    parser.add_argument("--direction-repeat-steps", type=int, default=8)
    parser.add_argument("--direction-limit", type=int, default=None)
    parser.add_argument("--displacement", type=float, default=1.0e-3)
    parser.add_argument(
        "--evaluation-mode",
        choices=("full", "none"),
        default="full",
        help=(
            "Use 'none' only for a zero-learning-rate gradient smoke; formal "
            "training must retain initial/final full internal-Hessian evaluation."
        ),
    )
    parser.add_argument(
        "--two-step-failure-reproduction",
        action="store_true",
        help=(
            "Allow at most two nonzero-LR steps without matrix evaluation, "
            "with a checkpoint every step, solely to reproduce a registered failure."
        ),
    )
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--early-stop-relative-frobenius", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostics-interval", type=int, default=25)
    parser.add_argument("--lambda-e", type=float, default=None)
    parser.add_argument("--lambda-f", type=float, default=None)
    parser.add_argument("--lambda-rho", type=float, default=None)
    parser.add_argument("--lambda-h", type=float, default=None)
    parser.add_argument(
        "--lambda-q",
        type=float,
        default=None,
        help="Weight for scalar relaxed-energy directional curvature v^T H v.",
    )
    parser.add_argument("--lambda-spec", type=float, default=None)
    parser.add_argument("--density-loss-scale", type=float, default=1.0e-2)
    parser.add_argument("--density-refresh-interval", type=int, default=25)
    parser.add_argument("--density-drift-trigger", type=float, default=1.0e-4)
    parser.add_argument(
        "--strict-active-density-refresh",
        action="store_true",
        help=(
            "After every parameter update, strictly re-relax the center and, "
            "for the legacy secant path, the next active displaced geometries."
        ),
    )
    parser.add_argument("--density-response-unroll-steps", type=int, default=0)
    parser.add_argument("--density-response-unroll-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--density-parameter-response-predictor",
        action="store_true",
        help=(
            "Use the exact KKT response to the accepted model-parameter step "
            "as the next strict density-corrector initial guess."
        ),
    )
    parser.add_argument(
        "--connect-lagrange-multiplier-response",
        action="store_true",
        help=(
            "Keep the stationary Lagrange multiplier connected to model parameters on "
            "the training graph; strict value evaluation retains the default detached path."
        ),
    )
    parser.add_argument(
        "--implicit-density-parameter-response",
        action="store_true",
        help="Use a constrained matrix-free implicit VJP for dc_star/dtheta.",
    )
    parser.add_argument("--implicit-response-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--implicit-response-max-iterations", type=int, default=500)
    parser.add_argument("--implicit-response-damping", type=float, default=1.0e-8)
    parser.add_argument("--implicit-response-diagonal-probes", type=int, default=0)
    parser.add_argument(
        "--implicit-response-solver",
        choices=("pcg", "direct"),
        default="pcg",
        help="Strict parameter-adjoint solver; no automatic fallback is used.",
    )
    parser.add_argument(
        "--implicit-response-warm-start",
        action="store_true",
        help="Reuse the previous detached adjoint solution for each exact geometry.",
    )
    parser.add_argument("--base-initialization", default="label_reference")
    parser.add_argument("--density-lr", type=float, default=1.0e-3)
    parser.add_argument("--density-max-cycles", type=int, default=1000)
    parser.add_argument("--density-first-threshold", type=float, default=1.0e-2)
    parser.add_argument("--density-fallback-lr", type=float, default=3.0e-4)
    parser.add_argument("--density-fallback-max-cycles", type=int, default=10000)
    parser.add_argument("--density-fallback-threshold", type=float, default=1.0e-5)
    parser.add_argument("--density-strict-threshold", type=float, default=1.0e-8)
    parser.add_argument(
        "--training-density-stationarity-threshold",
        type=float,
        default=None,
        help=(
            "Independent fail-closed gate for the density gradient recomputed "
            "on the training graph. Defaults to --density-strict-threshold; "
            "set the solver target tighter to leave numerical verification margin."
        ),
    )
    parser.add_argument("--lbfgs-max-iterations", type=int, default=500)
    parser.add_argument("--newton-max-iterations", type=int, default=6)
    parser.add_argument("--integral-derivative-step", type=float, default=1.0e-4)
    parser.add_argument(
        "--integral-directional-second-step",
        type=float,
        default=1.0e-4,
        help=(
            "Finite-difference step used only to differentiate PySCF first "
            "integral derivatives along the analytic HVP probe."
        ),
    )
    parser.add_argument("--integral-derivative-workers", type=int, default=4)
    parser.add_argument("--integral-cache-entries", type=int, default=6)
    parser.add_argument("--analytic-response-damping", type=float, default=0.0)
    parser.add_argument(
        "--analytic-response-residual-tolerance",
        type=float,
        default=1.0e-8,
    )
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
