#!/usr/bin/env python3
"""Isolate fixed-geometry and density-response errors in one relaxed HVP."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mldft.ofdft.implicit_response import ConstrainedResponseSystem
from mldft.ofdft.geometry_integrals import classical_energy_from_bundle
from qm9_complete_total_capacity_train import (
    IntegralBundleCache,
    RelaxedPoint,
    _analytic_center_response,
    _density_namespace,
    _evaluate_point_graph,
    _load_molecule,
    _parse_run,
    _refresh_base_densities,
    _relax,
    _sha256,
)
from qm9_graphformer_analytic_relaxed_hvp_audit import _common_args
from qm9_hessian_density_relaxed_eval import _load_context


def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    return float(
        (
            torch.linalg.vector_norm(prediction - reference) / denominator
        )
        .detach()
        .cpu()
    )


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    detached = value.detach()
    return {
        "norm": float(torch.linalg.vector_norm(detached).cpu()),
        "rms": float(torch.sqrt(torch.mean(detached.square())).cpu()),
        "max_abs": float(torch.max(torch.abs(detached)).cpu()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    protocol = yaml.safe_load(args.protocol.read_text())
    registration = json.loads(args.checkpoint_registration.read_text())
    protocol_hash = _sha256(args.protocol)
    if (
        registration["protocol_sha256"] != protocol_hash
        or registration["global_step"] != protocol["baseline"]["max_steps"]
        or registration["validation_accessed"] is not False
        or registration["test100_accessed"] is not False
    ):
        raise ValueError("checkpoint registration does not match the frozen protocol")
    checkpoint = Path(registration["checkpoint"])
    if _sha256(checkpoint) != registration["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")

    parent_manifest = json.loads(args.parent_manifest.read_text())
    direction_manifest = json.loads(args.direction_manifest.read_text())
    if (
        parent_manifest["protocol_sha256"] != protocol_hash
        or direction_manifest["protocol_sha256"] != protocol_hash
        or direction_manifest["parent_manifest_sha256"]
        != _sha256(args.parent_manifest)
    ):
        raise ValueError("parent or direction manifest provenance mismatch")
    parent_entry = next(
        row
        for row in parent_manifest["parents"]
        if str(row["molecule_id"]) == args.molecule
    )
    direction_entry = next(
        row
        for row in direction_manifest["parents"]
        if str(row["molecule_id"]) == args.molecule
    )
    molecule = _load_molecule(
        parent_entry,
        low_mode_count=0,
        direction_entry=direction_entry,
        direction_role="all",
    )
    direction = molecule.directions[args.direction_index]

    common = _common_args(args.output_dir, protocol, args.device)
    if args.force_fd_displacement is not None:
        common.displacement = float(args.force_fd_displacement)
    density_args = _density_namespace(common)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    run = (
        f"{protocol['identity']['baseline_name']}="
        f"{registration['run_dir']}={checkpoint}"
    )
    context = _load_context(_parse_run(run), density_args, device)
    context.model.to(torch.float64)
    cache = IntegralBundleCache(context, common)
    density_rows = _refresh_base_densities(
        context, [molecule], density_args, refresh_index=0
    )

    (
        center,
        position_direction,
        response_system,
        response,
        response_timings,
    ) = _analytic_center_response(
        context,
        cache,
        molecule,
        direction,
        create_graph=False,
    )
    partial = response_system.partial_position_hvp(
        position_direction, create_graph=False
    )
    response_correction = response_system.response_correction(
        response.density_response,
        response.multiplier_response,
        create_graph=False,
    )
    analytic = partial + response_correction
    center_component_hvps = {}
    for name, energy in center.energies.energies_dict.items():
        gradient = torch.autograd.grad(
            energy,
            center.geometry.positions,
            create_graph=True,
            retain_graph=True,
        )[0]
        center_component_hvps[name] = torch.autograd.grad(
            gradient,
            center.geometry.positions,
            grad_outputs=position_direction,
            create_graph=False,
            retain_graph=True,
        )[0].detach().cpu()

    def partial_position_hvp_at_integral_step(
        directional_step: float | None,
    ) -> torch.Tensor:
        step_common = copy.copy(common)
        if directional_step is not None:
            step_common.integral_directional_second_step = directional_step
        step_cache = IntegralBundleCache(context, step_common)
        point = _evaluate_point_graph(
            context,
            step_cache,
            molecule,
            molecule.base,
            create_graph=True,
            attach_density_parameter_response=False,
            directional_second_direction=(
                None if directional_step is None else direction.vector
            ),
        )
        system = ConstrainedResponseSystem(
            total_energy=point.energies.total_energy,
            coeffs=point.geometry.coeffs,
            positions=point.geometry.positions,
            normalization=torch.as_tensor(
                point.geometry.bundle.values.normalization,
                dtype=point.geometry.coeffs.dtype,
                device=point.geometry.coeffs.device,
            ),
            n_electron=int(
                np.sum(molecule.atomic_numbers) - step_common.charge
            ),
            multiplier=point.lagrange_multiplier,
        )
        value = system.partial_position_hvp(
            torch.as_tensor(
                direction.vector,
                dtype=system.positions.dtype,
                device=system.positions.device,
            ),
            create_graph=False,
        )
        return value.detach().cpu()

    displacement = float(common.displacement)
    endpoint_rows = []
    relaxed_forces = []
    fixed_forces = []
    relaxed_coefficients = []
    relaxed_multipliers = []
    fixed_energies = []
    fixed_component_energies = []
    for sign, side in ((1.0, "plus"), (-1.0, "minus")):
        positions = (
            molecule.positions_bohr
            + sign * displacement * direction.vector
        )
        predicted_start = (
            molecule.base.coefficients
            + sign * displacement * response.density_response.detach().cpu()
        )
        relaxed, metadata = _relax(
            context,
            molecule,
            positions,
            density_args,
            warm_start=None,
            explicit_start=predicted_start,
        )
        relaxed_total = _evaluate_point_graph(
            context,
            cache,
            molecule,
            relaxed,
            create_graph=False,
            attach_density_parameter_response=False,
        )
        fixed = RelaxedPoint(
            positions_bohr=np.asarray(positions, dtype=np.float64),
            coefficients=molecule.base.coefficients,
            final_gradient_norm=molecule.base.final_gradient_norm,
            cycles=0,
            total_energy=molecule.base.total_energy,
        )
        fixed_total = _evaluate_point_graph(
            context,
            cache,
            molecule,
            fixed,
            create_graph=False,
            attach_density_parameter_response=False,
        )
        relaxed_forces.append(relaxed_total.force.detach().cpu())
        fixed_forces.append(fixed_total.force.detach().cpu())
        fixed_energies.append(
            float(fixed_total.energies.total_energy.detach().cpu())
        )
        fixed_component_energies.append(
            {
                name: float(value.detach().cpu())
                for name, value in fixed_total.energies.energies_dict.items()
            }
        )
        relaxed_coefficients.append(relaxed.coefficients.detach().cpu())
        relaxed_multipliers.append(
            relaxed_total.lagrange_multiplier.detach().cpu()
        )
        endpoint_rows.append(
            {
                "side": side,
                "cycles": int(relaxed.cycles),
                "final_projected_density_gradient_norm": float(
                    relaxed.final_gradient_norm
                ),
                "predicted_start_to_final_coefficients_l2": float(
                    torch.linalg.vector_norm(
                        predicted_start - relaxed.coefficients
                    )
                ),
                "total_energy_hartree": float(
                    relaxed_total.energies.total_energy.detach().cpu()
                ),
                "fixed_density_total_energy_hartree": fixed_energies[-1],
                "optimization_metadata_converged": bool(metadata["converged"]),
            }
        )

    density_secant = (
        relaxed_coefficients[0] - relaxed_coefficients[1]
    ) / (2.0 * displacement)
    multiplier_secant = (
        relaxed_multipliers[0] - relaxed_multipliers[1]
    ) / (2.0 * displacement)
    relaxed_fd = -(
        relaxed_forces[0] - relaxed_forces[1]
    ) / (2.0 * displacement)
    fixed_fd = -(
        fixed_forces[0] - fixed_forces[1]
    ) / (2.0 * displacement)
    fd_response_correction = relaxed_fd - fixed_fd
    base_energy = float(center.energies.total_energy.detach().cpu())
    fixed_density_scalar_directional_curvature = (
        fixed_energies[0] - 2.0 * base_energy + fixed_energies[1]
    ) / displacement**2
    analytic_partial_directional_curvature = float(
        torch.sum(
            partial.detach().cpu()
            * torch.as_tensor(direction.vector, dtype=partial.dtype)
        )
    )
    fixed_force_fd_directional_curvature = float(
        torch.sum(
            fixed_fd
            * torch.as_tensor(direction.vector, dtype=fixed_fd.dtype)
        )
    )
    component_rows = []
    for name, center_energy in center.energies.energies_dict.items():
        scalar_curvature = (
            fixed_component_energies[0][name]
            - 2.0 * float(center_energy.detach().cpu())
            + fixed_component_energies[1][name]
        ) / displacement**2
        component_hvp = center_component_hvps[name]
        autograd_curvature = float(
            torch.sum(
                component_hvp
                * torch.as_tensor(
                    direction.vector, dtype=component_hvp.dtype
                )
            )
        )
        component_rows.append(
            {
                "name": name,
                "center_energy_hartree": float(center_energy.detach().cpu()),
                "scalar_directional_curvature": scalar_curvature,
                "autograd_directional_curvature": autograd_curvature,
                "autograd_minus_scalar_curvature": (
                    autograd_curvature - scalar_curvature
                ),
                "autograd_hvp": _tensor_summary(component_hvp),
            }
        )
    raw_classical_energy = classical_energy_from_bundle(
        center.geometry.coeffs,
        center.geometry.positions,
        torch.as_tensor(
            molecule.atomic_numbers,
            dtype=torch.long,
            device=center.geometry.positions.device,
        ),
        center.geometry.bundle,
    )
    raw_classical_gradient = torch.autograd.grad(
        raw_classical_energy,
        center.geometry.positions,
        create_graph=True,
        retain_graph=True,
    )[0]
    raw_classical_hvp = torch.autograd.grad(
        raw_classical_gradient,
        center.geometry.positions,
        grad_outputs=position_direction,
        create_graph=False,
        retain_graph=True,
    )[0].detach().cpu()
    raw_classical_directional_curvature = float(
        torch.sum(
            raw_classical_hvp
            * torch.as_tensor(
                direction.vector, dtype=raw_classical_hvp.dtype
            )
        )
    )
    scalar_classical_directional_curvature = sum(
        row["scalar_directional_curvature"]
        for row in component_rows
        if row["name"]
        in {"hartree", "nuclear_attraction", "nuclear_repulsion"}
    )
    secant_reconstruction = partial + response_system.response_correction(
        density_secant.to(response_system.coeffs),
        multiplier_secant.to(response_system.multiplier),
        create_graph=False,
    )
    first_derivative_only_partial = partial_position_hvp_at_integral_step(
        None
    )
    expected_directional_integral_correction = (
        fixed_fd - first_derivative_only_partial
    )
    directional_step_rows = []
    directional_step_partials = []
    for directional_step in args.directional_second_steps:
        if np.isclose(
            directional_step,
            common.integral_directional_second_step,
            rtol=0.0,
            atol=1.0e-15,
        ):
            step_partial = partial.detach().cpu()
        else:
            step_partial = partial_position_hvp_at_integral_step(
                directional_step
            )
        injected_correction = (
            step_partial - first_derivative_only_partial
        )
        directional_step_partials.append(step_partial)
        directional_step_rows.append(
            {
                "directional_second_step_bohr": directional_step,
                "partial_position_hvp": _tensor_summary(step_partial),
                "partial_vs_fixed_fd_relative_l2": _relative_l2(
                    step_partial, fixed_fd
                ),
                "injected_directional_integral_correction": (
                    _tensor_summary(injected_correction)
                ),
                "injected_vs_expected_correction_relative_l2": (
                    _relative_l2(
                        injected_correction,
                        expected_directional_integral_correction,
                    )
                ),
            }
        )

    result = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "molecule_id": args.molecule,
        "direction_index": int(direction.index),
        "direction_kind": direction.kind,
        "method_name": protocol["method_name"],
        "symmetric_matrix_power_mode": os.environ.get(
            "MLDFT_SYMMETRIC_MATRIX_POWER_MODE", "stable_first_order"
        ),
        "source_checkpoint_sha256": registration["checkpoint_sha256"],
        "protocol_sha256": protocol_hash,
        "displacement_bohr": displacement,
        "density_points": density_rows,
        "endpoints": endpoint_rows,
        "response_timings": response_timings,
        "analytic_density_response": _tensor_summary(
            response.density_response
        ),
        "strict_density_secant": _tensor_summary(density_secant),
        "density_response_vs_secant_relative_l2": _relative_l2(
            response.density_response.detach().cpu(), density_secant
        ),
        "analytic_partial_position_hvp": _tensor_summary(partial),
        "fixed_density_force_fd_hvp": _tensor_summary(fixed_fd),
        "partial_vs_fixed_fd_relative_l2": _relative_l2(
            partial.detach().cpu(), fixed_fd
        ),
        "fixed_density_scalar_directional_curvature": (
            fixed_density_scalar_directional_curvature
        ),
        "analytic_partial_directional_curvature": (
            analytic_partial_directional_curvature
        ),
        "fixed_force_fd_directional_curvature": (
            fixed_force_fd_directional_curvature
        ),
        "analytic_partial_vs_scalar_curvature_relative_abs": abs(
            analytic_partial_directional_curvature
            - fixed_density_scalar_directional_curvature
        )
        / max(abs(fixed_density_scalar_directional_curvature), 1.0e-15),
        "fixed_force_fd_vs_scalar_curvature_relative_abs": abs(
            fixed_force_fd_directional_curvature
            - fixed_density_scalar_directional_curvature
        )
        / max(abs(fixed_density_scalar_directional_curvature), 1.0e-15),
        "fixed_density_component_curvatures": component_rows,
        "raw_physical_basis_classical_hvp": _tensor_summary(
            raw_classical_hvp
        ),
        "raw_physical_basis_classical_directional_curvature": (
            raw_classical_directional_curvature
        ),
        "scalar_classical_directional_curvature": (
            scalar_classical_directional_curvature
        ),
        "raw_classical_vs_scalar_curvature_relative_abs": abs(
            raw_classical_directional_curvature
            - scalar_classical_directional_curvature
        )
        / max(abs(scalar_classical_directional_curvature), 1.0e-15),
        "first_derivative_only_partial": _tensor_summary(
            first_derivative_only_partial
        ),
        "first_derivative_only_vs_fixed_fd_relative_l2": _relative_l2(
            first_derivative_only_partial, fixed_fd
        ),
        "expected_directional_integral_correction": _tensor_summary(
            expected_directional_integral_correction
        ),
        "directional_second_step_scan": directional_step_rows,
        "analytic_response_correction": _tensor_summary(response_correction),
        "relaxed_minus_fixed_fd_response": _tensor_summary(
            fd_response_correction
        ),
        "response_correction_vs_fd_relative_l2": _relative_l2(
            response_correction.detach().cpu(), fd_response_correction
        ),
        "analytic_relaxed_hvp": _tensor_summary(analytic),
        "strict_relaxed_force_fd_hvp": _tensor_summary(relaxed_fd),
        "analytic_vs_relaxed_fd_relative_l2": _relative_l2(
            analytic.detach().cpu(), relaxed_fd
        ),
        "secant_response_reconstruction": _tensor_summary(
            secant_reconstruction
        ),
        "secant_reconstruction_vs_relaxed_fd_relative_l2": _relative_l2(
            secant_reconstruction.detach().cpu(), relaxed_fd
        ),
        "maximum_density_gradient_norm": max(
            [float(row["final_gradient_norm"]) for row in density_rows]
            + [
                float(row["final_projected_density_gradient_norm"])
                for row in endpoint_rows
            ]
        ),
        "wall_time_s": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else 0.0
        ),
        "proxy_fallback_used": False,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "components.npz",
        direction=direction.vector,
        analytic_density_response=response.density_response.detach().cpu().numpy(),
        strict_density_secant=density_secant.numpy(),
        analytic_partial_position_hvp=partial.detach().cpu().numpy(),
        fixed_density_force_fd_hvp=fixed_fd.numpy(),
        analytic_response_correction=response_correction.detach().cpu().numpy(),
        relaxed_minus_fixed_fd_response=fd_response_correction.numpy(),
        analytic_relaxed_hvp=analytic.detach().cpu().numpy(),
        strict_relaxed_force_fd_hvp=relaxed_fd.numpy(),
        secant_response_reconstruction=secant_reconstruction.detach().cpu().numpy(),
        first_derivative_only_partial=first_derivative_only_partial.numpy(),
        expected_directional_integral_correction=(
            expected_directional_integral_correction.numpy()
        ),
        directional_second_steps=np.asarray(
            args.directional_second_steps, dtype=np.float64
        ),
        directional_step_partials=np.stack(
            [value.numpy() for value in directional_step_partials]
        ),
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-registration", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecule", default="0028399")
    parser.add_argument("--direction-index", type=int, default=0)
    parser.add_argument("--force-fd-displacement", type=float)
    parser.add_argument(
        "--directional-second-steps",
        type=float,
        nargs="+",
        default=(3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
