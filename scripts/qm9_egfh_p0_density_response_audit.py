#!/usr/bin/env python3
"""Read-only P0 audit of EGFH density curvature and relaxed HVPs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
    prediction = prediction.detach().cpu()
    reference = reference.detach().cpu()
    denominator = torch.linalg.vector_norm(reference).clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    return float(torch.linalg.vector_norm(prediction - reference) / denominator)


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().cpu()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("Audited tensor contains a non-finite value")
    return {
        "norm": float(torch.linalg.vector_norm(value)),
        "rms": float(torch.sqrt(torch.mean(value.square()))),
        "max_abs": float(torch.max(torch.abs(value))),
    }


def _manifest_entry(
    manifest: dict[str, Any], molecule_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["parents"]
        if str(row["molecule_id"]) == molecule_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest entry for {molecule_id}, found {len(matches)}"
        )
    return matches[0]


def _density_spectrum(
    response_system: Any, relative_zero_threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    matrix = response_system.tangent_matrix(create_graph=False).detach()
    eigenvalues = torch.linalg.eigvalsh(matrix).detach().cpu().numpy()
    if not np.isfinite(eigenvalues).all():
        raise RuntimeError("Projected density Hessian spectrum is non-finite")
    maximum_abs = float(np.max(np.abs(eigenvalues)))
    absolute_zero_threshold = max(
        maximum_abs * relative_zero_threshold, np.finfo(np.float64).tiny
    )
    nonzero = np.abs(eigenvalues) > absolute_zero_threshold
    minimum_nonzero_abs = (
        float(np.min(np.abs(eigenvalues[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    condition = (
        maximum_abs / minimum_nonzero_abs
        if np.isfinite(minimum_nonzero_abs) and minimum_nonzero_abs > 0
        else float("inf")
    )
    summary = {
        "dimension": int(eigenvalues.size),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "maximum_absolute_eigenvalue": maximum_abs,
        "minimum_nonzero_absolute_eigenvalue": minimum_nonzero_abs,
        "condition_number_absolute": condition,
        "relative_zero_threshold": relative_zero_threshold,
        "absolute_zero_threshold": absolute_zero_threshold,
        "negative_count": int(np.sum(eigenvalues < -absolute_zero_threshold)),
        "near_zero_count": int(np.sum(~nonzero)),
        "positive_count": int(np.sum(eigenvalues > absolute_zero_threshold)),
        "lowest_eigenvalues": eigenvalues[: min(12, eigenvalues.size)].tolist(),
        "highest_eigenvalues": eigenvalues[-min(12, eigenvalues.size) :].tolist(),
    }
    return summary, eigenvalues


def _endpoint_pair(
    *,
    context: Any,
    cache: IntegralBundleCache,
    molecule: Any,
    direction: Any,
    response: Any,
    density_args: argparse.Namespace,
    displacement: float,
    strict_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    relaxed_forces: list[torch.Tensor] = []
    fixed_forces: list[torch.Tensor] = []
    relaxed_coefficients: list[torch.Tensor] = []
    endpoint_rows: list[dict[str, Any]] = []
    for sign, side in ((1.0, "plus"), (-1.0, "minus")):
        positions = (
            molecule.positions_bohr + sign * displacement * direction.vector
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
        relaxed_coefficients.append(relaxed.coefficients.detach().cpu())
        endpoint_rows.append(
            {
                "side": side,
                "cycles": int(relaxed.cycles),
                "final_projected_density_gradient_norm": float(
                    relaxed.final_gradient_norm
                ),
                "strict_density_pass": bool(
                    relaxed.final_gradient_norm < strict_threshold
                ),
                "optimizer_reported_converged": bool(metadata["converged"]),
                "predicted_start_to_final_coefficients_l2": float(
                    torch.linalg.vector_norm(
                        predicted_start - relaxed.coefficients
                    )
                ),
            }
        )
    density_secant = (
        relaxed_coefficients[0] - relaxed_coefficients[1]
    ) / (2.0 * displacement)
    relaxed_hvp = -(
        relaxed_forces[0] - relaxed_forces[1]
    ) / (2.0 * displacement)
    fixed_hvp = -(
        fixed_forces[0] - fixed_forces[1]
    ) / (2.0 * displacement)
    response_correction = relaxed_hvp - fixed_hvp
    arrays = {
        "density_secant": density_secant.numpy(),
        "relaxed_hvp": relaxed_hvp.numpy(),
        "fixed_hvp": fixed_hvp.numpy(),
        "response_correction": response_correction.numpy(),
    }
    return {"endpoints": endpoint_rows}, arrays


def _audit_checkpoint(
    *,
    checkpoint_cfg: dict[str, Any],
    protocol: dict[str, Any],
    base_protocol: dict[str, Any],
    parent_entry: dict[str, Any],
    direction_entry: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    name = str(checkpoint_cfg["name"])
    checkpoint_path = Path(checkpoint_cfg["checkpoint"])
    if _sha256(checkpoint_path) != checkpoint_cfg["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}")
    run_dir = Path(checkpoint_cfg["run_dir"])
    if not (run_dir / "hparams.yaml").is_file():
        raise FileNotFoundError(f"Missing hparams.yaml for {name}: {run_dir}")

    common = _common_args(output_dir, base_protocol, str(device))
    density_args = _density_namespace(common)
    strict_threshold = float(
        protocol["numerics"]["strict_density_gradient_threshold"]
    )
    diagnostic_threshold = float(
        protocol["numerics"]["diagnostic_acceptance_threshold"]
    )
    # The optimizer still targets 1e-8 because lbfgs_tolerance and
    # newton_tolerance were frozen by _density_namespace above. This relaxed
    # acceptance limit only lets the P0 diagnostic retain a non-strict point.
    density_args.density_strict_threshold = diagnostic_threshold

    molecule = _load_molecule(
        parent_entry,
        low_mode_count=0,
        direction_entry=direction_entry,
        direction_role=str(protocol["assets"]["direction_role"]),
    )
    selected = [
        row
        for row in molecule.directions
        if int(row.index) == int(protocol["assets"]["direction_index"])
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Direction {protocol['assets']['direction_index']} is unavailable"
        )
    direction = selected[0]
    run = f"{name}={run_dir}={checkpoint_path}"
    context = _load_context(_parse_run(run), density_args, device)
    context.model.to(torch.float64)
    cache = IntegralBundleCache(context, common)
    center_rows = _refresh_base_densities(
        context, [molecule], density_args, refresh_index=0
    )
    center_strict = bool(
        center_rows[0]["final_gradient_norm"] < strict_threshold
    )
    if not center_strict:
        raise RuntimeError(
            f"{name} center density is not strict: "
            f"{center_rows[0]['final_gradient_norm']:.3e}"
        )

    (
        _,
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
    analytic_correction = response_system.response_correction(
        response.density_response,
        response.multiplier_response,
        create_graph=False,
    )
    analytic_relaxed = partial + analytic_correction
    response_threshold = float(
        protocol["numerics"]["analytic_response_residual_threshold"]
    )
    if (
        response.stationarity_direction_residual > response_threshold
        or response.constraint_direction_residual > response_threshold
    ):
        raise RuntimeError(
            f"{name} analytic response residual exceeds "
            f"{response_threshold:.3e}: stationarity="
            f"{response.stationarity_direction_residual:.3e}, constraint="
            f"{response.constraint_direction_residual:.3e}"
        )
    pbe_hvp = torch.as_tensor(
        direction.target_hvp,
        dtype=analytic_relaxed.dtype,
        device=analytic_relaxed.device,
    )
    spectrum, eigenvalues = _density_spectrum(
        response_system,
        float(protocol["numerics"]["density_spectrum_relative_zero_threshold"]),
    )

    fd_rows = []
    fd_arrays: dict[str, np.ndarray] = {}
    for displacement in protocol["numerics"]["force_fd_displacements_bohr"]:
        displacement = float(displacement)
        pair, arrays = _endpoint_pair(
            context=context,
            cache=cache,
            molecule=molecule,
            direction=direction,
            response=response,
            density_args=density_args,
            displacement=displacement,
            strict_threshold=strict_threshold,
        )
        fixed_hvp = torch.as_tensor(arrays["fixed_hvp"])
        relaxed_hvp = torch.as_tensor(arrays["relaxed_hvp"])
        fd_correction = torch.as_tensor(arrays["response_correction"])
        density_secant = torch.as_tensor(arrays["density_secant"])
        key = f"h_{displacement:.1e}".replace(".", "p").replace("-", "m")
        for array_name, array in arrays.items():
            fd_arrays[f"{key}_{array_name}"] = array
        fd_rows.append(
            {
                "displacement_bohr": displacement,
                **pair,
                "all_endpoints_strict": all(
                    row["strict_density_pass"] for row in pair["endpoints"]
                ),
                "fixed_hvp": _tensor_summary(fixed_hvp),
                "relaxed_hvp": _tensor_summary(relaxed_hvp),
                "fd_response_correction": _tensor_summary(fd_correction),
                "density_secant": _tensor_summary(density_secant),
                "fixed_vs_analytic_partial_relative_l2": _relative_l2(
                    fixed_hvp, partial
                ),
                "relaxed_vs_analytic_relative_l2": _relative_l2(
                    relaxed_hvp, analytic_relaxed
                ),
                "relaxed_vs_pbe_relative_l2": _relative_l2(
                    relaxed_hvp, pbe_hvp
                ),
                "analytic_correction_vs_fd_relative_l2": _relative_l2(
                    fd_correction, analytic_correction
                ),
                "density_response_vs_secant_relative_l2": _relative_l2(
                    density_secant, response.density_response
                ),
            }
        )

    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_dir / "arrays.npz",
        direction=np.asarray(direction.vector),
        pbe_hvp=pbe_hvp.detach().cpu().numpy(),
        analytic_partial_hvp=partial.detach().cpu().numpy(),
        analytic_response_correction=analytic_correction.detach().cpu().numpy(),
        analytic_relaxed_hvp=analytic_relaxed.detach().cpu().numpy(),
        analytic_density_response=(
            response.density_response.detach().cpu().numpy()
        ),
        projected_density_hessian_eigenvalues=eigenvalues,
        **fd_arrays,
    )
    result = {
        "name": name,
        "training_step": int(checkpoint_cfg["training_step"]),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_cfg["checkpoint_sha256"],
        "center_density": center_rows[0],
        "center_density_strict": center_strict,
        "response_timings": response_timings,
        "response_stationarity_residual": float(
            response.stationarity_direction_residual
        ),
        "response_constraint_residual": float(
            response.constraint_direction_residual
        ),
        "projected_density_hessian_spectrum": spectrum,
        "pbe_hvp": _tensor_summary(pbe_hvp),
        "analytic_partial_hvp": _tensor_summary(partial),
        "analytic_response_correction": _tensor_summary(analytic_correction),
        "analytic_relaxed_hvp": _tensor_summary(analytic_relaxed),
        "analytic_partial_vs_pbe_relative_l2": _relative_l2(partial, pbe_hvp),
        "analytic_relaxed_vs_pbe_relative_l2": _relative_l2(
            analytic_relaxed, pbe_hvp
        ),
        "response_correction_fraction_of_relaxed_norm": float(
            torch.linalg.vector_norm(analytic_correction.detach())
            / torch.linalg.vector_norm(analytic_relaxed.detach()).clamp_min(
                torch.finfo(analytic_relaxed.dtype).tiny
            )
        ),
        "force_fd_scan": fd_rows,
        "wall_time_s": time.perf_counter() - started,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    del cache, context, molecule, response_system, response
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol["training_allowed"] is not False:
        raise ValueError("P0 protocol must prohibit training")
    if (
        protocol["validation_access_allowed"] is not False
        or protocol["test100_access_allowed"] is not False
    ):
        raise ValueError("P0 protocol must prohibit validation and Test100")
    expected_mode = protocol["implementation"]["symmetric_matrix_power_mode"]
    actual_mode = os.environ.get(
        "MLDFT_SYMMETRIC_MATRIX_POWER_MODE", "stable_first_order"
    )
    if actual_mode != expected_mode:
        raise ValueError(
            f"Symmetric matrix power mode mismatch: {actual_mode} != {expected_mode}"
        )

    parent_path = Path(protocol["assets"]["parent_manifest"])
    direction_path = Path(protocol["assets"]["direction_manifest"])
    if _sha256(parent_path) != protocol["assets"]["parent_manifest_sha256"]:
        raise ValueError("Parent manifest hash mismatch")
    if _sha256(direction_path) != protocol["assets"]["direction_manifest_sha256"]:
        raise ValueError("Direction manifest hash mismatch")
    parent_manifest = json.loads(parent_path.read_text())
    direction_manifest = json.loads(direction_path.read_text())
    if direction_manifest["parent_manifest_sha256"] != _sha256(parent_path):
        raise ValueError("Direction manifest parent provenance mismatch")
    molecule_id = str(protocol["assets"]["molecule_id"])
    parent_entry = _manifest_entry(parent_manifest, molecule_id)
    direction_entry = _manifest_entry(direction_manifest, molecule_id)

    base_protocol_path = args.repo / protocol["implementation"]["base_analytic_protocol"]
    base_protocol = yaml.safe_load(base_protocol_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    results = []
    failures = []
    for checkpoint_cfg in protocol["checkpoints"]:
        checkpoint_name = str(checkpoint_cfg["name"])
        print(f"[P0] starting {checkpoint_name}", flush=True)
        try:
            result = _audit_checkpoint(
                checkpoint_cfg=checkpoint_cfg,
                protocol=protocol,
                base_protocol=base_protocol,
                parent_entry=parent_entry,
                direction_entry=direction_entry,
                output_dir=args.output_dir,
                device=device,
            )
            results.append(result)
            print(
                f"[P0] completed {checkpoint_name}: "
                f"relaxed_vs_pbe="
                f"{result['analytic_relaxed_vs_pbe_relative_l2']:.6g}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "name": checkpoint_cfg["name"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            (args.output_dir / f"{checkpoint_cfg['name']}_failure.json").write_text(
                json.dumps(failures[-1], indent=2, sort_keys=True) + "\n"
            )
            if args.fail_fast:
                raise

    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "status": "complete" if not failures else "partial_failure",
        "molecule_id": molecule_id,
        "direction_index": int(protocol["assets"]["direction_index"]),
        "checkpoint_results": results,
        "failures": failures,
        "wall_time_s": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
            if device.type == "cuda"
            else 0.0
        ),
        "training_performed": False,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = audit(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
