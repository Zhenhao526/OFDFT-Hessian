#!/usr/bin/env python3
"""Fail-closed verification of trainable complete-total relaxed Graphformer HVPs."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mldft.ofdft.complete_total_training import (
    central_force_secant_hvp,
    consume_implicit_response_diagnostics,
)

from qm9_complete_total_capacity_train import (
    IntegralBundleCache,
    _density_namespace,
    _direction_prediction,
    _evaluate_point_graph,
    _full_hessian_metrics,
    _load_molecule,
    _parse_run,
    _relax,
    _refresh_densities,
    _sha256,
)
from qm9_hessian_density_relaxed_eval import _load_context
def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")
    return actual


def _require_finite(value: torch.Tensor | np.ndarray, label: str) -> None:
    if not bool(np.all(np.isfinite(np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)))):
        raise FloatingPointError(f"{label} contains NaN or Inf")


def _strict_force(
    context: Any,
    cache: IntegralBundleCache,
    molecule: Any,
    positions_bohr: np.ndarray,
    density_args: argparse.Namespace,
    initial_coefficients: torch.Tensor,
) -> dict[str, Any]:
    started = time.perf_counter()
    relaxed, metadata = _relax(
        context,
        molecule,
        positions_bohr,
        density_args,
        warm_start=None,
        explicit_start=initial_coefficients,
    )
    total = _evaluate_point_graph(
        context,
        cache,
        molecule,
        relaxed,
        create_graph=False,
        attach_density_parameter_response=False,
    )
    force = total.force.detach().cpu()
    _require_finite(force, "fresh strict force")
    return {
        "force": force,
        "coefficients": relaxed.coefficients,
        "cycles": relaxed.cycles,
        "final_gradient_norm": relaxed.final_gradient_norm,
        "total_energy": float(total.energies.total_energy.detach().cpu()),
        "wall_time_s": time.perf_counter() - started,
        "optimization_metadata": metadata,
    }


def _fresh_hvp(
    context: Any,
    cache: IntegralBundleCache,
    molecule: Any,
    direction: Any,
    density_args: argparse.Namespace,
    plus_start: torch.Tensor,
    minus_start: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    point_rows = []
    forces = []
    for sign, side, start in (
        (1.0, "plus", plus_start),
        (-1.0, "minus", minus_start),
    ):
        positions = (
            molecule.positions_bohr
            + sign * density_args.displacement * direction.vector
        )
        point = _strict_force(
            context, cache, molecule, positions, density_args, start
        )
        forces.append(point["force"])
        point_rows.append(
            {
                "side": side,
                "cycles": int(point["cycles"]),
                "final_projected_density_gradient_norm": float(
                    point["final_gradient_norm"]
                ),
                "total_energy_hartree": float(point["total_energy"]),
                "wall_time_s": float(point["wall_time_s"]),
            }
        )
    return (
        central_force_secant_hvp(
            forces[0], forces[1], density_args.displacement
        ),
        point_rows,
    )


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(first - second)
            / torch.linalg.vector_norm(second).clamp_min(
                torch.finfo(second.dtype).tiny
            )
        )
        .detach()
        .cpu()
    )


def _parameter_audit(
    context: Any,
    cache: IntegralBundleCache,
    molecule: Any,
    direction: Any,
    density_args: argparse.Namespace,
    parameter_fd_steps: list[float],
) -> dict[str, Any]:
    print("verification: building trainable relaxed-HVP graph", flush=True)
    consume_implicit_response_diagnostics()
    graph_hvp, graph_density_norms = _direction_prediction(
        context, cache, molecule, direction, create_graph=True
    )
    _require_finite(graph_hvp, "training-graph HVP")

    generator = torch.Generator(device=graph_hvp.device)
    generator.manual_seed(20260723)
    probe = torch.randn(
        graph_hvp.shape,
        generator=generator,
        device=graph_hvp.device,
        dtype=graph_hvp.dtype,
    )
    probe = probe / torch.linalg.vector_norm(probe)
    scalar = torch.sum(graph_hvp * probe)
    named_parameters = [
        (name, parameter)
        for name, parameter in context.model.net.named_parameters()
        if parameter.requires_grad
    ]
    print("verification: solving implicit parameter response", flush=True)
    gradients = torch.autograd.grad(
        scalar,
        [parameter for _, parameter in named_parameters],
        allow_unused=True,
    )
    implicit_diagnostics = consume_implicit_response_diagnostics()
    if not implicit_diagnostics:
        raise RuntimeError(
            "No implicit density parameter-response solve was recorded; "
            "refusing a detached/proxy HVP"
        )
    failed_solves = [
        item for item in implicit_diagnostics if not bool(item["converged"])
    ]
    if failed_solves:
        raise RuntimeError(f"Implicit response failed: {failed_solves}")

    selected: tuple[str, torch.Tensor, torch.Tensor, int, float] | None = None
    total_gradient_norm_squared = 0.0
    for (name, parameter), gradient in zip(
        named_parameters, gradients, strict=True
    ):
        if gradient is None:
            continue
        _require_finite(gradient, f"parameter gradient {name}")
        total_gradient_norm_squared += float(
            torch.sum(gradient.detach() ** 2).cpu()
        )
        flat_gradient = gradient.detach().reshape(-1)
        index = int(torch.argmax(torch.abs(flat_gradient)).cpu())
        magnitude = float(torch.abs(flat_gradient[index]).cpu())
        if selected is None or magnitude > selected[4]:
            selected = (name, parameter, gradient, index, magnitude)
    if selected is None or selected[4] == 0.0:
        raise RuntimeError("Complete-total relaxed HVP parameter gradient is zero")
    print(
        "verification: implicit response complete "
        f"({len(implicit_diagnostics)} solves)",
        flush=True,
    )

    parameter_name, parameter, gradient, flat_index, _ = selected
    analytic = float(gradient.detach().reshape(-1)[flat_index].cpu())
    original_value = float(parameter.detach().reshape(-1)[flat_index].cpu())
    plus_start = molecule.displaced[(direction.index, "plus")].coefficients
    minus_start = molecule.displaced[(direction.index, "minus")].coefficients

    fd_rows = []
    try:
        for step in parameter_fd_steps:
            print(f"verification: parameter finite difference h={step:g}", flush=True)
            scalar_values = {}
            hvp_values = {}
            point_rows = {}
            for sign, side in ((1.0, "parameter_plus"), (-1.0, "parameter_minus")):
                with torch.no_grad():
                    parameter.reshape(-1)[flat_index] = original_value + sign * step
                fresh_hvp, fresh_points = _fresh_hvp(
                    context,
                    cache,
                    molecule,
                    direction,
                    density_args,
                    plus_start,
                    minus_start,
                )
                _require_finite(fresh_hvp, f"{side} fresh HVP")
                hvp_values[side] = fresh_hvp
                scalar_values[side] = float(
                    torch.sum(fresh_hvp * probe.detach().cpu())
                )
                point_rows[side] = fresh_points
            finite_difference = (
                scalar_values["parameter_plus"]
                - scalar_values["parameter_minus"]
            ) / (2.0 * step)
            absolute_error = abs(analytic - finite_difference)
            relative_error = absolute_error / max(
                abs(analytic), abs(finite_difference), 1.0e-12
            )
            fd_rows.append(
                {
                    "parameter_step": step,
                    "analytic_gradient": analytic,
                    "finite_difference_gradient": finite_difference,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "parameter_plus_scalar": scalar_values["parameter_plus"],
                    "parameter_minus_scalar": scalar_values["parameter_minus"],
                    "parameter_plus_hvp_norm": float(
                        torch.linalg.vector_norm(
                            hvp_values["parameter_plus"]
                        )
                    ),
                    "parameter_minus_hvp_norm": float(
                        torch.linalg.vector_norm(
                            hvp_values["parameter_minus"]
                        )
                    ),
                    "points": point_rows,
                }
            )
    finally:
        with torch.no_grad():
            parameter.reshape(-1)[flat_index] = original_value

    return {
        "graph_hvp": graph_hvp.detach().cpu(),
        "probe": probe.detach().cpu(),
        "audit_scalar": float(scalar.detach().cpu()),
        "graph_density_gradient_norms": [
            float(value.detach().cpu()) for value in graph_density_norms
        ],
        "implicit_response_solve_count": len(implicit_diagnostics),
        "implicit_response_diagnostics": implicit_diagnostics,
        "parameter_name": parameter_name,
        "parameter_flat_index": flat_index,
        "parameter_value": original_value,
        "parameter_analytic_gradient": analytic,
        "total_parameter_gradient_norm": math.sqrt(total_gradient_norm_squared),
        "parameter_fd": fd_rows,
        "best_parameter_fd_relative_error": min(
            row["relative_error"] for row in fd_rows
        ),
    }


def _common_args(
    output_dir: Path,
    protocol: dict[str, Any],
    device: str,
    implicit_diagonal_probes: int | None,
    implicit_max_iterations: int | None,
) -> argparse.Namespace:
    density = protocol["numerics"]["density"]
    implicit = protocol["numerics"]["implicit_parameter_response"]
    return argparse.Namespace(
        output_dir=output_dir,
        device=device,
        transform_device="cpu",
        charge=0,
        negative_integrated_density_penalty_weight=0.0,
        base_initialization=density["initialization"],
        density_lr=float(density["stage1_adam_lr"]),
        density_max_cycles=int(density["stage1_max_cycles"]),
        density_first_threshold=float(density["stage1_threshold"]),
        density_fallback_lr=float(density["fallback_adam_lr"]),
        density_fallback_max_cycles=int(density["fallback_max_cycles"]),
        density_fallback_threshold=float(density["fallback_threshold"]),
        density_strict_threshold=float(
            density["strict_projected_gradient_threshold"]
        ),
        lbfgs_max_iterations=500,
        newton_max_iterations=6,
        displacement=float(protocol["numerics"]["displacement_bohr"]),
        integral_derivative_step=float(
            protocol["numerics"]["integral_derivative_step_bohr"]
        ),
        integral_derivative_workers=4,
        integral_cache_entries=6,
        implicit_density_parameter_response=True,
        implicit_response_tolerance=float(implicit["tolerance"]),
        implicit_response_max_iterations=(
            int(implicit["max_iterations"])
            if implicit_max_iterations is None
            else implicit_max_iterations
        ),
        implicit_response_damping=float(implicit["damping"]),
        implicit_response_diagonal_probes=(
            int(implicit["diagonal_probes"])
            if implicit_diagonal_probes is None
            else implicit_diagonal_probes
        ),
        implicit_response_warm_start=bool(implicit["warm_start"]),
        density_response_unroll_steps=0,
        density_response_unroll_lr=1.0e-3,
        connect_lagrange_multiplier_response=True,
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20260723)
    np.random.seed(20260723)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol["validation_access_allowed"] or protocol["test100_access_allowed"]:
        raise ValueError("Protocol does not freeze validation and Test100")
    source = protocol["source"]
    source_checkpoint = Path(source["checkpoint"])
    _require_hash(
        source_checkpoint, source["checkpoint_sha256"], "source checkpoint"
    )
    source_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    if source_payload.get("complete_total_capacity") is not None:
        raise ValueError("Source checkpoint is not untouched original-A")

    parent_manifest_path = Path(
        protocol["parent_sets"]["stable5"]["source_manifest"]
    )
    _require_hash(
        parent_manifest_path,
        protocol["parent_sets"]["stable5"]["source_manifest_sha256"],
        "stable5 parent manifest",
    )
    parent_manifest = json.loads(parent_manifest_path.read_text())
    validation_frozen = (
        parent_manifest.get("validation_accessed") is False
        or parent_manifest.get("stage1_role")
        == "capacity_only_not_validation"
    )
    if (
        not validation_frozen
        or parent_manifest.get("test100_accessed") is not False
        or int(parent_manifest.get("test100_label_reads", 0)) != 0
    ):
        raise ValueError("Stable5 parent manifest accessed validation or Test100")

    direction_manifest = json.loads(args.direction_manifest.read_text())
    if (
        direction_manifest.get("validation_accessed") is not False
        or direction_manifest.get("test100_accessed") is not False
    ):
        raise ValueError("Direction manifest accessed validation or Test100")
    parent_entries = {
        str(row["molecule_id"]): row for row in parent_manifest["parents"]
    }
    direction_entries = {
        str(row["molecule_id"]): row for row in direction_manifest["parents"]
    }
    if args.molecule not in parent_entries or args.molecule not in direction_entries:
        raise ValueError(f"{args.molecule} is absent from frozen stable5 artifacts")

    molecule = _load_molecule(
        parent_entries[args.molecule],
        low_mode_count=0,
        direction_entry=direction_entries[args.molecule],
        direction_role="all",
    )
    if args.mode == "derivative":
        if args.direction_index >= len(molecule.directions):
            raise ValueError("direction index is out of range")
        molecule.directions = [molecule.directions[args.direction_index]]

    run = (
        f"{source['name']}={source['run_dir']}={source['checkpoint']}"
    )
    common = _common_args(
        args.output_dir,
        protocol,
        args.device,
        args.implicit_diagonal_probes,
        args.implicit_max_iterations,
    )
    density_args = _density_namespace(common)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    context = _load_context(_parse_run(run), density_args, device)
    if context.model.net.__class__.__name__ != "Graphformer":
        raise TypeError(
            f"Expected Graphformer, got {context.model.net.__class__.__name__}"
        )
    cache = IntegralBundleCache(context, common)
    print(
        f"verification: relaxing {len(molecule.directions)} direction(s) "
        f"to {density_args.density_strict_threshold:g}",
        flush=True,
    )
    density_rows = _refresh_densities(
        context, [molecule], density_args, refresh_index=0
    )
    print("verification: initial strict relaxation complete", flush=True)

    result: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "mode": args.mode,
        "molecule_id": args.molecule,
        "source": source["name"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "direction_manifest": str(args.direction_manifest),
        "direction_manifest_sha256": _sha256(args.direction_manifest),
        "direction_count": len(molecule.directions),
        "strict_density_threshold": density_args.density_strict_threshold,
        "displacement_bohr": density_args.displacement,
        "scalar_owner": protocol["definitions"]["scalar_owner"],
        "hvp_definition": protocol["definitions"][
            "complete_total_relaxed_hvp"
        ],
        "implicit_density_parameter_response": True,
        "implicit_response_tolerance": common.implicit_response_tolerance,
        "implicit_response_max_iterations": (
            common.implicit_response_max_iterations
        ),
        "implicit_response_diagonal_probes": (
            common.implicit_response_diagonal_probes
        ),
        "density_response_unroll_steps": 0,
        "proxy_fallback_used": False,
        "independent_force_or_hessian_head_used": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "density_points": density_rows,
    }

    if args.mode == "derivative":
        direction = molecule.directions[0]
        audit = _parameter_audit(
            context,
            cache,
            molecule,
            direction,
            density_args,
            [
                float(value)
                for value in protocol["verification"]["parameter_fd_steps"]
            ],
        )
        fresh_hvp, fresh_rows = _fresh_hvp(
            context,
            cache,
            molecule,
            direction,
            density_args,
            molecule.displaced[(direction.index, "plus")].coefficients,
            molecule.displaced[(direction.index, "minus")].coefficients,
        )
        graph_hvp = audit.pop("graph_hvp")
        graph_vs_fresh = _relative_l2(graph_hvp, fresh_hvp)
        result.update(
            {
                "direction_index": direction.index,
                "direction_kind": direction.kind,
                "graph_hvp_norm": float(torch.linalg.vector_norm(graph_hvp)),
                "fresh_hvp_norm": float(torch.linalg.vector_norm(fresh_hvp)),
                "graph_vs_fresh_strict_fd_relative_l2": graph_vs_fresh,
                "fresh_strict_points": fresh_rows,
                **audit,
            }
        )
        np.savez_compressed(
            args.output_dir / "derivative_arrays.npz",
            graph_hvp=graph_hvp.numpy(),
            fresh_hvp=fresh_hvp.numpy(),
            probe=result["probe"].numpy(),
            direction=direction.vector,
            pbe_hvp=direction.target_hvp,
        )
        result.pop("probe")
        graph_gate = float(
            protocol["verification"][
                "graph_vs_fresh_strict_fd_relative_l2_max"
            ]
        )
        parameter_gate = float(
            protocol["verification"][
                "parameter_gradient_best_relative_error_max"
            ]
        )
        result["graph_vs_fresh_gate_passed"] = graph_vs_fresh <= graph_gate
        result["parameter_gradient_gate_passed"] = (
            result["best_parameter_fd_relative_error"] <= parameter_gate
        )
        result["passed"] = bool(
            result["graph_vs_fresh_gate_passed"]
            and result["parameter_gradient_gate_passed"]
        )
    else:
        metrics = _full_hessian_metrics(
            context, cache, [molecule], step=0
        )[0]
        result["full_hessian_metrics"] = metrics
        asym_gate = float(
            protocol["verification"]["full_basis_asym_over_sym_max"]
        )
        result["symmetry_gate_passed"] = (
            bool(metrics["full_hessian_complete"])
            and math.isfinite(
                metrics["antisymmetric_over_symmetric_frobenius"]
            )
            and metrics["antisymmetric_over_symmetric_frobenius"] <= asym_gate
        )
        result["passed"] = result["symmetry_gate_passed"]

    result.update(
        {
            "integral_bundle_build_count": cache.build_count,
            "integral_bundle_cache_hit_count": cache.hit_count,
            "wall_time_s": time.perf_counter() - started,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not result["passed"]:
        raise RuntimeError(
            f"Fail-closed {args.mode} verification did not pass; see "
            f"{args.output_dir / 'summary.json'}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecule", default="0028399")
    parser.add_argument("--direction-index", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("derivative", "full-basis"), default="derivative"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--implicit-diagonal-probes", type=int, default=None)
    parser.add_argument("--implicit-max-iterations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    try:
        summary = verify(arguments)
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        (arguments.output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "validation_accessed": False,
                    "test100_accessed": False,
                    "proxy_fallback_used": False,
                    "passed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    main()
