#!/usr/bin/env python3
"""Read-only fixed/response/relaxed Hessian decomposition for canonical QM9.

This wrapper deliberately leaves the frozen capacity runner unchanged.  It
captures the two tensors that the runner already computes for every internal
direction, assembles their Cartesian matrices, and writes component metrics
next to the ordinary relaxed-Hessian audit output.

The wrapper is an evaluation utility, not a registered training protocol.  It
overrides only the in-memory registered step count to zero so that the frozen
runner performs its initial full-Hessian evaluation without parameter updates.
The override and both runner/wrapper hashes are recorded in audit_metadata.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from scripts import qm9_complete_total_capacity_train as runner


AUDIT_DEFINITION = "fixed_response_relaxed_internal_hessian_decomposition_v1"
_captured_components: list[dict[str, Any]] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_direction_prediction(
    context: Any,
    cache: Any,
    molecule: Any,
    direction: Any,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, Any]]:
    (
        center,
        position_direction,
        response_system,
        response,
        diagnostics,
    ) = runner._analytic_center_response(
        context,
        cache,
        molecule,
        direction,
        create_graph=create_graph,
    )
    started = time.perf_counter()
    partial_hvp = response_system.partial_position_hvp(
        position_direction, create_graph=create_graph
    )
    response_correction = response_system.response_correction(
        response.density_response,
        response.multiplier_response,
        create_graph=create_graph,
    )
    relaxed_hvp = partial_hvp + response_correction
    runner._synchronize(relaxed_hvp)
    if not bool(torch.isfinite(relaxed_hvp).all()):
        raise RuntimeError("component audit produced a non-finite relaxed HVP")
    diagnostics["relaxed_hvp_seconds"] = time.perf_counter() - started
    diagnostics["density_response_norm"] = float(
        torch.linalg.vector_norm(response.density_response.detach()).cpu()
    )
    diagnostics.update(
        runner._response_cancellation_diagnostics(
            partial_hvp, response_correction
        )
    )
    _captured_components.append(
        {
            "molecule_id": molecule.molecule_id,
            "direction_index": int(direction.index),
            "partial_hvp": partial_hvp.detach().cpu().reshape(-1),
            "response_correction": (
                response_correction.detach().cpu().reshape(-1)
            ),
            "relaxed_hvp": relaxed_hvp.detach().cpu().reshape(-1),
        }
    )
    return (
        relaxed_hvp,
        [center.projected_density_gradient_norm],
        diagnostics,
    )


def _prefixed_metrics(
    prefix: str, prediction: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    values = runner._to_float_metrics(
        runner.hessian_error_metrics(prediction, reference)
    )
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _component_full_hessian_metrics(
    context: Any,
    cache: Any,
    molecules: list[Any],
    step: int,
) -> list[dict[str, Any]]:
    del _captured_components[:]
    original_direction = runner._analytic_relaxed_direction_prediction
    runner._analytic_relaxed_direction_prediction = _capture_direction_prediction
    try:
        rows = _original_full_hessian_metrics(
            context, cache, molecules, step
        )
    finally:
        runner._analytic_relaxed_direction_prediction = original_direction

    cursor = 0
    component_dir = cache.args.output_dir / "component_hessian_arrays"
    component_dir.mkdir(parents=True, exist_ok=True)
    for molecule, row in zip(molecules, rows, strict=True):
        count = len(molecule.directions)
        captured = _captured_components[cursor : cursor + count]
        cursor += count
        if len(captured) != count or any(
            item["molecule_id"] != molecule.molecule_id for item in captured
        ):
            raise RuntimeError("component capture order does not match molecules")
        expected_indices = [int(item.index) for item in molecule.directions]
        actual_indices = [int(item["direction_index"]) for item in captured]
        if actual_indices != expected_indices:
            raise RuntimeError("component capture order does not match directions")

        direction_matrix = torch.as_tensor(
            np.stack(
                [item.vector.reshape(-1) for item in molecule.directions]
            ),
            dtype=captured[0]["partial_hvp"].dtype,
        )
        projector = direction_matrix.T @ direction_matrix

        def assemble(key: str) -> torch.Tensor:
            columns = torch.stack([item[key] for item in captured], dim=1)
            return projector @ (columns @ direction_matrix) @ projector

        fixed = assemble("partial_hvp")
        response = assemble("response_correction")
        relaxed = assemble("relaxed_hvp")
        reference_cartesian = torch.as_tensor(
            molecule.pbe_hessian, dtype=fixed.dtype
        )
        reference = projector @ reference_cartesian @ projector
        ordinary_path = (
            cache.args.output_dir
            / "hessian_arrays"
            / f"step_{step:07d}_{molecule.molecule_id}.npz"
        )
        with np.load(ordinary_path) as ordinary:
            runner_relaxed = torch.as_tensor(
                ordinary["predicted_hessian"], dtype=fixed.dtype
            )
        closure = runner_relaxed - fixed - response
        relaxed_norm = torch.linalg.vector_norm(runner_relaxed).clamp_min(
            torch.finfo(runner_relaxed.dtype).tiny
        )
        fixed_norm = torch.linalg.vector_norm(fixed)
        response_norm = torch.linalg.vector_norm(response)
        cosine = torch.sum(fixed * response) / (
            fixed_norm.clamp_min(torch.finfo(fixed.dtype).tiny)
            * response_norm.clamp_min(torch.finfo(response.dtype).tiny)
        )
        row.update(
            {
                "component_audit_definition": AUDIT_DEFINITION,
                "component_closure_max_abs": float(
                    torch.max(torch.abs(closure)).cpu()
                ),
                "component_closure_relative_frobenius": float(
                    (torch.linalg.vector_norm(closure) / relaxed_norm).cpu()
                ),
                "fixed_density_frobenius_norm": float(fixed_norm.cpu()),
                "response_correction_frobenius_norm": float(
                    response_norm.cpu()
                ),
                "relaxed_frobenius_norm": float(relaxed_norm.cpu()),
                "response_fraction_of_relaxed_frobenius": float(
                    (response_norm / relaxed_norm).cpu()
                ),
                "fixed_response_cosine": float(cosine.cpu()),
                "component_cancellation_index": float(
                    ((fixed_norm + response_norm) / relaxed_norm).cpu()
                ),
                **_prefixed_metrics(
                    "fixed_density_vs_reference", fixed, reference
                ),
                **_prefixed_metrics(
                    "relaxed_reconstructed_vs_reference", relaxed, reference
                ),
                **_prefixed_metrics(
                    "fixed_density_vs_relaxed", fixed, runner_relaxed
                ),
            }
        )
        np.savez_compressed(
            component_dir / f"step_{step:07d}_{molecule.molecule_id}.npz",
            fixed_density_hessian=fixed.numpy(),
            density_response_correction_hessian=response.numpy(),
            relaxed_reconstructed_hessian=relaxed.numpy(),
            relaxed_runner_hessian=runner_relaxed.numpy(),
            comparison_reference_hessian=reference.numpy(),
            raw_pbe_hessian=(
                molecule.raw_pbe_hessian
                if molecule.raw_pbe_hessian is not None
                else molecule.pbe_hessian
            ),
            direction_matrix=direction_matrix.numpy(),
            direction_indices=np.asarray(actual_indices, dtype=np.int64),
        )
    if cursor != len(_captured_components):
        raise RuntimeError("unconsumed component captures remain")
    return rows


def _audit_safe_load(stream: Any) -> Any:
    payload = _original_safe_load(stream)
    if (
        isinstance(payload, dict)
        and payload.get("protocol_id")
        == "qm9_graphformer_egfh_torch_autograd_relaxed_hvp_v9"
    ):
        payload = copy.deepcopy(payload)
        formal = payload.setdefault("formal_training", {})
        formal["steps"] = 0
        formal["resume_additional_steps"] = 0
    return payload


def main() -> None:
    args = runner.parse_args()
    metadata = {
        "audit_definition": AUDIT_DEFINITION,
        "non_normative_read_only_evaluation": True,
        "parameter_updates": 0,
        "in_memory_schedule_override": {
            "formal_training.steps": 0,
            "formal_training.resume_additional_steps": 0,
        },
        "wrapper_path": Path(__file__).resolve().as_posix(),
        "wrapper_sha256": _sha256(Path(__file__).resolve()),
        "frozen_runner_path": Path(runner.__file__).resolve().as_posix(),
        "frozen_runner_sha256": _sha256(Path(runner.__file__).resolve()),
        "protocol_path": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "root_source_checkpoint_sha256": args.root_source_checkpoint_sha256,
        "argv": sys.argv,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    summary = runner.train(args)
    summary["component_audit"] = metadata
    (args.output_dir / "component_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


_original_safe_load = runner.yaml.safe_load
_original_full_hessian_metrics = runner._analytic_full_hessian_metrics
runner.yaml.safe_load = _audit_safe_load
runner._analytic_full_hessian_metrics = _component_full_hessian_metrics


if __name__ == "__main__":
    main()
