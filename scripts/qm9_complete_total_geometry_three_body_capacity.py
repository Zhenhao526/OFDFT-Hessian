#!/usr/bin/env python3
"""Audit conservative three-body residual capacity on a frozen training parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from mldft.ml.models.components.three_body_geometry_residual import (
    build_triplet_groups,
    make_three_body_feature_function,
)


def constrained_row_space_fit(
    design: np.ndarray,
    target: np.ndarray,
    constraints: np.ndarray,
    relative_svd_tolerance: float,
    constraint_target: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit through the design row space after removing the constraint row space."""
    constraint_u, constraint_singular, constraint_vt = np.linalg.svd(
        constraints, full_matrices=False
    )
    constraint_threshold = relative_svd_tolerance * max(constraints.shape) * (
        constraint_singular[0] if constraint_singular.size else 1.0
    )
    constraint_rank = int(np.sum(constraint_singular > constraint_threshold))
    row_basis = constraint_vt[:constraint_rank]
    if constraint_target is None:
        constraint_target = np.zeros(constraints.shape[0], dtype=np.float64)
    constraint_target = np.asarray(constraint_target, dtype=np.float64)
    if constraint_target.shape != (constraints.shape[0],):
        raise ValueError("constraint target has incompatible shape")
    particular = constraint_vt[:constraint_rank].T @ (
        (constraint_u[:, :constraint_rank].T @ constraint_target)
        / constraint_singular[:constraint_rank]
    )
    null_design = design - (design @ row_basis.T) @ row_basis
    design_u, design_singular, design_vt = np.linalg.svd(
        null_design, full_matrices=False
    )
    design_threshold = relative_svd_tolerance * max(null_design.shape) * (
        design_singular[0] if design_singular.size else 1.0
    )
    design_rank = int(np.sum(design_singular > design_threshold))
    residual_target = target - design @ particular
    coefficients = particular + design_vt[:design_rank].T @ (
        (design_u[:, :design_rank].T @ residual_target)
        / design_singular[:design_rank]
    )
    nonzero = design_singular[:design_rank]
    return coefficients, {
        "constraint_rank": constraint_rank,
        "null_space_dimension": int(design.shape[1] - constraint_rank),
        "reduced_design_rank": design_rank,
        "reduced_design_condition": (
            float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
        ),
    }


def hessian_metrics(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float | None]:
    difference = prediction - reference
    reference_norm = max(np.linalg.norm(reference), np.finfo(float).tiny)
    metrics: dict[str, float | None] = {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / reference_norm),
        "symmetry_max_abs": None,
        "antisymmetric_over_symmetric_frobenius": None,
    }
    if prediction.shape[0] == prediction.shape[1]:
        symmetric = 0.5 * (prediction + prediction.T)
        antisymmetric = 0.5 * (prediction - prediction.T)
        metrics["symmetry_max_abs"] = float(
            np.max(np.abs(prediction - prediction.T))
        )
        metrics["antisymmetric_over_symmetric_frobenius"] = float(
            np.linalg.norm(antisymmetric)
            / max(np.linalg.norm(symmetric), np.finfo(float).tiny)
        )
    return metrics


def write_hessian_comparison_plot(
    output_path: Path,
    prediction: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    correction: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = (
        ("PBE reference", reference),
        ("Baseline error", prediction - reference),
        ("Corrected error", corrected - reference),
        ("Scalar correction", correction),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    reference_limit = max(np.max(np.abs(reference)), np.finfo(float).tiny)
    error_limit = max(
        np.max(np.abs(prediction - reference)),
        np.max(np.abs(corrected - reference)),
        np.max(np.abs(correction)),
        np.finfo(float).tiny,
    )
    for axis, (title, matrix) in zip(axes.flat, matrices, strict=True):
        limit = reference_limit if title == "PBE reference" else error_limit
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(title)
        axis.set_xlabel("Cartesian column")
        axis.set_ylabel("Cartesian row")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("manifest does not certify frozen Test100")
    entries = {str(row["molecule_id"]): row for row in manifest["parents"]}
    if args.molecule_id not in entries:
        raise ValueError("molecule is absent from frozen Stage-1 manifest")
    entry = entries[args.molecule_id]
    label = zarr.open(entry["label_path"], mode="r")
    atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
    positions_numpy = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
    pbe_force = np.asarray(label["metadata/pbe_derivatives/forces"], dtype=np.float64)
    energy_trace = np.asarray(label["ks_labels/energies/e_tot"], dtype=np.float64)
    has_energy = np.asarray(
        label["ks_labels/energies/has_energy_label"], dtype=np.bool_
    )
    pbe_total_energy = float(energy_trace[has_energy][-1])
    with np.load(args.capacity_array) as payload:
        prediction = np.asarray(payload["predicted_hessian"], dtype=np.float64)
        reference = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        predicted_base_force = (
            np.asarray(payload["predicted_base_force"], dtype=np.float64)
            if "predicted_base_force" in payload
            else None
        )
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    reference = reference[:, : prediction.shape[1]]

    groups = build_triplet_groups(atomic_numbers)
    centers = np.linspace(args.center_min, args.center_max, args.center_count)
    feature_function, feature_keys = make_three_body_feature_function(
        groups, centers, args.sigma, args.angular_order
    )
    positions = torch.tensor(positions_numpy, dtype=torch.float64, requires_grad=True)
    energies = feature_function(positions)
    energy_jacobian = torch.func.jacfwd(feature_function)(positions).reshape(
        energies.numel(), -1
    )
    hessian_columns = []
    for column in range(prediction.shape[1]):
        direction = torch.zeros_like(positions).reshape(-1)
        direction[column] = 1.0
        direction = direction.reshape_as(positions)
        plus_jacobian = torch.func.jacfwd(feature_function)(
            positions + args.derivative_step * direction
        ).reshape(energies.numel(), -1)
        minus_jacobian = torch.func.jacfwd(feature_function)(
            positions - args.derivative_step * direction
        ).reshape(energies.numel(), -1)
        hessian_columns.append(
            ((plus_jacobian - minus_jacobian) / (2 * args.derivative_step)).T
        )
    design_tensor = torch.stack(hessian_columns, dim=1)
    design = design_tensor.detach().cpu().numpy().reshape(-1, energies.numel())
    target = (reference - prediction).reshape(-1)
    constraints = np.concatenate(
        (
            energies.detach().cpu().numpy()[None, :],
            energy_jacobian.detach().cpu().numpy().T,
        ),
        axis=0,
    )
    constraint_target = None
    if args.baseline_total_energy is not None:
        if predicted_base_force is None:
            raise ValueError(
                "energy/force fitting requires predicted_base_force in the capacity array"
            )
        force_correction = pbe_force - predicted_base_force
        constraint_target = np.concatenate(
            (
                np.asarray([pbe_total_energy - args.baseline_total_energy]),
                -force_correction.reshape(-1),
            )
        )
    coefficients, solver = constrained_row_space_fit(
        design,
        target,
        constraints,
        args.svd_tolerance,
        constraint_target=constraint_target,
    )
    correction = (design @ coefficients).reshape(prediction.shape)
    corrected = prediction + correction
    baseline_metrics = hessian_metrics(prediction, reference)
    corrected_metrics = hessian_metrics(corrected, reference)

    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float64)

    def scalar_adapter(candidate_positions: torch.Tensor) -> torch.Tensor:
        return torch.dot(feature_function(candidate_positions), coefficient_tensor)

    adapter_energy = scalar_adapter(positions)
    adapter_gradient = torch.autograd.grad(adapter_energy, positions)[0]
    adapter_hessian = torch.func.hessian(scalar_adapter)(positions).reshape(
        positions.numel(), positions.numel()
    )
    reference_norm = max(np.linalg.norm(reference), np.finfo(float).tiny)
    result: dict[str, object] = {
        "definition": (
            "train-only invariant scalar three-body RBF-Legendre geometry residual, "
            + (
                "constrained to the PBE anchor energy and force"
                if args.baseline_total_energy is not None
                else "constrained to zero anchor energy and force"
            )
        ),
        "molecule_id": args.molecule_id,
        "natoms": int(atomic_numbers.size),
        "column_count": int(prediction.shape[1]),
        "triplet_group_count": len(groups),
        "feature_count": len(feature_keys),
        "center_count": args.center_count,
        "center_range_bohr": [args.center_min, args.center_max],
        "sigma_bohr": args.sigma,
        "angular_order": args.angular_order,
        "derivative_step_bohr": args.derivative_step,
        "source_capacity_array": str(args.capacity_array.resolve()),
        "fit_anchor_energy_force": args.baseline_total_energy is not None,
        "baseline_relative_frobenius": float(
            np.linalg.norm(prediction - reference) / reference_norm
        ),
        "corrected_relative_frobenius": float(
            np.linalg.norm(corrected - reference) / reference_norm
        ),
        "adapter_energy_abs": float(abs(adapter_energy.detach().cpu())),
        "adapter_force_max_abs": float(
            torch.max(torch.abs(adapter_gradient)).detach().cpu()
        ),
        "adapter_hessian_symmetry_max_abs": float(
            torch.max(torch.abs(adapter_hessian - adapter_hessian.T)).detach().cpu()
        ),
        "adapter_hvp_fd_vs_autograd_max_abs": float(
            torch.max(
                torch.abs(
                    adapter_hessian[:, : prediction.shape[1]]
                    - torch.as_tensor(
                        correction, dtype=adapter_hessian.dtype
                    )
                )
            ).detach().cpu()
        ),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "constraint_residual_max_abs": float(
            np.max(
                np.abs(
                    constraints @ coefficients
                    - (
                        constraint_target
                        if constraint_target is not None
                        else np.zeros(constraints.shape[0])
                    )
                )
            )
        ),
        "pbe_total_energy_hartree": pbe_total_energy,
        "baseline_total_energy_hartree": args.baseline_total_energy,
        "corrected_total_energy_hartree": (
            float(args.baseline_total_energy + adapter_energy.detach().cpu())
            if args.baseline_total_energy is not None
            else None
        ),
        "baseline_force_mae_hartree_per_bohr": (
            float(np.mean(np.abs(predicted_base_force - pbe_force)))
            if predicted_base_force is not None
            else None
        ),
        "corrected_force_mae_hartree_per_bohr": (
            float(
                np.mean(
                    np.abs(
                        predicted_base_force
                        - adapter_gradient.detach().cpu().numpy()
                        - pbe_force
                    )
                )
            )
            if predicted_base_force is not None
            else None
        ),
        "baseline_metrics": baseline_metrics,
        "corrected_metrics": corrected_metrics,
        "adapter_force_fd_symmetry_max_abs": (
            float(np.max(np.abs(correction - correction.T)))
            if correction.shape[0] == correction.shape[1]
            else None
        ),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **solver,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "geometry_three_body_capacity.npz",
        coefficients=coefficients,
        predicted_hessian=prediction,
        correction_hessian_columns=correction,
        corrected_hessian=corrected,
        pbe_hessian_columns=reference,
        adapter_full_hessian=adapter_hessian.detach().cpu().numpy(),
        centers_bohr=centers,
        feature_keys=np.asarray(feature_keys, dtype=np.int64),
    )
    write_hessian_comparison_plot(
        args.output_dir / "hessian_capacity_comparison.png",
        prediction,
        corrected,
        reference,
        correction,
    )
    (args.output_dir / "geometry_three_body_capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument("--capacity-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-min", type=float, default=0.5)
    parser.add_argument("--center-max", type=float, default=8.0)
    parser.add_argument("--center-count", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--angular-order", type=int, default=3)
    parser.add_argument("--derivative-step", type=float, default=1.0e-4)
    parser.add_argument("--baseline-total-energy", type=float, default=None)
    parser.add_argument("--svd-tolerance", type=float, default=1.0e-10)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
