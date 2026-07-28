#!/usr/bin/env python3
"""Audit conservative four-body torsion residual capacity on a frozen train parent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zarr

try:
    from scripts.qm9_complete_total_geometry_three_body_capacity import (
        constrained_row_space_fit,
        hessian_metrics,
        write_hessian_comparison_plot,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qm9_complete_total_geometry_three_body_capacity import (
        constrained_row_space_fit,
        hessian_metrics,
        write_hessian_comparison_plot,
    )


_COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class ChainGroup:
    element_key: tuple[int, int, int, int]
    first_indices: tuple[int, ...]
    second_indices: tuple[int, ...]
    third_indices: tuple[int, ...]
    fourth_indices: tuple[int, ...]
    reversal_symmetric: bool


def build_bonded_chain_groups(
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    bond_scale: float,
) -> tuple[list[ChainGroup], list[tuple[int, int]]]:
    """Build unique length-three bonded paths with reversal-canonical element order."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions_bohr = np.asarray(positions_bohr, dtype=np.float64)
    if bond_scale <= 0:
        raise ValueError("bond_scale must be positive")
    bonds = []
    neighbors = [[] for _ in range(atomic_numbers.size)]
    for first in range(atomic_numbers.size):
        for second in range(first + 1, atomic_numbers.size):
            radius = (
                _COVALENT_RADII_ANGSTROM[int(atomic_numbers[first])]
                + _COVALENT_RADII_ANGSTROM[int(atomic_numbers[second])]
            ) * _ANGSTROM_TO_BOHR
            distance = np.linalg.norm(positions_bohr[first] - positions_bohr[second])
            if distance <= bond_scale * radius:
                bonds.append((first, second))
                neighbors[first].append(second)
                neighbors[second].append(first)

    grouped: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for second, third in bonds:
        for first in neighbors[second]:
            if first == third:
                continue
            for fourth in neighbors[third]:
                if fourth in {second, first}:
                    continue
                chain = (first, second, third, fourth)
                reverse_chain = tuple(reversed(chain))
                element_key = tuple(int(atomic_numbers[index]) for index in chain)
                reverse_key = tuple(reversed(element_key))
                if reverse_key < element_key or (
                    reverse_key == element_key and reverse_chain < chain
                ):
                    chain = reverse_chain
                    element_key = reverse_key
                grouped.setdefault(element_key, []).append(chain)
    groups = [
        ChainGroup(
            element_key=key,
            first_indices=tuple(row[0] for row in rows),
            second_indices=tuple(row[1] for row in rows),
            third_indices=tuple(row[2] for row in rows),
            fourth_indices=tuple(row[3] for row in rows),
            reversal_symmetric=key == tuple(reversed(key)),
        )
        for key, rows in sorted(grouped.items())
    ]
    return groups, bonds


def _cosine_multiples(cosine: torch.Tensor, maximum_order: int) -> torch.Tensor:
    values = [torch.ones_like(cosine)]
    if maximum_order >= 1:
        values.append(cosine)
    for _ in range(2, maximum_order + 1):
        values.append(2.0 * cosine * values[-1] - values[-2])
    return torch.stack(values, dim=1)


def make_four_body_feature_function(
    groups: list[ChainGroup],
    centers_bohr: np.ndarray,
    sigma_bohr: float,
    torsion_order: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], list[tuple[int, ...]]]:
    if sigma_bohr <= 0 or torsion_order < 0:
        raise ValueError("sigma must be positive and torsion_order non-negative")
    centers_numpy = np.asarray(centers_bohr, dtype=np.float64)
    center_triples_by_group: list[list[tuple[int, int, int]]] = []
    feature_keys = []
    for group in groups:
        triples = []
        for first in range(centers_numpy.size):
            for second in range(centers_numpy.size):
                for third in range(centers_numpy.size):
                    triple = (first, second, third)
                    if group.reversal_symmetric and tuple(reversed(triple)) < triple:
                        continue
                    triples.append(triple)
                    for order in range(torsion_order + 1):
                        feature_keys.append((*group.element_key, *triple, order))
        center_triples_by_group.append(triples)

    def features(positions_bohr: torch.Tensor) -> torch.Tensor:
        centers = torch.as_tensor(
            centers_numpy, dtype=positions_bohr.dtype, device=positions_bohr.device
        )
        output = []
        for group, center_triples in zip(
            groups, center_triples_by_group, strict=True
        ):
            first = positions_bohr[list(group.first_indices)]
            second = positions_bohr[list(group.second_indices)]
            third = positions_bohr[list(group.third_indices)]
            fourth = positions_bohr[list(group.fourth_indices)]
            first_bond = second - first
            central_bond = third - second
            third_bond = fourth - third
            first_normal = torch.linalg.cross(first_bond, central_bond, dim=1)
            second_normal = torch.linalg.cross(central_bond, third_bond, dim=1)
            first_normal_squared = torch.sum(first_normal * first_normal, dim=1)
            second_normal_squared = torch.sum(second_normal * second_normal, dim=1)
            normal_floor = 64.0 * torch.finfo(positions_bohr.dtype).eps
            valid_torsion = (first_normal_squared > normal_floor**2) & (
                second_normal_squared > normal_floor**2
            )
            # A collinear chain has no defined torsion. Keep that feature constant instead
            # of differentiating norm(0), whose second derivative is NaN.
            safe_first_squared = torch.where(
                valid_torsion, first_normal_squared, torch.ones_like(first_normal_squared)
            )
            safe_second_squared = torch.where(
                valid_torsion,
                second_normal_squared,
                torch.ones_like(second_normal_squared),
            )
            cosine_value = torch.sum(first_normal * second_normal, dim=1) / (
                torch.sqrt(safe_first_squared) * torch.sqrt(safe_second_squared)
            )
            cosine = torch.where(
                valid_torsion, cosine_value, torch.zeros_like(cosine_value)
            )
            torsion = _cosine_multiples(cosine, torsion_order)
            distances = [
                torch.linalg.vector_norm(first_bond, dim=1),
                torch.linalg.vector_norm(central_bond, dim=1),
                torch.linalg.vector_norm(third_bond, dim=1),
            ]
            rbfs = [
                torch.exp(
                    -0.5 * ((distance[:, None] - centers[None, :]) / sigma_bohr) ** 2
                )
                for distance in distances
            ]
            radial_columns = []
            for first_center, second_center, third_center in center_triples:
                radial = (
                    rbfs[0][:, first_center]
                    * rbfs[1][:, second_center]
                    * rbfs[2][:, third_center]
                )
                reverse = (third_center, second_center, first_center)
                if group.reversal_symmetric and reverse != (
                    first_center,
                    second_center,
                    third_center,
                ):
                    radial = radial + (
                        rbfs[0][:, third_center]
                        * rbfs[1][:, second_center]
                        * rbfs[2][:, first_center]
                    )
                radial_columns.append(radial)
            radial_matrix = torch.stack(radial_columns, dim=1)
            output.append(
                torch.einsum("tp,tl->pl", radial_matrix, torsion).reshape(-1)
            )
        return torch.cat(output)

    return features, feature_keys


def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("test100_accessed") is not False:
        raise ValueError("manifest does not certify frozen Test100")
    entries = {str(row["molecule_id"]): row for row in manifest["parents"]}
    if args.molecule_id not in entries:
        raise ValueError("molecule is absent from frozen Stage-1 manifest")
    label = zarr.open(entries[args.molecule_id]["label_path"], mode="r")
    atomic_numbers = np.asarray(label["geometry/atomic_numbers"], dtype=np.int64)
    positions_numpy = np.asarray(label["geometry/atom_pos"], dtype=np.float64)
    with np.load(args.capacity_array) as payload:
        if "corrected_hessian" in payload:
            prediction = np.asarray(payload["corrected_hessian"], dtype=np.float64)
        else:
            prediction = np.asarray(payload["predicted_hessian"], dtype=np.float64)
        reference = np.asarray(payload["pbe_hessian_columns"], dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    reference = reference[:, : prediction.shape[1]]

    groups, bonds = build_bonded_chain_groups(
        atomic_numbers, positions_numpy, args.bond_scale
    )
    if not groups:
        raise ValueError("no bonded four-atom chains were found")
    centers = np.linspace(args.center_min, args.center_max, args.center_count)
    feature_function, feature_keys = make_four_body_feature_function(
        groups, centers, args.sigma, args.torsion_order
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
        plus = torch.func.jacfwd(feature_function)(
            positions + args.derivative_step * direction
        ).reshape(energies.numel(), -1)
        minus = torch.func.jacfwd(feature_function)(
            positions - args.derivative_step * direction
        ).reshape(energies.numel(), -1)
        hessian_columns.append(
            ((plus - minus) / (2.0 * args.derivative_step)).T
        )
    design_tensor = torch.stack(hessian_columns, dim=1)
    design = design_tensor.detach().cpu().numpy().reshape(-1, energies.numel())
    constraints = np.concatenate(
        (energies.detach().cpu().numpy()[None, :], energy_jacobian.detach().cpu().numpy().T),
        axis=0,
    )
    coefficients, solver = constrained_row_space_fit(
        design,
        (reference - prediction).reshape(-1),
        constraints,
        args.svd_tolerance,
    )
    correction = (design @ coefficients).reshape(prediction.shape)
    corrected = prediction + correction
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float64)

    def scalar_adapter(candidate_positions: torch.Tensor) -> torch.Tensor:
        return torch.dot(feature_function(candidate_positions), coefficient_tensor)

    adapter_energy = scalar_adapter(positions)
    adapter_gradient = torch.autograd.grad(adapter_energy, positions)[0]
    adapter_hessian = torch.func.hessian(scalar_adapter)(positions).reshape(
        positions.numel(), positions.numel()
    )
    result: dict[str, object] = {
        "definition": (
            "train-only parity-even scalar four-body torsion RBF residual, constrained "
            "to zero incremental anchor energy and force"
        ),
        "molecule_id": args.molecule_id,
        "natoms": int(atomic_numbers.size),
        "bond_count": len(bonds),
        "chain_group_count": len(groups),
        "feature_count": len(feature_keys),
        "center_count": args.center_count,
        "center_range_bohr": [args.center_min, args.center_max],
        "sigma_bohr": args.sigma,
        "torsion_order": args.torsion_order,
        "bond_scale": args.bond_scale,
        "derivative_step_bohr": args.derivative_step,
        "source_capacity_array": str(args.capacity_array.resolve()),
        "baseline_metrics": hessian_metrics(prediction, reference),
        "corrected_metrics": hessian_metrics(corrected, reference),
        "adapter_energy_abs": float(abs(adapter_energy.detach().cpu())),
        "adapter_force_max_abs": float(
            torch.max(torch.abs(adapter_gradient)).detach().cpu()
        ),
        "constraint_residual_max_abs": float(
            np.max(np.abs(constraints @ coefficients))
        ),
        "adapter_hessian_symmetry_max_abs": float(
            torch.max(torch.abs(adapter_hessian - adapter_hessian.T)).detach().cpu()
        ),
        "adapter_force_fd_symmetry_max_abs": float(
            np.max(np.abs(correction - correction.T))
        ),
        "adapter_hvp_fd_vs_autograd_max_abs": float(
            torch.max(
                torch.abs(adapter_hessian - torch.as_tensor(correction))
            ).detach().cpu()
        ),
        "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        **solver,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "geometry_four_body_capacity.npz",
        coefficients=coefficients,
        feature_keys=np.asarray(feature_keys, dtype=np.int64),
        predicted_hessian=prediction,
        correction_hessian_columns=correction,
        corrected_hessian=corrected,
        pbe_hessian_columns=reference,
        adapter_full_hessian=adapter_hessian.detach().cpu().numpy(),
        centers_bohr=centers,
    )
    (args.output_dir / "geometry_four_body_capacity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_hessian_comparison_plot(
        args.output_dir / "hessian_capacity_comparison.png",
        prediction,
        corrected,
        reference,
        correction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--molecule-id", default="0028399")
    parser.add_argument("--capacity-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-min", type=float, default=1.0)
    parser.add_argument("--center-max", type=float, default=4.0)
    parser.add_argument("--center-count", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--torsion-order", type=int, default=4)
    parser.add_argument("--bond-scale", type=float, default=1.35)
    parser.add_argument("--derivative-step", type=float, default=1.0e-3)
    parser.add_argument("--svd-tolerance", type=float, default=1.0e-10)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
