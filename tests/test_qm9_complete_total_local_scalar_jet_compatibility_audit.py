from __future__ import annotations

import numpy as np
import torch

from scripts.qm9_complete_total_local_scalar_jet_compatibility_audit import (
    audit_scalar_jet,
)


def _autograd_jet(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = torch.tensor(positions, dtype=torch.float64, requires_grad=True)

    def energy(flat: torch.Tensor) -> torch.Tensor:
        geometry = flat.reshape_as(coordinates)
        value = geometry.new_zeros(())
        for first in range(geometry.shape[0]):
            for second in range(first + 1, geometry.shape[0]):
                distance_squared = torch.sum((geometry[first] - geometry[second]) ** 2)
                value = value + 0.2 * distance_squared + 0.03 * distance_squared**2
        return value

    flat = coordinates.reshape(-1)
    value = energy(flat)
    gradient = torch.autograd.grad(value, flat, create_graph=True)[0]
    hessian = torch.autograd.functional.hessian(energy, flat)
    return -gradient.detach().numpy().reshape(-1, 3), hessian.detach().numpy()


def test_exact_rigid_invariant_scalar_has_negligible_jet_floor() -> None:
    positions = np.asarray(
        [[-0.8, 0.1, 0.2], [0.5, -0.4, 0.3], [0.2, 0.9, -0.5]],
        dtype=np.float64,
    )
    force, hessian = _autograd_jet(positions)

    result = audit_scalar_jet(positions, force, hessian)

    assert result["force_rigid_projection_relative"] < 1e-12
    assert result["translation_hessian_response_relative"] < 1e-12
    assert result["rotation_jet_response_relative"] < 1e-12
    assert result["exact_invariant_force_hessian_floor_relative_frobenius"] < 1e-11
    assert result["free_invariant_force_hessian_floor_relative_frobenius"] < 1e-11


def test_forbidden_translation_curvature_has_positive_floor() -> None:
    positions = np.asarray(
        [[-0.8, 0.1, 0.2], [0.5, -0.4, 0.3], [0.2, 0.9, -0.5]],
        dtype=np.float64,
    )
    force, hessian = _autograd_jet(positions)
    translation = np.tile([1.0, 0.0, 0.0], positions.shape[0])
    translation /= np.linalg.norm(translation)
    incompatible = hessian + 0.5 * np.outer(translation, translation)

    result = audit_scalar_jet(positions, force, incompatible)

    assert result["translation_hessian_response_relative"] > 1e-2
    assert result["exact_invariant_force_hessian_floor_relative_frobenius"] > 1e-2
    assert result["free_invariant_force_hessian_floor_relative_frobenius"] > 1e-2
