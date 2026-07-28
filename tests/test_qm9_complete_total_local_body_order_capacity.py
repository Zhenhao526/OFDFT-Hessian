from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mldft.ml.models.components.local_body_order_residual import (
    LocalBodyOrderResidual,
    build_body_order_topology,
)
from scripts.qm9_complete_total_local_body_order_capacity import (
    LocalParent,
    _evaluate,
    _relative_hvp_loss,
    _sample_batch,
    _task_gradient_norm,
)


def test_relative_hvp_loss_is_zero_for_exact_prediction_and_finite_at_floor() -> None:
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]], dtype=torch.float64
    )
    exact = _relative_hvp_loss(
        reference,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.05,
    )
    perturbed = _relative_hvp_loss(
        reference + 0.01,
        reference,
        absolute_scale=0.1,
        relative_fraction=0.5,
        reference_floor=0.05,
    )

    assert float(exact) == 0.0
    assert torch.isfinite(perturbed)
    assert float(perturbed) > 0.0


def test_zero_local_correction_reproduces_source_parent() -> None:
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0], [-0.4, 1.6, 0.0]],
        dtype=torch.float64,
    )
    coordinate_count = positions.numel()
    source_hessian = np.eye(coordinate_count, dtype=np.float64) * 0.2
    directions = torch.eye(coordinate_count, dtype=torch.float64)[:2]
    parent = LocalParent(
        molecule_id="synthetic",
        natoms=3,
        positions=positions,
        topology=build_body_order_topology(atomic_numbers),
        pbe_energy=-75.0,
        pbe_force=np.zeros((3, 3), dtype=np.float64),
        pbe_hessian=source_hessian.copy(),
        source_energy=-75.0,
        source_force=np.zeros((3, 3), dtype=np.float64),
        source_hessian=source_hessian.copy(),
        train_directions=directions,
        heldout_directions=directions,
        low_mode_directions=directions[:1],
    )
    model = LocalBodyOrderResidual(seed=31)

    selection_gates = SimpleNamespace(
        energy_median_gate=1e-3,
        energy_max_gate=2e-3,
        force_median_gate=1e-3,
        force_max_gate=3e-3,
        anchor_energy_force_at_parent=False,
    )
    scalar, rows, arrays = _evaluate(model, [parent], selection_gates)

    assert scalar["median_relative_frobenius"] < 1e-13
    assert scalar["median_train_hvp_relative_frobenius"] < 1e-13
    assert scalar["median_heldout_hvp_relative_frobenius"] < 1e-13
    assert rows[0]["energy_abs_error_hartree"] < 1e-13
    assert rows[0]["force_mae_hartree_per_bohr"] < 1e-13
    assert scalar["energy_force_gate_penalty"] == 0.0
    assert scalar["selection_score"] == scalar["curvature_selection_score"]
    assert np.allclose(arrays["synthetic"]["predicted_hessian"], source_hessian)


def test_task_gradient_norm_handles_unused_parameters() -> None:
    first = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
    norm = _task_gradient_norm(first.square(), [first, unused])
    assert torch.allclose(norm, torch.tensor(6.0, dtype=torch.float64))


def test_sample_batch_is_without_replacement_when_it_fits() -> None:
    parents = [
        SimpleNamespace(
            molecule_id=str(index),
            positions=torch.zeros((1, 3)),
            train_directions=torch.eye(3)[:2],
        )
        for index in range(5)
    ]
    args = SimpleNamespace(parent_batch_size=5, directions_per_parent=1)
    batch = _sample_batch(parents, args, torch.Generator().manual_seed(17))

    assert len(batch) == 5
    assert len({parent.molecule_id for parent, _ in batch}) == 5
