import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.qm9_complete_total_capacity_train import (
    Direction,
    MoleculeState,
    _assert_training_density_stationarity,
    _density_refresh_scope,
    _hutchinson_direction,
    _save_checkpoint,
    train,
)


def test_strict_active_density_refresh_is_due_after_every_parameter_update():
    assert (
        _density_refresh_scope(
            strict_active=True,
            current_parameter_step=0,
            density_parameter_step=0,
            refresh_needed=False,
            refresh_interval=25,
        )
        is None
    )
    assert (
        _density_refresh_scope(
            strict_active=True,
            current_parameter_step=1,
            density_parameter_step=0,
            refresh_needed=False,
            refresh_interval=25,
        )
        == "active"
    )


def test_legacy_density_refresh_policy_remains_interval_based():
    assert (
        _density_refresh_scope(
            strict_active=False,
            current_parameter_step=24,
            density_parameter_step=0,
            refresh_needed=False,
            refresh_interval=25,
        )
        is None
    )
    assert (
        _density_refresh_scope(
            strict_active=False,
            current_parameter_step=25,
            density_parameter_step=0,
            refresh_needed=False,
            refresh_interval=25,
        )
        == "all"
    )


@pytest.mark.parametrize("value", [1.0e-8, 1.0e-3, math.nan])
def test_strict_training_graph_rejects_nonstationary_density(value):
    with pytest.raises(RuntimeError, match="non-stationary density"):
        _assert_training_density_stationarity(value, 1.0e-8, enabled=True)


def test_strict_training_graph_accepts_density_below_threshold():
    _assert_training_density_stationarity(9.9e-9, 1.0e-8, enabled=True)


def test_analytic_training_rejects_first_order_only_matrix_power():
    with pytest.raises(
        ValueError, match="second-order-connected symmetric matrix power"
    ):
        train(
            SimpleNamespace(
                analytic_relaxed_hvp=True,
                symmetric_matrix_power_mode="stable_first_order",
            )
        )


def test_hutchinson_direction_is_fresh_internal_and_has_matching_target():
    basis = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    external = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    hessian = np.diag([2.0, 3.0, 4.0, 5.0])
    directions = [
        Direction(
            index=index,
            kind=("bond" if index == 0 else "angle"),
            vector=vector.reshape(2, 2),
            target_hvp=(hessian @ vector).reshape(2, 2),
        )
        for index, vector in enumerate(basis)
    ]
    molecule = MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1, 1]),
        positions_bohr=np.zeros((2, 2), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((2, 2), dtype=np.float64),
        pbe_hessian=hessian,
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=directions,
        direction_basis_definition="structured_internal_orthonormal",
        external_basis=external,
        expected_internal_dimension=2,
    )

    first = _hutchinson_direction(molecule, step=1, seed=17)
    probes = [
        _hutchinson_direction(molecule, step=step, seed=17)
        for step in range(1, 9)
    ]

    assert first.kind == "hutchinson_internal"
    assert np.isclose(np.sum(first.vector**2), 2.0)
    np.testing.assert_allclose(
        first.target_hvp.reshape(-1),
        hessian @ first.vector.reshape(-1),
    )
    assert len({probe.vector.tobytes() for probe in probes}) > 1
    np.testing.assert_array_equal(
        first.vector,
        _hutchinson_direction(molecule, step=1, seed=17).vector,
    )


def test_checkpoint_records_analytic_definition_and_frozen_provenance(tmp_path):
    source = tmp_path / "source.ckpt"
    output = tmp_path / "analytic.ckpt"
    torch.save({"state_dict": {}}, source)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6)
    provenance = {
        "protocol_id": "analytic_v2",
        "root_source_checkpoint_sha256": "root-hash",
        "direction_manifest_sha256": "direction-hash",
        "analytic_relaxed_hvp": True,
        "proxy_hvp_fallback_allowed": False,
        "validation_accessed": False,
        "test100_accessed": False,
    }

    _save_checkpoint(
        source,
        SimpleNamespace(model=model),
        optimizer,
        None,
        output,
        3,
        definition="complete-total analytic test",
        provenance_update=provenance,
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    recorded = payload["complete_total_capacity"]
    assert recorded["step"] == 3
    assert recorded["definition"] == "complete-total analytic test"
    assert recorded["protocol_id"] == "analytic_v2"
    assert recorded["direction_manifest_sha256"] == "direction-hash"
    assert recorded["analytic_relaxed_hvp"] is True
    assert recorded["proxy_hvp_fallback_allowed"] is False
