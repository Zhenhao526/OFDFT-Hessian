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
    RelaxedPoint,
    _assert_training_density_stationarity,
    _assert_density_refresh_cost_gates,
    _center_density_checkpoint_state,
    _density_refresh_scope,
    _hutchinson_direction,
    _label_density_replay_point,
    _response_cancellation_diagnostics,
    _response_predictor_trial_scales,
    _restore_center_density_checkpoint_state,
    _save_checkpoint,
    _training_density_stationarity_threshold,
    train,
)
from scripts import qm9_hessian_density_relaxed_eval as density_eval
from mldft.ofdft.energies import TensorEnergies


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


def test_training_density_gate_can_leave_solver_verification_margin():
    args = SimpleNamespace(
        density_strict_threshold=5.0e-9,
        training_density_stationarity_threshold=1.0e-8,
    )
    assert _training_density_stationarity_threshold(args) == pytest.approx(1.0e-8)


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


def test_label_density_replay_point_preserves_frozen_ks_density():
    molecule = MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        pbe_total_energy=-1.25,
        pbe_force=np.zeros((1, 3), dtype=np.float64),
        pbe_hessian=np.eye(3, dtype=np.float64),
        label_coefficients=torch.tensor([0.4, 0.6], dtype=torch.float64),
        directions=[],
    )

    point = _label_density_replay_point(molecule)

    np.testing.assert_array_equal(point.positions_bohr, molecule.positions_bohr)
    torch.testing.assert_close(point.coefficients, molecule.label_coefficients)
    assert point.total_energy == molecule.pbe_total_energy
    assert point.cycles == 0


def test_response_cancellation_diagnostics_detect_large_antiparallel_terms():
    diagnostics = _response_cancellation_diagnostics(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([-0.9], dtype=torch.float64),
    )

    assert diagnostics["partial_response_cosine"] == pytest.approx(-1.0)
    assert diagnostics["response_correction_fraction_of_relaxed_norm"] == pytest.approx(
        9.0
    )
    assert diagnostics["cancellation_index"] == pytest.approx(19.0)


def test_response_predictor_trust_scales_are_bounded_halvings():
    assert _response_predictor_trial_scales(1.0 / 8.0) == [
        1.0,
        0.5,
        0.25,
        0.125,
    ]
    with pytest.raises(ValueError, match="minimum scale"):
        _response_predictor_trial_scales(0.0)


def test_density_refresh_cost_gates_fail_closed_on_long_transition():
    protocol = {
        "gates": {
            "density_refresh_max_cycles": 500,
            "density_fallback_full_count": 0,
        }
    }
    density_args = SimpleNamespace(fallback_max_cycle=10000)
    with pytest.raises(RuntimeError, match="cycle gate"):
        _assert_density_refresh_cost_gates(
            [{"parameter_step": 1, "kind": "base", "cycles": 501}],
            protocol,
            density_args,
            source_capacity_step=0,
        )
    with pytest.raises(RuntimeError, match="full-fallback gate"):
        _assert_density_refresh_cost_gates(
            [
                {
                    "parameter_step": 1,
                    "kind": "base",
                    "cycles": 12,
                    "fallback_cycles": 10000,
                }
            ],
            protocol,
            density_args,
            source_capacity_step=0,
        )


def test_checkpoint_density_state_preserves_certification_boundary():
    strict = MoleculeState(
        molecule_id="strict",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((1, 3)),
        pbe_hessian=np.eye(3),
        label_coefficients=torch.zeros(2),
        directions=[],
        base=RelaxedPoint(
            positions_bohr=np.zeros((1, 3)),
            coefficients=torch.tensor([0.4, 0.6]),
            final_gradient_norm=1.0e-10,
            cycles=7,
            total_energy=-1.0,
        ),
    )
    predicted = MoleculeState(
        molecule_id="predicted",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((1, 3)),
        pbe_hessian=np.eye(3),
        label_coefficients=torch.zeros(2),
        directions=[],
        base=RelaxedPoint(
            positions_bohr=np.asarray([[0.1, 0.0, 0.0]]),
            coefficients=torch.tensor([0.3, 0.7]),
            final_gradient_norm=1.0e-10,
            cycles=0,
            total_energy=-0.9,
            initialization_mode="parameter_response_prediction",
            predictor_projected_gradient_norm=2.0e-6,
            predictor_total_energy=-0.95,
            predictor_trust_scale=0.5,
        ),
    )
    state = _center_density_checkpoint_state(
        [strict, predicted], parameter_step=4, strict_threshold=5.0e-9
    )
    assert state["molecules"]["strict"]["certified_strict"] is True
    assert state["molecules"]["predicted"]["certified_strict"] is False

    strict.base = None
    predicted.base = None
    assert _restore_center_density_checkpoint_state(
        [strict, predicted], {"center_density_state": state}, parameter_step=4
    ) == 2
    assert strict.base.initialization_mode == "checkpoint_density_warm_start"
    assert predicted.base.initialization_mode == "parameter_response_prediction"
    assert predicted.base.predictor_projected_gradient_norm == pytest.approx(2.0e-6)


def test_response_predictor_fast_path_skips_legacy_adam(monkeypatch):
    class QuadraticFactory:
        @staticmethod
        def evaluate_tensor_functional(sample, *_):
            target = torch.tensor([0.5, 0.5], dtype=sample.coeffs.dtype)
            return TensorEnergies(
                quadratic=0.5 * torch.sum((sample.coeffs - target) ** 2),
                nuclear_repulsion=torch.zeros((), dtype=sample.coeffs.dtype),
            )

    sample = SimpleNamespace(
        coeffs=torch.tensor([0.5, 0.5], dtype=torch.float64),
        dual_basis_integrals=torch.ones(2, dtype=torch.float64),
        coulomb_matrix=torch.eye(2, dtype=torch.float64),
        nuclear_attraction_vector=torch.zeros(2, dtype=torch.float64),
        mol=SimpleNamespace(nelectron=1),
    )
    monkeypatch.setattr(
        density_eval,
        "transform_tensor_with_sample",
        lambda _sample, tensor, *_args, **_kwargs: tensor,
    )
    _, final_coefficients, metadata, _ = density_eval._optimize_density(
        SimpleNamespace(functional_factory=QuadraticFactory()),
        sample,
        SimpleNamespace(
            response_predictor_fast_refine=True,
            response_predictor_fast_refine_threshold=5.0e-6,
            convergence_tolerance=1.0e-2,
        ),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        "parameter_response_prediction",
    )

    torch.testing.assert_close(
        final_coefficients, torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    assert metadata["response_predictor_fast_path_used"] is True
    assert metadata["first_stage_cycles"] == 0
    assert metadata["fallback_cycles"] == 0


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
