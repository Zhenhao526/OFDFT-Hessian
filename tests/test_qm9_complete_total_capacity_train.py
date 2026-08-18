import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.qm9_complete_total_capacity_train import (
    CANONICAL_PHYSICAL_DEFINITION_ID,
    LEGACY_PHYSICAL_DEFINITION_ID,
    Direction,
    IntegralBundleCache,
    MoleculeState,
    RelaxedPoint,
    _assert_canonical_replay_closures,
    _assert_canonical_internal_basis,
    _assert_training_density_stationarity,
    _assert_density_refresh_cost_gates,
    _canonical_structures25_replay_terms,
    _center_density_checkpoint_state,
    _clip_parameter_gradients_with_diagnostics,
    _cyclic_full_rank_rademacher_internal_directions,
    _cyclic_orthogonal_internal_directions,
    _density_refresh_scope,
    _hvp_updates_before_step,
    _hutchinson_direction,
    _fixed_full_rank_rademacher_internal_probe_pool,
    _label_density_replay_point,
    _optimizer_parameter_step_diagnostics,
    _physical_definition_id,
    _response_cancellation_diagnostics,
    _response_predictor_trial_scales,
    _restore_center_density_checkpoint_state,
    _save_checkpoint,
    _select_backward_terms,
    _symmetrize_canonical_reference_hessian,
    _training_density_stationarity_threshold,
    _validate_canonical_calibration_artifact,
    train,
)
from scripts import qm9_hessian_density_relaxed_eval as density_eval
from scripts.calibrate_qm9_implicit_hvp_weight import calibrate as calibrate_hvp_weight
from mldft.ofdft.energies import TensorEnergies
from mldft.ofdft.complete_total_training import (
    hutchinson_internal_projected_frobenius_squared_loss,
)


def test_joint_optimizer_backpropagates_egfh_in_one_loss():
    parameter = torch.tensor(2.0, requires_grad=True)
    weighted = {
        "energy": parameter,
        "density": 2.0 * parameter,
        "force": 3.0 * parameter,
        "hvp": 5.0 * parameter,
        "curvature": 0.0 * parameter,
        "spectrum": 0.0 * parameter,
    }

    selected = _select_backward_terms(
        weighted,
        alternating_hvp_updates=False,
        update_kind="joint",
    )
    torch.stack(list(selected.values())).sum().backward()

    assert set(selected) == set(weighted)
    assert parameter.grad == pytest.approx(11.0)


def test_alternating_optimizer_controls_remain_separated():
    weighted = {
        name: torch.tensor(float(index))
        for index, name in enumerate(
            ("energy", "density", "force", "hvp", "curvature", "spectrum"),
            start=1,
        )
    }

    assert set(
        _select_backward_terms(
            weighted,
            alternating_hvp_updates=True,
            update_kind="hvp",
        )
    ) == {"hvp"}
    assert "hvp" not in _select_backward_terms(
        weighted,
        alternating_hvp_updates=True,
        update_kind="replay",
    )


def test_autodiff_integral_cache_fails_closed_on_backend_mismatch():
    cache = IntegralBundleCache(
        None,
        SimpleNamespace(integral_derivative_backend="pyscfad_autodiff"),
    )
    with pytest.raises(RuntimeError, match="backend mismatch"):
        cache._record_and_validate_backend(
            SimpleNamespace(derivative_backend="finite_difference_pyscf"),
            directional=False,
        )


def test_autodiff_integral_cache_records_zero_step_directional_backend():
    cache = IntegralBundleCache(
        None,
        SimpleNamespace(integral_derivative_backend="pyscfad_autodiff"),
    )
    cache._record_and_validate_backend(
        SimpleNamespace(
            derivative_backend="pyscfad_jax_autodiff",
            directional_second_backend=(
                "pyscfad_jacfwd_of_jvp_with_traceable_rinv"
            ),
            derivative_step_bohr=0.0,
            directional_second_step_bohr=0.0,
        ),
        directional=True,
    )
    assert cache.observed_derivative_backends == {"pyscfad_jax_autodiff"}
    assert cache.observed_directional_second_backends == {
        "pyscfad_jacfwd_of_jvp_with_traceable_rinv"
    }


def test_torch_integral_cache_registers_direct_double_backward_provider(monkeypatch):
    class FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "scripts.qm9_complete_total_capacity_train."
        "TorchAutogradLibcintIntegralProvider",
        FakeProvider,
    )
    context = SimpleNamespace(
        sample_generator=SimpleNamespace(basis_info=SimpleNamespace(basis_dict="sto-3g"))
    )
    args = SimpleNamespace(
        integral_derivative_backend="torch_autograd_dqc",
        charge=0,
        integral_derivative_step=0.0,
        integral_derivative_workers=1,
        integral_cache_entries=2,
    )
    cache = IntegralBundleCache(context, args)
    provider = cache.get_torch_provider(np.asarray([1, 1]))
    assert provider is cache.get_torch_provider(np.asarray([1, 1]))
    assert cache.observed_derivative_backends == {
        "torch_autograd_dqc_libcint"
    }
    assert cache.observed_directional_second_backends == {
        "torch_autograd_double_backward"
    }


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


def test_cyclic_orthogonal_blocks_cover_linear_d13_continuously():
    cartesian_dimension = 18
    internal_dimension = 13
    basis = np.eye(cartesian_dimension, dtype=np.float64)[:internal_dimension]
    external = np.eye(cartesian_dimension, dtype=np.float64)[
        internal_dimension:
    ].T
    hessian = np.diag(np.linspace(1.0, 2.0, cartesian_dimension))
    molecule = MoleculeState(
        molecule_id="0000023",
        atomic_numbers=np.asarray([6, 6, 6, 6, 1, 1]),
        positions_bohr=np.zeros((6, 3), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((6, 3), dtype=np.float64),
        pbe_hessian=hessian,
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=[
            Direction(
                index=index,
                kind="bond",
                vector=vector.reshape(6, 3),
                target_hvp=(hessian @ vector).reshape(6, 3),
            )
            for index, vector in enumerate(basis)
        ],
        direction_basis_definition="structured_internal_orthonormal",
        external_basis=external,
        expected_internal_dimension=internal_dimension,
    )

    blocks = [
        _cyclic_orthogonal_internal_directions(
            molecule,
            hvp_update_ordinal=ordinal,
            count=3,
            seed=20260803,
        )
        for ordinal in range(13)
    ]
    repeated = _cyclic_orthogonal_internal_directions(
        molecule,
        hvp_update_ordinal=4,
        count=3,
        seed=20260803,
    )
    np.testing.assert_array_equal(blocks[4][0].vector, repeated[0].vector)

    flattened_indices = [item.index for block in blocks for item in block]
    assert len(flattened_indices) == 39
    assert {index: flattened_indices.count(index) for index in range(13)} == {
        index: 3 for index in range(13)
    }
    for block in blocks:
        matrix = np.stack([item.vector.reshape(-1) for item in block])
        np.testing.assert_allclose(
            matrix @ matrix.T,
            13.0 * np.eye(3),
            atol=1.0e-12,
            rtol=0.0,
        )
        for item in block:
            np.testing.assert_allclose(
                item.target_hvp.reshape(-1),
                hessian @ item.vector.reshape(-1),
            )

    error_hessian = np.diag(np.linspace(0.1, 0.9, cartesian_dimension))
    basis_tensor = torch.as_tensor(basis, dtype=torch.float64)
    losses = []
    for item in (item for block in blocks for item in block):
        target = torch.as_tensor(item.target_hvp, dtype=torch.float64)
        prediction = target + torch.as_tensor(
            (error_hessian @ item.vector.reshape(-1)).reshape(6, 3),
            dtype=torch.float64,
        )
        losses.append(
            hutchinson_internal_projected_frobenius_squared_loss(
                prediction,
                target,
                basis_tensor,
                reduction="mean_internal_matrix",
            )
        )
    projected_error = basis @ error_hessian @ basis.T
    expected = np.linalg.norm(projected_error) ** 2 / internal_dimension**2
    assert float(torch.stack(losses).mean()) == pytest.approx(expected)


def test_fixed_rademacher_pool_is_full_rank_conditioned_and_cycles():
    cartesian_dimension = 18
    internal_dimension = 13
    pool_size = 16
    basis = np.eye(cartesian_dimension, dtype=np.float64)[:internal_dimension]
    external = np.eye(cartesian_dimension, dtype=np.float64)[
        internal_dimension:
    ].T
    hessian = np.diag(np.linspace(1.0, 2.0, cartesian_dimension))
    molecule = MoleculeState(
        molecule_id="0000023",
        atomic_numbers=np.asarray([6, 6, 6, 6, 1, 1]),
        positions_bohr=np.zeros((6, 3), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((6, 3), dtype=np.float64),
        pbe_hessian=hessian,
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=[
            Direction(
                index=index,
                kind="bond",
                vector=vector.reshape(6, 3),
                target_hvp=(hessian @ vector).reshape(6, 3),
            )
            for index, vector in enumerate(basis)
        ],
        direction_basis_definition="structured_internal_orthonormal",
        external_basis=external,
        expected_internal_dimension=internal_dimension,
    )

    signs, metrics = _fixed_full_rank_rademacher_internal_probe_pool(
        molecule,
        pool_size=pool_size,
        seed=20260803,
        condition_number_max=50.0,
    )
    repeated_signs, repeated_metrics = (
        _fixed_full_rank_rademacher_internal_probe_pool(
            molecule,
            pool_size=pool_size,
            seed=20260803,
            condition_number_max=50.0,
        )
    )
    np.testing.assert_array_equal(signs, repeated_signs)
    assert metrics == repeated_metrics
    assert signs.shape == (pool_size, internal_dimension)
    assert set(np.unique(signs)) == {-1, 1}
    assert metrics["rank"] == internal_dimension
    assert metrics["condition_number"] <= 50.0

    blocks = [
        _cyclic_full_rank_rademacher_internal_directions(
            molecule,
            hvp_update_ordinal=ordinal,
            count=4,
            pool_size=pool_size,
            seed=20260803,
            condition_number_max=50.0,
        )
        for ordinal in range(4)
    ]
    flattened = [item for block in blocks for item in block]
    assert [item.index for item in flattened] == list(range(pool_size))
    for item in flattened:
        assert item.kind == "fixed_rademacher_internal"
        assert float(np.sum(item.vector**2)) == pytest.approx(
            float(internal_dimension)
        )
        np.testing.assert_allclose(
            item.target_hvp.reshape(-1),
            hessian @ item.vector.reshape(-1),
        )

    wrapped = _cyclic_full_rank_rademacher_internal_directions(
        molecule,
        hvp_update_ordinal=4,
        count=4,
        pool_size=pool_size,
        seed=20260803,
        condition_number_max=50.0,
    )
    np.testing.assert_array_equal(wrapped[0].vector, flattened[0].vector)


def test_hvp_update_ordinal_does_not_advance_on_replay_and_resumes():
    settings = {"hvp_update_period": 2, "replay_update_period": 5}
    assert _hvp_updates_before_step(1, **settings) == 0
    assert _hvp_updates_before_step(5, **settings) == 4
    # Step 5 is replay, so step 6 consumes the same next H-direction block.
    assert _hvp_updates_before_step(6, **settings) == 4
    # A checkpoint at cumulative step 50 has completed exactly 40 H updates.
    assert _hvp_updates_before_step(51, **settings) == 40


def _synthetic_internal_basis_molecule(
    *,
    basis: np.ndarray | None = None,
    external: np.ndarray | None = None,
) -> MoleculeState:
    if basis is None:
        basis = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    if external is None:
        external = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        )
    directions = [
        Direction(
            index=index,
            kind="bond",
            vector=vector.reshape(2, 2),
            target_hvp=vector.reshape(2, 2),
        )
        for index, vector in enumerate(basis)
    ]
    return MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1, 1]),
        positions_bohr=np.zeros((2, 2), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((2, 2), dtype=np.float64),
        pbe_hessian=np.eye(4, dtype=np.float64),
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=directions,
        direction_basis_definition="structured_internal_orthonormal",
        external_basis=external,
        expected_internal_dimension=2,
    )


def test_canonical_internal_basis_records_all_projection_closures():
    molecule = _synthetic_internal_basis_molecule()
    metrics = _assert_canonical_internal_basis(
        molecule,
        {
            "internal_basis_orthonormality_max_abs": 1.0e-12,
            "internal_external_overlap_max_abs": 1.0e-12,
            "projector_idempotence_max_abs": 1.0e-12,
        },
    )

    assert metrics["internal_dimension"] == 2
    assert metrics["external_rank"] == 2
    assert metrics["internal_basis_orthonormality_max_abs"] == pytest.approx(0.0)
    assert metrics["internal_external_overlap_max_abs"] == pytest.approx(0.0)
    assert metrics["projector_idempotence_max_abs"] == pytest.approx(0.0)
    assert molecule.internal_basis_metrics == metrics


@pytest.mark.parametrize(
    ("basis", "external", "failed_metric"),
    [
        (
            np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            None,
            "internal_basis_orthonormality_max_abs",
        ),
        (
            None,
            np.asarray(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=np.float64,
            ),
            "internal_external_overlap_max_abs",
        ),
    ],
)
def test_canonical_internal_basis_gates_are_fail_closed(
    basis, external, failed_metric
):
    molecule = _synthetic_internal_basis_molecule(
        basis=basis,
        external=external,
    )
    with pytest.raises(ValueError, match=failed_metric):
        _assert_canonical_internal_basis(
            molecule,
            {
                "internal_basis_orthonormality_max_abs": 1.0e-12,
                "internal_external_overlap_max_abs": 1.0e-12,
                "projector_idempotence_max_abs": 1.0e-12,
            },
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


def test_physical_definition_is_explicit_and_legacy_remains_reproducible():
    assert _physical_definition_id({}) == LEGACY_PHYSICAL_DEFINITION_ID
    assert _physical_definition_id(
        {
            "definitions": {
                "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID
            }
        }
    ) == CANONICAL_PHYSICAL_DEFINITION_ID
    with pytest.raises(ValueError, match="Unsupported physical definition"):
        _physical_definition_id(
            {"definitions": {"physical_definition_id": "silent_semantic_drift"}}
        )


def test_canonical_calibration_artifact_binds_protocol_curve_and_weight(tmp_path):
    curve = tmp_path / "training_curve.csv"
    curve.write_text("step,physical_definition_id\n1,qm9_complete_total_relaxed_egfh_v1\n")
    curve_sha256 = hashlib.sha256(curve.read_bytes()).hexdigest()
    code_provenance = {
        "runner_sha256": "runner-sha",
        "complete_total_training_sha256": "helper-sha",
        "normative_specification": "docs/spec.md",
        "normative_specification_sha256": "spec-sha",
    }
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID,
                "semantic_version": CANONICAL_PHYSICAL_DEFINITION_ID,
                "protocol_id": "canonical_v4",
                "protocol_sha256": "protocol-sha",
                "code_provenance": code_provenance,
            }
        )
    )
    summary_sha256 = hashlib.sha256(summary.read_bytes()).hexdigest()
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "protocol_id": "canonical_v4",
                "protocol_sha256": "protocol-sha",
                "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID,
                "training_curve": curve.as_posix(),
                "training_curve_sha256": curve_sha256,
                "training_summary": summary.as_posix(),
                "training_summary_sha256": summary_sha256,
                "code_provenance": code_provenance,
                "formal_lambda_h": 0.125,
            }
        )
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

    provenance = _validate_canonical_calibration_artifact(
        artifact,
        expected_sha256=artifact_sha256,
        protocol_id="canonical_v4",
        protocol_sha256="protocol-sha",
        requested_lambda_h=0.125,
        expected_code_provenance=code_provenance,
    )

    assert provenance["sha256"] == artifact_sha256
    assert provenance["training_curve_sha256"] == curve_sha256
    assert provenance["formal_lambda_h"] == pytest.approx(0.125)
    with pytest.raises(ValueError, match="lambda_H"):
        _validate_canonical_calibration_artifact(
            artifact,
            expected_sha256=artifact_sha256,
            protocol_id="canonical_v4",
            protocol_sha256="protocol-sha",
            requested_lambda_h=0.25,
            expected_code_provenance=code_provenance,
        )


def test_canonical_calibration_artifact_is_recomputed_from_bound_curve(tmp_path):
    protocol = {
        "definitions": {
            "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID
        },
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        },
    }
    row = {
        "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID,
        "computed_components": "E;G;F;H",
        "gradient_norm/energy": "3",
        "gradient_norm/density": "4",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.5",
        "gradient_norm/aggregate_egf": "1",
    }
    curve = tmp_path / "training_curve.csv"
    with curve.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    code_provenance = {
        "runner_sha256": "runner-sha",
        "complete_total_training_sha256": "helper-sha",
        "normative_specification": "docs/spec.md",
        "normative_specification_sha256": "spec-sha",
    }
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID,
                "semantic_version": CANONICAL_PHYSICAL_DEFINITION_ID,
                "protocol_id": "canonical_v4",
                "protocol_sha256": "protocol-sha",
                "code_provenance": code_provenance,
            }
        )
    )
    calibrated = calibrate_hvp_weight(protocol, row)
    payload = {
        "protocol_id": "canonical_v4",
        "protocol_sha256": "protocol-sha",
        "physical_definition_id": CANONICAL_PHYSICAL_DEFINITION_ID,
        "training_curve": curve.as_posix(),
        "training_curve_sha256": hashlib.sha256(curve.read_bytes()).hexdigest(),
        "training_summary": summary.as_posix(),
        "training_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "code_provenance": code_provenance,
        **calibrated,
    }
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps(payload))
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

    _validate_canonical_calibration_artifact(
        artifact,
        expected_sha256=artifact_sha256,
        protocol_id="canonical_v4",
        protocol_sha256="protocol-sha",
        requested_lambda_h=float(calibrated["formal_lambda_h"]),
        expected_code_provenance=code_provenance,
        calibration_protocol=protocol,
    )

    payload["aggregate_weighted_egf_gradient_norm"] = 999.0
    artifact.write_text(json.dumps(payload))
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="recomputation mismatch"):
        _validate_canonical_calibration_artifact(
            artifact,
            expected_sha256=artifact_sha256,
            protocol_id="canonical_v4",
            protocol_sha256="protocol-sha",
            requested_lambda_h=float(calibrated["formal_lambda_h"]),
            expected_code_provenance=code_provenance,
            calibration_protocol=protocol,
        )


@pytest.mark.parametrize("torch_direct", [False, True])
def test_canonical_replay_uses_structures25_functional_jet_and_exact_closures(
    torch_direct,
):
    coefficients = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    normalization = np.asarray([1.0, 1.0], dtype=np.float64)
    coulomb = np.eye(2, dtype=np.float64)
    attraction = np.zeros(2, dtype=np.float64)
    target_gradient = torch.tensor([1.6, 1.4], dtype=torch.float64)
    tangent_error = torch.tensor([0.2, -0.2], dtype=torch.float64)
    predicted_gradient = target_gradient + tangent_error
    target_energy = -2.0
    model_energy = (
        torch.tensor(target_energy + 0.05, dtype=torch.float64)
        + torch.dot(predicted_gradient, coefficients - coefficients.detach())
    )
    pbe_total_energy = -1.25
    geometry = SimpleNamespace(
        coeffs=coefficients,
        bundle=SimpleNamespace(
            values=SimpleNamespace(
                normalization=normalization,
                coulomb=coulomb,
                nuclear_attraction=attraction,
            )
        ),
        normalization_untransformed=None,
        coulomb_untransformed=None,
        nuclear_attraction_untransformed=None,
    )
    if torch_direct:
        geometry.bundle = None
        geometry.normalization_untransformed = torch.as_tensor(normalization)
        geometry.coulomb_untransformed = torch.as_tensor(coulomb)
        geometry.nuclear_attraction_untransformed = torch.as_tensor(attraction)
    replay = SimpleNamespace(
        energies=TensorEnergies(
            kin_plus_xc=model_energy,
            classical=torch.tensor(
                pbe_total_energy - target_energy, dtype=torch.float64
            ),
            nuclear_repulsion=torch.zeros((), dtype=torch.float64),
        ),
        geometry=geometry,
        constraint_residual=torch.zeros((), dtype=torch.float64),
    )
    molecule = MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.zeros((1, 3), dtype=np.float64),
        pbe_total_energy=pbe_total_energy,
        pbe_force=np.zeros((1, 3), dtype=np.float64),
        pbe_hessian=np.eye(3, dtype=np.float64),
        label_coefficients=coefficients.detach(),
        directions=[],
        structures25_kin_plus_xc_energy=target_energy,
        structures25_kin_plus_xc_gradient=target_gradient,
    )
    loss_config = {
        "energy": {"absolute_scale_hartree": 0.1},
        "density": {"scale_hartree_per_density_coefficient": 0.1},
    }

    terms = _canonical_structures25_replay_terms(
        replay, molecule, loss_config
    )

    torch.testing.assert_close(
        terms["energy_loss"], torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        terms["gradient_loss"], torch.tensor(4.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        terms["reference_euler_residual_norm"],
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        terms["gradient_identity_max_abs"],
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        terms["matched_total_energy_closure_abs"],
        torch.zeros((), dtype=torch.float64),
    )
    _assert_canonical_replay_closures(
        terms,
        {
            "reference_euler_residual_rms_max": 1.0e-12,
            "gradient_identity_max_abs": 1.0e-12,
            "matched_total_energy_closure_max_hartree": 1.0e-12,
            "label_electron_number_residual_abs_max": 1.0e-12,
        },
        molecule.molecule_id,
    )


def test_canonical_replay_closure_is_fail_closed():
    terms = {
        "reference_euler_residual_norm": torch.tensor(1.0e-3),
        "reference_euler_residual_rms": torch.tensor(1.0e-3),
        "gradient_identity_max_abs": torch.tensor(0.0),
        "matched_total_energy_closure_abs": torch.tensor(0.0),
        "label_electron_number_residual_abs": torch.tensor(0.0),
    }
    with pytest.raises(RuntimeError, match="replay closure failed"):
        _assert_canonical_replay_closures(
            terms,
            {
                "reference_euler_residual_rms_max": 1.0e-6,
                "gradient_identity_max_abs": 1.0e-12,
                "matched_total_energy_closure_max_hartree": 1.0e-12,
                "label_electron_number_residual_abs_max": 1.0e-12,
            },
            "synthetic",
        )


def test_canonical_reference_hessian_is_symmetrized_before_hvp_targets():
    raw_hessian = np.asarray([[2.0, 0.3], [0.1, 4.0]], dtype=np.float64)
    direction = Direction(
        index=0,
        kind="bond",
        vector=np.asarray([[1.0, 0.0]], dtype=np.float64),
        target_hvp=(raw_hessian @ np.asarray([1.0, 0.0])).reshape(1, 2),
    )
    molecule = MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.zeros((1, 2), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((1, 2), dtype=np.float64),
        pbe_hessian=raw_hessian,
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=[direction],
    )

    _symmetrize_canonical_reference_hessian(
        molecule,
        antisymmetric_over_symmetric_frobenius_max=0.1,
    )

    expected = 0.5 * (raw_hessian + raw_hessian.T)
    np.testing.assert_allclose(molecule.pbe_hessian, expected)
    np.testing.assert_allclose(
        molecule.directions[0].target_hvp.reshape(-1),
        expected @ direction.vector.reshape(-1),
    )
    np.testing.assert_array_equal(molecule.raw_pbe_hessian, raw_hessian)


def test_canonical_reference_hessian_symmetry_gate_is_fail_closed():
    molecule = MoleculeState(
        molecule_id="synthetic",
        atomic_numbers=np.asarray([1]),
        positions_bohr=np.zeros((1, 2), dtype=np.float64),
        pbe_total_energy=0.0,
        pbe_force=np.zeros((1, 2), dtype=np.float64),
        pbe_hessian=np.asarray([[1.0, 2.0], [-2.0, 1.0]], dtype=np.float64),
        label_coefficients=torch.zeros(2, dtype=torch.float64),
        directions=[],
    )

    with pytest.raises(ValueError, match="symmetry gate"):
        _symmetrize_canonical_reference_hessian(
            molecule,
            antisymmetric_over_symmetric_frobenius_max=0.1,
        )


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


def test_gradient_clip_diagnostics_distinguish_clipped_and_unclipped_updates():
    clipped = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    clipped.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    clipped_diagnostics = _clip_parameter_gradients_with_diagnostics(
        [clipped], 1.0
    )

    assert clipped_diagnostics["parameter_gradient_norm_before_clip"] == pytest.approx(
        5.0
    )
    assert clipped_diagnostics["parameter_gradient_norm_after_clip"] == pytest.approx(
        1.0, rel=1.0e-6
    )
    assert clipped_diagnostics["gradient_clip_scale"] == pytest.approx(
        0.2, rel=1.0e-6
    )
    assert clipped_diagnostics["gradient_clipping_enabled"] is True
    assert clipped_diagnostics["gradient_clipping_active"] is True

    unclipped = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    unclipped.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    unclipped_diagnostics = _clip_parameter_gradients_with_diagnostics(
        [unclipped], 0.0
    )

    assert unclipped_diagnostics["parameter_gradient_norm_before_clip"] == pytest.approx(
        5.0
    )
    assert unclipped_diagnostics["parameter_gradient_norm_after_clip"] == pytest.approx(
        5.0
    )
    assert unclipped_diagnostics["gradient_clip_scale"] == pytest.approx(1.0)
    assert unclipped_diagnostics["gradient_clipping_enabled"] is False
    assert unclipped_diagnostics["gradient_clipping_active"] is False
    torch.testing.assert_close(
        unclipped.grad, torch.tensor([3.0, 4.0], dtype=torch.float64)
    )


def test_optimizer_parameter_step_diagnostics_report_groups_and_total():
    first = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    diagnostics = _optimizer_parameter_step_diagnostics(
        [("gnn_module.weight", first), ("energy_mlp.bias", second)],
        [torch.zeros(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)],
        [
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.tensor([12.0], dtype=torch.float64),
        ],
    )

    assert diagnostics["optimizer_raw_parameter_step_norm"] == pytest.approx(13.0)
    assert diagnostics["optimizer_step_group_norm/gnn_module"] == pytest.approx(5.0)
    assert diagnostics["optimizer_step_group_norm/energy_mlp"] == pytest.approx(12.0)
    assert diagnostics[
        "optimizer_step_group_squared_fraction/gnn_module"
    ] == pytest.approx(25.0 / 169.0)
    assert diagnostics[
        "optimizer_step_group_squared_fraction/energy_mlp"
    ] == pytest.approx(144.0 / 169.0)


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
