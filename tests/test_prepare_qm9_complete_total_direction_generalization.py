from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "prepare_qm9_complete_total_direction_generalization.py"
    spec = spec_from_file_location("prepare_direction_generalization", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_bank_is_deterministic_internal_orthonormal_and_split():
    module = _load_script()
    atomic_numbers = np.asarray([6, 1, 1, 1, 1])
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 1.2, 1.2],
            [-1.2, -1.2, 1.2],
            [-1.2, 1.2, -1.2],
            [1.2, -1.2, -1.2],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(15, 15))
    hessian = 0.5 * (matrix + matrix.T)

    first = module.build_direction_bank(
        atomic_numbers,
        positions,
        hessian,
        seed=17,
        maximum=9,
        structured_per_kind=2,
        heldout_fraction=1.0 / 3.0,
    )
    second = module.build_direction_bank(
        atomic_numbers,
        positions,
        hessian,
        seed=17,
        maximum=9,
        structured_per_kind=2,
        heldout_fraction=1.0 / 3.0,
    )

    np.testing.assert_array_equal(first["directions"], second["directions"])
    np.testing.assert_array_equal(first["roles"], second["roles"])
    assert first["directions"].shape == (9, 15)
    assert first["orthonormality_max_abs"] < 1.0e-12
    assert first["external_overlap_max_abs"] < 1.0e-12
    assert set(first["roles"]) == {"train", "heldout"}
    assert {"bond", "angle", "random_internal"}.issubset(set(first["kinds"]))


def test_near_complete_direction_bank_is_stably_reorthogonalized():
    module = _load_script()
    rng = np.random.default_rng(101)
    atomic_numbers = np.ones(10, dtype=np.int64)
    positions = rng.normal(size=(10, 3)) * 4.0
    matrix = rng.normal(size=(30, 30))
    hessian = 0.5 * (matrix + matrix.T)

    bank = module.build_direction_bank(
        atomic_numbers,
        positions,
        hessian,
        seed=103,
        maximum=24,
        structured_per_kind=8,
        heldout_fraction=0.1,
        minimum_random_fraction=0.25,
    )

    assert bank["directions"].shape == (24, 30)
    assert bank["orthonormality_max_abs"] < 1.0e-12
    assert bank["external_overlap_max_abs"] < 1.0e-12
    assert np.sum(bank["kinds"] == "random_internal") >= 6


def test_frozen_protocol_contains_strict_evaluator_compatibility_keys():
    protocol = yaml.safe_load(
        (ROOT / "configs/audit/qm9_complete_total_hessian_direction_generalization_v1.yaml").read_text()
    )

    assert protocol["test100_access_allowed"] is False
    assert protocol["loss"]["initial_weights"]["lambda_H"] == 1.0
    assert protocol["stage1"]["gate"]["antisymmetric_over_symmetric_fro_max"] == 0.005


def test_dense_v2_protocol_uses_new_frozen_seed_and_holds_out_directions():
    protocol = yaml.safe_load(
        (
            ROOT
            / "configs/audit/qm9_complete_total_hessian_direction_generalization_v2_dense.yaml"
        ).read_text()
    )

    assert protocol["test100_access_allowed"] is False
    assert protocol["directions"]["seed"] == 20260721
    assert protocol["directions"]["maximum_per_parent"] == 48
    assert protocol["directions"]["heldout_fraction"] == 0.2
    assert protocol["data"]["v1_direction_coverage_audit"][
        "median_unidentified_qhq_relative_frobenius"
    ] > 0.68


def test_near_full_v3_protocol_keeps_a_frozen_heldout_split():
    protocol = yaml.safe_load(
        (
            ROOT
            / "configs/audit/qm9_complete_total_hessian_direction_generalization_v3_near_full.yaml"
        ).read_text()
    )

    assert protocol["test100_access_allowed"] is False
    assert protocol["directions"]["seed"] == 20260722
    assert protocol["directions"]["maximum_per_parent"] == 64
    assert 0.0 < protocol["directions"]["heldout_fraction"] < 0.2
    assert protocol["directions"]["minimum_random_fraction"] == 0.2
    assert protocol["data"]["coverage_ladder_audit"][
        "v2_median_unidentified_qhq_relative_frobenius"
    ] > 0.4
