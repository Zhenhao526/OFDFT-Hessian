import numpy as np

from mldft.ofdft.internal_directions import (
    build_structured_internal_direction_bank,
    sample_internal_rademacher_direction,
)


def _nonlinear_chain():
    atomic_numbers = np.asarray([6, 6, 6, 6, 1, 1], dtype=np.int64)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.65, 0.0, 0.0],
            [5.1, 1.0, 0.2],
            [7.4, 0.4, 1.1],
            [-0.7, 1.5, 0.2],
            [8.1, -0.8, 1.6],
        ],
        dtype=np.float64,
    )
    return atomic_numbers, positions


def test_structured_internal_bank_is_complete_deterministic_and_orthogonal():
    atomic_numbers, positions = _nonlinear_chain()
    first = build_structured_internal_direction_bank(
        atomic_numbers,
        positions,
        seed=20260723,
        structured_per_kind=8,
    )
    second = build_structured_internal_direction_bank(
        atomic_numbers,
        positions,
        seed=20260723,
        structured_per_kind=8,
    )

    assert first.external_rank == 6
    assert first.internal_dimension == 3 * atomic_numbers.size - 6
    assert first.directions.shape == (first.internal_dimension, positions.size)
    np.testing.assert_array_equal(first.directions, second.directions)
    np.testing.assert_array_equal(first.kinds, second.kinds)
    np.testing.assert_array_equal(first.partial_roles, second.partial_roles)
    assert first.orthonormality_max_abs < 1.0e-12
    assert first.external_overlap_max_abs < 1.0e-12
    assert first.projector_idempotence_max_abs < 1.0e-12
    np.testing.assert_allclose(
        first.directions.T @ first.directions,
        first.projector,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_partial_split_is_exact_and_stratified_where_available():
    atomic_numbers, positions = _nonlinear_chain()
    bank = build_structured_internal_direction_bank(
        atomic_numbers,
        positions,
        seed=17,
        heldout_fraction=0.2,
        structured_per_kind=8,
    )

    expected_held = round(0.2 * bank.internal_dimension)
    assert np.count_nonzero(bank.partial_roles == "heldout") == expected_held
    assert np.count_nonzero(bank.partial_roles == "train") == (
        bank.internal_dimension - expected_held
    )
    populated = set(bank.kinds.tolist())
    held_kinds = set(bank.kinds[bank.partial_roles == "heldout"].tolist())
    assert len(held_kinds) == min(expected_held, len(populated))
    if expected_held >= len(populated):
        assert held_kinds == populated


def test_internal_hutchinson_probe_is_deterministic_complete_and_unscaled():
    atomic_numbers, positions = _nonlinear_chain()
    bank = build_structured_internal_direction_bank(
        atomic_numbers,
        positions,
        seed=23,
    )
    first = sample_internal_rademacher_direction(bank, seed=101)
    second = sample_internal_rademacher_direction(bank, seed=101)

    np.testing.assert_array_equal(first.signs, second.signs)
    np.testing.assert_array_equal(first.vector, second.vector)
    assert set(first.signs.tolist()) <= {-1.0, 1.0}
    np.testing.assert_allclose(
        first.squared_norm,
        bank.internal_dimension,
        atol=1.0e-10,
        rtol=0.0,
    )
    assert first.external_overlap_max_abs < 1.0e-10
