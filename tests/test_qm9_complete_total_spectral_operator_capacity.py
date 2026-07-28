from __future__ import annotations

import numpy as np

from scripts.qm9_complete_total_spectral_operator_capacity import (
    block_parent_conditioned_basis,
)


def test_block_parent_conditioning_has_registered_low_rank_size() -> None:
    block = np.arange(510 * 4, dtype=np.float64).reshape(510, 2, 2)
    features = np.asarray([1.0, 0.5, -0.25, 0.0, 1.5, -1.0, 0.2, 0.7])
    result = block_parent_conditioned_basis(block, features)

    expected_per_channel = 4 * 8 + 15 * 8 + 6
    pair = block[:480].reshape(15, 4, 8, 2, 2)
    diagonal = block[480:].reshape(5, 6, 2, 2)
    shared = np.concatenate(
        (
            np.sum(pair, axis=0).reshape(-1, 2, 2),
            np.sum(pair, axis=1).reshape(-1, 2, 2),
            np.sum(diagonal, axis=0).reshape(-1, 2, 2),
        )
    )
    assert result.shape == (7 * expected_per_channel, 2, 2)
    assert np.allclose(result[:expected_per_channel], 0.5 * shared)
    zero_channel_start = 2 * expected_per_channel
    assert np.count_nonzero(result[zero_channel_start : zero_channel_start + expected_per_channel]) == 0
