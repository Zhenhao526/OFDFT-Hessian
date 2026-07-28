from __future__ import annotations

import numpy as np

import pytest

from scripts.qm9_complete_total_replay_descriptor_cache import (
    DESCRIPTOR_SETTING_NAMES,
    _map_energy_gradient,
    _validate_descriptor_settings,
)


def test_map_energy_gradient_uses_frozen_active_order_and_norms() -> None:
    descriptor, jacobian, coverage = _map_energy_gradient(
        [(1, 2), (3, 4), (5, 6)],
        np.asarray([2.0, 6.0, 10.0]),
        np.asarray([[4.0, 12.0, 20.0]]),
        np.asarray([[3, 4], [1, 2]], dtype=np.int64),
        np.asarray([3.0, 2.0]),
    )
    np.testing.assert_allclose(descriptor, [2.0, 1.0])
    np.testing.assert_allclose(jacobian, [[4.0, 2.0]])
    assert coverage["matched_active_feature_count"] == 2
    assert coverage["matched_active_feature_fraction"] == 2 / 3


def test_descriptor_cache_rejects_settings_that_do_not_match_schema() -> None:
    settings = {name: index for index, name in enumerate(DESCRIPTOR_SETTING_NAMES)}
    manifest = {"descriptor_settings": dict(settings)}
    _validate_descriptor_settings(manifest, settings)

    manifest["descriptor_settings"]["four_body_sigma"] = 0.4
    with pytest.raises(ValueError, match="four_body_sigma"):
        _validate_descriptor_settings(manifest, settings)
