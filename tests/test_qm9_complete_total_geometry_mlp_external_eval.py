from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.qm9_complete_total_geometry_mlp_external_eval import (
    _map_local_design,
    _model_from_state,
)
from scripts.qm9_complete_total_geometry_mlp_capacity import (
    DeepSmoothDescriptorResidual,
    SmoothDescriptorResidual,
)
from scripts.qm9_complete_total_geometry_shared_capacity import FeatureDesign


def test_map_local_design_uses_only_frozen_active_keys() -> None:
    local = FeatureDesign(
        molecule_id="0000001",
        keys=[(1, 2), (3, 4), (5, 6)],
        energy=np.asarray([2.0, 6.0, 10.0]),
        gradient=np.asarray([[4.0, 12.0, 20.0]]),
        hessian=np.asarray([[8.0, 24.0, 40.0]]),
        metadata={},
    )
    descriptor, jacobian, hessian, coverage = _map_local_design(
        local,
        np.asarray([[3, 4], [1, 2]], dtype=np.int64),
        np.asarray([3.0, 2.0]),
    )

    np.testing.assert_allclose(descriptor, [2.0, 1.0])
    np.testing.assert_allclose(jacobian, [[4.0, 2.0]])
    np.testing.assert_allclose(hessian, [[8.0, 4.0]])
    assert coverage["matched_active_feature_count"] == 2
    assert coverage["matched_active_feature_fraction"] == 2 / 3


@pytest.mark.parametrize("deep_hidden_size", [0, 4])
def test_model_from_state_rebuilds_shallow_and_deep_scalars(deep_hidden_size) -> None:
    if deep_hidden_size:
        source = DeepSmoothDescriptorResidual(
            5,
            hidden_size=3,
            deep_hidden_size=deep_hidden_size,
            initial_linear=torch.zeros(5, dtype=torch.float64),
            seed=3,
        )
    else:
        source = SmoothDescriptorResidual(
            5,
            hidden_size=3,
            initial_linear=torch.zeros(5, dtype=torch.float64),
            seed=3,
        )
    rebuilt, feature_count, hidden_size, rebuilt_deep_size = _model_from_state(
        source.state_dict(), torch.device("cpu")
    )
    assert type(rebuilt) is type(source)
    assert (feature_count, hidden_size, rebuilt_deep_size) == (
        5,
        3,
        deep_hidden_size,
    )


def test_model_from_state_rejects_partial_deep_state() -> None:
    source = SmoothDescriptorResidual(
        5,
        hidden_size=3,
        initial_linear=torch.zeros(5, dtype=torch.float64),
        seed=3,
    )
    state = dict(source.state_dict())
    state["deep_output"] = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="incomplete deep residual"):
        _model_from_state(state, torch.device("cpu"))
