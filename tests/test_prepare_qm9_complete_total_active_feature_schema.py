from __future__ import annotations

import numpy as np

from scripts.prepare_qm9_complete_total_active_feature_schema import (
    _feature_key_container,
)


def test_feature_key_container_supports_standalone_and_shared_archive(tmp_path):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    np.save(standalone / "feature_keys.npy", np.asarray([[1, 2]]))
    path, dataset = _feature_key_container(standalone)
    assert path == standalone / "feature_keys.npy"
    assert dataset is None

    shared = tmp_path / "shared"
    shared.mkdir()
    np.savez(shared / "shared_coefficients.npz", feature_keys=np.asarray([[3, 4]]))
    path, dataset = _feature_key_container(shared)
    assert path == shared / "shared_coefficients.npz"
    assert dataset == "feature_keys"
