from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.prepare_qm9_complete_total_relaxed_q_stage2p5 import _heldout_index


def test_heldout_index_is_deterministic_and_eligible() -> None:
    indices = np.asarray([0, 2, 3], dtype=np.int64)
    kinds = np.asarray(["random_internal", "bond_stretch", "low_frequency"])
    first = _heldout_index(20260723, "0012345", indices, kinds)
    second = _heldout_index(20260723, "0012345", indices, kinds)
    assert first == second
    assert first in indices


def test_heldout_index_depends_on_parent_identity() -> None:
    indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    kinds = np.asarray(["random_internal", "bond_stretch", "angle_bend", "low_frequency"])
    values = {
        _heldout_index(20260723, f"{index:07d}", indices, kinds)
        for index in range(20)
    }
    assert len(values) > 1
