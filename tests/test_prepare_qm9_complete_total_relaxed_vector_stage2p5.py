from __future__ import annotations

import numpy as np

from scripts.prepare_qm9_complete_total_relaxed_vector_stage2p5 import (
    _atom_count_bin,
    _composition,
    _round_robin_select,
)


def test_round_robin_selection_is_deterministic_and_stratum_diverse() -> None:
    candidates = [
        {"molecule_id": "a1", "stratum": "a"},
        {"molecule_id": "a2", "stratum": "a"},
        {"molecule_id": "a3", "stratum": "a"},
        {"molecule_id": "b1", "stratum": "b"},
        {"molecule_id": "c1", "stratum": "c"},
    ]
    selected = _round_robin_select(candidates, 4, seed=17)
    assert selected == _round_robin_select(candidates, 4, seed=17)
    assert {row["stratum"] for row in selected[:3]} == {"a", "b", "c"}
    assert len({row["molecule_id"] for row in selected}) == 4


def test_composition_and_atom_count_bins() -> None:
    assert _composition(np.asarray([6, 1, 8, 7, 1, 6])) == "CNO"
    bins = [
        {"name": "small", "minimum": 0, "maximum": 15},
        {"name": "medium", "minimum": 16, "maximum": 19},
        {"name": "large", "minimum": 20, "maximum": 999},
    ]
    assert _atom_count_bin(15, bins) == "small"
    assert _atom_count_bin(16, bins) == "medium"
    assert _atom_count_bin(20, bins) == "large"
