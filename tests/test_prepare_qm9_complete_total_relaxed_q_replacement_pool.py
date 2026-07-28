from __future__ import annotations

from scripts.prepare_qm9_complete_total_relaxed_q_replacement_pool import _rank_key


def test_replacement_rank_prefers_exact_stratum_before_hash() -> None:
    slot = {
        "parent_id": "0000001",
        "composition_class": "N+O",
        "natoms_bin": 2,
        "difficulty_bin": 1,
        "natoms": 18,
    }
    exact = {
        "parent_id": "0000002",
        "composition_class": "N+O",
        "natoms_bin": 2,
        "difficulty_bin": 1,
        "natoms": 20,
    }
    near = {
        "parent_id": "0000003",
        "composition_class": "N+O",
        "natoms_bin": 2,
        "difficulty_bin": 2,
        "natoms": 18,
    }
    mismatch = {
        "parent_id": "0000004",
        "composition_class": "O-only",
        "natoms_bin": 2,
        "difficulty_bin": 1,
        "natoms": 18,
    }
    assert _rank_key(slot, exact, 7) < _rank_key(slot, near, 7)
    assert _rank_key(slot, near, 7) < _rank_key(slot, mismatch, 7)
