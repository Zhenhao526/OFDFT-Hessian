from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_suf_pair_ti import analyze


def write_window(
    root: Path,
    coupling: float,
    *,
    minimum_distance: float,
    du_offset: float,
) -> None:
    window = root / f"lambda_{coupling:.3f}".replace(".", "p")
    window.mkdir(parents=True)
    rows = []
    for index in range(40):
        rows.append(
            {
                "lambda": coupling,
                "temperature_k": 900.0 + (index % 3) - 1.0,
                "du_pair_minus_suf_ev_per_atom": du_offset
                + 0.0001 * ((index % 5) - 2),
                "nearest_neighbor_angstrom": minimum_distance
                if index == 5
                else 2.3,
                "msd_angstrom2": 2.0 * index / 39.0,
            }
        )
    (window / "trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (window / "summary.json").write_text(
        json.dumps(
            {
                "natoms": 108,
                "target_temperature_k": 900.0,
                "minimum_distance_angstrom": minimum_distance,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_auxiliary_close_contacts_do_not_invalidate_safe_pair_endpoint(tmp_path: Path):
    write_window(tmp_path, 0.0, minimum_distance=1.8, du_offset=-0.0570)
    write_window(tmp_path, 0.5, minimum_distance=1.9, du_offset=-0.0571)
    write_window(tmp_path, 1.0, minimum_distance=2.1, du_offset=-0.0572)

    result = analyze(tmp_path, blocks=5)

    assert result["status"] == "verified"
    assert result["all_windows_stable"] is True
    assert result["target_windows_stable"] is True
    assert len(result["adjacent_overlap"]) == 2
    assert result["minimum_adjacent_effective_sample_fraction"] > 0.9


def test_unsafe_pair_endpoint_fails_target_distance_gate(tmp_path: Path):
    write_window(tmp_path, 0.0, minimum_distance=1.8, du_offset=-0.0570)
    write_window(tmp_path, 0.5, minimum_distance=1.9, du_offset=-0.0571)
    write_window(tmp_path, 1.0, minimum_distance=1.99, du_offset=-0.0572)

    result = analyze(tmp_path, blocks=5)

    assert result["status"] == "production_gate_failed"
    assert result["checks"]["numerical_stability"] is True
    assert result["checks"]["target_windows_stable"] is False
