from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_local_overlap_bridge import analyze_bridge


def write_window(root: Path, coupling: float, offset: float) -> Path:
    window = root / f"lambda_{coupling:.12f}".replace(".", "p")
    window.mkdir()
    rows = [
        {
            "lambda": coupling,
            "du": offset + 1.0e-5 * ((index % 5) - 2),
        }
        for index in range(40)
    ]
    (window / "trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (window / "summary.json").write_text(
        json.dumps({"natoms": 128, "target_temperature_k": 900.0}) + "\n",
        encoding="utf-8",
    )
    return window


def test_local_bridge_reports_both_adjacent_pairs(tmp_path: Path):
    windows = [
        write_window(tmp_path, 0.75, -1.0),
        write_window(tmp_path, 0.875, -1.0),
        write_window(tmp_path, 1.0, -1.0),
    ]

    result = analyze_bridge(
        windows,
        du_key="du",
        discard_fraction=0.5,
        minimum_overlap_ess=0.05,
        maximum_overlap_closure_mev_per_atom=2.0,
    )

    assert result["status"] == "verified"
    assert len(result["adjacent_overlap"]) == 2
    assert result["minimum_adjacent_effective_sample_fraction"] > 0.99
    assert result["maximum_adjacent_closure_mev_per_atom"] < 1.0e-5
