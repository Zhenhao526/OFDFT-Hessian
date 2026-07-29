from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_suf_pair_ti import analyze


def write_window(
    root: Path,
    coupling: float,
    *,
    minimum_distance: float,
    du_offset: float,
    include_msd: bool = True,
) -> None:
    window = root / f"lambda_{coupling:.12f}".replace(".", "p")
    window.mkdir(parents=True)
    rows = []
    for index in range(40):
        row = {
            "lambda": coupling,
            "temperature_k": 900.0 + (index % 3) - 1.0,
            "du_pair_minus_suf_ev_per_atom": du_offset
            + 0.0001 * ((index % 5) - 2),
            "nearest_neighbor_angstrom": minimum_distance
            if index == 5
            else 2.3,
        }
        if include_msd:
            row["msd_angstrom2"] = 2.0 * index / 39.0
        rows.append(row)
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


def test_einstein_windows_can_disable_liquid_diffusion_gate(tmp_path: Path):
    for coupling in (0.0, 0.5, 1.0):
        write_window(
            tmp_path,
            coupling,
            minimum_distance=2.1,
            du_offset=-0.0570 - 0.0001 * coupling,
            include_msd=False,
        )

    result = analyze(
        tmp_path,
        blocks=5,
        reference_label="Einstein",
        minimum_liquid_msd=0.0,
    )

    assert result["status"] == "verified"
    assert result["checks"]["liquid_diffusion"] is True
    assert all(window["msd_last_angstrom2"] is None for window in result["windows"])


def test_missing_liquid_msd_fails_when_diffusion_is_required(tmp_path: Path):
    for coupling in (0.0, 0.5, 1.0):
        write_window(
            tmp_path,
            coupling,
            minimum_distance=2.1,
            du_offset=-0.0570 - 0.0001 * coupling,
            include_msd=False,
        )

    result = analyze(tmp_path, blocks=5)

    assert result["status"] == "production_gate_failed"
    assert result["checks"]["liquid_diffusion"] is False


def test_power_transformed_coordinate_integrates_nonuniform_lambda_grid(
    tmp_path: Path,
):
    for coordinate in (0.0, 0.25, 0.5, 0.75, 1.0):
        write_window(
            tmp_path,
            coordinate**2,
            minimum_distance=2.1,
            du_offset=1.0,
        )

    result = analyze(
        tmp_path,
        blocks=5,
        integration_coordinate_power=2.0,
    )

    assert result["status"] == "verified"
    assert result["delta_f_pair_minus_suf_simpson_ev_per_atom"] == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert result["integration_coordinate"] == {
        "lambda_equals_x_to_power": 2.0,
        "spacing": 0.25,
    }
    assert result["windows"][0]["integration_jacobian"] == 0.0
    assert result["windows"][-1]["integration_jacobian"] == 2.0


def test_power_grid_accepts_serialized_lambda_rounding(tmp_path: Path):
    for index in range(17):
        coordinate = index / 16
        write_window(
            tmp_path,
            round(coordinate**4, 12),
            minimum_distance=2.1,
            du_offset=1.0,
        )

    result = analyze(
        tmp_path,
        blocks=5,
        integration_coordinate_power=4.0,
        max_quadrature_difference=5.0,
    )

    assert result["status"] == "verified"
    assert result["delta_f_pair_minus_suf_simpson_ev_per_atom"] == pytest.approx(
        1.0, abs=1.0e-4
    )
    assert result["integration_coordinate"] == {
        "lambda_equals_x_to_power": 4.0,
        "spacing": 0.0625,
    }


def test_nonuniform_lambda_grid_requires_matching_coordinate_power(tmp_path: Path):
    for coupling in (0.0, 0.0625, 0.25, 0.5625, 1.0):
        write_window(
            tmp_path,
            coupling,
            minimum_distance=2.1,
            du_offset=-0.057,
        )

    with pytest.raises(
        ValueError, match="integration-coordinate windows must be evenly spaced"
    ):
        analyze(tmp_path, blocks=5)
