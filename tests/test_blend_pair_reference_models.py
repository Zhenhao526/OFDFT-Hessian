from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.blend_pair_reference_models import blend_models


def write_model(
    path: Path,
    coefficients: list[float],
    *,
    target_kedf: str = "lkt",
    sigma: float = 0.1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "target_kedf": target_kedf,
                "reference_kind": "suf_radial_proxy",
                "reference_gate_passed": True,
                "short_range_guard_passed": True,
                "model": {
                    "centers_angstrom": [1.8, 2.0],
                    "sigma_angstrom": sigma,
                    "cutoff_angstrom": 6.5,
                    "coefficients_ev": coefficients,
                    "repulsive_core": {
                        "amplitude_ev": 5000.0,
                        "cutoff_angstrom": 2.03,
                        "power": 2,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_blends_compatible_coefficients_and_records_provenance(tmp_path: Path):
    base = tmp_path / "base.json"
    correction = tmp_path / "correction.json"
    output = tmp_path / "blend.json"
    write_model(base, [1.0, 2.0, 3.0])
    write_model(correction, [5.0, 6.0, 7.0])

    result = blend_models(
        base,
        correction,
        output,
        alpha=0.75,
        target_kedf="lkt",
        phase="liquid",
    )

    assert result["status"] == "verified"
    assert result["alpha_correction"] == 0.75
    assert result["model"]["coefficients_ev"] == [4.0, 5.0, 6.0]
    assert result["model"]["repulsive_core"]["amplitude_ev"] == 5000.0
    assert result["base"]["sha256"]
    assert result["correction"]["sha256"]


def test_rejects_incompatible_basis(tmp_path: Path):
    base = tmp_path / "base.json"
    correction = tmp_path / "correction.json"
    write_model(base, [1.0, 2.0, 3.0])
    write_model(correction, [1.0, 2.0, 3.0], sigma=0.2)

    with pytest.raises(ValueError, match="sigma_angstrom"):
        blend_models(
            base,
            correction,
            tmp_path / "blend.json",
            alpha=0.5,
            target_kedf="lkt",
            phase="liquid",
        )


def test_rejects_cross_method_input(tmp_path: Path):
    base = tmp_path / "base.json"
    correction = tmp_path / "correction.json"
    write_model(base, [1.0, 2.0, 3.0])
    write_model(correction, [1.0, 2.0, 3.0], target_kedf="xwm")

    with pytest.raises(ValueError, match="correction model target_kedf"):
        blend_models(
            base,
            correction,
            tmp_path / "blend.json",
            alpha=0.5,
            target_kedf="lkt",
            phase="liquid",
        )
