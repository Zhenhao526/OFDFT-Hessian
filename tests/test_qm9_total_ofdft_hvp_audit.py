from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qm9_total_ofdft_hvp_audit.py"
    sys.path.insert(0, str(path.parent))
    spec = spec_from_file_location("qm9_total_ofdft_hvp_audit", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_directional_curvatures_match_quadratic_energy_and_force():
    module = _load_script()
    step = 3e-4
    expected_curvature = 2.5
    direction = np.array([[1.0, 0.0, 0.0]])
    base_energy = 7.0
    plus_energy = base_energy + 0.5 * expected_curvature * step**2
    minus_energy = base_energy + 0.5 * expected_curvature * step**2
    plus_force = np.array([[-expected_curvature * step, 0.0, 0.0]])
    minus_force = np.array([[expected_curvature * step, 0.0, 0.0]])

    result = module._directional_curvatures(
        plus_energy,
        base_energy,
        minus_energy,
        plus_force,
        minus_force,
        direction,
        step,
    )

    assert result["energy_curvature"] == pytest.approx(expected_curvature, rel=1e-8)
    assert result["force_curvature"] == pytest.approx(expected_curvature)
    assert result["relative_difference"] < 1e-8
