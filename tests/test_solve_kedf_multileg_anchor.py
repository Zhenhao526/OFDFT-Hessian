from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from scripts.solve_wt_melting_gibbs_helmholtz import main


def test_solver_accepts_multileg_kedf_anchor(tmp_path: Path) -> None:
    combination = {
        "schema": "kedf-melting-multileg-free-energy-anchor-v1",
        "status": "anchor_temperature_free_energy_verified",
        "target_kedf": "xwm",
        "checks": {"anchor": True},
        "temperature_k": 900.0,
        "free_energy_ev_per_atom": {"liquid_minus_solid": 0.01},
        "uncertainty_budget_mev_per_atom": {
            "statistical_rss": 1.0,
            "combined_conservative": 2.0,
        },
    }
    points = [
        {
            "temperature_k": temperature,
            "delta_h_ev_per_atom": 0.1,
            "block_standard_error_mev_per_atom": 1.0,
            "half_drift_mev_per_atom": 0.5,
            "status": "verified",
        }
        for temperature in (900.0, 1050.0)
    ]
    series = {
        "schema": "kedf-zero-pressure-fusion-enthalpy-series-v1",
        "target_kedf": "xwm",
        "status": "verified",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "discard_fraction": 0.5,
        "points": points,
    }
    convergence = {
        "schema": "kedf-enthalpy-discard-convergence-summary-v1",
        "target_kedf": "xwm",
        "status": "verified",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "points": [
            {
                "temperature_k": point["temperature_k"],
                "status": "verified",
                "delta_h_discard_spread_mev_per_atom": 0.5,
                "reports": [
                    {
                        "discard_fraction": 0.5,
                        "status": "verified",
                        "delta_h_mev_per_atom": (
                            1000.0 * point["delta_h_ev_per_atom"]
                        ),
                        "block_standard_error_mev_per_atom": 1.0,
                        "half_drift_mev_per_atom": 0.5,
                    }
                ],
            }
            for point in points
        ],
    }
    paths = {
        "combination": tmp_path / "combination.json",
        "series": tmp_path / "series.json",
        "convergence": tmp_path / "convergence.json",
        "output": tmp_path / "result.json",
    }
    for name, payload in (
        ("combination", combination),
        ("series", series),
        ("convergence", convergence),
    ):
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    argv = [
        "solve_wt_melting_gibbs_helmholtz.py",
        "--combination",
        str(paths["combination"]),
        "--enthalpy-series",
        str(paths["series"]),
        "--enthalpy-convergence",
        str(paths["convergence"]),
        "--out",
        str(paths["output"]),
    ]
    with mock.patch("sys.argv", argv), mock.patch("builtins.print"):
        main()
    result = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert result["schema"] == "kedf-gibbs-helmholtz-melting-v1"
    assert result["target_kedf"] == "xwm"
    assert result["status"] == "verified"
