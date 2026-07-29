import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mpn_melting.gibbs_helmholtz import (
    gibbs_over_temperature,
    integrate_linear_enthalpy_over_t2,
    solve_melting_temperature,
)
from scripts.solve_wt_melting_gibbs_helmholtz import (
    convergence_summary_matches_series,
    optional_root,
    uncertainty_interval,
)


class GibbsHelmholtzTests(unittest.TestCase):
    def test_constant_enthalpy_integral(self):
        value = integrate_linear_enthalpy_over_t2(900.0, 1000.0, 0.1, 0.1)
        self.assertAlmostEqual(value, 0.1 * (1.0 / 900.0 - 1.0 / 1000.0))

    def test_constant_enthalpy_recovers_known_root(self):
        root = solve_melting_temperature(
            900.0, 0.01, [(900.0, 0.1), (1050.0, 0.1)]
        )
        self.assertAlmostEqual(root, 1000.0, places=5)

    def test_anchor_in_middle_recovers_lower_temperature_root(self):
        root = solve_melting_temperature(
            900.0,
            -0.01,
            [(800.0, 0.1), (900.0, 0.1), (1000.0, 0.1)],
        )
        self.assertAlmostEqual(root, 900.0 / 1.1, places=5)
        profile = gibbs_over_temperature(
            900.0,
            -0.01,
            [(800.0, 0.1), (900.0, 0.1), (1000.0, 0.1)],
        )
        self.assertGreater(profile[0][1], 0.0)
        self.assertLess(profile[-1][1], 0.0)

    def test_anchor_must_be_present_once(self):
        with self.assertRaisesRegex(ValueError, "at the anchor temperature"):
            gibbs_over_temperature(900.0, 0.0, [(800.0, 0.1), (1000.0, 0.1)])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            gibbs_over_temperature(
                900.0, 0.0, [(900.0, 0.1), (900.0, 0.1), (1000.0, 0.1)]
            )

    def test_linear_enthalpy_matches_dense_numerical_integral(self):
        exact = integrate_linear_enthalpy_over_t2(900.0, 1000.0, 0.08, 0.1)
        count = 100000
        spacing = 100.0 / count
        numerical = 0.0
        for index in range(count):
            temperature = 900.0 + (index + 0.5) * spacing
            enthalpy = 0.08 + 0.02 * (temperature - 900.0) / 100.0
            numerical += enthalpy / temperature**2 * spacing
        self.assertTrue(math.isclose(exact, numerical, rel_tol=1.0e-10))

    def test_requires_a_bracket(self):
        values = gibbs_over_temperature(
            900.0, 0.05, [(900.0, 0.08), (950.0, 0.08)]
        )
        self.assertGreater(values[-1][1], 0.0)
        with self.assertRaises(ValueError):
            solve_melting_temperature(
                900.0, 0.05, [(900.0, 0.08), (950.0, 0.08)]
            )

    def test_uncertainty_interval_contains_nominal_root(self):
        points = [
            {
                "temperature_k": 900.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 2.0,
            },
            {
                "temperature_k": 1050.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 2.0,
            },
        ]
        interval = uncertainty_interval(900.0, 0.01, 1.0, points)
        self.assertLess(interval["lower_k"], 1000.0)
        self.assertGreater(interval["upper_k"], 1000.0)

    def test_uncertainty_interval_contains_lower_temperature_root(self):
        points = [
            {
                "temperature_k": 800.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 2.0,
            },
            {
                "temperature_k": 900.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 2.0,
            },
        ]
        nominal = 900.0 / 1.1
        interval = uncertainty_interval(900.0, -0.01, 1.0, points)
        self.assertLess(interval["lower_k"], nominal)
        self.assertGreater(interval["upper_k"], nominal)
        self.assertEqual(len(interval["scenarios"]), 4)

    def test_optional_root_reports_missing_bracket(self):
        self.assertIsNone(optional_root(900.0, 0.05, [(900.0, 0.08), (950.0, 0.08)]))

    def test_convergence_summary_must_match_selected_discard_report(self):
        point = {
            "temperature_k": 900.0,
            "delta_h_ev_per_atom": 0.1,
            "block_standard_error_mev_per_atom": 2.0,
            "half_drift_mev_per_atom": 1.5,
            "status": "verified",
        }
        series = {"status": "verified", "discard_fraction": 0.5, "points": [point]}
        summary = {
            "status": "verified",
            "points": [
                {
                    "temperature_k": 900.0,
                    "status": "verified",
                    "delta_h_discard_spread_mev_per_atom": 0.75,
                    "reports": [
                        {
                            "discard_fraction": 0.5,
                            "status": "verified",
                            "delta_h_mev_per_atom": 100.0,
                            "block_standard_error_mev_per_atom": 2.0,
                            "half_drift_mev_per_atom": 1.5,
                        }
                    ],
                }
            ],
        }
        matches, uncertainties = convergence_summary_matches_series(summary, series)
        self.assertTrue(matches)
        self.assertEqual(uncertainties[0]["conservative_mev_per_atom"], 4.25)

        summary["points"][0]["reports"][0]["delta_h_mev_per_atom"] = 101.0
        matches, uncertainties = convergence_summary_matches_series(summary, series)
        self.assertFalse(matches)
        self.assertEqual(uncertainties, [])

    def test_uncertainty_interval_supports_conservative_enthalpy_field(self):
        points = [
            {
                "temperature_k": 900.0,
                "delta_h_ev_per_atom": 0.1,
                "enthalpy_conservative_uncertainty_mev_per_atom": 5.0,
            },
            {
                "temperature_k": 1050.0,
                "delta_h_ev_per_atom": 0.1,
                "enthalpy_conservative_uncertainty_mev_per_atom": 5.0,
            },
        ]
        interval = uncertainty_interval(
            900.0,
            0.01,
            1.0,
            points,
            enthalpy_uncertainty_field=(
                "enthalpy_conservative_uncertainty_mev_per_atom"
            ),
        )
        self.assertLess(interval["lower_k"], 1000.0)
        self.assertGreater(interval["upper_k"], 1000.0)

    def test_solver_output_records_thermodynamic_convention(self):
        combination = {
            "schema": "wt-melting-free-energy-combination-v2",
            "status": "anchor_temperature_free_energy_verified",
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
                "temperature_k": 900.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 1.0,
                "half_drift_mev_per_atom": 0.5,
                "status": "verified",
            },
            {
                "temperature_k": 1050.0,
                "delta_h_ev_per_atom": 0.1,
                "block_standard_error_mev_per_atom": 1.0,
                "half_drift_mev_per_atom": 0.5,
                "status": "verified",
            },
        ]
        series = {
            "schema": "wt-zero-pressure-fusion-enthalpy-series-v2",
            "status": "verified",
            "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
            "discard_fraction": 0.5,
            "points": points,
        }
        convergence = {
            "schema": "wt-enthalpy-discard-convergence-summary-v2",
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combination_path = root / "combination.json"
            series_path = root / "series.json"
            convergence_path = root / "convergence.json"
            output_path = root / "result.json"
            for path, payload in (
                (combination_path, combination),
                (series_path, series),
                (convergence_path, convergence),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "solve_wt_melting_gibbs_helmholtz.py",
                "--combination",
                str(combination_path),
                "--enthalpy-series",
                str(series_path),
                "--enthalpy-convergence",
                str(convergence_path),
                "--out",
                str(output_path),
            ]
            with mock.patch("sys.argv", argv), mock.patch("builtins.print"):
                from scripts.solve_wt_melting_gibbs_helmholtz import main

                main()
            result = json.loads(output_path.read_text(encoding="utf-8"))
        convention = result["thermodynamic_convention"]
        self.assertEqual(convention["delta_g"], "G_liquid_minus_G_solid")
        self.assertEqual(convention["delta_h"], "H_liquid_minus_H_solid")
        self.assertFalse(
            result["uncertainty_interpretation"][
                "probabilistic_confidence_interval"
            ]
        )

    def test_solver_accepts_method_specific_kedf_input_family(self):
        combination = {
            "schema": "kedf-melting-free-energy-anchor-v1",
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "combination": root / "combination.json",
                "series": root / "series.json",
                "convergence": root / "convergence.json",
                "output": root / "result.json",
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
                from scripts.solve_wt_melting_gibbs_helmholtz import main

                main()
            result = json.loads(paths["output"].read_text(encoding="utf-8"))

        self.assertEqual(result["schema"], "kedf-gibbs-helmholtz-melting-v1")
        self.assertEqual(result["target_kedf"], "xwm")
        self.assertEqual(result["status"], "verified")


if __name__ == "__main__":
    unittest.main()
