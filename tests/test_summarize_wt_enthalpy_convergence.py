from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.summarize_wt_enthalpy_convergence import summarize


def report(discard: float, delta_h_mev: float = 84.0) -> dict:
    return {
        "schema": "wt-zero-pressure-fusion-enthalpy-series-v2",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "discard_fraction": discard,
        "status": "verified",
        "points": [
            {
                "temperature_k": 900.0,
                "status": "verified",
                "delta_h_ev_per_atom": delta_h_mev / 1000.0,
                "block_standard_error_mev_per_atom": 1.0,
                "half_drift_mev_per_atom": 0.5,
                "liquid_minus_solid_temperature_mean_difference_k": -2.0,
                "checks": {"half_drift_within_tolerance": True},
            }
        ],
    }


def kedf_report(discard: float, method: str = "xwm") -> dict:
    result = report(discard)
    result["schema"] = "kedf-zero-pressure-fusion-enthalpy-series-v1"
    result["target_kedf"] = method
    return result


class EnthalpyConvergenceTests(unittest.TestCase):
    def test_verified_when_discard_reports_agree(self):
        result = summarize([report(0.25), report(0.5, 84.4), report(0.75, 83.8)])

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["critical_temperatures"], {})
        self.assertAlmostEqual(
            result["points"][0]["delta_h_discard_spread_mev_per_atom"], 0.6
        )
        self.assertEqual(
            result["points"][0]["reports"][0][
                "liquid_minus_solid_temperature_mean_difference_k"
            ],
            -2.0,
        )

    def test_identifies_temporal_and_statistical_extension_reasons(self):
        reports = [report(0.25, 80.0), report(0.5, 84.0), report(0.75, 84.5)]
        reports[1]["points"][0]["block_standard_error_mev_per_atom"] = 4.0
        reports[2]["status"] = "gate_failed"
        reports[2]["points"][0]["status"] = "gate_failed"
        reports[2]["points"][0]["checks"]["half_drift_within_tolerance"] = False

        result = summarize(reports)

        self.assertEqual(result["status"], "extension_required")
        self.assertEqual(
            result["critical_temperatures"]["900.000000000"],
            ["report_gate", "discard_sensitivity", "block_standard_error"],
        )
        self.assertEqual(
            result["points"][0]["failed_checks"],
            ["half_drift_within_tolerance"],
        )

    def test_rejects_mismatched_temperature_grids_without_mutating_inputs(self):
        reports = [report(0.25), report(0.5)]
        reports[1]["points"][0]["temperature_k"] = 920.0
        original = deepcopy(reports)

        with self.assertRaisesRegex(ValueError, "different temperature grids"):
            summarize(reports)

        self.assertEqual(original, reports)

    def test_requires_unique_discard_fractions(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            summarize([report(0.5), report(0.5)])

    def test_rejects_legacy_potential_only_schema(self):
        reports = [report(0.25), report(0.5)]
        reports[0]["schema"] = "wt-zero-pressure-fusion-enthalpy-series-v1"

        with self.assertRaisesRegex(ValueError, "total-energy schema"):
            summarize(reports)

    def test_kedf_reports_preserve_method_specific_schema(self):
        result = summarize(
            [kedf_report(0.25), kedf_report(0.5), kedf_report(0.75)]
        )

        self.assertEqual(
            result["schema"], "kedf-enthalpy-discard-convergence-summary-v1"
        )
        self.assertEqual(result["target_kedf"], "xwm")
        self.assertEqual(result["status"], "verified")

    def test_rejects_cross_method_kedf_reports(self):
        with self.assertRaisesRegex(ValueError, "different target KEDFs"):
            summarize([kedf_report(0.25, "xwm"), kedf_report(0.5, "lkt")])


if __name__ == "__main__":
    unittest.main()
