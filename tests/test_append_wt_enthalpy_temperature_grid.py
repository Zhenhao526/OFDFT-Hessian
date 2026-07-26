import unittest

from scripts.append_wt_enthalpy_temperature_grid import append_temperature_grid
from scripts.summarize_wt_enthalpy_convergence import summarize


def series(discard, temperatures):
    return {
        "schema": "wt-zero-pressure-fusion-enthalpy-series-v2",
        "status": "verified",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "discard_fraction": discard,
        "points": [
            {
                "temperature_k": temperature,
                "delta_h_ev_per_atom": (80.0 + temperature / 100.0 + discard)
                / 1000.0,
                "block_standard_error_mev_per_atom": 1.0,
                "half_drift_mev_per_atom": 0.5,
                "liquid_minus_solid_temperature_mean_difference_k": 0.0,
                "checks": {"all": True},
                "status": "verified",
            }
            for temperature in temperatures
        ],
    }


def grid(temperatures):
    reports = {
        discard: series(discard, temperatures)
        for discard in (0.25, 0.5, 0.75)
    }
    return reports, summarize(list(reports.values()))


class AppendWtEnthalpyTemperatureGridTests(unittest.TestCase):
    def test_appends_disjoint_verified_grid(self):
        base, base_convergence = grid((900.0, 975.0, 1050.0))
        appended, appended_convergence = grid((1100.0,))
        merged, convergence = append_temperature_grid(
            base_series_by_discard=base,
            base_convergence=base_convergence,
            appended_series_by_discard=appended,
            appended_convergence=appended_convergence,
        )
        self.assertEqual(
            [point["temperature_k"] for point in merged[0.5]["points"]],
            [900.0, 975.0, 1050.0, 1100.0],
        )
        self.assertEqual(convergence["status"], "verified")
        self.assertEqual(len(convergence["points"]), 4)

    def test_rejects_overlapping_temperature(self):
        base, base_convergence = grid((900.0, 975.0))
        appended, appended_convergence = grid((975.0, 1100.0))
        with self.assertRaisesRegex(ValueError, "overlap"):
            append_temperature_grid(
                base_series_by_discard=base,
                base_convergence=base_convergence,
                appended_series_by_discard=appended,
                appended_convergence=appended_convergence,
            )

    def test_rejects_series_convergence_mismatch(self):
        base, base_convergence = grid((900.0, 975.0))
        appended, appended_convergence = grid((1100.0,))
        appended_convergence["points"][0]["reports"][0][
            "delta_h_mev_per_atom"
        ] += 1.0
        with self.assertRaisesRegex(ValueError, "enthalpies differ"):
            append_temperature_grid(
                base_series_by_discard=base,
                base_convergence=base_convergence,
                appended_series_by_discard=appended,
                appended_convergence=appended_convergence,
            )

    def test_rejects_different_gates(self):
        base, base_convergence = grid((900.0, 975.0))
        appended, appended_convergence = grid((1100.0,))
        appended_convergence["maximum_discard_spread_mev_per_atom"] = 3.0
        with self.assertRaisesRegex(ValueError, "different maximum_discard"):
            append_temperature_grid(
                base_series_by_discard=base,
                base_convergence=base_convergence,
                appended_series_by_discard=appended,
                appended_convergence=appended_convergence,
            )


if __name__ == "__main__":
    unittest.main()
