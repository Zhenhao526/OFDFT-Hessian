import unittest

from scripts.merge_wt_enthalpy_temperature_grid import (
    merge_temperature_grid,
)


def series(discard, values):
    return {
        "schema": "wt-zero-pressure-fusion-enthalpy-series-v2",
        "status": "verified",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "discard_fraction": discard,
        "points": [
            {
                "temperature_k": temperature,
                "delta_h_ev_per_atom": value / 1000.0,
                "block_standard_error_mev_per_atom": 1.0,
                "half_drift_mev_per_atom": 0.5,
                "liquid_minus_solid_temperature_mean_difference_k": 0.0,
                "checks": {"all": True},
                "status": "verified",
            }
            for temperature, value in values
        ],
    }


def convergence(points):
    return {
        "schema": "wt-enthalpy-discard-convergence-summary-v2",
        "status": "verified",
        "enthalpy_energy_definition": "sampled_total_energy_plus_external_pv",
        "maximum_discard_spread_mev_per_atom": 2.0,
        "maximum_block_standard_error_mev_per_atom": 3.0,
        "points": points,
    }


class MergeWtEnthalpyTemperatureGridTests(unittest.TestCase):
    def setUp(self):
        self.current_series = {
            discard: series(
                discard,
                [(975.0, 84.0 + discard), (1050.0, 88.0 + discard)],
            )
            for discard in (0.25, 0.5, 0.75)
        }
        self.current_convergence = convergence(
            [
                {
                    "temperature_k": temperature,
                    "status": "verified",
                    "reports": [
                        {
                            "discard_fraction": discard,
                            "status": "verified",
                            "delta_h_mev_per_atom": value,
                            "block_standard_error_mev_per_atom": 1.0,
                            "half_drift_mev_per_atom": 0.5,
                        }
                        for discard, value in (
                            (0.25, base + 0.25),
                            (0.5, base + 0.5),
                            (0.75, base + 0.75),
                        )
                    ],
                }
                for temperature, base in ((975.0, 84.0), (1050.0, 88.0))
            ]
        )
        self.archived_convergence = convergence(
            [
                {
                    "temperature_k": 900.0,
                    "status": "verified",
                    "reports": [
                        {
                            "discard_fraction": discard,
                            "status": "verified",
                            "delta_h_mev_per_atom": 82.0 + discard,
                            "block_standard_error_mev_per_atom": 2.0,
                            "half_drift_mev_per_atom": 1.0,
                            "liquid_minus_solid_temperature_mean_difference_k": -1.0,
                        }
                        for discard in (0.25, 0.5, 0.75)
                    ],
                },
                {
                    "temperature_k": 975.0,
                    "status": "extension_required",
                    "reports": [],
                },
            ]
        )

    def test_merges_only_verified_archived_anchor_point(self):
        merged, summary = merge_temperature_grid(
            anchor_temperature_k=900.0,
            archived_convergence=self.archived_convergence,
            current_series_by_discard=self.current_series,
            current_convergence=self.current_convergence,
        )
        self.assertEqual(
            [point["temperature_k"] for point in merged[0.5]["points"]],
            [900.0, 975.0, 1050.0],
        )
        self.assertAlmostEqual(
            merged[0.5]["points"][0]["delta_h_ev_per_atom"], 0.0825
        )
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(len(summary["points"]), 3)

    def test_rejects_unverified_archived_anchor_report(self):
        self.archived_convergence["points"][0]["reports"][1][
            "status"
        ] = "gate_failed"
        with self.assertRaisesRegex(ValueError, "not verified"):
            merge_temperature_grid(
                anchor_temperature_k=900.0,
                archived_convergence=self.archived_convergence,
                current_series_by_discard=self.current_series,
                current_convergence=self.current_convergence,
            )

    def test_rejects_mismatched_current_series_and_convergence(self):
        self.current_convergence["points"][0]["reports"][0][
            "delta_h_mev_per_atom"
        ] += 1.0
        with self.assertRaisesRegex(ValueError, "enthalpies differ"):
            merge_temperature_grid(
                anchor_temperature_k=900.0,
                archived_convergence=self.archived_convergence,
                current_series_by_discard=self.current_series,
                current_convergence=self.current_convergence,
            )


if __name__ == "__main__":
    unittest.main()
