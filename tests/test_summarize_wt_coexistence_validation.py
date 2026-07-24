import unittest

from scripts.summarize_wt_coexistence_validation import summarize


def melting():
    return {
        "schema": "wt-gibbs-helmholtz-melting-v2",
        "status": "verified",
        "checks": {"root": True},
        "melting_temperature_k": 1000.0,
    }


def run(change, temperature, status="two_phase_verified"):
    return {
        "run_dir": f"run-{temperature}",
        "status": status,
        "md": {
            "max_step": 500,
            "temperature_last_100_steps_K": {"mean": temperature},
        },
        "trajectory": {
            "atom_count": 1728,
            "nearest_neighbor_A": 2.2,
            "minimum_nearest_neighbor_A": 2.1,
        },
        "interface_migration_indicators": {
            "integrated_ordered_fraction_initial": 0.5,
            "integrated_ordered_fraction_final": 0.5 + change,
            "integrated_ordered_fraction_change": change,
        },
    }


class WtCoexistenceValidationSummaryTests(unittest.TestCase):
    def test_verifies_expected_interface_directions(self):
        result = summarize(
            melting(),
            run(0.04, 975.0),
            run(0.01, 1000.0),
            run(-0.03, 1025.0),
            low_temperature=975.0,
            nominal_temperature=1000.0,
            high_temperature=1025.0,
        )

        self.assertEqual(result["status"], "verified")
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_wrong_high_temperature_direction(self):
        result = summarize(
            melting(),
            run(0.04, 975.0),
            run(0.01, 1000.0),
            run(0.03, 1025.0),
            low_temperature=975.0,
            nominal_temperature=1000.0,
            high_temperature=1025.0,
        )

        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["checks"]["high_temperature_liquid_growth"])

    def test_requires_nominal_two_phase_state(self):
        result = summarize(
            melting(),
            run(0.04, 975.0),
            run(0.01, 1000.0, status="two_phase_not_verified"),
            run(-0.03, 1025.0),
            low_temperature=975.0,
            nominal_temperature=1000.0,
            high_temperature=1025.0,
        )

        self.assertFalse(result["checks"]["nominal_two_phase_verified"])

    def test_rejects_temperature_mean_outside_tolerance(self):
        high = run(-0.03, 1060.0)
        result = summarize(
            melting(),
            run(0.04, 975.0),
            run(0.01, 1000.0),
            high,
            low_temperature=975.0,
            nominal_temperature=1000.0,
            high_temperature=1025.0,
        )

        self.assertFalse(result["checks"]["all_run_quality_verified"])

    def test_rejects_transient_short_neighbor(self):
        high = run(-0.03, 1025.0)
        high["trajectory"]["minimum_nearest_neighbor_A"] = 1.95
        result = summarize(
            melting(),
            run(0.04, 975.0),
            run(0.01, 1000.0),
            high,
            low_temperature=975.0,
            nominal_temperature=1000.0,
            high_temperature=1025.0,
        )

        self.assertFalse(result["checks"]["all_run_quality_verified"])
        self.assertFalse(
            result["runs"]["high"]["checks"][
                "all_frames_nearest_neighbor_gt_2_A"
            ]
        )


if __name__ == "__main__":
    unittest.main()
