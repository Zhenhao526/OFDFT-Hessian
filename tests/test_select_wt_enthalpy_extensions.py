import unittest

from scripts.select_wt_enthalpy_extensions import select_extensions


def summary(*, failed_checks=None, reasons=None, status="extension_required"):
    return {
        "schema": "wt-enthalpy-discard-convergence-summary-v2",
        "status": status,
        "points": [
            {
                "temperature_k": 975.0,
                "status": "extension_required",
                "extension_reasons": reasons or ["report_gate", "discard_sensitivity"],
                "failed_checks": failed_checks
                if failed_checks is not None
                else ["half_drift_within_tolerance"],
            }
        ],
    }


class WtEnthalpyExtensionSelectionTests(unittest.TestCase):
    def test_selects_statistical_and_half_drift_failure(self):
        result = select_extensions(summary())

        self.assertEqual(result["status"], "statistical_extension_required")
        self.assertEqual(result["critical_temperatures_k"], [975.0])

    def test_allows_pure_block_or_discard_failure(self):
        result = select_extensions(
            summary(failed_checks=[], reasons=["block_standard_error"])
        )

        self.assertEqual(result["critical_temperatures_k"], [975.0])

    def test_allows_solid_liquid_temperature_mismatch_to_extend(self):
        result = select_extensions(
            summary(
                failed_checks=["solid_liquid_temperature_means_match"],
                reasons=["report_gate"],
            )
        )

        self.assertEqual(result["status"], "statistical_extension_required")
        self.assertEqual(result["critical_temperatures_k"], [975.0])

    def test_rejects_phase_failure(self):
        with self.assertRaisesRegex(ValueError, "non-statistical"):
            select_extensions(
                summary(failed_checks=["liquid_phase_verified"])
            )

    def test_rejects_unexplained_report_gate(self):
        with self.assertRaisesRegex(ValueError, "no specific"):
            select_extensions(summary(failed_checks=[], reasons=["report_gate"]))

    def test_verified_summary_needs_no_extension(self):
        result = select_extensions(
            {
                "schema": "wt-enthalpy-discard-convergence-summary-v2",
                "status": "verified",
            }
        )

        self.assertEqual(result["status"], "no_extension_required")
        self.assertEqual(result["critical_temperatures_k"], [])


if __name__ == "__main__":
    unittest.main()
