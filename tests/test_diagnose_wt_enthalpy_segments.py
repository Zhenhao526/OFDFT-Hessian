import unittest

from scripts.diagnose_wt_enthalpy_segments import (
    integrated_autocorrelation_time,
    phase_drift_attribution,
    series_diagnostics,
)


class DiagnoseWtEnthalpySegmentsTests(unittest.TestCase):
    def test_constant_series_has_minimum_iat(self):
        self.assertEqual(
            integrated_autocorrelation_time([2.0] * 20),
            0.5,
        )

    def test_drift_attribution_uses_signed_phase_changes(self):
        result = phase_drift_attribution(
            {"signed_half_change_mev_per_atom": 2.0},
            {"signed_half_change_mev_per_atom": -5.0},
        )
        self.assertEqual(result["fusion_signed_half_change_mev_per_atom"], -7.0)
        self.assertEqual(result["dominant_phase"], "liquid")
        self.assertEqual(result["phases_with_material_drift"], ["liquid"])

    def test_series_diagnostics_reports_signed_drift(self):
        result = series_diagnostics(
            [0.0, 0.0, 2.0, 2.0],
            sample_stride_md_steps=5.0,
            blocks=2,
            maximum_lag=2,
        )
        self.assertEqual(result["signed_half_change_mev_per_atom"], 2.0)
        self.assertEqual(result["absolute_half_drift_mev_per_atom"], 2.0)
        self.assertEqual(result["quarter_means_mev_per_atom"], [0.0, 0.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
