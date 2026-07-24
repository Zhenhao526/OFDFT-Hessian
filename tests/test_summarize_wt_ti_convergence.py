from __future__ import annotations

from copy import deepcopy
import math
import unittest

from scripts.summarize_wt_ti_convergence import summarize


def report(discard: float, integral: float = 0.5):
    return {
        "schema": "wt-pair-ti-production-analysis-v1",
        "phase": "solid",
        "natoms": 100,
        "target_temperature_K": 900.0,
        "temperature_tolerance_k": 50.0,
        "minimum_distance_gate_angstrom": 2.0,
        "discard_fraction": discard,
        "status": "verified",
        "checks": {
            "structural_gate": True,
            "temperature_gate": True,
            "block_standard_error_gate": True,
            "half_drift_gate": True,
            "quadrature_gate": True,
            "adjacent_overlap_gate": True,
            "adjacent_overlap_closure_gate": True,
        },
        "delta_f_wt_minus_pair_simpson_mev_per_atom": integral,
        "block_standard_error_mev_per_atom": 0.1,
        "half_drift_mev_per_atom": 0.2,
        "quadrature_difference_mev_per_atom": 0.3,
        "minimum_adjacent_effective_sample_fraction": 0.5,
        "maximum_adjacent_overlap_closure_mev_per_atom": 0.2,
        "windows": [
            {
                "label": "lambda_0p000",
                "lambda": 0.0,
                "status": "complete",
                "max_step": 1000,
                "phase_status": "solid_verified",
                "minimum_pair_distance_angstrom": 2.3,
                "temperature_mean_k": 900.0,
                "du_mean_ev_system": 0.05,
                "du_block_standard_error_ev_system": 0.01,
                "du_first_half_ev_system": 0.04,
                "du_second_half_ev_system": 0.06,
            },
            {
                "label": "lambda_1p000",
                "lambda": 1.0,
                "status": "complete",
                "max_step": 1000,
                "phase_status": "solid_verified",
                "minimum_pair_distance_angstrom": 2.25,
                "temperature_mean_k": 905.0,
                "du_mean_ev_system": -0.02,
                "du_block_standard_error_ev_system": 0.02,
                "du_first_half_ev_system": -0.03,
                "du_second_half_ev_system": -0.01,
            },
        ],
    }


class TiConvergenceTests(unittest.TestCase):
    def test_verified_when_reports_and_windows_are_stable(self):
        result = summarize(
            [report(0.25, 0.4), report(0.50, 0.5), report(0.75, 0.6)]
        )

        self.assertEqual(result["status"], "verified")
        self.assertTrue(
            math.isclose(result["integral_discard_spread_mev_per_atom"], 0.2)
        )
        self.assertEqual(result["critical_windows"], [])
        self.assertFalse(result["lambda_refinement_required"])

    def test_identifies_window_extension_and_lambda_refinement_separately(self):
        reports = [report(0.25), report(0.50), report(0.75)]
        reports[0]["windows"][0]["temperature_mean_k"] = 840.0
        reports[2]["windows"][0]["du_mean_ev_system"] = 0.30
        reports[1]["checks"]["quadrature_gate"] = False
        reports[1]["checks"]["adjacent_overlap_closure_gate"] = False
        reports[1]["status"] = "production_gate_failed"

        result = summarize(reports)

        self.assertEqual(result["status"], "extension_or_refinement_required")
        self.assertTrue(result["lambda_refinement_required"])
        critical = {
            item["label"]: item["reasons"] for item in result["critical_windows"]
        }
        self.assertIn("temperature_gate", critical["lambda_0p000"])
        self.assertIn("window_discard_spread", critical["lambda_0p000"])

    def test_liquid_diffusion_only_failure_is_separated_from_structural_failure(self):
        reports = [report(0.25), report(0.50), report(0.75)]
        for item in reports:
            item["phase"] = "liquid"
            for window in item["windows"]:
                window["phase_status"] = "liquid_verified"
            window = item["windows"][-1]
            window["phase_status"] = "liquid_not_verified"
            window["phase_gate_checks"] = {
                "all_frames_nearest_neighbor_gt_2_A": True,
                "final_MSD_gt_0_3_A2": True,
                "late_MSD_slope_gt_0_001_A2_per_step": False,
                "median_CSP_gt_6_A2": True,
                "nearest_neighbor_gt_2_A": True,
                "ordered_fraction_lt_0_1": True,
            }
            item["checks"]["structural_gate"] = False
            item["status"] = "production_gate_failed"

        result = summarize(reports)

        critical = {
            item["label"]: item["reasons"] for item in result["critical_windows"]
        }
        self.assertEqual(
            critical["lambda_1p000"], ["liquid_diffusion_slope_gate"]
        )

    def test_liquid_order_failure_remains_a_structural_phase_failure(self):
        reports = [report(0.25), report(0.50), report(0.75)]
        for item in reports:
            item["phase"] = "liquid"
            for window in item["windows"]:
                window["phase_status"] = "liquid_verified"
            window = item["windows"][-1]
            window["phase_status"] = "liquid_not_verified"
            window["phase_gate_checks"] = {
                "all_frames_nearest_neighbor_gt_2_A": True,
                "final_MSD_gt_0_3_A2": True,
                "late_MSD_slope_gt_0_001_A2_per_step": False,
                "median_CSP_gt_6_A2": True,
                "nearest_neighbor_gt_2_A": True,
                "ordered_fraction_lt_0_1": False,
            }
            item["checks"]["structural_gate"] = False
            item["status"] = "production_gate_failed"

        result = summarize(reports)

        critical = {
            item["label"]: item["reasons"] for item in result["critical_windows"]
        }
        self.assertEqual(critical["lambda_1p000"], ["phase_gate"])

    def test_does_not_mutate_input_reports(self):
        reports = [report(0.25), report(0.50)]
        original = deepcopy(reports)

        summarize(reports)

        self.assertEqual(reports, original)


if __name__ == "__main__":
    unittest.main()
