import math
import unittest
from pathlib import Path

from scripts.combine_wt_melting_free_energy import (
    convergence_summary_matches_analysis,
    finite_size_allowance_mev_per_atom,
    fusion_enthalpy_from_ti_endpoints,
    linearized_melting_temperature,
    melting_temperature_uncertainty,
    rss,
)
from scripts.analyze_wt_pair_ti_production import parse_window_overrides
from scripts.analyze_wt_pair_ti_production import overlap_closure_mev_per_atom


class MeltingFreeEnergyCombinationTests(unittest.TestCase):
    def test_convergence_summary_must_match_selected_analysis(self):
        analysis = {
            "natoms": 108,
            "discard_fraction": 0.5,
            "delta_f_wt_minus_pair_simpson_mev_per_atom": 12.5,
        }
        summary = {
            "status": "verified",
            "phase": "solid",
            "natoms": 108,
            "integral_consensus_mev_per_atom": 12.5,
            "reports": [{"discard_fraction": 0.5, "status": "verified"}],
        }
        self.assertTrue(
            convergence_summary_matches_analysis(summary, analysis, "solid")
        )
        summary["status"] = "extension_or_refinement_required"
        self.assertFalse(
            convergence_summary_matches_analysis(summary, analysis, "solid")
        )

    def test_convergence_summary_rejects_mismatched_integral(self):
        analysis = {
            "natoms": 108,
            "discard_fraction": 0.5,
            "delta_f_wt_minus_pair_simpson_mev_per_atom": 12.5,
        }
        summary = {
            "status": "verified",
            "phase": "solid",
            "natoms": 108,
            "integral_consensus_mev_per_atom": 12.6,
            "reports": [{"discard_fraction": 0.5, "status": "verified"}],
        }
        self.assertFalse(
            convergence_summary_matches_analysis(summary, analysis, "solid")
        )

    def test_rss(self):
        self.assertEqual(rss(3.0, 4.0), 5.0)

    def test_linearized_melting_temperature(self):
        self.assertAlmostEqual(linearized_melting_temperature(900.0, 0.01, 0.1), 1000.0)
        with self.assertRaises(ValueError):
            linearized_melting_temperature(900.0, 0.1, 0.1)

    def test_finite_size_allowance_uses_larger_audited_scale(self):
        audit = {
            "finite_size_sensitivity": {
                "exact_finite_n_ideal_minus_thermodynamic_limit_mev_per_atom": 2.34,
                "polson_kbt_ln_n_over_n_mev_per_atom": 3.36,
            }
        }
        self.assertEqual(finite_size_allowance_mev_per_atom(audit), 3.36)

    def test_uncertainty_matches_finite_difference_derivatives(self):
        temperature = 900.0
        delta_g = 0.006
        delta_h = 0.09
        sigma_g = 0.001
        sigma_h = 0.002
        step = 1.0e-7
        derivative_g = (
            linearized_melting_temperature(temperature, delta_g + step, delta_h)
            - linearized_melting_temperature(temperature, delta_g - step, delta_h)
        ) / (2.0 * step)
        derivative_h = (
            linearized_melting_temperature(temperature, delta_g, delta_h + step)
            - linearized_melting_temperature(temperature, delta_g, delta_h - step)
        ) / (2.0 * step)
        expected = math.sqrt((derivative_g * sigma_g) ** 2 + (derivative_h * sigma_h) ** 2)
        self.assertAlmostEqual(
            melting_temperature_uncertainty(
                temperature, delta_g, delta_h, sigma_g, sigma_h
            ),
            expected,
            places=6,
        )

    def test_window_overrides_are_explicit_and_unique(self):
        overrides = parse_window_overrides(["lambda_0p000=/tmp/extended"])
        self.assertEqual(overrides["lambda_0p000"], Path("/tmp/extended").resolve())
        with self.assertRaises(ValueError):
            parse_window_overrides(["lambda_0p000"])
        with self.assertRaises(ValueError):
            parse_window_overrides(
                ["lambda_0p000=/tmp/one", "lambda_0p000=/tmp/two"]
            )

    def test_overlap_closure_uses_forward_plus_reverse(self):
        self.assertAlmostEqual(
            overlap_closure_mev_per_atom(0.54, -0.432, 108), 1.0
        )
        with self.assertRaises(ValueError):
            overlap_closure_mev_per_atom(0.0, 0.0, 0)

    def test_preliminary_fusion_enthalpy_uses_total_energy(self):
        solid = {
            "lambda_one_target_potential_ev_per_atom": -57.05,
            "lambda_one_target_kinetic_ev_per_atom": 0.12,
            "lambda_one_target_total_energy_ev_per_atom": -56.93,
        }
        liquid = {
            "lambda_one_target_potential_ev_per_atom": -56.96,
            "lambda_one_target_kinetic_ev_per_atom": 0.117,
            "lambda_one_target_total_energy_ev_per_atom": -56.843,
        }

        result = fusion_enthalpy_from_ti_endpoints(solid, liquid)

        self.assertAlmostEqual(result["potential_difference_ev_per_atom"], 0.09)
        self.assertAlmostEqual(result["kinetic_difference_ev_per_atom"], -0.003)
        self.assertAlmostEqual(result["total_energy_difference_ev_per_atom"], 0.087)


if __name__ == "__main__":
    unittest.main()
