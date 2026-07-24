import math
import unittest

from mpn_melting.trajectory import MdFrame, non_affine_msd_series
from scripts.analyze_phase_run import late_msd_window, linear_slope, solid_initial_checks
from scripts.check_mpn_pressure_finite_difference import finite_difference_pressure_kbar


class PhaseAnalysisTests(unittest.TestCase):
    def test_non_affine_msd_ignores_isotropic_cell_scaling(self):
        frames = [
            MdFrame(0, ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)), [(2.0, 3.0, 4.0)]),
            MdFrame(1, ((11.0, 0.0, 0.0), (0.0, 11.0, 0.0), (0.0, 0.0, 11.0)), [(2.2, 3.3, 4.4)]),
        ]

        self.assertAlmostEqual(non_affine_msd_series(frames)[-1][1], 0.0)

    def test_non_affine_msd_unwraps_periodic_crossing(self):
        lattice = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
        frames = [
            MdFrame(0, lattice, [(9.8, 0.0, 0.0)]),
            MdFrame(1, lattice, [(0.2, 0.0, 0.0)]),
        ]

        self.assertTrue(math.isclose(non_affine_msd_series(frames)[-1][1], 0.16, abs_tol=1e-12))

    def test_linear_slope_distinguishes_plateau_from_diffusion(self):
        self.assertAlmostEqual(linear_slope([(0.0, 0.29), (10.0, 0.295), (20.0, 0.296)]), 0.0003)
        self.assertAlmostEqual(linear_slope([(0.0, 0.55), (10.0, 0.58), (20.0, 0.61)]), 0.003)

    def test_late_msd_window_uses_half_of_long_trajectory(self):
        series = [(step, float(step)) for step in range(200)]

        window = late_msd_window(series)

        self.assertEqual(len(window), 100)
        self.assertEqual(window[0][0], 100)

    def test_late_msd_window_keeps_at_least_six_short_points(self):
        series = [(step, float(step)) for step in range(10)]

        window = late_msd_window(series)

        self.assertEqual(len(window), 6)
        self.assertEqual(window[0][0], 4)

    def test_fresh_solid_requires_pristine_initial_order(self):
        checks = solid_initial_checks(0.4, 2.8, thermalized_initial=False)

        self.assertEqual(checks, {"initial_ordered_fraction_gt_0_9": False})

    def test_thermalized_solid_accepts_ordered_hot_initial_state(self):
        checks = solid_initial_checks(0.4, 2.8, thermalized_initial=True)

        self.assertTrue(all(checks.values()))

    def test_thermalized_solid_rejects_disordered_initial_state(self):
        checks = solid_initial_checks(0.05, 7.0, thermalized_initial=True)

        self.assertFalse(any(checks.values()))

    def test_pressure_finite_difference_uses_negative_energy_derivative(self):
        pressure = finite_difference_pressure_kbar(-10.0, -10.2, 99.0, 101.0)

        self.assertAlmostEqual(pressure, 160.21766208)


if __name__ == "__main__":
    unittest.main()
