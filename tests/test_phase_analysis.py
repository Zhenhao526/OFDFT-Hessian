import math
import random
import unittest
from unittest import mock

from mpn_melting.trajectory import (
    MdFrame,
    non_affine_msd_series,
    time_origin_averaged_msd_series,
)
from scripts.analyze_phase_run import (
    late_msd_window,
    linear_slope,
    nearest_neighbor_summary,
    recent_structure_summary,
    solid_initial_checks,
)
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

    def test_time_origin_averaged_msd_is_zero_for_static_solid(self):
        lattice = ((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0))
        positions = [(2.0, 2.0, 2.0), (6.0, 6.0, 6.0)]
        frames = [MdFrame(step, lattice, positions) for step in range(30)]

        series = time_origin_averaged_msd_series(frames)

        self.assertTrue(series)
        self.assertTrue(all(math.isclose(value, 0.0, abs_tol=1e-12) for _, value in series))

    def test_time_origin_averaged_msd_detects_random_walk(self):
        random_generator = random.Random(20260722)
        lattice = ((50.0, 0.0, 0.0), (0.0, 50.0, 0.0), (0.0, 0.0, 50.0))
        positions = [[5.0 + index, 10.0, 15.0] for index in range(16)]
        frames = []
        for step in range(120):
            frames.append(MdFrame(step, lattice, [tuple(position) for position in positions]))
            for position in positions:
                for axis in range(3):
                    position[axis] = (
                        position[axis] + random_generator.gauss(0.0, 0.08)
                    ) % 50.0

        series = time_origin_averaged_msd_series(frames)
        slope = linear_slope(series)

        self.assertGreater(slope, 0.001)

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

    def test_nearest_neighbor_summary_detects_midtrajectory_close_approach(self):
        lattice = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
        frames = [
            MdFrame(0, lattice, [(1.0, 1.0, 1.0), (4.0, 1.0, 1.0)]),
            MdFrame(5, lattice, [(1.0, 1.0, 1.0), (2.5, 1.0, 1.0)]),
            MdFrame(10, lattice, [(1.0, 1.0, 1.0), (3.8, 1.0, 1.0)]),
        ]

        summary = nearest_neighbor_summary(frames)

        self.assertAlmostEqual(summary["initial_A"], 3.0)
        self.assertAlmostEqual(summary["final_A"], 2.8)
        self.assertAlmostEqual(summary["minimum_A"], 1.5)
        self.assertEqual(summary["minimum_step"], 5)

    def test_recent_structure_summary_averages_endpoint_fluctuations(self):
        lattice = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
        frames = [
            MdFrame(step, lattice, [(1.0, 1.0, 1.0), (4.0, 1.0, 1.0)])
            for step in (0, 5, 10)
        ]

        with mock.patch(
            "scripts.analyze_phase_run.centrosymmetry_parameters",
            side_effect=([1.0, 3.0], [1.0, 1.0], [3.0, 3.0]),
        ):
            summary = recent_structure_summary(frames)

        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["first_step"], 0)
        self.assertEqual(summary["last_step"], 10)
        self.assertAlmostEqual(summary["median_CSP_A2"]["mean"], 2.0)
        self.assertAlmostEqual(
            summary["ordered_fraction_CSP_lt_2_5"]["mean"], 0.5
        )

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
