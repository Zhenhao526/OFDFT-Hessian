from __future__ import annotations

import unittest

from mpn_melting.structures import AtomSet
from scripts.prepare_al108_volume_scan import (
    bracket_linear_fit,
    point_seed,
    point_seed_provenance,
    pressure_bracket,
    pressure_monotonically_decreases,
    pressure_monotonicity_diagnostic,
    scaled_to_volume,
    zero_pressure_checks,
)


class VolumeScanSeedTests(unittest.TestCase):
    def test_volume_points_use_independent_reproducible_streams(self) -> None:
        self.assertEqual([point_seed(73000, index) for index in range(3)], [73000, 73001, 73002])

    def test_executed_seed_is_available_for_point_and_run_metadata(self) -> None:
        self.assertEqual(point_seed_provenance(73000, 2), {"md_seed": 73002})


class ScaledToVolumeTests(unittest.TestCase):
    def test_preserves_velocities_and_movements(self) -> None:
        atoms = AtomSet(
            ["Al"],
            [(0.25, 0.5, 0.75)],
            [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)],
            velocities=[(0.1, -0.2, 0.3)],
            movements=[(1, 0, 1)],
        )
        scaled = scaled_to_volume(atoms, 27.0)
        self.assertEqual(scaled.velocities, atoms.velocities)
        self.assertEqual(scaled.movements, atoms.movements)
        self.assertEqual(scaled.scaled_positions, atoms.scaled_positions)
        self.assertAlmostEqual(scaled.lattice_vectors[0][0], 3.0)


class ZeroPressureVolumeGateTests(unittest.TestCase):
    def test_rejects_nonmonotonic_pressures_despite_negative_global_slope(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 18.0,
                "pressure_last_half_kbar": {"mean": 5.0},
            },
            {
                "volume_per_atom_A3": 18.2,
                "pressure_last_half_kbar": {"mean": 6.0},
            },
            {
                "volume_per_atom_A3": 18.4,
                "pressure_last_half_kbar": {"mean": -5.0},
            },
        ]
        self.assertFalse(pressure_monotonically_decreases(rows))

    def test_accepts_small_pressure_reversal_within_sampling_uncertainty(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 18.0,
                "pressure_last_half_kbar": {"mean": 4.0, "sd": 2.0, "n": 60},
            },
            {
                "volume_per_atom_A3": 18.2,
                "pressure_last_half_kbar": {"mean": -1.0, "sd": 2.0, "n": 60},
            },
            {
                "volume_per_atom_A3": 18.4,
                "pressure_last_half_kbar": {"mean": -0.8, "sd": 2.0, "n": 60},
            },
        ]
        diagnostic = pressure_monotonicity_diagnostic(rows)
        self.assertTrue(pressure_monotonically_decreases(rows))
        self.assertLess(diagnostic[1]["pressure_drop_kbar"], 0.0)
        self.assertTrue(diagnostic[1]["consistent_with_decrease"])

    def test_rejects_pressure_reversal_beyond_sampling_uncertainty(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 18.0,
                "pressure_last_half_kbar": {"mean": 4.0, "sd": 1.0, "n": 60},
            },
            {
                "volume_per_atom_A3": 18.2,
                "pressure_last_half_kbar": {"mean": -2.0, "sd": 1.0, "n": 60},
            },
            {
                "volume_per_atom_A3": 18.4,
                "pressure_last_half_kbar": {"mean": 0.0, "sd": 1.0, "n": 60},
            },
        ]
        self.assertFalse(pressure_monotonically_decreases(rows))

    def test_zero_pressure_fit_uses_local_bracket_when_scan_is_curved(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 18.65,
                "pressure_last_half_kbar": {"mean": 8.6313},
            },
            {
                "volume_per_atom_A3": 18.90,
                "pressure_last_half_kbar": {"mean": -0.3010},
            },
            {
                "volume_per_atom_A3": 19.15,
                "pressure_last_half_kbar": {"mean": -6.9841},
            },
        ]
        bracket = pressure_bracket(rows)
        fit = bracket_linear_fit(rows, bracket)
        self.assertEqual(bracket, [18.65, 18.9])
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertGreater(fit["zero_pressure_volume_per_atom_A3"], 18.65)
        self.assertLess(fit["zero_pressure_volume_per_atom_A3"], 18.9)

    def test_accepts_physical_bracket_and_fit(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 17.8,
                "max_step": 200,
                "nearest_neighbor_A": 2.4,
                "pressure_last_half_kbar": {"mean": 4.0},
            },
            {
                "volume_per_atom_A3": 18.0,
                "max_step": 200,
                "nearest_neighbor_A": 2.4,
                "pressure_last_half_kbar": {"mean": -2.0},
            },
        ]
        bracket = pressure_bracket(rows)
        fit = {
            "slope_kbar_per_A3_per_atom": -30.0,
            "zero_pressure_volume_per_atom_A3": 17.93,
        }
        checks = zero_pressure_checks(rows, rows, 200, fit, bracket)
        self.assertEqual(bracket, [17.8, 18.0])
        self.assertTrue(all(checks.values()))

    def test_rejects_unbracketed_extrapolation(self) -> None:
        rows = [
            {
                "volume_per_atom_A3": 17.8,
                "max_step": 200,
                "nearest_neighbor_A": 2.4,
                "pressure_last_half_kbar": {"mean": 4.0},
            },
            {
                "volume_per_atom_A3": 18.0,
                "max_step": 200,
                "nearest_neighbor_A": 2.4,
                "pressure_last_half_kbar": {"mean": 1.0},
            },
        ]
        fit = {
            "slope_kbar_per_A3_per_atom": -15.0,
            "zero_pressure_volume_per_atom_A3": 18.07,
        }
        checks = zero_pressure_checks(rows, rows, 200, fit, pressure_bracket(rows))
        self.assertFalse(checks["pressure_bracket_found"])
        self.assertFalse(checks["zero_pressure_fit_inside_bracket"])


if __name__ == "__main__":
    unittest.main()
