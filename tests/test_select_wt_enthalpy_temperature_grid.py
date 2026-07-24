import unittest

from scripts.select_wt_enthalpy_temperature_grid import select_grid


class EnthalpyTemperatureGridTests(unittest.TestCase):
    def test_uses_three_point_grid_when_1050_brackets_interval(self):
        result = select_grid(990.0, 40.0)

        self.assertEqual(result["status"], "grid_verified")
        self.assertEqual(result["enthalpy_temperature_grid_k"], [900.0, 975.0, 1050.0])

    def test_adds_1100_point_when_conservative_upper_bound_exceeds_1050(self):
        result = select_grid(1008.0, 56.0)

        self.assertEqual(result["status"], "grid_verified")
        self.assertEqual(
            result["enthalpy_temperature_grid_k"],
            [900.0, 975.0, 1050.0, 1100.0],
        )

    def test_rejects_interval_outside_all_prepared_grids(self):
        result = select_grid(1040.0, 70.0)

        self.assertEqual(result["status"], "grid_incomplete")
        self.assertFalse(result["checks"]["upper_interval_not_above_grid"])


if __name__ == "__main__":
    unittest.main()
