import math
import unittest

from scripts.prepare_al108_ti_windows import bennett_acceptance_ratio, simpson


def ti_row(lambda_value, mean):
    return {
        "lambda": lambda_value,
        "delta_U_last_half_eV": {"mean": mean},
    }


class TiAnalysisTests(unittest.TestCase):
    def test_simpson_integrates_quadratic_exactly(self):
        rows = [ti_row(value, value * value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(math.isclose(simpson(rows), 1.0 / 3.0, rel_tol=1e-12))

    def test_simpson_requires_uniform_odd_grid(self):
        self.assertIsNone(simpson([ti_row(0.0, 0.0), ti_row(1.0, 1.0)]))
        self.assertIsNone(
            simpson([ti_row(0.0, 0.0), ti_row(0.4, 0.4), ti_row(1.0, 1.0)])
        )

    def test_bar_recovers_constant_energy_offset(self):
        value = bennett_acceptance_ratio([0.4] * 20, [0.4] * 20, 0.25, 750.0)
        self.assertTrue(math.isclose(value, 0.1, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
