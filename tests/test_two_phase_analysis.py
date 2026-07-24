import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_two_phase_run.py"
SPEC = importlib.util.spec_from_file_location("analyze_two_phase_run", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class TwoPhaseAnalysisTests(unittest.TestCase):
    def test_late_series_slope_uses_latter_half(self):
        series = [(0, 0.0), (1, 10.0), (2, 20.0), (3, 22.0), (4, 24.0)]

        self.assertAlmostEqual(ANALYSIS.late_series_slope(series), 2.0)

    def test_late_series_slope_requires_two_tail_samples(self):
        self.assertIsNone(ANALYSIS.late_series_slope([(0, 0.0)]))

    def test_incomplete_trailing_frame_is_discarded(self):
        frames = [
            SimpleNamespace(step=0, positions=[object()] * 4),
            SimpleNamespace(step=1, positions=[object()] * 4),
            SimpleNamespace(step=2, positions=[object()] * 2),
        ]

        complete, dropped = ANALYSIS.complete_trajectory_frames(frames)

        self.assertEqual([frame.step for frame in complete], [0, 1])
        self.assertEqual(dropped, 1)

    def test_log_step_count_is_independent_of_observable_frequency(self):
        text = """
STEP OF MOLECULAR DYNAMICS: 496
Energy (Ry) Potential (Ry) Kinetic (Ry) Temperature (K)
-1.0 -2.0 1.0 940.0
STEP OF MOLECULAR DYNAMICS: 497
STEP OF MOLECULAR DYNAMICS: 498
STEP OF MOLECULAR DYNAMICS: 499
STEP OF MOLECULAR DYNAMICS: 500
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_md.log"
            path.write_text(text, encoding="utf-8")
            rows, max_step = ANALYSIS.parse_md_log(path)

        self.assertEqual(max_step, 500)
        self.assertEqual([row["step"] for row in rows], [496])

    def test_phase_cores_exclude_both_interfaces(self):
        self.assertEqual(ANALYSIS.phase_regions(0.25, 0.5), ("solid_half", "solid_core"))
        self.assertEqual(ANALYSIS.phase_regions(0.75, 0.5), ("liquid_half", "liquid_core"))
        self.assertEqual(ANALYSIS.phase_regions(0.02, 0.5), ("solid_half", None))
        self.assertEqual(ANALYSIS.phase_regions(0.52, 0.5), ("liquid_half", None))

    def test_integrated_ordered_fraction_weights_bins_by_atom_count(self):
        profile = {
            "bins": [
                {"n": 10, "fraction_CSP_lt_2_5": 0.8},
                {"n": 30, "fraction_CSP_lt_2_5": 0.2},
            ]
        }

        value = ANALYSIS.integrated_ordered_fraction(profile)

        self.assertAlmostEqual(value, 0.35)

    def test_npt_pressure_column_is_parsed(self):
        text = """
STEP OF MOLECULAR DYNAMICS: 7
Energy (Ry) Potential (Ry) Kinetic (Ry) Temperature (K) Pressure (kbar)
-452.7 -453.6 0.9 931.5 -1.09
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_md.log"
            path.write_text(text, encoding="utf-8")
            rows, max_step = ANALYSIS.parse_md_log(path)

        self.assertEqual(max_step, 7)
        self.assertEqual(rows[0]["temperature_K"], 931.5)
        self.assertEqual(rows[0]["pressure_kbar"], -1.09)

    def test_energy_drift_uses_actual_atom_count(self):
        rows = [{"total_Ry": -10.0}, {"total_Ry": -9.0}]

        drift = ANALYSIS.energy_drift_ev_per_atom(rows, 64)

        self.assertAlmostEqual(drift, ANALYSIS.RY_TO_EV / 64.0)

    def test_high_temperature_solid_can_pass_by_phase_contrast(self):
        result = {
            "trajectory": {
                "nearest_neighbor_A": 2.22,
                "minimum_nearest_neighbor_A": 2.18,
            },
            "structure": {
                "solid_core": {"median_A2": 2.77, "fraction_CSP_lt_2_5": 0.43},
                "liquid_core": {"median_A2": 3.60, "fraction_CSP_lt_2_5": 0.26},
            },
            "identity_region_displacement": {
                "solid_seed": {"lindemann_ratio": 0.139},
                "liquid_seed": {"lindemann_ratio": 0.256},
            },
        }

        verified, checks = ANALYSIS.evaluate_two_phase(result)

        self.assertTrue(verified)
        self.assertTrue(checks["core_median_CSP_contrast_gt_0_5_A2"])

    def test_two_phase_gate_rejects_transient_short_neighbor(self):
        result = {
            "trajectory": {
                "nearest_neighbor_A": 2.22,
                "minimum_nearest_neighbor_A": 1.95,
            },
            "structure": {
                "solid_core": {"median_A2": 2.0, "fraction_CSP_lt_2_5": 0.6},
                "liquid_core": {"median_A2": 6.0, "fraction_CSP_lt_2_5": 0.1},
            },
            "identity_region_displacement": {
                "solid_seed": {"lindemann_ratio": 0.10},
                "liquid_seed": {"lindemann_ratio": 0.25},
            },
        }

        verified, checks = ANALYSIS.evaluate_two_phase(result)

        self.assertFalse(verified)
        self.assertFalse(checks["all_frames_nearest_neighbor_gt_2_A"])


if __name__ == "__main__":
    unittest.main()
