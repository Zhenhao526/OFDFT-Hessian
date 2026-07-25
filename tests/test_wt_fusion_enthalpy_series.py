import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.analyze_wt_fusion_enthalpy_series import (
    KBAR_A3_TO_EV,
    block_standard_error,
    concatenate_segment_rows,
    fusion_enthalpy_components,
    main,
)


class FusionEnthalpySeriesTests(unittest.TestCase):
    def test_constant_blocks_have_zero_error(self):
        self.assertEqual(block_standard_error([2.0] * 20, 10), 0.0)

    def test_block_error_uses_block_means(self):
        values = [value for value in range(20)]
        result = block_standard_error(values, 10)
        block_means = [0.5 + 2.0 * index for index in range(10)]
        average = sum(block_means) / 10
        expected = math.sqrt(
            sum((value - average) ** 2 for value in block_means) / 9
        ) / math.sqrt(10)
        self.assertAlmostEqual(result, expected)

    def test_block_error_includes_remainder_samples(self):
        values = [0.0] * 20 + [21.0]

        result = block_standard_error(values, 10)

        block_means = [0.0] * 9 + [7.0]
        average = sum(block_means) / 10
        expected = math.sqrt(
            sum((value - average) ** 2 for value in block_means) / 9
        ) / math.sqrt(10)
        self.assertAlmostEqual(result, expected)

    def test_pressure_volume_conversion(self):
        self.assertAlmostEqual(1.0 * 10.0 * KBAR_A3_TO_EV, 0.00624150907446)

    def test_fusion_enthalpy_uses_sampled_total_energy(self):
        solid = {
            "potential_mean_ev_per_atom": -57.05,
            "kinetic_mean_ev_per_atom": 0.12,
            "total_mean_ev_per_atom": -56.93,
        }
        liquid = {
            "potential_mean_ev_per_atom": -56.96,
            "kinetic_mean_ev_per_atom": 0.117,
            "total_mean_ev_per_atom": -56.843,
        }

        result = fusion_enthalpy_components(solid, liquid, 0.001)

        self.assertAlmostEqual(result["potential_difference_ev_per_atom"], 0.09)
        self.assertAlmostEqual(result["kinetic_difference_ev_per_atom"], -0.003)
        self.assertAlmostEqual(result["delta_h_ev_per_atom"], 0.088)

    def test_restart_segments_accumulate_and_drop_duplicate_step_zero(self):
        first = ([{"step": 0}, {"step": 5}, {"step": 10}], 10)
        second = ([{"step": 0}, {"step": 5}, {"step": 10}], 10)

        rows, completed_steps, segment_steps = concatenate_segment_rows(
            [first, second]
        )

        self.assertEqual([row["step"] for row in rows], [0, 5, 10, 5, 10])
        self.assertEqual(completed_steps, 20)
        self.assertEqual(segment_steps, [10, 10])

    def test_phase_temperature_difference_gate_rejects_opposite_offsets(self):
        manifest = {
            "target_kedf": "wt",
            "points": [
                {
                    "temperature_k": 900.0,
                    "target_pressure_kbar": 0.0,
                    "steps": 100,
                    "solid_run": "solid",
                    "liquid_run": "liquid",
                    "solid_volume_per_atom_A3": 18.0,
                    "liquid_volume_per_atom_A3": 19.0,
                }
            ],
        }

        def phase_result(phase):
            temperature = 919.0 if phase == "solid" else 881.0
            total = -56.9 if phase == "solid" else -56.8
            return {
                "phase_status": f"{phase}_verified",
                "max_step": 100,
                "natoms": 108,
                "minimum_nearest_neighbor_angstrom": 2.2,
                "temperature_k": {"mean": temperature},
                "pressure_kbar": {"mean": 0.0},
                "potential_mean_ev_per_atom": total - 0.12,
                "kinetic_mean_ev_per_atom": 0.12,
                "total_mean_ev_per_atom": total,
                "total_block_standard_error_mev_per_atom": 1.0,
                "total_first_half_ev_per_atom": total,
                "total_second_half_ev_per_atom": total,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            output_path = root / "analysis.json"
            manifest_path.write_text(__import__("json").dumps(manifest))
            with mock.patch(
                "scripts.analyze_wt_fusion_enthalpy_series.analyze_run",
                side_effect=lambda runs, phase, discard, blocks: phase_result(phase),
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_wt_fusion_enthalpy_series.py",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()

            result = __import__("json").loads(output_path.read_text())
            point = result["points"][0]
            self.assertTrue(point["checks"]["temperature_means_within_tolerance"])
            self.assertFalse(
                point["checks"]["solid_liquid_temperature_means_match"]
            )
            self.assertEqual(
                point["liquid_minus_solid_temperature_mean_difference_k"],
                -38.0,
            )
            self.assertEqual(point["status"], "gate_failed")
            self.assertEqual(result["status"], "gate_failed")

    def test_phase_specific_requested_steps_are_gated_independently(self):
        manifest = {
            "target_kedf": "wt",
            "points": [
                {
                    "temperature_k": 900.0,
                    "target_pressure_kbar": 0.0,
                    "steps": 100,
                    "solid_steps": 150,
                    "liquid_steps": 100,
                    "solid_run": "solid",
                    "liquid_run": "liquid",
                    "solid_volume_per_atom_A3": 18.0,
                    "liquid_volume_per_atom_A3": 19.0,
                }
            ],
        }

        def phase_result(phase):
            total = -56.9 if phase == "solid" else -56.8
            return {
                "phase_status": f"{phase}_verified",
                "max_step": 149 if phase == "solid" else 100,
                "natoms": 108,
                "minimum_nearest_neighbor_angstrom": 2.2,
                "temperature_k": {"mean": 900.0},
                "pressure_kbar": {"mean": 0.0},
                "potential_mean_ev_per_atom": total - 0.12,
                "kinetic_mean_ev_per_atom": 0.12,
                "total_mean_ev_per_atom": total,
                "total_block_standard_error_mev_per_atom": 1.0,
                "total_first_half_ev_per_atom": total,
                "total_second_half_ev_per_atom": total,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            output_path = root / "analysis.json"
            manifest_path.write_text(__import__("json").dumps(manifest))
            with mock.patch(
                "scripts.analyze_wt_fusion_enthalpy_series.analyze_run",
                side_effect=lambda runs, phase, discard, blocks: phase_result(phase),
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_wt_fusion_enthalpy_series.py",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()

            point = __import__("json").loads(output_path.read_text())["points"][0]

        self.assertEqual(point["requested_solid_steps"], 150)
        self.assertEqual(point["requested_liquid_steps"], 100)
        self.assertFalse(point["checks"]["requested_steps_reached"])


if __name__ == "__main__":
    unittest.main()
