import unittest
from pathlib import Path

from scripts.build_wt_targeted_enthalpy_manifest import (
    build_targeted_manifest,
)


class BuildWtTargetedEnthalpyManifestTests(unittest.TestCase):
    def test_appends_only_verified_solid_segment(self):
        parent = {
            "schema": "wt-zero-pressure-fusion-enthalpy-manifest-v1",
            "target_kedf": "wt",
            "points": [
                {
                    "temperature_k": 1050.0,
                    "target_pressure_kbar": 0.0,
                    "steps": 15000,
                    "solid_run": "/solid-r3",
                    "liquid_run": "/liquid-r3",
                    "solid_runs": ["/solid-base", "/solid-r3"],
                    "liquid_runs": ["/liquid-base", "/liquid-r3"],
                    "segment_steps": [9000, 6000],
                    "solid_volume_per_atom_A3": 18.25,
                    "liquid_volume_per_atom_A3": 19.10,
                }
            ],
        }
        extension = {
            "extension_schema": "wt-zero-pressure-enthalpy-phase-extension-v1",
            "target_kedf": "wt",
            "temperature_K": 1050.0,
            "target_pressure_kbar": 0.0,
            "steps": 6000,
            "phases": [
                {
                    "phase": "solid",
                    "run": "/solid-r4",
                    "volume_per_atom_A3": 18.25,
                }
            ],
        }
        summary = {
            "status": "all_confirmations_passed",
            "phase_results": [
                {
                    "phase": "solid",
                    "status": "confirmation_passed",
                    "checks": {"phase_verified": True},
                }
            ],
        }

        result = build_targeted_manifest(
            parent,
            extension,
            summary,
            parent_path=Path("/parent.json"),
            extension_root=Path("/extension"),
        )
        point = result["points"][0]

        self.assertEqual(point["solid_runs"][-1], "/solid-r4")
        self.assertEqual(point["liquid_runs"], ["/liquid-base", "/liquid-r3"])
        self.assertEqual(point["solid_steps"], 21000)
        self.assertEqual(point["liquid_steps"], 15000)
        self.assertEqual(point["steps"], 15000)

    def test_appends_liquid_and_keeps_both_d25_windows_beyond_recovered_parent(self):
        parent = {
            "schema": "wt-zero-pressure-fusion-enthalpy-manifest-v1",
            "target_kedf": "wt",
            "points": [
                {
                    "temperature_k": 1100.0,
                    "target_pressure_kbar": 0.0,
                    "steps": 7700,
                    "solid_run": "/solid-r1",
                    "liquid_run": "/liquid-r1",
                    "solid_runs": ["/solid-parent", "/solid-r1"],
                    "liquid_runs": ["/liquid-parent", "/liquid-r1"],
                    "segment_steps": [1700, 6000],
                    "solid_volume_per_atom_A3": 18.37,
                    "liquid_volume_per_atom_A3": 19.11,
                }
            ],
        }
        extension = {
            "extension_schema": "wt-zero-pressure-enthalpy-phase-extension-v1",
            "target_kedf": "wt",
            "temperature_K": 1100.0,
            "target_pressure_kbar": 0.0,
            "steps": 3000,
            "phases": [
                {
                    "phase": "liquid",
                    "run": "/liquid-r2",
                    "volume_per_atom_A3": 19.11,
                }
            ],
        }
        summary = {
            "status": "all_confirmations_passed",
            "phase_results": [
                {
                    "phase": "liquid",
                    "status": "confirmation_passed",
                    "checks": {"phase_verified": True},
                }
            ],
        }

        result = build_targeted_manifest(
            parent,
            extension,
            summary,
            parent_path=Path("/parent.json"),
            extension_root=Path("/extension"),
        )
        point = result["points"][0]

        self.assertEqual(point["solid_segment_steps"], [1700, 6000])
        self.assertEqual(point["liquid_segment_steps"], [1700, 6000, 3000])
        self.assertEqual(point["solid_steps"], 7700)
        self.assertEqual(point["liquid_steps"], 10700)
        self.assertGreater(int(0.25 * point["solid_steps"]), 1700)
        self.assertGreater(int(0.25 * point["liquid_steps"]), 1700)
        self.assertEqual(point["targeted_extension"]["phase"], "liquid")


if __name__ == "__main__":
    unittest.main()
