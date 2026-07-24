from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_wt_fusion_enthalpy_manifest import build_manifest, merge_manifests


def inputs(status: str = "all_confirmations_passed"):
    manifest = {
        "target_kedf": "wt",
        "temperature_K": 900.0,
        "target_pressure_kbar": 0.0,
        "steps": 3000,
        "phases": [
            {"phase": "solid", "run": "/solid", "volume_per_atom_A3": 17.94},
            {"phase": "liquid", "run": "/liquid", "volume_per_atom_A3": 18.73},
        ],
    }
    summary = {
        "status": status,
        "phase_results": [
            {"phase": "solid", "status": "confirmation_passed"},
            {"phase": "liquid", "status": "confirmation_passed"},
        ],
    }
    return manifest, summary


class BuildEnthalpyManifestTests(unittest.TestCase):
    def test_builds_verified_pair(self):
        result = build_manifest(*inputs())

        self.assertEqual(result["points"][0]["steps"], 3000)
        self.assertEqual(result["points"][0]["solid_run"], "/solid")
        self.assertEqual(result["points"][0]["liquid_run"], "/liquid")
        self.assertEqual(result["points"][0]["solid_runs"], ["/solid"])
        self.assertEqual(result["points"][0]["segment_steps"], [3000])

    def test_extension_accumulates_parent_and_continuation_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            parent_manifest, parent_summary = inputs()
            (parent / "confirmation_manifest.json").write_text(
                json.dumps(parent_manifest)
            )
            (parent / "confirmation_summary.json").write_text(
                json.dumps(parent_summary)
            )
            extension_manifest, extension_summary = inputs()
            extension_manifest.update(
                {
                    "extension_schema": "wt-zero-pressure-enthalpy-extension-v1",
                    "parent_confirmation": str(parent),
                }
            )
            extension_manifest["phases"][0]["run"] = "/solid-extension"
            extension_manifest["phases"][1]["run"] = "/liquid-extension"

            point = build_manifest(extension_manifest, extension_summary)["points"][0]

        self.assertEqual(point["steps"], 6000)
        self.assertEqual(point["segment_steps"], [3000, 3000])
        self.assertEqual(point["solid_run"], "/solid-extension")
        self.assertEqual(point["solid_runs"], ["/solid", "/solid-extension"])
        self.assertEqual(point["liquid_runs"], ["/liquid", "/liquid-extension"])

    def test_rejects_failed_confirmation(self):
        with self.assertRaisesRegex(ValueError, "not verified"):
            build_manifest(*inputs("confirmation_gate_failed"))

    def test_merges_temperature_points_in_order(self):
        first = build_manifest(*inputs())
        manifest, summary = inputs()
        manifest["temperature_K"] = 1000.0
        manifest["phases"][0]["run"] = "/solid-1000"
        manifest["phases"][1]["run"] = "/liquid-1000"
        second = build_manifest(manifest, summary)

        result = merge_manifests([second, first])

        self.assertEqual(
            [point["temperature_k"] for point in result["points"]],
            [900.0, 1000.0],
        )
        self.assertEqual(result["points"][1]["solid_run"], "/solid-1000")

    def test_rejects_duplicate_temperature(self):
        first = build_manifest(*inputs())
        with self.assertRaisesRegex(ValueError, "duplicate temperatures"):
            merge_manifests([first, first])

    def test_rejects_mixed_kedf(self):
        first = build_manifest(*inputs())
        second = {**first, "target_kedf": "mpn"}
        with self.assertRaisesRegex(ValueError, "different target KEDFs"):
            merge_manifests([first, second])


if __name__ == "__main__":
    unittest.main()
