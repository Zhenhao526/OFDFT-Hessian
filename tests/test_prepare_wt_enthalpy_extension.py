import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mpn_melting.structures import AtomSet
from scripts.prepare_wt_enthalpy_extension import (
    load_verified_source,
    prepare_extension,
)


class WtEnthalpyExtensionTests(unittest.TestCase):
    def make_source(self, root: Path) -> None:
        phases = []
        phase_results = []
        for phase, volume in (("solid", 18.0), ("liquid", 18.8)):
            run = root / phase
            run.mkdir(parents=True)
            phases.append(
                {
                    "phase": phase,
                    "run": str(run),
                    "volume_per_atom_A3": volume,
                }
            )
            phase_results.append(
                {"phase": phase, "status": "confirmation_passed"}
            )
        (root / "confirmation_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "wt-zero-pressure-confirmation-v1",
                    "target_kedf": "wt",
                    "temperature_K": 975.0,
                    "target_pressure_kbar": 0.0,
                    "steps": 3000,
                    "phases": phases,
                }
            )
        )
        (root / "confirmation_summary.json").write_text(
            json.dumps(
                {
                    "status": "all_confirmations_passed",
                    "phase_results": phase_results,
                }
            )
        )

    def test_rejects_unverified_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_source(root)
            summary = root / "confirmation_summary.json"
            data = json.loads(summary.read_text())
            data["status"] = "confirmation_failed"
            summary.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "not verified"):
                load_verified_source(root)

    def test_continues_both_phases_with_velocities_and_same_volumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            self.make_source(source_root)
            config = root / "wt.json"
            config.write_text(
                json.dumps(
                    {
                        "of_kinetic": "wt",
                        "esolver_type": "ofdft",
                        "of_method": "tn",
                        "pseudo_dir": "/tmp",
                        "kmesh": [1, 1, 1],
                    }
                )
            )
            atoms = AtomSet(
                ["Al"] * 4,
                [(0.0, 0.0, 0.0)] * 4,
                [(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
                velocities=[(0.01, 0.02, 0.03)] * 4,
            )

            def fake_source(path, frame, element, include_velocities=False):
                return {"source": f"{path}/MD_dump", "step": 2995, "atoms": atoms}

            out = root / "extension"
            with patch(
                "scripts.prepare_wt_enthalpy_extension.load_atom_source",
                side_effect=fake_source,
            ):
                manifest = prepare_extension(
                    source_root, out, config, 3000, 5.0, 100
                )

            solid_metadata = json.loads((out / "solid" / "metadata.json").read_text())
            liquid_metadata = json.loads((out / "liquid" / "metadata.json").read_text())
            solid_stru = (out / "solid" / "STRU").read_text()

        self.assertEqual(manifest["temperature_K"], 975.0)
        self.assertFalse(manifest["source_velocities_discarded"])
        self.assertEqual(solid_metadata["volume_per_atom_A3"], 18.0)
        self.assertEqual(liquid_metadata["volume_per_atom_A3"], 18.8)
        self.assertEqual(solid_metadata["source_step"], 2995)
        self.assertIn(" v ", solid_stru)


if __name__ == "__main__":
    unittest.main()
