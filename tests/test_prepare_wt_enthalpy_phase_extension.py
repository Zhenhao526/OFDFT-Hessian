import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mpn_melting.structures import AtomSet
from scripts.prepare_wt_enthalpy_phase_extension import prepare_phase_extension
from tests.test_prepare_wt_enthalpy_extension import WtEnthalpyExtensionTests


class WtEnthalpyPhaseExtensionTests(unittest.TestCase):
    def test_continues_only_selected_phase_with_velocities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            WtEnthalpyExtensionTests().make_source(source_root)
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
                self.assertTrue(include_velocities)
                return {
                    "source": f"{path}/MD_dump",
                    "step": 5995,
                    "atoms": atoms,
                }

            out = root / "extension"
            with patch(
                "scripts.prepare_wt_enthalpy_phase_extension.load_atom_source",
                side_effect=fake_source,
            ):
                manifest = prepare_phase_extension(
                    source_root,
                    out,
                    config,
                    "solid",
                    6000,
                    5.0,
                    100,
                )

            metadata = json.loads((out / "solid" / "metadata.json").read_text())
            stru = (out / "solid" / "STRU").read_text()

        self.assertEqual([row["phase"] for row in manifest["phases"]], ["solid"])
        self.assertFalse(manifest["source_velocities_discarded"])
        self.assertEqual(metadata["source_step"], 5995)
        self.assertEqual(metadata["volume_per_atom_A3"], 18.0)
        self.assertIn(" v ", stru)

    def test_continues_only_liquid_for_requested_short_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            WtEnthalpyExtensionTests().make_source(source_root)
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
                self.assertTrue(include_velocities)
                self.assertTrue(str(path).endswith("liquid"))
                return {
                    "source": f"{path}/MD_dump",
                    "step": 5995,
                    "atoms": atoms,
                }

            out = root / "extension"
            with patch(
                "scripts.prepare_wt_enthalpy_phase_extension.load_atom_source",
                side_effect=fake_source,
            ):
                manifest = prepare_phase_extension(
                    source_root,
                    out,
                    config,
                    "liquid",
                    3000,
                    5.0,
                    101,
                )

            metadata = json.loads((out / "liquid" / "metadata.json").read_text())

        self.assertEqual([row["phase"] for row in manifest["phases"]], ["liquid"])
        self.assertEqual(manifest["steps"], 3000)
        self.assertFalse(manifest["source_velocities_discarded"])
        self.assertEqual(metadata["source_step"], 5995)
        self.assertEqual(metadata["volume_per_atom_A3"], 18.8)


if __name__ == "__main__":
    unittest.main()
