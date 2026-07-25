import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mpn_melting.structures import AtomSet
from scripts.prepare_wt_recovered_confirmation import (
    prepare_recovered_confirmation,
)


class WtRecoveredConfirmationTests(unittest.TestCase):
    def make_source(self, root: Path) -> dict[str, str]:
        shas = {}
        for phase in ("solid", "liquid"):
            dump = root / phase / "OUT.test" / "MD_dump"
            dump.parent.mkdir(parents=True)
            dump.write_bytes(f"recovered-{phase}".encode())
            shas[phase] = hashlib.sha256(dump.read_bytes()).hexdigest()
        return shas

    @staticmethod
    def atoms(with_velocities: bool = True) -> AtomSet:
        velocities = [(0.01, 0.02, 0.03)] * 4 if with_velocities else None
        return AtomSet(
            ["Al"] * 4,
            [(0.0, 0.0, 0.0)] * 4,
            [(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
            velocities=velocities,
        )

    def test_builds_traceable_symlink_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shas = self.make_source(source)

            def fake_source(path, frame, element, include_velocities=False):
                return {
                    "source": f"{path}/OUT.test/MD_dump",
                    "step": 1695,
                    "atoms": self.atoms(),
                }

            out = root / "parent"
            with patch(
                "scripts.prepare_wt_recovered_confirmation.load_atom_source",
                side_effect=fake_source,
            ):
                manifest = prepare_recovered_confirmation(
                    source,
                    out,
                    temperature_k=1100.0,
                    steps=1700,
                    expected_last_step=1695,
                    expected_natoms=4,
                    solid_volume_per_atom_a3=18.3,
                    liquid_volume_per_atom_a3=19.1,
                    expected_sha256=shas,
                )

            preflight = json.loads((out / "recovery_preflight.json").read_text())
            stored = json.loads((out / "confirmation_manifest.json").read_text())
            self.assertEqual(manifest, stored)
            self.assertEqual(
                manifest["recovery_schema"],
                "wt-recovered-confirmation-wrapper-v1",
            )
            self.assertFalse(manifest["source_velocities_discarded"])
            self.assertEqual(manifest["phases"][0]["source_step"], 1695)
            self.assertEqual(preflight["status"], "verified")
            self.assertTrue((out / "solid").is_symlink())
            self.assertEqual(
                (out / "solid").resolve(), (source / "solid").resolve()
            )

    def test_rejects_wrong_md_dump_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shas = self.make_source(source)
            shas["solid"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare_recovered_confirmation(
                    source,
                    root / "out",
                    temperature_k=1100.0,
                    steps=1700,
                    expected_last_step=1695,
                    expected_natoms=4,
                    solid_volume_per_atom_a3=18.3,
                    liquid_volume_per_atom_a3=19.1,
                    expected_sha256=shas,
                )

    def test_rejects_missing_velocities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shas = self.make_source(source)

            def fake_source(path, frame, element, include_velocities=False):
                return {
                    "source": f"{path}/OUT.test/MD_dump",
                    "step": 1695,
                    "atoms": self.atoms(with_velocities=False),
                }

            with patch(
                "scripts.prepare_wt_recovered_confirmation.load_atom_source",
                side_effect=fake_source,
            ), self.assertRaisesRegex(ValueError, "velocities"):
                prepare_recovered_confirmation(
                    source,
                    root / "out",
                    temperature_k=1100.0,
                    steps=1700,
                    expected_last_step=1695,
                    expected_natoms=4,
                    solid_volume_per_atom_a3=18.3,
                    liquid_volume_per_atom_a3=19.1,
                    expected_sha256=shas,
                )


if __name__ == "__main__":
    unittest.main()
