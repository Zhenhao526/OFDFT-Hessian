import tempfile
import unittest
from pathlib import Path

from mpn_melting.abacus_input import write_job
from mpn_melting.structures import AtomSet, build_fcc


class AbacusInputTests(unittest.TestCase):
    def test_write_job_files(self):
        atoms = build_fcc("Al", 4.05, (1, 1, 1))
        element_config = {
            "element": "Al",
            "structure": "fcc",
            "mass": 26.9815385,
            "pseudopotential": "Al.fake.upf",
            "pseudo_type": "blps",
        }
        abacus_config = {
            "abacus_executable": "/tmp/fake-abacus",
            "calculation": "scf",
            "esolver_type": "ofdft",
            "basis_type": "pw",
            "of_kinetic": "mpn",
            "kmesh": [1, 1, 1],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model = out / "source_net.pt"
            model.write_text("fake model", encoding="utf-8")
            abacus_config["mpn_model"] = str(model)
            write_job(out, atoms, element_config, abacus_config, "static", "test")
            self.assertTrue((out / "INPUT").exists())
            self.assertTrue((out / "STRU").exists())
            self.assertTrue((out / "KPT").exists())
            self.assertTrue((out / "net.pt").exists())
            self.assertTrue((out / "metadata.json").exists())
            self.assertIn("of_kinetic mpn", (out / "INPUT").read_text())
            self.assertIn("Al 26.9815385 Al.fake.upf blps", (out / "STRU").read_text())
            self.assertIn('"/tmp/fake-abacus"', (out / "run_local.sh").read_text())

    def test_write_job_velocities(self):
        atoms = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.0, 0.0, 0.0)],
            lattice_vectors=[(4.05, 0.0, 0.0), (0.0, 4.05, 0.0), (0.0, 0.0, 4.05)],
            velocities=[(0.1, 0.2, 0.3)],
            movements=[(0, 0, 0)],
        )
        element_config = {
            "element": "Al",
            "structure": "fcc",
            "mass": 26.9815385,
            "pseudopotential": "Al.fake.upf",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_job(out, atoms, element_config, {"kmesh": [1, 1, 1]}, "md_restart", "test")
            text = (out / "STRU").read_text()
            self.assertIn("0 0 0 v 0.100000000000 0.200000000000 0.300000000000", text)


if __name__ == "__main__":
    unittest.main()
