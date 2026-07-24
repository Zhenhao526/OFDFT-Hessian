import tempfile
import unittest
from pathlib import Path

from scripts.prepare_wt_enthalpy_resume_from_stru import read_restart_stru


class RestartStruTests(unittest.TestCase):
    def test_reads_cartesian_restart_with_movements_and_velocities(self):
        text = """\
ATOMIC_SPECIES
Al 26.9815 al.gga.psp blps

LATTICE_CONSTANT
1.8897261254578281

LATTICE_VECTORS
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0

ATOMIC_POSITIONS
Cartesian

Al #label
0.0
2
1.0 2.0 3.0 m 1 0 1 v 0.1 0.2 0.3
3.0 2.0 1.0 m 1 1 1 v -0.1 -0.2 -0.3
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "STRU_MD_1300"
            path.write_text(text, encoding="utf-8")
            atoms = read_restart_stru(path)

        self.assertEqual(atoms.symbols, ["Al", "Al"])
        self.assertEqual(atoms.scaled_positions[0], (0.25, 0.5, 0.75))
        self.assertEqual(atoms.scaled_positions[1], (0.75, 0.5, 0.25))
        self.assertEqual(atoms.movements, [(1, 0, 1), (1, 1, 1)])
        self.assertEqual(atoms.velocities[0], (0.1, 0.2, 0.3))
        self.assertEqual(atoms.velocities[1], (-0.1, -0.2, -0.3))

    def test_rejects_incomplete_velocities(self):
        text = """\
ATOMIC_SPECIES
Al 26.9815 al.gga.psp blps
LATTICE_CONSTANT
1.8897261254578281
LATTICE_VECTORS
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
ATOMIC_POSITIONS
Direct
Al
0.0
1
0.0 0.0 0.0 1 1 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "STRU"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete velocities"):
                read_restart_stru(path)


if __name__ == "__main__":
    unittest.main()
