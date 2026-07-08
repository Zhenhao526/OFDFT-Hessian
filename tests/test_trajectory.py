import tempfile
import unittest
from pathlib import Path

from mpn_melting.trajectory import summarize_md_dump


class TrajectoryTests(unittest.TestCase):
    def test_md_dump_minimum_image_displacement(self):
        text = """MDSTEP:  0
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  9.8 0.0 0.0  0 0 0  0 0 0

MDSTEP:  1
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  0.2 0.0 0.0  0 0 0  0 0 0

"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MD_dump"
            path.write_text(text, encoding="utf-8")
            summary = summarize_md_dump(path)
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.rms_displacement, 0.4, places=12)
        self.assertAlmostEqual(summary.nearest_neighbor, 0.0, places=12)
        self.assertEqual(summary.frames, 2)

    def test_parse_md_dump_velocities(self):
        from mpn_melting.trajectory import parse_md_dump

        text = """MDSTEP:  0
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  1.0 0.0 0.0  0.1 0.2 0.3  0.4 0.5 0.6

"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MD_dump"
            path.write_text(text, encoding="utf-8")
            frames = parse_md_dump(path)
        self.assertEqual(frames[0].velocities, [(0.4, 0.5, 0.6)])

    def test_lindemann_ratio_for_two_atoms(self):
        text = """MDSTEP:  0
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  0.0 0.0 0.0  0 0 0  0 0 0
  1  Al  2.0 0.0 0.0  0 0 0  0 0 0

MDSTEP:  1
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  0.2 0.0 0.0  0 0 0  0 0 0
  1  Al  2.2 0.0 0.0  0 0 0  0 0 0

"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MD_dump"
            path.write_text(text, encoding="utf-8")
            summary = summarize_md_dump(path)
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.nearest_neighbor, 2.0, places=12)
        self.assertAlmostEqual(summary.final_nearest_neighbor, 2.0, places=12)
        self.assertAlmostEqual(summary.mean_nearest_neighbor, 2.0, places=12)
        self.assertAlmostEqual(summary.final_mean_nearest_neighbor, 2.0, places=12)
        self.assertAlmostEqual(summary.lindemann_ratio, 0.1, places=12)


if __name__ == "__main__":
    unittest.main()
