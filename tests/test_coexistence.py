import tempfile
import unittest
from pathlib import Path

from mpn_melting.coexistence import summarize_coexistence


class CoexistenceTests(unittest.TestCase):
    def test_summarize_coexistence_regions(self):
        text = """MDSTEP:  0
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  0.0 0.0 0.0  0 0 0  0 0 0
  1  Al  2.0 0.0 0.0  0 0 0  0 0 0
  2  Al  0.0 0.0 5.0  0 0 0  0 0 0
  3  Al  2.0 0.0 5.0  0 0 0  0 0 0

MDSTEP:  1
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  0.05 0.0 0.0  0 0 0  0 0 0
  1  Al  2.05 0.0 0.0  0 0 0  0 0 0
  2  Al  0.5 0.0 5.0  0 0 0  0 0 0
  3  Al  2.5 0.0 5.0  0 0 0  0 0 0

"""
        regions = (
            "index,symbol,fx,fy,fz,region\n"
            "1,Al,0,0,0,solid_seed\n"
            "2,Al,0.2,0,0,solid_seed\n"
            "3,Al,0,0,0.5,liquid_seed\n"
            "4,Al,0.2,0,0.5,liquid_seed\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            out = run / "OUT.test"
            out.mkdir()
            (out / "MD_dump").write_text(text, encoding="utf-8")
            (run / "regions.csv").write_text(regions, encoding="utf-8")
            summary = summarize_coexistence(run)
        self.assertIn("solid_seed", summary)
        self.assertIn("liquid_seed", summary)
        self.assertIn("two_phase_persisting", summary)


if __name__ == "__main__":
    unittest.main()
