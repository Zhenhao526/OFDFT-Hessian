import tempfile
import unittest
from pathlib import Path

from mpn_melting.cli import (
    ANGSTROM_PER_FS_PER_ATOMIC_UNIT_VELOCITY,
    load_atom_source,
    load_region_labels,
    resolve_region_source,
)


class CliSourceTests(unittest.TestCase):
    def test_load_atom_source_uses_selected_frame(self):
        text = """MDSTEP:  0
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  1.0 0.0 0.0  0 0 0  0 0 0

MDSTEP:  7
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  8.0 0.0 0.0
  0.0 8.0 0.0
  0.0 0.0 8.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  9.0 0.0 0.0  0 0 0  21.876932361 0.0 -10.9384661805

"""
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "MD_dump"
            dump.write_text(text, encoding="utf-8")
            source = load_atom_source(str(dump), "last", "Al", include_velocities=True)
        atoms = source["atoms"]
        self.assertEqual(source["step"], 7)
        self.assertEqual(atoms.natoms, 1)
        self.assertEqual(atoms.lattice_vectors[0], (8.0, 0.0, 0.0))
        self.assertAlmostEqual(atoms.scaled_positions[0][0], 0.125)
        self.assertAlmostEqual(atoms.velocities[0][0], 1.0)
        self.assertAlmostEqual(atoms.velocities[0][1], 0.0)
        self.assertAlmostEqual(atoms.velocities[0][2], -0.5)

    def test_velocity_conversion_constant_matches_abacus_units(self):
        self.assertAlmostEqual(ANGSTROM_PER_FS_PER_ATOMIC_UNIT_VELOCITY, 21.876932361)

    def test_resolve_region_source_from_run_dir(self):
        regions = "index,symbol,fx,fy,fz,region\n1,Al,0,0,0,solid_seed\n"
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run / "regions.csv"
            path.write_text(regions, encoding="utf-8")
            resolved = resolve_region_source(str(run))
            labels = load_region_labels(resolved)
        self.assertEqual(resolved, path)
        self.assertEqual(labels, ["solid_seed"])


if __name__ == "__main__":
    unittest.main()
