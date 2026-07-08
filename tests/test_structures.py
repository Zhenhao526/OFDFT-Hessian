import unittest

from mpn_melting.structures import build_coexistence_seed, build_fcc, build_hcp


class StructureTests(unittest.TestCase):
    def test_fcc_atom_count(self):
        atoms = build_fcc("Al", 4.05, (2, 3, 4))
        self.assertEqual(atoms.natoms, 4 * 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Al"])

    def test_hcp_atom_count(self):
        atoms = build_hcp("Mg", 3.2, 5.2, (2, 3, 4))
        self.assertEqual(atoms.natoms, 2 * 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Mg"])

    def test_coexistence_seed_uses_liquid_source_half(self):
        element = {
            "element": "Al",
            "structure": "fcc",
            "lattice_a_angstrom": 4.05,
        }
        source = build_fcc("Al", 4.05, (2, 2, 2)).scaled_positions
        atoms, labels = build_coexistence_seed(element, (2, 2, 4), liquid_scaled_positions=source)
        self.assertEqual(atoms.natoms, 64)
        self.assertEqual(labels.count("solid_seed"), 32)
        self.assertEqual(labels.count("liquid_seed"), 32)
        liquid_z = [pos[2] for pos, label in zip(atoms.scaled_positions, labels) if label == "liquid_seed"]
        self.assertGreaterEqual(min(liquid_z), 0.5)
        self.assertLess(max(liquid_z), 1.0)

    def test_coexistence_seed_wraps_unwrapped_liquid_source_axis(self):
        element = {
            "element": "Al",
            "structure": "fcc",
            "lattice_a_angstrom": 4.05,
        }
        source = [(0.0, 0.0, 1.2)] * 32
        atoms, labels = build_coexistence_seed(element, (2, 2, 4), liquid_scaled_positions=source)
        liquid_z = [pos[2] for pos, label in zip(atoms.scaled_positions, labels) if label == "liquid_seed"]
        self.assertTrue(all(abs(z - 0.6) < 1e-12 for z in liquid_z))

    def test_coexistence_seed_applies_liquid_shift(self):
        element = {
            "element": "Al",
            "structure": "fcc",
            "lattice_a_angstrom": 4.05,
        }
        source = [(0.9, 0.8, 0.7)] * 32
        atoms, labels = build_coexistence_seed(
            element,
            (2, 2, 4),
            liquid_scaled_positions=source,
            liquid_shift=(0.25, 0.5, 0.1),
        )
        liquid = [pos for pos, label in zip(atoms.scaled_positions, labels) if label == "liquid_seed"]
        self.assertTrue(all(abs(pos[0] - 0.15) < 1e-12 for pos in liquid))
        self.assertTrue(all(abs(pos[1] - 0.3) < 1e-12 for pos in liquid))
        self.assertTrue(all(abs(pos[2] - 0.9) < 1e-12 for pos in liquid))


if __name__ == "__main__":
    unittest.main()
