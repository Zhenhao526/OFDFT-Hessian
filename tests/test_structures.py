import unittest

from mpn_melting.structures import (
    AtomSet,
    build_bcc,
    build_coexistence_seed,
    build_fcc,
    build_hcp,
    build_sc,
    reshape_orthorhombic_cell,
)


class StructureTests(unittest.TestCase):
    def test_fcc_atom_count(self):
        atoms = build_fcc("Al", 4.05, (2, 3, 4))
        self.assertEqual(atoms.natoms, 4 * 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Al"])

    def test_hcp_atom_count(self):
        atoms = build_hcp("Mg", 3.2, 5.2, (2, 3, 4))
        self.assertEqual(atoms.natoms, 2 * 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Mg"])

    def test_bcc_atom_count(self):
        atoms = build_bcc("Al", 3.2, (2, 3, 4))
        self.assertEqual(atoms.natoms, 2 * 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Al"])

    def test_sc_atom_count(self):
        atoms = build_sc("Al", 2.55, (2, 3, 4))
        self.assertEqual(atoms.natoms, 2 * 3 * 4)
        self.assertEqual(atoms.species, ["Al"])

    def test_reshape_cell_preserves_fractional_state(self):
        atoms = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.25, 0.5, 0.75)],
            lattice_vectors=[(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)],
            velocities=[(0.1, 0.2, 0.3)],
            movements=[(1, 0, 1)],
        )
        reshaped = reshape_orthorhombic_cell(atoms, (8.0, 9.0, 12.0))
        self.assertEqual(reshaped.lattice_vectors, [(8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (0.0, 0.0, 12.0)])
        self.assertEqual(reshaped.scaled_positions, atoms.scaled_positions)
        self.assertEqual(reshaped.velocities, atoms.velocities)
        self.assertEqual(reshaped.movements, atoms.movements)

    def test_reshape_cell_rejects_nonpositive_lengths(self):
        atoms = build_fcc("Al", 4.05, (1, 1, 1))
        with self.assertRaises(ValueError):
            reshape_orthorhombic_cell(atoms, (4.0, 0.0, 4.0))

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
