import unittest

from mpn_melting.structures import (
    AtomSet,
    constrain_regions,
    join_phase_sources,
    lattice_volume,
    reshape_coexistence_z_lengths,
)


class PaperCoexistenceTests(unittest.TestCase):
    def test_join_preserves_each_phase_volume(self):
        solid = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.25, 0.25, 0.5)],
            lattice_vectors=[(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
        )
        liquid = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.75, 0.75, 0.5)],
            lattice_vectors=[(5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)],
        )

        joined, labels = join_phase_sources(solid, liquid)

        self.assertEqual(labels, ["solid_seed", "liquid_seed"])
        self.assertAlmostEqual(lattice_volume(joined.lattice_vectors), 64.0 + 125.0)
        self.assertAlmostEqual(joined.scaled_positions[0][2], 0.5 * 64.0 / 189.0)
        self.assertAlmostEqual(joined.scaled_positions[1][2], 64.0 / 189.0 + 0.5 * 125.0 / 189.0)

    def test_constraint_flags_only_selected_region(self):
        atoms = AtomSet(
            symbols=["Al", "Al"],
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            lattice_vectors=[(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
        )
        constrained = constrain_regions(atoms, ["solid_seed", "liquid_seed"], ["solid_seed"])
        self.assertEqual(constrained.movements, [(0, 0, 0), (1, 1, 1)])

    def test_join_applies_periodic_liquid_shift(self):
        solid = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.1, 0.2, 0.3)],
            lattice_vectors=[(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
        )
        liquid = AtomSet(
            symbols=["Al"],
            scaled_positions=[(0.9, 0.8, 0.7)],
            lattice_vectors=[(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
        )
        joined, _ = join_phase_sources(solid, liquid, liquid_shift=(0.25, 0.5, 0.4))
        self.assertAlmostEqual(joined.scaled_positions[1][0], 0.15)
        self.assertAlmostEqual(joined.scaled_positions[1][1], 0.3)
        self.assertAlmostEqual(joined.scaled_positions[1][2], 0.55)

    def test_reshape_coexistence_sets_independent_slab_lengths(self):
        atoms = AtomSet(
            symbols=["Al", "Al"],
            scaled_positions=[(0.0, 0.0, 0.25), (0.0, 0.0, 0.75)],
            lattice_vectors=[(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 8.0)],
        )
        reshaped = reshape_coexistence_z_lengths(
            atoms,
            ["solid_seed", "liquid_seed"],
            solid_length=4.0,
            liquid_length=6.0,
        )
        self.assertEqual(reshaped.lattice_vectors[2], (0.0, 0.0, 10.0))
        self.assertAlmostEqual(reshaped.scaled_positions[0][2], 0.2)
        self.assertAlmostEqual(reshaped.scaled_positions[1][2], 0.7)


if __name__ == "__main__":
    unittest.main()
