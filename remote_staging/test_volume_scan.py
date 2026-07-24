import unittest

from mpn_melting.structures import AtomSet
from scripts.prepare_al108_volume_scan import scaled_to_volume


class VolumeScalingTests(unittest.TestCase):
    def test_scaled_to_volume_preserves_dynamical_state(self):
        atoms = AtomSet(
            symbols=["Al", "Al"],
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            lattice_vectors=[(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)],
            velocities=[(0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)],
            movements=[(1, 1, 1), (0, 0, 0)],
        )

        scaled = scaled_to_volume(atoms, 64.0)

        self.assertEqual(scaled.velocities, atoms.velocities)
        self.assertEqual(scaled.movements, atoms.movements)
        self.assertEqual(scaled.scaled_positions, atoms.scaled_positions)
        self.assertAlmostEqual(scaled.lattice_vectors[0][0], 4.0)
        self.assertAlmostEqual(scaled.lattice_vectors[1][1], 4.0)
        self.assertAlmostEqual(scaled.lattice_vectors[2][2], 4.0)


if __name__ == "__main__":
    unittest.main()
