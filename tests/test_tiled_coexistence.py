import tempfile
import unittest
from pathlib import Path

from mpn_melting.abacus_input import stru_text
from mpn_melting.stru import read_stru
from mpn_melting.structures import build_coexistence_seed, tile_coexistence_template


class TiledCoexistenceTests(unittest.TestCase):
    def setUp(self):
        self.element = {
            "element": "Al",
            "structure": "fcc",
            "lattice_a_angstrom": 4.05,
            "mass": 26.9815385,
            "pseudopotential": "al.gga.psp",
            "pseudo_type": "blps",
        }

    def test_read_stru_round_trip(self):
        source, _ = build_coexistence_seed(self.element, (2, 2, 4), seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "STRU"
            path.write_text(stru_text(source, [self.element]), encoding="utf-8")
            loaded = read_stru(path)
        self.assertEqual(loaded.natoms, 64)
        self.assertEqual(loaded.lattice_vectors, source.lattice_vectors)
        self.assertAlmostEqual(loaded.scaled_positions[-1][2], source.scaled_positions[-1][2])

    def test_tiles_each_phase_without_density_scaling(self):
        source, labels = build_coexistence_seed(self.element, (2, 2, 4), seed=7)
        tiled, tiled_labels = tile_coexistence_template(source, labels, (3, 3, 3))
        self.assertEqual(tiled.natoms, 1728)
        self.assertEqual(tiled_labels.count("solid_seed"), 864)
        self.assertEqual(tiled_labels.count("liquid_seed"), 864)
        self.assertEqual(
            tiled.lattice_vectors,
            [(24.299999999999997, 0.0, 0.0), (0.0, 24.299999999999997, 0.0), (0.0, 0.0, 48.599999999999994)],
        )
        solid_z = [pos[2] for pos, label in zip(tiled.scaled_positions, tiled_labels) if label == "solid_seed"]
        liquid_z = [pos[2] for pos, label in zip(tiled.scaled_positions, tiled_labels) if label == "liquid_seed"]
        self.assertLess(max(solid_z), 0.5)
        self.assertGreaterEqual(min(liquid_z), 0.5)

    def test_reflects_liquid_atom_that_crossed_internal_interface(self):
        source, labels = build_coexistence_seed(self.element, (1, 1, 2), liquid_displacement_angstrom=0.0)
        positions = list(source.scaled_positions)
        liquid_index = labels.index("liquid_seed")
        positions[liquid_index] = (positions[liquid_index][0], positions[liquid_index][1], 0.49)
        crossed = type(source)(source.symbols, positions, source.lattice_vectors)
        tiled, tiled_labels = tile_coexistence_template(crossed, labels, (1, 1, 2))
        liquid_z = [
            position[2]
            for position, label in zip(tiled.scaled_positions, tiled_labels)
            if label == "liquid_seed"
        ]
        self.assertGreaterEqual(min(liquid_z), 0.5)
        self.assertLess(min(liquid_z), 0.52)


if __name__ == "__main__":
    unittest.main()
