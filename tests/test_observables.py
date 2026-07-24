import unittest

from mpn_melting.observables import centrosymmetry_parameters
from mpn_melting.structures import build_fcc
from mpn_melting.trajectory import matmul_row


class ObservableTests(unittest.TestCase):
    def test_fcc_centrosymmetry_is_zero(self):
        atoms = build_fcc("Al", 4.05, (2, 2, 2))
        positions = [matmul_row(position, atoms.lattice_vectors) for position in atoms.scaled_positions]
        values = centrosymmetry_parameters(positions, atoms.lattice_vectors)
        self.assertEqual(len(values), 32)
        self.assertLess(max(values), 1e-20)


if __name__ == "__main__":
    unittest.main()
