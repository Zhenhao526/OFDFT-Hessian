import json
import tempfile
import unittest
from pathlib import Path

from mpn_melting.pair_reference import (
    RadialBasis,
    SmoothRepulsiveCore,
    evaluate_pair_virial,
    evenly_spaced_centers,
)
from scripts.export_pair_reference_for_abacus import export_model
from scripts.analyze_pair_reference_pressure import parse_logged_virials


class PairReferenceTests(unittest.TestCase):
    def test_logged_pair_virial_parser(self):
        text = (
            " PAIR_REFERENCE_COMPONENTS step=12 U_REF_eV=-10.25 "
            "W_REF_eV=-3.5 RMIN_A=2.31\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_md.log"
            path.write_text(text)
            parsed = parse_logged_virials(path)
        self.assertEqual(list(parsed), [12])
        self.assertEqual(parsed[12].potential_energy_ev, -10.25)
        self.assertEqual(parsed[12].configurational_virial_ev, -3.5)
        self.assertEqual(parsed[12].nearest_neighbor_angstrom, 2.31)

    def test_radial_derivative_matches_finite_difference(self):
        basis = RadialBasis([2.0, 3.0], sigma_angstrom=0.4, cutoff_angstrom=5.0)
        distance = 2.4
        step = 1.0e-6
        values, derivatives = basis.values_and_derivatives(distance)
        plus, _ = basis.values_and_derivatives(distance + step)
        minus, _ = basis.values_and_derivatives(distance - step)
        for index in range(len(values)):
            numerical = (plus[index] - minus[index]) / (2 * step)
            self.assertAlmostEqual(derivatives[index], numerical, places=8)

    def test_basis_is_zero_at_and_beyond_cutoff(self):
        basis = RadialBasis([2.0], sigma_angstrom=0.4, cutoff_angstrom=5.0)
        self.assertEqual(basis.values_and_derivatives(5.0), ([0.0], [0.0]))
        self.assertEqual(basis.values_and_derivatives(6.0), ([0.0], [0.0]))

    def test_evenly_spaced_centers(self):
        self.assertEqual(evenly_spaced_centers(2.0, 4.0, 3), [2.0, 3.0, 4.0])

    def test_repulsive_core_derivative_and_smooth_cutoff(self):
        core = SmoothRepulsiveCore(amplitude_ev=500.0, cutoff_angstrom=2.2, power=4)
        distance = 1.5
        step = 1.0e-6
        _, derivative = core.value_and_derivative(distance)
        plus, _ = core.value_and_derivative(distance + step)
        minus, _ = core.value_and_derivative(distance - step)
        self.assertAlmostEqual(derivative, (plus - minus) / (2 * step), places=7)
        self.assertEqual(core.value_and_derivative(2.2), (0.0, 0.0))

    def test_guarded_model_export_has_one_coefficient_per_center(self):
        document = {
            "reference_gate_passed": True,
            "short_range_guard_passed": True,
            "model": {
                "centers_angstrom": [2.0, 3.0],
                "coefficients_ev": [-57.0, 1.0, -0.5],
                "sigma_angstrom": 0.3,
                "cutoff_angstrom": 6.5,
                "repulsive_core": {"amplitude_ev": 500.0, "cutoff_angstrom": 2.2, "power": 4},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.json"
            output = Path(directory) / "model.dat"
            source.write_text(json.dumps(document))
            export_model(source, output)
            text = output.read_text()
        self.assertIn("n_centers 2", text)
        self.assertIn("constant_ev_per_atom -57", text)
        self.assertIn("coefficients_ev 1 -0.5", text)

    def test_configurational_virial_matches_isotropic_energy_derivative(self):
        model = {
            "centers_angstrom": [2.5],
            "coefficients_ev": [0.0, 1.25],
            "sigma_angstrom": 0.4,
            "cutoff_angstrom": 6.5,
            "repulsive_core": {
                "amplitude_ev": 0.0,
                "cutoff_angstrom": 2.2,
                "power": 4,
            },
        }
        positions = [(0.0, 0.0, 0.0), (2.4, 0.0, 0.0)]
        lattice = ((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0))
        result = evaluate_pair_virial(positions, lattice, model)
        volume = 20.0**3
        epsilon = 1.0e-6

        def scaled_energy(volume_factor):
            scale = volume_factor ** (1.0 / 3.0)
            scaled_positions = [tuple(scale * value for value in row) for row in positions]
            scaled_lattice = tuple(
                tuple(scale * value for value in row) for row in lattice
            )
            return evaluate_pair_virial(scaled_positions, scaled_lattice, model).potential_energy_ev

        numerical_pressure = -(
            scaled_energy(1.0 + epsilon) - scaled_energy(1.0 - epsilon)
        ) / (2.0 * epsilon * volume)
        analytical_pressure = result.configurational_virial_ev / (3.0 * volume)
        self.assertAlmostEqual(analytical_pressure, numerical_pressure, places=10)


if __name__ == "__main__":
    unittest.main()
