import tempfile
import unittest
from pathlib import Path

from mpn_melting.parser import RY_TO_EV, parse_directory, parse_text, summarize


class ParserTests(unittest.TestCase):
    def test_parse_ofdft_tn_energy(self):
        parsed = parse_text(
            "ITER       ETOT/eV           EDIFF/eV\n"
            " TN0     -4.66993349e+01   -4.66993349e+01\n"
            " TN10    -4.92869812e+01   -1.45506904e-10\n"
        )
        self.assertEqual(parsed.energies[-1], -49.2869812)

    def test_parse_abacus_e_total_ev_column(self):
        parsed = parse_text("              E_Total     -28.980200045864    -394.295849803608\n")
        self.assertEqual(parsed.energies, [-394.295849803608])

    def test_temperature_ignores_dates(self):
        parsed = parse_text(
            "Start Time is Tue Jul  7 16:31:02 2026\n"
            "md step 10 T = 1200.0 K\n"
        )
        self.assertEqual(parsed.temperatures, [1200.0])

    def test_parse_md_observable_row(self):
        parsed = parse_text(
            " Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)\n"
            " ------------------------------------------------------------------------------------------------\n"
            " -3.61112431         -3.62244013         0.01131582          1191.08262643\n"
        )
        self.assertEqual(parsed.temperatures, [1191.08262643])
        self.assertEqual(parsed.md_energies, [-3.61112431 * RY_TO_EV])
        self.assertEqual(parsed.md_potential_energies, [-3.62244013 * RY_TO_EV])
        self.assertEqual(parsed.md_kinetic_energies, [0.01131582 * RY_TO_EV])
        self.assertAlmostEqual(parsed.energies[-1], -3.61112431 * RY_TO_EV, places=9)

    def test_ignore_timing_temp_label(self):
        parsed = parse_text(
            "                 DiagSub::temp          4.822\n"
            " Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)\n"
            " -132.5469           -132.90032          0.3534166           1200\n"
        )
        self.assertEqual(parsed.temperatures, [1200.0])

    def test_summarize_md_statistics(self):
        parsed = parse_text(
            " Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)\n"
            " -3.0                -4.0                1.0                 1000.0\n"
            " Energy (Ry)         Potential (Ry)      Kinetic (Ry)        Temperature (K)\n"
            " -2.0                -3.0                1.0                 1100.0\n"
        )
        text = summarize({"running_md.log": parsed}, natoms=2)
        self.assertIn("md_steps=2", text)
        self.assertIn("md_energy_drift_per_atom=6.80285 eV/atom", text)
        self.assertIn("temperature_stats=mean 1050 K", text)

    def test_find_abacus_output_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            out_dir = run_dir / "OUT.test"
            out_dir.mkdir()
            (out_dir / "running_scf.log").write_text(
                " TN1     -2.28651006e+02   -1.45388478e-09\n",
                encoding="utf-8",
            )
            parsed = parse_directory(run_dir)
            self.assertIn("OUT.test/running_scf.log", parsed)
            self.assertEqual(parsed["OUT.test/running_scf.log"].energies[-1], -228.651006)


if __name__ == "__main__":
    unittest.main()
