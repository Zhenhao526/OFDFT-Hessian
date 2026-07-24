import json
import math
import tempfile
import unittest
from pathlib import Path

from mpn_melting.stru import read_stru
from scripts.prepare_wt_coexistence_validation import (
    coexistence_geometry,
    prepare_hot_template,
    verified_melting_report,
)


def melting_report(temperature=1000.0):
    return {
        "schema": "wt-gibbs-helmholtz-melting-v2",
        "status": "verified",
        "checks": {"all": True},
        "melting_temperature_k": temperature,
        "enthalpy_points": [
            {
                "temperature_k": 900.0,
                "solid_volume_per_atom_A3": 17.8,
                "liquid_volume_per_atom_A3": 18.6,
                "status": "verified",
            },
            {
                "temperature_k": 1100.0,
                "solid_volume_per_atom_A3": 18.2,
                "liquid_volume_per_atom_A3": 19.0,
                "status": "verified",
            },
        ],
    }


class WtCoexistenceValidationTests(unittest.TestCase):
    def test_interpolates_phase_volumes_and_preserves_them(self):
        geometry = coexistence_geometry(melting_report())
        self.assertAlmostEqual(geometry["solid_volume_per_atom_A3"], 18.0)
        self.assertAlmostEqual(geometry["liquid_volume_per_atom_A3"], 18.8)
        self.assertEqual(geometry["tiled_atoms"], 1728)
        self.assertEqual(geometry["tiled_solid_atoms"], 864)
        self.assertEqual(geometry["tiled_liquid_atoms"], 864)
        area = geometry["template_cell_lengths_A"][0] ** 2
        self.assertAlmostEqual(
            area * geometry["solid_slab_length_A"] / 32.0, 18.0
        )
        self.assertAlmostEqual(
            area * geometry["liquid_slab_length_A"] / 32.0, 18.8
        )
        self.assertAlmostEqual(
            geometry["solid_slab_length_A"],
            2.0 * geometry["fcc_lattice_a_A"],
        )

    def test_rejects_temperature_outside_volume_sampling(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            coexistence_geometry(melting_report(1150.0))

    def test_rejects_unverified_melting_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "melting.json"
            report = melting_report()
            report["status"] = "temperature_bracket_incomplete"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not fully verified"):
                verified_melting_report(path)

    def test_prepares_fixed_solid_wt_hot_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            melting = root / "melting.json"
            report = melting_report()
            melting.write_text(json.dumps(report), encoding="utf-8")
            config = root / "wt.json"
            config.write_text(
                json.dumps(
                    {
                        "of_kinetic": "wt",
                        "esolver_type": "ofdft",
                        "of_method": "tn",
                        "pseudo_dir": "/tmp",
                        "kmesh": [1, 1, 1],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "hot"
            geometry = coexistence_geometry(report)
            plan = prepare_hot_template(
                out,
                melting,
                report,
                geometry,
                config,
                hot_temperature=1600.0,
                steps=500,
                csvr_tau=10.0,
                seed=7,
                mpi_ranks=16,
            )
            atoms = read_stru(out / "STRU")
            metadata = json.loads((out / "metadata.json").read_text())
            input_text = (out / "INPUT").read_text()

        self.assertEqual(atoms.natoms, 64)
        self.assertEqual(atoms.movements.count((0, 0, 0)), 32)
        self.assertEqual(atoms.movements.count((1, 1, 1)), 32)
        self.assertEqual(metadata["abacus"]["of_kinetic"], "wt")
        self.assertEqual(metadata["geometry"]["tiled_atoms"], 1728)
        self.assertEqual(plan["status"], "hot_template_prepared")
        self.assertIn("of_kinetic wt", input_text)
        self.assertIn("md_nstep 500", input_text)


if __name__ == "__main__":
    unittest.main()
