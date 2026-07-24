import tempfile
import unittest
from pathlib import Path

from mpn_melting.cli import (
    ANGSTROM_PER_FS_PER_ATOMIC_UNIT_VELOCITY,
    apply_md_options,
    build_parser,
    configure_restart_velocities,
    load_atom_source,
    load_region_labels,
    resolve_region_source,
)


class CliSourceTests(unittest.TestCase):
    def test_npt_options_enable_stress_and_zero_pressure(self):
        args = build_parser().parse_args(
            [
                "make-liquid",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "940",
                "--ensemble",
                "npt",
                "--pmode",
                "iso",
                "--pcouple",
                "xyz",
            ]
        )
        config = {"cal_stress": 0}
        apply_md_options(config, args)
        self.assertEqual(config["md_type"], "npt")
        self.assertEqual(config["cal_stress"], 1)
        self.assertEqual(config["md_pfirst"], 0.0)
        self.assertEqual(config["md_plast"], 0.0)
        self.assertEqual(config["md_pmode"], "iso")
        self.assertEqual(config["md_pcouple"], "xyz")

    def test_nve_restart_preserves_snapshot_temperature(self):
        config = {"md_tfirst": 940.0, "md_tlast": 940.0}
        configure_restart_velocities(config, ensemble="nve", has_velocities=True)
        self.assertEqual(config["init_vel"], 1)
        self.assertEqual(config["md_tfirst"], -1)
        self.assertEqual(config["md_tlast"], -1)

    def test_restart_accepts_cell_lengths(self):
        args = build_parser().parse_args(
            [
                "make-restart",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "980",
                "--source",
                "source",
                "--cell-lengths",
                "24.3",
                "24.3",
                "27.6",
            ]
        )
        self.assertEqual(args.cell_lengths, [24.3, 24.3, 27.6])

    def test_joined_coexist_accepts_liquid_shift(self):
        args = build_parser().parse_args(
            [
                "make-joined-coexist",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "940",
                "--solid-source",
                "solid",
                "--liquid-source",
                "liquid",
                "--liquid-shift",
                "0.25",
                "0.5",
                "0.75",
            ]
        )
        self.assertEqual(args.liquid_shift, [0.25, 0.5, 0.75])

    def test_coexist_seed_accepts_phase_z_lengths(self):
        args = build_parser().parse_args(
            [
                "make-coexist",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "1600",
                "--phase-z-lengths",
                "8.1",
                "9.213",
            ]
        )
        self.assertEqual(args.phase_z_lengths, [8.1, 9.213])

    def test_tiled_coexist_accepts_template_frame(self):
        args = build_parser().parse_args(
            [
                "make-tiled-coexist",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "940",
                "--template",
                "template-run",
                "--template-frame",
                "35",
            ]
        )
        self.assertEqual(args.template_frame, "35")

    def test_restart_can_discard_source_velocities(self):
        args = build_parser().parse_args(
            [
                "make-restart",
                "--element",
                "Al",
                "--out",
                "unused",
                "--temperature",
                "940",
                "--source",
                "source",
                "--discard-velocities",
            ]
        )
        self.assertTrue(args.discard_velocities)

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
