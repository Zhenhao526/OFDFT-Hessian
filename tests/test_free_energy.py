import json
import tempfile
import unittest
from pathlib import Path

from mpn_melting.free_energy import (
    exponential_free_energy_difference,
    load_run_frames,
    select_frames,
    write_dataset,
)
from mpn_melting.parser import RY_TO_EV


def md_frame(step: int, x: float, force: float) -> str:
    return f"""MDSTEP:  {step}
LATTICE_CONSTANT: 1.0 Angstrom
LATTICE_VECTORS
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
INDEX    LABEL    POSITION (Angstrom)    FORCE (eV/Angstrom)    VELOCITY (Angstrom/fs)
  0  Al  {x} 0.0 0.0  {force} 0.0 0.0  0.0 0.0 0.0

"""


def md_observable(total: float, potential: float, kinetic: float, temperature: float) -> str:
    return f"""Energy (Ry) Potential (Ry) Kinetic (Ry) Temperature (K)
{total} {potential} {kinetic} {temperature}
"""


class FreeEnergyDatasetTests(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "run"
        output = run / "OUT.ABACUS"
        output.mkdir(parents=True)
        (output / "MD_dump").write_text(
            md_frame(0, 1.0, 0.1) + md_frame(5, 1.1, 0.2), encoding="utf-8"
        )
        (output / "running_md.log").write_text(
            md_observable(-2.0, -2.1, 0.1, 800.0)
            + md_observable(-1.9, -2.05, 0.15, 850.0),
            encoding="utf-8",
        )
        return run

    def test_load_and_select_aligned_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames, source = load_run_frames(self.make_run(Path(tmp)), "solid")
        self.assertEqual(source["available_frames"], 2)
        self.assertEqual(frames[1].md_step, 5)
        self.assertAlmostEqual(frames[1].potential_energy_ev, -2.05 * RY_TO_EV)
        self.assertEqual(frames[1].forces_ev_per_angstrom[0], (0.2, 0.0, 0.0))
        self.assertEqual([frame.source_index for frame in select_frames(frames, 0.5)], [1])

    def test_write_manifest_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            manifest = write_dataset([("liquid", run)], root / "dataset", discard_fraction=0.0, stride=2)
            record = json.loads((root / "dataset" / "frames.jsonl").read_text(encoding="utf-8"))
            disk_manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase_counts"], {"liquid": 1})
        self.assertEqual(record["natoms"], 1)
        self.assertEqual(record["volume_angstrom3"], 1000.0)
        self.assertEqual(len(disk_manifest["frames_sha256"]), 64)
        self.assertEqual(len(disk_manifest["sources"][0]["md_dump_sha256"]), 64)

    def test_exponential_free_energy_constant_and_stable_extremes(self):
        result = exponential_free_energy_difference([2.0, 2.0, 2.0], 900.0)
        self.assertAlmostEqual(result["delta_f_ev"], 2.0)
        self.assertAlmostEqual(result["effective_sample_size"], 3.0)
        extreme = exponential_free_energy_difference([-1000.0, 1000.0], 900.0)
        self.assertTrue(extreme["delta_f_ev"] < -999.0)
        self.assertAlmostEqual(extreme["effective_sample_size"], 1.0)


if __name__ == "__main__":
    unittest.main()
