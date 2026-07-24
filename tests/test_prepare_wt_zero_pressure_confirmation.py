import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_wt_zero_pressure_confirmation import volume_scan_specifications


class PrepareWtZeroPressureConfirmationTests(unittest.TestCase):
    def make_phase(self, root: Path, phase: str, temperature: float = 975.0) -> None:
        phase_root = root / phase
        rows = []
        for label, volume, pressure in (
            ("vpa_17p900", 17.9, 1.0),
            ("vpa_18p050", 18.05, -1.0),
        ):
            dump = phase_root / label / f"OUT.{label}" / "MD_dump"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text("trajectory\n", encoding="utf-8")
            rows.append(
                {
                    "label": label,
                    "volume_per_atom_A3": volume,
                    "phase_status": f"{phase}_verified",
                    "temperature_mean_within_25_K": True,
                    "pressure_last_half_kbar": {"mean": pressure},
                }
            )
        report = {
            "target_kedf": "wt",
            "target_temperature_K": temperature,
            "rows": rows,
            "zero_pressure_bracket_A3_per_atom": [17.9, 18.05],
            "linear_fit": {"zero_pressure_volume_per_atom_A3": 18.01},
            "checks": {"strict_gate": True},
            "status": "zero_pressure_volume_verified",
        }
        (phase_root / "nvt_volume_scan_result.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def test_uses_nearest_verified_source_and_fitted_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_phase(root, "solid")
            self.make_phase(root, "liquid")

            temperature, specifications = volume_scan_specifications(root)

            self.assertEqual(temperature, 975.0)
            self.assertEqual([item[0] for item in specifications], ["solid", "liquid"])
            self.assertTrue(all("vpa_18p050" in item[1] for item in specifications))
            self.assertTrue(all(item[2] == 18.01 for item in specifications))

    def test_rejects_unverified_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_phase(root, "solid")
            self.make_phase(root, "liquid")
            path = root / "liquid" / "nvt_volume_scan_result.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["status"] = "zero_pressure_volume_not_verified"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "liquid zero-pressure"):
                volume_scan_specifications(root)


if __name__ == "__main__":
    unittest.main()
