import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.report_wt_enthalpy_progress import report


class WtEnthalpyProgressTests(unittest.TestCase):
    def test_discovers_base_and_extension_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in (
                "T0975_steps3000",
                "T1050_steps3000_round1",
                "T0975_steps6000_round3",
            ):
                for phase in ("solid", "liquid"):
                    (root / label / phase).mkdir(parents=True)

            result = report(root, 3000)

        self.assertEqual(len(result["runs"]), 6)
        self.assertEqual(
            {row["temperature_label"] for row in result["runs"]},
            {
                "T0975_steps3000",
                "T1050_steps3000_round1",
                "T0975_steps6000_round3",
            },
        )
        self.assertTrue(all(row["status"] == "initializing" for row in result["runs"]))

    def test_reports_kedf_log_without_analytic_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "T0900_steps3000" / "solid"
            output = run / "OUT.test"
            output.mkdir(parents=True)
            (output / "running_md.log").write_text("")
            samples = [
                {"step": step, "temperature_K": 900.0 + step}
                for step in range(11)
            ]
            with mock.patch(
                "scripts.report_wt_enthalpy_progress.parse_md_log",
                return_value=(samples, 10),
            ):
                result = report(root, 3000)

        row = result["runs"][0]
        self.assertEqual(row["pressure_recent_kbar"], {})
        self.assertFalse(row["trajectory_pressure_available"])
        self.assertEqual(row["temperature_recent_K"]["last"], 910.0)


if __name__ == "__main__":
    unittest.main()
