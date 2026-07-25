import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
