import unittest
from pathlib import Path

from mpn_melting.scan import ScanPoint, estimate_threshold_temperature, summarize_scan


class ScanTests(unittest.TestCase):
    def test_threshold_interpolation(self):
        points = [
            ScanPoint(Path("T1000"), 1000.0, "rescale_v", 1000.0, 1000.0, 0.08, 0.2, 2.5, 0.0),
            ScanPoint(Path("T1400"), 1400.0, "rescale_v", 1400.0, 1400.0, 0.12, 0.3, 2.5, 0.0),
        ]
        estimate = estimate_threshold_temperature(points, threshold=0.1)
        self.assertIn("1200", estimate)

    def test_summarize_scan_table(self):
        points = [
            ScanPoint(Path("T1000"), 1000.0, "rescale_v", 1000.0, 1000.0, 0.08, 0.2, 2.5, 0.0),
            ScanPoint(Path("T1400"), 1400.0, "rescale_v", 1400.0, 1400.0, 0.12, 0.3, 2.5, 0.0),
        ]
        text = summarize_scan(points, threshold=0.1)
        self.assertIn("| T1000 |", text)
        self.assertIn("Estimated threshold crossing", text)


if __name__ == "__main__":
    unittest.main()
