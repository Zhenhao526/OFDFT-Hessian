from __future__ import annotations

import unittest

from scripts.prepare_kedf_temperature_continuation import validate_protocol


class TemperatureProtocolTests(unittest.TestCase):
    def test_accepts_cooling_ramp(self) -> None:
        validate_protocol(1200.0, 900.0, 1000, 2.0, 72)

    def test_rejects_nonpositive_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperatures"):
            validate_protocol(0.0, 900.0, 1000, 2.0, 72)

    def test_rejects_nonpositive_steps_tau_or_ranks(self) -> None:
        for arguments in (
            (1200.0, 900.0, 0, 2.0, 72),
            (1200.0, 900.0, 1000, 0.0, 72),
            (1200.0, 900.0, 1000, 2.0, 0),
        ):
            with self.assertRaises(ValueError):
                validate_protocol(*arguments)


if __name__ == "__main__":
    unittest.main()
