import tempfile
import unittest
from pathlib import Path

from scripts.finalize_wt_gibbs_helmholtz_melting import (
    aggregate_results,
    write_plot,
)


def result(discard, root, offset):
    grid = (900.0, 1000.0, 1100.0)
    return {
        "schema": "wt-gibbs-helmholtz-melting-v2",
        "status": "verified",
        "checks": {"all": True},
        "thermodynamic_convention": {
            "delta_g": "G_liquid_minus_G_solid",
            "delta_h": "H_liquid_minus_H_solid",
            "root_definition": "delta_g(T_m)=0",
        },
        "anchor_temperature_k": 900.0,
        "anchor_delta_g_ev_per_atom": 0.01,
        "melting_temperature_k": root,
        "statistical_interval": {
            "lower_k": root - 5.0,
            "upper_k": root + 5.0,
            "scenarios": [{"root_k": root - 5.0}, {"root_k": root + 5.0}],
        },
        "conservative_interval": {
            "lower_k": root - 40.0,
            "upper_k": root + 40.0,
            "scenarios": [{"root_k": root - 40.0}, {"root_k": root + 40.0}],
        },
        "enthalpy_points": [
            {
                "temperature_k": temperature,
                "discard_fraction": discard,
            }
            for temperature in grid
        ],
        "gibbs_profile": [
            {
                "temperature_k": temperature,
                "delta_g_ev_per_atom": value + offset,
            }
            for temperature, value in zip(grid, (0.01, 0.0, -0.01))
        ],
    }


class FinalizeWtGibbsHelmholtzMeltingTests(unittest.TestCase):
    def setUp(self):
        self.results = {
            "d25": result(0.25, 1002.0, 0.001),
            "d50": result(0.5, 1000.0, 0.0),
            "d75": result(0.75, 999.0, -0.001),
        }

    def test_aggregates_verified_roots_and_envelopes(self):
        summary = aggregate_results(self.results)
        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["melting_temperature_k"], 1000.0)
        self.assertEqual(
            summary["nominal_discard_envelope_k"]["lower"], 999.0
        )
        self.assertEqual(
            summary["nominal_discard_envelope_k"]["upper"], 1002.0
        )
        self.assertEqual(
            summary["all_discard_conservative_envelope_k"]["lower"], 959.0
        )
        self.assertEqual(
            summary["all_discard_conservative_envelope_k"]["upper"], 1042.0
        )

    def test_rejects_unbracketed_scenario(self):
        self.results["d50"]["conservative_interval"]["scenarios"][0][
            "root_k"
        ] = None
        with self.assertRaisesRegex(ValueError, "not temperature-bracketed"):
            aggregate_results(self.results)

    def test_rejects_nonmonotonic_profile(self):
        self.results["d75"]["gibbs_profile"][1]["delta_g_ev_per_atom"] = 0.02
        with self.assertRaisesRegex(ValueError, "profile_monotonic"):
            aggregate_results(self.results)

    def test_writes_dependency_free_svg_plot(self):
        summary = aggregate_results(self.results)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curve.svg"
            write_plot(output, self.results, summary)
            text = output.read_text(encoding="utf-8")
        self.assertIn("<svg", text)
        self.assertIn("Gibbs-Helmholtz", text)


if __name__ == "__main__":
    unittest.main()
