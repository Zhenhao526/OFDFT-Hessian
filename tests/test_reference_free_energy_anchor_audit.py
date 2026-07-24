from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_reference_free_energy_anchor import (
    finite_n_ideal_gas_correction_ev_per_atom,
    inspect_anchor_window_provenance,
    reconstruct_analytic,
)


class AnalyticReferenceAuditTests(unittest.TestCase):
    def test_finite_n_ideal_correction_matches_exact_factorial_expression(self):
        value = finite_n_ideal_gas_correction_ev_per_atom(108, 900.0)

        self.assertTrue(
            math.isclose(value, 0.002341599277213511, rel_tol=1.0e-13)
        )

    def test_reconstructs_saved_analytic_reference_values(self):
        analytic = {
            "temperature_k": 900.0,
            "natoms": 108,
            "mass_amu": 26.9815385,
            "einstein": {
                "solid_volume_per_atom_angstrom3": 17.94,
                "spring_constant_ev_per_angstrom2": 1.6715519282032714,
            },
            "suf": {
                "liquid_volume_per_atom_angstrom3": 18.736615407000613,
                "sigma_angstrom": 1.54,
                "beta_excess_free_energy": 11.69739979491142,
            },
        }

        result = reconstruct_analytic(analytic)

        self.assertTrue(
            math.isclose(
                result["einstein_total_ev_per_atom"], -0.3743492846424327
            )
        )
        self.assertTrue(
            math.isclose(result["suf_reduced_density_x"], 0.5427075125697992)
        )
        self.assertTrue(
            math.isclose(result["suf_ideal_ev_per_atom"], -0.8141339981410283)
        )
        self.assertTrue(
            math.isclose(result["suf_total_ev_per_atom"], 0.09306953285865627)
        )


class AnchorWindowProvenanceTests(unittest.TestCase):
    def test_recovers_atom_count_temperature_and_lambda_from_trajectories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows = []
            for index, coupling in enumerate((0.0, 0.5, 1.0)):
                path = root / f"window-{index}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "lambda": coupling,
                            "temperature_k": 900.0,
                            "positions_angstrom": [[0.0, 0.0, 0.0]] * 108,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                windows.append(
                    {
                        "lambda": coupling,
                        "path": str(path),
                        "temperature_mean_k": 900.0 + index,
                    }
                )

            result = inspect_anchor_window_provenance({"windows": windows})

        self.assertEqual(result["natoms"], [108])
        self.assertEqual(result["lambdas"], [0.0, 0.5, 1.0])
        self.assertTrue(result["lambda_labels_match_trajectories"])
        self.assertTrue(
            all(
                row["initial_temperature_k"] == 900.0
                for row in result["windows"]
            )
        )

    def test_detects_mismatched_lambda_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "window.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "lambda": 0.25,
                        "temperature_k": 900.0,
                        "positions_angstrom": [[0.0, 0.0, 0.0]] * 4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = inspect_anchor_window_provenance(
                {
                    "windows": [
                        {
                            "lambda": 0.5,
                            "path": str(path),
                            "temperature_mean_k": 900.0,
                        }
                    ]
                }
            )

        self.assertFalse(result["lambda_labels_match_trajectories"])


if __name__ == "__main__":
    unittest.main()
