from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_kedf_ti_pilot_replacements import (
    parse_replacement_specs,
    select_window_runs,
)


class PilotReplacementTests(unittest.TestCase):
    def test_parses_phase_and_label(self) -> None:
        parsed = parse_replacement_specs(
            ["solid:lambda_0p000=/tmp/solid-zero"]
        )
        self.assertEqual(
            parsed,
            {("solid", "lambda_0p000"): Path("/tmp/solid-zero")},
        )

    def test_rejects_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_replacement_specs(
                [
                    "liquid:lambda_0p500=/tmp/a",
                    "liquid:lambda_0p500=/tmp/b",
                ]
            )

    def test_selects_only_requested_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ("solid", "liquid"):
                phase_root = root / phase
                phase_root.mkdir()
                (phase_root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "windows": [
                                {"label": "lambda_0p000", "lambda": 0.0},
                                {"label": "lambda_1p000", "lambda": 1.0},
                            ]
                        }
                    )
                )
            replacement = root / "replacement"
            selected = select_window_runs(
                root,
                {("liquid", "lambda_1p000"): replacement},
            )

        self.assertFalse(selected["solid"][0]["replaced"])
        self.assertFalse(selected["liquid"][0]["replaced"])
        self.assertTrue(selected["liquid"][1]["replaced"])
        self.assertEqual(selected["liquid"][1]["run"], replacement.resolve())

    def test_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ("solid", "liquid"):
                phase_root = root / phase
                phase_root.mkdir()
                (phase_root / "manifest.json").write_text(
                    json.dumps(
                        {"windows": [{"label": "lambda_0p000", "lambda": 0.0}]}
                    )
                )
            with self.assertRaisesRegex(ValueError, "not present"):
                select_window_runs(
                    root,
                    {("solid", "lambda_0p500"): root / "replacement"},
                )


if __name__ == "__main__":
    unittest.main()
