from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ti_window_extensions import resolve_pair_model, resolve_phase_roots


class ResolvePhaseRootsTests(unittest.TestCase):
    def test_uses_legacy_parent_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            roots = resolve_phase_roots(parent, ["solid", "liquid"], None)

        self.assertEqual(roots["solid"], parent.resolve() / "solid")
        self.assertEqual(roots["liquid"], parent.resolve() / "liquid")

    def test_accepts_independent_liquid_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "liquid-v3"
            roots = resolve_phase_roots(
                None,
                ["solid", "liquid"],
                [f"liquid={root}"],
            )

        self.assertEqual(roots, {"liquid": root.resolve()})

    def test_rejects_duplicate_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_phase_roots(
                None,
                ["liquid"],
                ["liquid=/tmp/a", "liquid=/tmp/b"],
            )


class ResolvePairModelTests(unittest.TestCase):
    def test_accepts_verified_lambda_one_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.json"
            override = root / "bridge.json"
            override.write_text(
                """{
  "target_kedf": "lkt",
  "phase": "liquid",
  "reference_gate_passed": true,
  "short_range_guard_passed": true
}
"""
            )
            selected, changed = resolve_pair_model(
                parent,
                override,
                target_kedf="lkt",
                phase="liquid",
                selected_windows=[{"lambda": 1.0}],
            )

        self.assertEqual(selected, override.resolve())
        self.assertTrue(changed)

    def test_rejects_override_away_from_lambda_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "bridge.json"
            override.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "lambda=1"):
                resolve_pair_model(
                    Path(directory) / "parent.json",
                    override,
                    target_kedf="lkt",
                    phase="liquid",
                    selected_windows=[{"lambda": 0.875}],
                )


if __name__ == "__main__":
    unittest.main()
