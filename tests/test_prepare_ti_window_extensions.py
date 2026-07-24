from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ti_window_extensions import resolve_phase_roots


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


if __name__ == "__main__":
    unittest.main()
