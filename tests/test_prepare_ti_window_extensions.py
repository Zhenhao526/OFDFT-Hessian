from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ti_window_extensions import (
    parse_source_run_overrides,
    resolve_pair_model,
    resolve_phase_roots,
    validate_source_run_override,
)


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


class SourceRunOverrideTests(unittest.TestCase):
    def test_accepts_verified_same_volume_lambda_one_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "liquid"
            run.mkdir()
            (run / "metadata.json").write_text(
                """{
  "target_kedf": "lkt",
  "phase": "liquid",
  "volume_per_atom_A3": 18.735
}
"""
            )
            (run / "phase_analysis.json").write_text(
                '{"status": "liquid_verified"}\n'
            )
            selected = validate_source_run_override(
                run,
                target_kedf="lkt",
                phase="liquid",
                volume_per_atom_A3=18.735,
                selected_windows=[{"lambda": 1.0}],
            )

        self.assertEqual(selected, run.resolve())

    def test_rejects_source_override_away_from_lambda_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "lambda=1"):
            validate_source_run_override(
                Path("/tmp/not-read"),
                target_kedf="lkt",
                phase="liquid",
                volume_per_atom_A3=18.735,
                selected_windows=[{"lambda": 0.875}],
            )

    def test_parses_phase_specific_source_override(self) -> None:
        roots = parse_source_run_overrides(["liquid=/tmp/liquid"])
        self.assertEqual(roots, {"liquid": Path("/tmp/liquid").resolve()})


if __name__ == "__main__":
    unittest.main()
