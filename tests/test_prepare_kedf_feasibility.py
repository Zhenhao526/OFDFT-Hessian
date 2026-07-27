from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.prepare_kedf_feasibility import prepare


@pytest.mark.parametrize(
    ("method", "stress"),
    (("xwm", False), ("lkt", True)),
)
def test_prepare_records_method_and_stress_availability(
    tmp_path: Path,
    method: str,
    stress: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / f"{method}.json"
    config.write_text(
        json.dumps(
            {
                "of_kinetic": method,
                "cal_stress": int(stress),
                "pseudo_dir": ".",
                "kmesh": [1, 1, 1],
            }
        )
    )

    class Atoms:
        natoms = 1

    monkeypatch.setattr(
        "scripts.prepare_kedf_feasibility.load_atom_source",
        lambda *args, **kwargs: {"atoms": Atoms(), "source": "source", "step": 10},
    )
    monkeypatch.setattr(
        "scripts.prepare_kedf_feasibility.load_json",
        lambda path: json.loads(path.read_text()) if path == config else {"element": "Al"},
    )
    captured = []
    monkeypatch.setattr(
        "scripts.prepare_kedf_feasibility.write_job",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    out = tmp_path / "out"
    prepare(
        argparse.Namespace(
            out=out,
            solid_source="solid",
            liquid_source="liquid",
            source_frame="last",
            config=config,
            temperature=975.0,
            steps=200,
            dt=1.0,
            csvr_tau=2.0,
            seed=5,
            ranks=36,
        )
    )

    manifest = json.loads((out / "feasibility_manifest.json").read_text())
    assert manifest["target_kedf"] == method
    assert manifest["stress_available"] is stress
    assert len(captured) == 2
    assert all(
        bool(call[1]["extra_metadata"]["stress_available"]) is stress
        for call in captured
    )
    assert all(call[0][3]["init_vel"] == 0 for call in captured)
    assert all(
        call[1]["extra_metadata"]["source_velocities_discarded"] is True
        and call[1]["extra_metadata"]["abacus_init_vel"] == 0
        for call in captured
    )
