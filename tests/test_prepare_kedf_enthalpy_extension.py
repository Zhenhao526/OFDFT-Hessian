from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mpn_melting.structures import AtomSet
from scripts.prepare_kedf_enthalpy_extension import (
    load_verified_source,
    prepare_extension,
)


def write_source(root: Path, *, method: str = "xwm") -> None:
    phases = []
    results = []
    for phase, volume in (("solid", 18.0), ("liquid", 18.8)):
        run = root / phase
        run.mkdir(parents=True)
        phases.append(
            {
                "phase": phase,
                "run": str(run),
                "volume_per_atom_A3": volume,
            }
        )
        results.append({"phase": phase, "status": "passed"})
    (root / "confirmation_manifest.json").write_text(
        json.dumps(
            {
                "schema": "kedf-volume-confirmation-v1",
                "target_kedf": method,
                "temperature_K": 900.0,
                "target_pressure_kbar": 0.0,
                "stress_available": method == "lkt",
                "steps": 3000,
                "phases": phases,
            }
        )
    )
    (root / "confirmation_summary.json").write_text(
        json.dumps(
            {
                "status": "volume_confirmation_verified",
                "target_kedf": method,
                "results": results,
            }
        )
    )


def test_rejects_unverified_source(tmp_path: Path) -> None:
    write_source(tmp_path)
    summary = tmp_path / "confirmation_summary.json"
    document = json.loads(summary.read_text())
    document["status"] = "volume_confirmation_failed"
    summary.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not verified"):
        load_verified_source(tmp_path)


@pytest.mark.parametrize("method", ["xwm", "lkt"])
def test_preserves_velocities_and_parent_provenance(
    tmp_path: Path,
    method: str,
) -> None:
    source_root = tmp_path / "source"
    write_source(source_root, method=method)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "of_kinetic": method,
                "cal_stress": method == "lkt",
                "esolver_type": "ofdft",
                "of_method": "tn",
                "pseudo_dir": "/tmp",
                "kmesh": [1, 1, 1],
            }
        )
    )
    source_dump = tmp_path / "MD_dump"
    source_dump.write_text("source")
    atoms = AtomSet(
        ["Al"] * 4,
        [(0.0, 0.0, 0.0)] * 4,
        [(4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)],
        velocities=[(0.01, 0.02, 0.03)] * 4,
    )

    def fake_source(path, frame, element, include_velocities=False):
        assert include_velocities is True
        return {"source": str(source_dump), "step": 2995, "atoms": atoms}

    out = tmp_path / "extension"
    with patch(
        "scripts.prepare_kedf_enthalpy_extension.load_atom_source",
        side_effect=fake_source,
    ):
        manifest = prepare_extension(
            source_root,
            out,
            config,
            3000,
            5.0,
            100,
            36,
        )

    metadata = json.loads((out / "solid" / "metadata.json").read_text())
    assert manifest["target_kedf"] == method
    assert manifest["parent_confirmation"] == str(source_root.resolve())
    assert manifest["source_velocities_discarded"] is False
    assert metadata["source_step"] == 2995
    assert metadata["source_velocities_discarded"] is False
    assert " v " in (out / "solid" / "STRU").read_text()
