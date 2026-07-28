from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_kedf_volume_confirmation import prepare


@pytest.mark.parametrize(
    ("method", "stress"),
    (("xwm", False), ("lkt", True)),
)
def test_prepare_uses_fresh_velocities_and_method_stress(
    tmp_path: Path,
    method: str,
    stress: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
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

    source_file = tmp_path / "source"
    source_file.write_text("source structure\n")
    source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.load_atom_source",
        lambda *args, **kwargs: {
            "atoms": Atoms(),
            "source": str(source_file),
            "step": 20,
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.scaled_to_volume",
        lambda atoms, volume: atoms,
    )
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.load_json",
        lambda path: (
            json.loads(path.read_text())
            if path == config
            else {"element": "Al"}
        ),
    )
    captured = []
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.write_job",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    prepare(
        argparse.Namespace(
            out=tmp_path / "out",
            solid_source="solid",
            liquid_source="liquid",
            solid_volume=18.0,
            liquid_volume=19.0,
            config=config,
            temperature=975.0,
            steps=300,
            csvr_tau=5.0,
            seed=1,
            ranks=36,
            phases=("solid", "liquid"),
        )
    )
    assert len(captured) == 2
    assert all(call[0][3]["init_vel"] == 0 for call in captured)
    assert all(
        bool(call[0][3]["cal_stress"]) is stress for call in captured
    )
    assert all(
        call[1]["extra_metadata"]["source_velocities_discarded"] is True
        for call in captured
    )
    assert all(
        call[1]["extra_metadata"]["source_structure_sha256"]
        == source_sha256
        for call in captured
    )


def test_generated_run_script_disables_mpi_binding(tmp_path: Path) -> None:
    from mpn_melting.abacus_input import run_script_text

    script = run_script_text(
        "/path/to/abacus",
        "/path/to/mpirun",
        36,
        ("--bind-to", "none"),
    )
    assert (
        '"/path/to/mpirun" "--bind-to" "none" -np 36 '
        '"/path/to/abacus"'
    ) in script


def test_prepare_can_select_one_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "of_kinetic": "lkt",
                "cal_stress": 1,
                "pseudo_dir": ".",
                "kmesh": [1, 1, 1],
            }
        )
    )

    class Atoms:
        natoms = 1

    source_file = tmp_path / "source"
    source_file.write_text("source structure\n")
    source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.load_atom_source",
        lambda *args, **kwargs: {
            "atoms": Atoms(),
            "source": str(source_file),
            "step": 20,
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.scaled_to_volume",
        lambda atoms, volume: atoms,
    )
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.load_json",
        lambda path: (
            json.loads(path.read_text())
            if path == config
            else {"element": "Al"}
        ),
    )
    captured = []
    monkeypatch.setattr(
        "scripts.prepare_kedf_volume_confirmation.write_job",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    out = tmp_path / "solid_only"
    prepare(
        argparse.Namespace(
            out=out,
            solid_source="solid",
            liquid_source="liquid",
            solid_volume=18.3,
            liquid_volume=19.0,
            config=config,
            temperature=975.0,
            steps=300,
            csvr_tau=5.0,
            seed=2,
            ranks=72,
            phases=("solid",),
        )
    )
    manifest = json.loads(
        (out / "confirmation_manifest.json").read_text()
    )
    assert [item["phase"] for item in manifest["phases"]] == ["solid"]
    assert manifest["phases"][0]["source_structure_sha256"] == source_sha256
    assert len(captured) == 1
    assert captured[0][0][3]["mpirun_np"] == 72


def test_prepare_rejects_seed_outside_abacus_integer_range(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "of_kinetic": "xwm",
                "cal_stress": 0,
                "pseudo_dir": ".",
                "kmesh": [1, 1, 1],
            }
        )
    )

    with pytest.raises(ValueError, match="signed 32-bit"):
        prepare(
            argparse.Namespace(
                out=tmp_path / "out",
                solid_source="solid",
                liquid_source="liquid",
                solid_volume=18.0,
                liquid_volume=19.0,
                config=config,
                temperature=1050.0,
                steps=300,
                csvr_tau=1.0,
                seed=2_147_483_647,
                ranks=1,
                phases=("solid", "liquid"),
            )
        )

    assert not (tmp_path / "out").exists()
