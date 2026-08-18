import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from scripts.prepare_al108_ti_windows import apply_element_phase_model, parse_components, prepare


class PrepareAl108TiWindowsTests(unittest.TestCase):
    def test_component_parser_accepts_optional_reference_virial(self):
        rows = [
            "MPN_TI_COMPONENTS step=1 lambda=0.5 U_MPN_eV=-10 "
            "U_REF_eV=-11 DELTA_U_eV=1 RMIN_A=2.3",
            "MPN_TI_COMPONENTS step=2 lambda=0.5 U_MPN_eV=-9 "
            "U_REF_eV=-10 W_REF_eV=-3.2 DELTA_U_eV=1 RMIN_A=2.2",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_md.log"
            path.write_text("\n".join(rows) + "\n")
            parsed = parse_components(path)
        self.assertEqual([row["step"] for row in parsed], [1, 2])
        self.assertEqual([row["U_target_eV"] for row in parsed], [-10.0, -9.0])
        self.assertEqual([row["delta_U_eV"] for row in parsed], [1.0, 1.0])


@pytest.mark.parametrize("target_kedf", ("xwm", "lkt"))
def test_prepare_records_non_wt_target(
    tmp_path: Path,
    target_kedf: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "of_kinetic": target_kedf,
                "pseudo_dir": ".",
                "kmesh": [1, 1, 1],
                "mpirun_np": 36,
            }
        )
    )
    pair_model = tmp_path / "pair.json"
    pair_model.write_text("{}\n")

    class Atoms:
        natoms = 108

    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.load_atom_source",
        lambda *args, **kwargs: {
            "atoms": Atoms(),
            "source": "source",
            "step": 20,
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.scaled_to_volume",
        lambda atoms, volume: atoms,
    )
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.load_json",
        lambda path: (
            json.loads(path.read_text())
            if path == config
            else {"element": "Al"}
        ),
    )
    captured = []
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.write_job",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    out = tmp_path / "out"
    prepare(
        argparse.Namespace(
            out=out,
            source="source",
            source_frame="last",
            phase="solid",
            temperature=900.0,
            volume_per_atom=18.0,
            lambdas=[0.0, 1.0],
            steps=10,
            dt=1.0,
            csvr_tau=5.0,
            dumpfreq=1,
            restartfreq=10,
            seed=10,
            pair_model=pair_model,
            config=config,
            ranks=36,
        )
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == "kedf-pair-ti-windows-v1"
    assert manifest["target_kedf"] == target_kedf
    assert len(captured) == 2
    assert all(call[0][3]["of_kinetic"] == target_kedf for call in captured)
    assert all(call[0][3]["init_vel"] == 0 for call in captured)
    assert all(
        call[1]["extra_metadata"]["target_kedf"] == target_kedf
        for call in captured
    )


def test_prepare_supports_explicit_mg_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"of_kinetic": "wt", "mpirun_np": 12}))
    element_config = tmp_path / "mg.json"
    element_config.write_text(json.dumps({"element": "Mg"}))
    pair_model = tmp_path / "pair.json"
    pair_model.write_text("{}\n")

    class Atoms:
        natoms = 128

    source_calls = []
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.load_atom_source",
        lambda source, frame, symbol, **kwargs: (
            source_calls.append(symbol)
            or {"atoms": Atoms(), "source": "source", "step": 20}
        ),
    )
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.scaled_to_volume", lambda atoms, volume: atoms
    )
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.load_json",
        lambda path: json.loads(path.read_text()),
    )
    captured = []
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.write_job",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    out = tmp_path / "out"
    prepare(
        argparse.Namespace(
            out=out,
            source="source",
            source_frame="last",
            phase="solid",
            temperature=900.0,
            volume_per_atom=23.18,
            lambdas=[0.0, 1.0],
            steps=10,
            dt=1.0,
            csvr_tau=5.0,
            dumpfreq=1,
            restartfreq=10,
            seed=10,
            pair_model=pair_model,
            config=config,
            ranks=12,
            element_symbol="Mg",
            element_config=element_config,
        )
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert source_calls == ["Mg"]
    assert manifest["element"] == "Mg"
    assert manifest["natoms"] == 128
    assert all(call[0][2]["element"] == "Mg" for call in captured)
    assert all(call[1]["suffix"].startswith("mg128_wt_solid") for call in captured)


def test_mg_phase_status_uses_hcp_dynamics(monkeypatch):
    monkeypatch.setattr(
        "scripts.prepare_al108_ti_windows.hcp_phase_checks",
        lambda phase, expected, thermalized_initial: {
            "bounded": thermalized_initial,
            "nearest": True,
        },
    )
    phase = {"status": "solid_not_verified"}
    result = apply_element_phase_model(
        phase, {"element": "Mg", "phase": "solid"}
    )
    assert result["status"] == "solid_verified"
    assert result["legacy_structure_status"] == "solid_not_verified"


if __name__ == "__main__":
    unittest.main()
