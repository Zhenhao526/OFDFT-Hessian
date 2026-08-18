from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_mg_phase_md.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_mg_phase_md", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepares_independent_ks_melt_job(tmp_path, monkeypatch) -> None:
    module = load_script()
    out = tmp_path / "job"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--repo",
            str(ROOT),
            "--out",
            str(out),
            "--method",
            "ksdft",
            "--phase-role",
            "melt",
            "--volume-per-atom",
            "22.5",
            "--temperature-k",
            "1800",
            "--steps",
            "1000",
            "--seed",
            "2026080401",
        ],
    )
    module.main()
    manifest = json.loads((out / "mg_phase_md_manifest.json").read_text())
    assert manifest["method"] == "ksdft"
    assert manifest["natoms"] == 128
    assert "suffix mg128_ksdft_melt_T1800" in (out / "INPUT").read_text()
    assert manifest["source"]["kind"] == "generated_hcp"
    expected_sigma = 1800.0 * module.KB_RY_PER_K
    assert f"smearing_sigma {expected_sigma}" in (out / "INPUT").read_text()
    assert "init_vel 0" in (out / "INPUT").read_text()
    assert manifest["abacus_init_vel"] == 0


def test_generated_hcp_cell_uses_requested_ratio_and_kmesh(tmp_path, monkeypatch) -> None:
    module = load_script()
    out = tmp_path / "job"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--repo",
            str(ROOT),
            "--out",
            str(out),
            "--method",
            "ksdft",
            "--phase-role",
            "solid",
            "--volume-per-atom",
            "21.2",
            "--temperature-k",
            "900",
            "--steps",
            "20",
            "--seed",
            "7",
            "--c-over-a",
            "1.623",
            "--kmesh",
            "2",
            "2",
            "2",
        ],
    )
    module.main()
    manifest = json.loads((out / "mg_phase_md_manifest.json").read_text())
    assert manifest["hcp_c_over_a"] == 1.623
    assert manifest["kmesh"] == [2, 2, 2]
    assert manifest["source"]["c_over_a"] == 1.623
    assert "2 2 2 0 0 0" in (out / "KPT").read_text()


def test_scaled_to_volume_preserves_velocities() -> None:
    module = load_script()
    atoms = module.AtomSet(
        symbols=["Mg"],
        scaled_positions=[(0.0, 0.0, 0.0)],
        lattice_vectors=[(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)],
        velocities=[(1.0, 2.0, 3.0)],
    )
    scaled = module.scaled_to_volume(atoms, 27.0)
    assert abs(module.lattice_volume(scaled.lattice_vectors) - 27.0) < 1.0e-12
    assert scaled.velocities == [(1.0, 2.0, 3.0)]
