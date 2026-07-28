from __future__ import annotations

from pathlib import Path

from scripts import qm9_pbe_hessian_reference_set as reference


def test_checkpoint_restoration_disables_hessian_writes(
    monkeypatch, tmp_path: Path
) -> None:
    class MeanField:
        def __init__(self, molecule, xc):
            self.molecule = molecule
            self.xc = xc
            self.chkfile = "must-not-remain-set"
            self.grids = type("Grids", (), {})()

    molecule = object()
    record = {
        "mo_coeff": "coefficients",
        "mo_occ": "occupations",
        "mo_energy": "orbital-energies",
        "e_tot": -1.0,
    }
    values = {
        "Results/name_xc_functional": "PBE",
        "Results/grid_level": 3,
    }
    monkeypatch.setattr(
        reference.scf.chkfile,
        "load_scf",
        lambda _: (molecule, record),
    )
    monkeypatch.setattr(
        reference.scf.chkfile,
        "load",
        lambda _, key: values[key],
    )
    monkeypatch.setattr(reference.dft, "RKS", MeanField)

    restored_molecule, mean_field = reference._mf_from_chk(
        tmp_path / "frozen.chk"
    )

    assert restored_molecule is molecule
    assert mean_field.chkfile is None
    assert mean_field.mo_coeff == "coefficients"
    assert mean_field.mo_occ == "occupations"
    assert mean_field.mo_energy == "orbital-energies"
