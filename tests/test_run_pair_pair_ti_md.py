import json

import pytest

torch = pytest.importorskip("torch")

from scripts import run_pair_pair_ti_md as module


def test_pair_interpolation_has_exact_endpoints(monkeypatch):
    reference_model = {"name": "reference"}
    target_model = {"name": "target"}

    def fake_evaluate(positions, lattice, model):
        if model is reference_model:
            return (
                torch.tensor(2.0),
                torch.full_like(positions, 3.0),
                torch.tensor(2.1),
            )
        return (
            torch.tensor(6.0),
            torch.full_like(positions, 7.0),
            torch.tensor(2.2),
        )

    monkeypatch.setattr(module, "evaluate_model", fake_evaluate)
    positions = torch.zeros((2, 3))
    lattice = torch.eye(3)

    zero = module.evaluate_pair_interpolation(
        positions, lattice, reference_model, target_model, 0.0
    )
    one = module.evaluate_pair_interpolation(
        positions, lattice, reference_model, target_model, 1.0
    )
    middle = module.evaluate_pair_interpolation(
        positions, lattice, reference_model, target_model, 0.25
    )

    assert zero[0].item() == 2.0
    assert torch.all(zero[1] == 3.0)
    assert one[0].item() == 6.0
    assert torch.all(one[1] == 7.0)
    assert middle[0].item() == 3.0
    assert torch.all(middle[1] == 4.0)
    assert middle[2].item() == 2.0
    assert middle[3].item() == 6.0
    assert middle[4].item() == pytest.approx(2.1)


def test_pair_document_requires_gates_and_provenance(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "reference_gate_passed": True,
                "short_range_guard_passed": True,
                "target_kedf": "lkt",
                "phase": "liquid",
                "model": {},
            }
        )
    )

    assert module.load_pair_document(
        path, target_kedf="lkt", phase="liquid"
    )["target_kedf"] == "lkt"
    with pytest.raises(ValueError, match="wrong target KEDF"):
        module.load_pair_document(path, target_kedf="xwm", phase="liquid")
    with pytest.raises(ValueError, match="wrong phase"):
        module.load_pair_document(path, target_kedf="lkt", phase="solid")


def test_pair_document_accepts_reference_phase_provenance(tmp_path):
    path = tmp_path / "proxy.json"
    path.write_text(
        json.dumps(
            {
                "reference_gate_passed": True,
                "short_range_guard_passed": True,
                "target_kedf": "lkt",
                "reference_phase": "liquid",
                "model": {},
            }
        )
    )

    assert module.load_pair_document(
        path, target_kedf="lkt", phase="liquid"
    )["reference_phase"] == "liquid"


def test_pair_document_rejects_missing_short_range_gate(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "reference_gate_passed": True,
                "short_range_guard_passed": False,
                "target_kedf": "lkt",
                "phase": "liquid",
                "model": {},
            }
        )
    )

    with pytest.raises(ValueError, match="short-range"):
        module.load_pair_document(path, target_kedf="lkt", phase="liquid")
