import json
import sys

from scripts import prepare_ti_pilot_from_endpoints as module


def test_explicit_csvr_tau_is_used_and_recorded(tmp_path, monkeypatch):
    endpoint = tmp_path / "endpoints"
    output = tmp_path / "pilot"
    solid_model = tmp_path / "solid.dat"
    liquid_model = tmp_path / "liquid.dat"
    config = tmp_path / "config.json"
    solid_model.write_text("solid\n")
    liquid_model.write_text("liquid\n")
    config.write_text(json.dumps({"of_kinetic": "lkt"}))
    endpoint.mkdir()
    (endpoint / "endpoint_manifest.json").write_text(
        json.dumps(
            {
                "target_kedf": "lkt",
                "temperature_K": 900.0,
                "phases": [
                    {
                        "phase": phase,
                        "source": str(tmp_path / phase),
                        "zero_pressure_volume_per_atom_A3": volume,
                    }
                    for phase, volume in (("solid", 18.4), ("liquid", 18.735))
                ],
            }
        )
    )
    for phase in ("solid", "liquid"):
        phase_root = endpoint / phase
        phase_root.mkdir()
        (phase_root / "endpoint_validation.json").write_text(
            json.dumps({"status": "endpoint_validation_passed"})
        )

    prepared = []

    def fake_prepare(arguments):
        prepared.append(arguments)
        arguments.out.mkdir(parents=True)

    monkeypatch.setattr(module, "prepare", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_ti_pilot_from_endpoints.py",
            "--endpoint-root",
            str(endpoint),
            "--out",
            str(output),
            "--solid-pair-model",
            str(solid_model),
            "--liquid-pair-model",
            str(liquid_model),
            "--steps",
            "300",
            "--csvr-tau",
            "5",
            "--config",
            str(config),
        ],
    )

    module.main()

    assert [item.csvr_tau for item in prepared] == [5.0, 5.0]
    manifest = json.loads((output / "pilot_manifest.json").read_text())
    assert manifest["steps"] == 300
    assert manifest["csvr_tau_fs"] == 5.0
