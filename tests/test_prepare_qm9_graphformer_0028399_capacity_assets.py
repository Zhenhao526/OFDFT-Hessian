import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.prepare_qm9_graphformer_0028399_capacity_assets import (
    _load_protocol,
)


def _closed_protocol() -> dict:
    return {
        "protocol_id": "qm9_graphformer_0028399_full39_capacity_only_v1",
        "access_boundary": {
            "molecule_ids": ["0028399"],
            "sample_ids": [0],
            "stable5_access_allowed": False,
            "train20_access_allowed": False,
            "held_direction_access_allowed": False,
            "validation_access_allowed": False,
            "test100_access_allowed": False,
        },
        "loss": {
            "lambda_energy": 0.0,
            "lambda_force": 0.0,
            "lambda_density": 0.0,
            "lambda_hessian": 1.0,
        },
    }


@pytest.mark.parametrize(
    "protocol_id",
    [
        "qm9_graphformer_0028399_full39_capacity_only_v1",
        "qm9_graphformer_0028399_full39_capacity_only_v2",
    ],
)
def test_capacity_asset_protocol_accepts_only_closed_single_parent(
    tmp_path, protocol_id
):
    protocol = _closed_protocol()
    protocol["protocol_id"] = protocol_id
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    loaded = _load_protocol(path)
    assert loaded["access_boundary"]["molecule_ids"] == ["0028399"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("access_boundary", "molecule_ids", ["0028399", "0031108"]),
        ("access_boundary", "held_direction_access_allowed", True),
        ("access_boundary", "validation_access_allowed", True),
        ("access_boundary", "test100_access_allowed", True),
        ("loss", "lambda_energy", 0.1),
        ("loss", "lambda_force", 1.0),
    ],
)
def test_capacity_asset_protocol_fails_closed(
    tmp_path, section, key, value
):
    protocol = _closed_protocol()
    protocol[section][key] = value
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError):
        _load_protocol(path)
