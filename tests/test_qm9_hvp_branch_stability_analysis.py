import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def _load_module():
    path = Path("scripts/qm9_hvp_branch_stability_analysis.py")
    spec = importlib.util.spec_from_file_location("hvp_branch_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_frozen_protocol_gates_are_numeric_after_explicit_conversion(version):
    protocol = yaml.safe_load(
        Path(f"configs/audit/qm9_hvp_branch_stability_{version}.yaml").read_text()
    )
    gates = {key: float(value) for key, value in protocol["gates"].items()}

    assert gates["tangent_condition_number_max"] == 1.0e10
    assert gates["projected_density_gradient_max"] == 1.0e-8


def test_v3_hierarchical_aggregation_keeps_parent_and_direction_gates_separate():
    protocol = yaml.safe_load(
        Path("configs/audit/qm9_hvp_branch_stability_v3.yaml").read_text()
    )
    aggregation = protocol["aggregation"]

    assert aggregation["minimum_stable_directions"] == 3
    assert "density_gradient" in aggregation["parent_exclusion_gates"]
    assert "step_stability" in aggregation["direction_exclusion_gates"]
    assert not set(aggregation["parent_exclusion_gates"]) & set(
        aggregation["direction_exclusion_gates"]
    )


def test_task_inventory_keeps_paths_not_lazy_npz_handles(tmp_path):
    module = _load_module()
    task = tmp_path / "task_0000"
    task.mkdir()
    (task / "summary.json").write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "molecule_id": "1",
                        "direction_index": 0,
                        "base_initialization": "configured",
                        "displaced_initialization": "base_continuation",
                    }
                ]
            }
        )
    )
    (task / "points.csv").write_text("point\n")
    np.savez_compressed(task / "audit_hvp_arrays.npz", placeholder=np.ones(1))
    grouped, errors = module._load_tasks(tmp_path)
    record = grouped[("0000001", 0)]["sad_continuation"]
    assert errors == []
    assert record["arrays_path"].name == "audit_hvp_arrays.npz"
    assert "arrays" not in record


def test_filtered_sidecars_have_authoritative_hash_manifest(tmp_path):
    module = _load_module()
    source = tmp_path / "source"
    output = tmp_path / "stable"
    source.mkdir()
    np.savez_compressed(source / "0000001.0000000.npz", direction=np.ones((4, 2, 3)))
    manifest_path = module._write_filtered_sidecars(
        source,
        output,
        [
            {"molecule_id": "0000001", "direction_index": index, "stable": stable}
            for index, stable in enumerate((True, False, True, True))
        ],
        [{"molecule_id": "0000001", "parent_stable": True}],
        4,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sidecar_count"] == 1
    assert manifest["entries"][0]["stability_mask"] == [1, 0, 1, 1]
    assert manifest["entries"][0]["sha256"] == module._sha256(
        output / "0000001.0000000.npz"
    )
