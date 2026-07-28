import json
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qm9_complete_total_stage2_select_baselines.py"
    spec = spec_from_file_location("select_stage2_baselines", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_uses_frozen_order_and_excludes_curl_failure(tmp_path):
    module = _load_script()
    candidate_path = tmp_path / "candidates.json"
    candidates = []
    directions = []
    for order, molecule_id in enumerate(("a", "b", "c")):
        candidates.append(
            {
                "molecule_id": molecule_id,
                "candidate_order": order,
                "natoms": 2,
            }
        )
        directions.append(
            {
                "molecule_id": molecule_id,
                "direction_path": f"/{molecule_id}.npz",
                "direction_sha256": molecule_id,
            }
        )
    candidate_path.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "source_split_sha256": "split",
                "protocol_id": "stage2",
                "final_parent_count": 2,
                "numerical_gate": {
                    "baseline_asym_over_pbe_frobenius_max": 0.005,
                    "strict_density_gradient_max": 1.0e-8,
                },
                "candidates": candidates,
                "directions": directions,
            }
        )
    )
    baseline_path = tmp_path / "baseline.json"
    rows = []
    for molecule_id, curl in (("a", 0.02), ("b", 1.0e-5), ("c", 2.0e-5)):
        rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": 2,
                "baseline_asym_over_pbe_frobenius": curl,
                "baseline_max_density_gradient_norm": 1.0e-10,
                "baseline_hessian_relative_frobenius": 1.0,
                "capacity_array": f"/{molecule_id}.npz",
                "full_hessian_complete": True,
            }
        )
    baseline_path.write_text(json.dumps({"test100_accessed": False, "parents": rows}))

    result = module.select(
        Namespace(
            candidate_manifest=candidate_path,
            known_baseline_manifest=[baseline_path],
            run_root=[],
            run_dir=[],
            output_dir=tmp_path / "out",
        )
    )

    assert [row["molecule_id"] for row in result["parents"]] == ["b", "c"]
    assert result["audit_rows"][0]["failure_reasons"] == "pbe_normalized_curl"
    assert result["test100_accessed"] is False


def test_legacy_baseline_manifest_derives_missing_pbe_normalized_curl(tmp_path):
    module = _load_script()
    array = tmp_path / "baseline.npz"
    predicted = np.asarray([[1.0, 0.2], [0.0, 1.0]])
    reference = np.eye(2)
    np.savez_compressed(array, predicted_hessian=predicted, pbe_hessian=reference)

    normalized = module._normalize_manifest_parent(
        {
            "capacity_array": array.as_posix(),
            "baseline_max_density_gradient_norm": 1.0e-10,
        }
    )

    expected = np.linalg.norm(0.5 * (predicted - predicted.T)) / np.linalg.norm(
        reference
    )
    assert normalized["baseline_asym_over_pbe_frobenius"] == expected


def test_provenance_audit_requires_frozen_test_and_indexes_resolved_runs(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    audit = tmp_path / "provenance.json"
    audit.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "records": [
                    {
                        "baseline_run_dir": run_dir.as_posix(),
                        "status": "pass",
                        "base_state_exact_match": True,
                    }
                ],
            }
        )
    )
    indexed = module._load_provenance_audit(audit)
    assert indexed[run_dir.resolve().as_posix()]["base_state_exact_match"] is True

    audit.write_text(json.dumps({"test100_accessed": True, "records": []}))
    try:
        module._load_provenance_audit(audit)
    except ValueError as error:
        assert "frozen Test100" in str(error)
    else:
        raise AssertionError("provenance audit accepted Test100 access")
