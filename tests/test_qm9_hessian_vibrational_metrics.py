from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qm9_hessian_vibrational_metrics.py"
    spec = spec_from_file_location("qm9_hessian_vibrational_metrics", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_model_hessian_accepts_total_ofdft_key(tmp_path):
    module = _load_script()
    expected = np.arange(9, dtype=np.float64).reshape(3, 3)
    path = tmp_path / "total_hessian.npz"
    np.savez_compressed(path, total_ofdft_hessian=expected)

    np.testing.assert_array_equal(module._load_model_hessian(path), expected)


def test_load_model_hessian_accepts_capacity_prediction_key(tmp_path):
    module = _load_script()
    expected = np.arange(9, dtype=np.float64).reshape(3, 3)
    path = tmp_path / "capacity_hessian.npz"
    np.savez_compressed(path, predicted_hessian=expected)

    np.testing.assert_array_equal(module._load_model_hessian(path), expected)


def test_result_rows_accepts_total_ofdft_metric_rows():
    module = _load_script()
    rows = [{"run": "EG", "success": True}]

    assert module._result_rows({"metric_rows": rows}) is rows


def test_capacity_result_rows_adapts_frozen_capacity_directory(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "capacity"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        '{"test100_accessed": false, "per_parent": [{"molecule_id": "0000001"}]}'
    )
    np.savez_compressed(run_dir / "0000001_result.npz", predicted_hessian=np.eye(3))

    rows = module._capacity_result_rows("capacity", run_dir)

    assert rows == [
        {
            "run": "capacity",
            "molecule_id": "0000001",
            "sample_id": 0,
            "success": True,
            "hessian_npz": (run_dir / "0000001_result.npz").resolve().as_posix(),
        }
    ]


def test_capacity_result_rows_accepts_external_metrics_csv(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "external"
    run_dir.mkdir()
    metrics = run_dir / "per_parent_metrics.csv"
    metrics.write_text("molecule_id,natoms\n0000002,3\n")
    (run_dir / "summary.json").write_text(
        '{"test100_accessed": false, "per_parent_metrics": "'
        + metrics.as_posix()
        + '"}'
    )
    np.savez_compressed(run_dir / "0000002_result.npz", predicted_hessian=np.eye(3))

    rows = module._capacity_result_rows("external", run_dir)

    assert rows[0]["molecule_id"] == "0000002"
    assert rows[0]["hessian_npz"] == (
        run_dir / "0000002_result.npz"
    ).resolve().as_posix()


def test_baseline_result_rows_adapts_frozen_baseline_manifest(tmp_path):
    module = _load_script()
    hessian = tmp_path / "baseline.npz"
    np.savez_compressed(hessian, predicted_hessian=np.eye(3))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"test100_accessed": false, "parents": [{"molecule_id": "0000003", '
        '"capacity_array": "' + hessian.as_posix() + '"}]}'
    )

    rows = module._baseline_result_rows("original_A", manifest)

    assert rows == [
        {
            "run": "original_A",
            "molecule_id": "0000003",
            "sample_id": 0,
            "success": True,
            "hessian_npz": hessian.resolve().as_posix(),
        }
    ]


def test_capacity_result_rows_rejects_nonzero_test100_count(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "capacity"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        '{"test100_accessed": false, "test100_evaluations_used": 1, '
        '"per_parent": [{"molecule_id": "0000001"}]}'
    )

    with np.testing.assert_raises_regex(ValueError, "Test100 evaluations"):
        module._capacity_result_rows("capacity", run_dir)


def test_reference_manifest_rows_accepts_capacity_baseline_manifest(tmp_path):
    module = _load_script()
    path = tmp_path / "baseline.json"
    path.write_text(
        '{"parents": [{"molecule_id": "0000001", "natoms": 2, '
        '"capacity_array": "/tmp/reference.npz"}]}'
    )

    assert module._reference_manifest_rows(path) == [
        {
            "molecule_id": "0000001",
            "natoms": 2,
            "capacity_array": "/tmp/reference.npz",
            "sample_id": 0,
            "cache_path": "/tmp/reference.npz",
            "success": True,
        }
    ]
