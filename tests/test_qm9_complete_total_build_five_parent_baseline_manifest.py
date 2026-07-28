import csv
import hashlib
import json
import numpy as np

from scripts.qm9_complete_total_build_five_parent_baseline_manifest import (
    build_manifest,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_manifest_requires_and_records_all_frozen_parents(tmp_path):
    label = tmp_path / "label.zarr.zip"
    hessian = tmp_path / "pbe.npz"
    label.write_bytes(b"label")
    hessian.write_bytes(b"pbe")
    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "split_sha256": "split",
                "protocol_sha256": "protocol",
                "parents": [
                    {
                        "molecule_id": "0000001",
                        "label_path": str(label),
                        "label_sha256": _sha(label),
                        "pbe_hessian_path": str(hessian),
                        "pbe_hessian_sha256": _sha(hessian),
                    }
                ],
            }
        )
    )
    run_dir = tmp_path / "run"
    (run_dir / "hessian_arrays").mkdir(parents=True)
    array = run_dir / "hessian_arrays" / "step_0000000_0000001.npz"
    np.savez(
        array,
        predicted_hessian=np.eye(2),
        pbe_hessian=np.eye(2),
    )
    fields = {
        "molecule_id": "0000001",
        "natoms": "3",
        "step": "0",
        "full_hessian_complete": "True",
        "total_energy_hartree": "-1.2",
        "total_energy_abs_error_hartree": "0.1",
        "complete_total_force_mae_hartree_per_bohr": "0.2",
        "relative_frobenius": "0.3",
        "max_cached_density_gradient_norm": "1e-10",
    }
    with (run_dir / "full_hessian_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerow(fields)
    output = tmp_path / "baseline.json"
    result = build_manifest(stage1, [run_dir], output, 0.005)
    assert result["test100_accessed"] is False
    assert result["parents"][0]["capacity_array_sha256"] == _sha(array)
    assert json.loads(output.read_text())["source_split_sha256"] == "split"
    assert result["parents"][0]["baseline_asym_over_pbe_frobenius"] == 0.0
