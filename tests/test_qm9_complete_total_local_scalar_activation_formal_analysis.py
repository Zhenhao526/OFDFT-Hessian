from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.qm9_complete_total_local_scalar_activation_formal_analysis import run


def _write_formal_run(
    root: Path, arm_id: str, protocol_sha256: str, passed: bool
) -> None:
    run_dir = root / arm_id
    run_dir.mkdir(parents=True)
    final = {
        "hessian_relative_frobenius": {"median": 0.04, "p90": 0.045, "max": 0.05},
        "energy_median_ratio_to_source": 1.0,
        "force_median_ratio_to_source": 1.0,
        "max_antisymmetric_over_symmetric_frobenius": 1e-12,
        "gate": {"passed": passed},
    }
    summary = {
        "protocol_sha256": protocol_sha256,
        "arm_id": arm_id,
        "run_mode": "formal",
        "selected_parent_ids": ["a", "b", "c", "d", "e"],
        "unselected_parent_artifacts_opened": 0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "parent_cv_authorized": passed,
        "architecture": {"trainable_parameter_count": 10},
        "best_step": 10,
        "final": final,
        "per_parent": [
            {"molecule_id": value, "relative_frobenius": 0.04}
            for value in ["a", "b", "c", "d", "e"]
        ],
        "wall_time_s": 1.0,
        "max_rss_mb": 2.0,
        "gpu_peak_memory_mb": 3.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "training_metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 10,
                "median_relative_frobenius": 0.04,
                "max_relative_frobenius": 0.05,
                "energy_median_ratio_to_source": 1.0,
                "force_median_ratio_to_source": 1.0,
            }
        )
        + "\n"
    )
    (run_dir / "best.ckpt").write_bytes(b"checkpoint")
    for index, molecule_id in enumerate(["a", "b", "c", "d", "e"]):
        source = np.eye(3) * (1.0 + 0.1 * index)
        reference = source + np.eye(3) * 0.2
        predicted = source + np.eye(3) * 0.1
        np.savez_compressed(
            run_dir / f"{molecule_id}_result.npz",
            predicted_hessian=predicted,
            source_hessian_symmetric=source,
            pbe_hessian=reference,
        )


def test_analysis_reads_only_smoke_selected_formal_arms(tmp_path: Path) -> None:
    selected_arm = "J5"
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [
            {"id": "J3", "activation": "softplus", "softplus_beta": 5.0},
            {"id": selected_arm, "activation": "tanh"},
        ],
        "smoke_selection": {"maximum_formal_non_control_arms": 1},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_sha256,
                "selected_formal_arms": [selected_arm],
            }
        )
    )
    run_root = tmp_path / "formal"
    _write_formal_run(run_root, selected_arm, protocol_sha256, True)
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "smoke_summary": smoke_path,
            "run_root": run_root,
            "output_dir": tmp_path / "analysis",
        },
    )()

    result = run(args)

    assert result["selected_formal_arms"] == [selected_arm]
    assert result["passing_arms"] == [selected_arm]
    assert result["selected_capacity_arm"] == selected_arm
    assert result["parent_cv_design_authorized"] is True
    assert result["validation_accessed"] is False
    assert result["test100_evaluations_used"] == 0


def test_analysis_accepts_generic_maximum_formal_arms_field(tmp_path: Path) -> None:
    selected_arm = "K5"
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [{"id": selected_arm, "activation": "tanh"}],
        "smoke_selection": {"maximum_formal_arms": 1},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_sha256,
                "selected_formal_arms": [selected_arm],
            }
        )
    )
    run_root = tmp_path / "formal"
    _write_formal_run(run_root, selected_arm, protocol_sha256, False)
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "smoke_summary": smoke_path,
            "run_root": run_root,
            "output_dir": tmp_path / "analysis",
        },
    )()

    result = run(args)

    assert result["selected_formal_arms"] == [selected_arm]
    assert result["passing_arms"] == []
    assert result["parent_cv_design_authorized"] is False
