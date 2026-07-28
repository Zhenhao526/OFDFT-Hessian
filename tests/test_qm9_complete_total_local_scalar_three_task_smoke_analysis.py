from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.qm9_complete_total_local_scalar_three_task_smoke_analysis import run


def _write_smoke(
    root: Path,
    arm_id: str,
    protocol_sha256: str,
    final_score: float,
    positive_norms: bool = True,
    positive_finite: bool = True,
) -> None:
    run_dir = root / arm_id
    run_dir.mkdir(parents=True)
    summary = {
        "protocol_sha256": protocol_sha256,
        "arm_id": arm_id,
        "run_mode": "smoke",
        "selected_parent_ids": ["a", "b", "c", "d", "e"],
        "unselected_parent_artifacts_opened": 0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    norm = 1.0 if positive_norms else 0.0
    metrics = [
        {
            "step": 0,
            "selection_score": 8.0,
            "median_relative_frobenius": 3.0,
            "max_relative_frobenius": 4.0,
            "sampled_energy_loss": math.nan,
            "gradient_norm": math.nan,
        },
        {
            "step": 20,
            "selection_score": final_score,
            "median_relative_frobenius": 2.0,
            "max_relative_frobenius": 3.0,
            "pcgrad/energy_gradient_norm": norm,
            "pcgrad/force_gradient_norm": norm,
            "pcgrad/hessian_gradient_norm": norm,
            "sampled_energy_loss": 1.0 if positive_finite else math.nan,
        },
    ]
    (run_dir / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )


def test_three_task_smoke_selects_only_best_eligible_arm(tmp_path: Path) -> None:
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [
            {"id": "K3", "paired_activation_smoke_final_selection_score": 6.0},
            {"id": "K5", "paired_activation_smoke_final_selection_score": 6.0},
        ],
        "smoke_selection": {"maximum_formal_arms": 1},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    run_root = tmp_path / "runs"
    _write_smoke(run_root, "K3", protocol_sha256, final_score=5.0)
    _write_smoke(run_root, "K5", protocol_sha256, final_score=4.0)
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "run_root": run_root,
            "output_dir": tmp_path / "analysis",
        },
    )()

    result = run(args)

    assert result["eligible_arms"] == ["K5", "K3"]
    assert result["selected_formal_arms"] == ["K5"]
    assert result["formal_run_authorized"] is True
    assert result["parent_cv_design_authorized"] is False
    assert result["test100_evaluations_used"] == 0


def test_three_task_smoke_rejects_zero_named_task_gradient(tmp_path: Path) -> None:
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [
            {"id": "K3", "paired_activation_smoke_final_selection_score": 6.0}
        ],
        "smoke_selection": {"maximum_formal_arms": 1},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    run_root = tmp_path / "runs"
    _write_smoke(
        run_root, "K3", protocol_sha256, final_score=5.0, positive_norms=False
    )
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "run_root": run_root,
            "output_dir": tmp_path / "analysis",
        },
    )()

    result = run(args)

    assert result["eligible_arms"] == []
    assert result["selected_formal_arms"] == []
    assert result["formal_run_authorized"] is False


def test_three_task_smoke_ignores_expected_step_zero_nan_but_rejects_positive_nan(
    tmp_path: Path,
) -> None:
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [
            {"id": "finite", "paired_activation_smoke_final_selection_score": 6.0},
            {"id": "nan", "paired_activation_smoke_final_selection_score": 6.0},
        ],
        "smoke_selection": {"maximum_formal_arms": 1},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    run_root = tmp_path / "runs"
    _write_smoke(run_root, "finite", protocol_sha256, final_score=5.0)
    _write_smoke(
        run_root,
        "nan",
        protocol_sha256,
        final_score=4.0,
        positive_finite=False,
    )
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "run_root": run_root,
            "output_dir": tmp_path / "analysis",
        },
    )()

    result = run(args)

    by_arm = {row["arm_id"]: row for row in result["arms"]}
    assert by_arm["finite"]["all_diagnostics_finite"] is True
    assert by_arm["nan"]["all_diagnostics_finite"] is False
    assert result["selected_formal_arms"] == ["finite"]
