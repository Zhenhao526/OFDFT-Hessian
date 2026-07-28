from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.qm9_complete_total_local_angular_smoke_analysis import run


def _fixture(tmp_path: Path, *, positive_nan: bool = False):
    protocol = {
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b", "c", "d", "e"]},
        "arms": [{"id": "M1"}],
        "smoke_gate": {
            "final_median_relative_frobenius_max": 2.5,
            "final_selection_score_max": 5.5,
            "peak_gpu_memory_mb_max": 75000,
        },
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "arm_id": "M1",
        "run_mode": "smoke",
        "selected_parent_ids": ["a", "b", "c", "d", "e"],
        "unselected_parent_artifacts_opened": 0,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    metrics = [
        {
            "step": 0,
            "median_relative_frobenius": 3.0,
            "sampled_energy_loss": math.nan,
        },
        {
            "step": 20,
            "median_relative_frobenius": 2.0,
            "max_relative_frobenius": 2.5,
            "selection_score": 4.5,
            "gpu_peak_memory_mb": 1000.0,
            "gradnorm/raw_hessian_gradient_norm": 1.0,
            "gradnorm/energy_force_gradient_norm": 2.0,
            "sampled_energy_loss": math.nan if positive_nan else 1.0,
        },
    ]
    (run_dir / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )
    args = type(
        "Args",
        (),
        {
            "protocol": protocol_path,
            "run_dir": run_dir,
            "output_dir": tmp_path / "analysis",
        },
    )()
    return args


def test_local_angular_smoke_authorizes_finite_improving_run(tmp_path: Path) -> None:
    result = run(_fixture(tmp_path))
    assert result["formal_run_authorized"] is True
    assert result["test100_evaluations_used"] == 0


def test_local_angular_smoke_rejects_positive_step_nan(tmp_path: Path) -> None:
    result = run(_fixture(tmp_path, positive_nan=True))
    assert result["checks"]["positive_steps_finite"] is False
    assert result["formal_run_authorized"] is False
