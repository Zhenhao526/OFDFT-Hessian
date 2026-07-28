from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.qm9_complete_total_local_scalar_activation_smoke_analysis import (
    _sha256,
    run,
)


def _write_arm(
    root: Path,
    protocol: Path,
    arm_id: str,
    *,
    score: float,
    raw_ratio: float,
) -> None:
    arm = root / arm_id
    arm.mkdir(parents=True)
    summary = {
        "protocol_sha256": _sha256(protocol),
        "arm_id": arm_id,
        "run_mode": "smoke",
        "best_step": 10,
        "unselected_parent_artifacts_opened": 0,
        "validation_accessed": False,
        "test100_accessed": False,
        "final": {
            "selection_score": score,
            "hessian_relative_frobenius": {"median": 1.0, "max": 1.2},
            "energy_median_ratio_to_source": 1.0,
            "force_median_ratio_to_source": 1.0,
        },
    }
    (arm / "summary.json").write_text(json.dumps(summary))
    metrics = [
        {
            "step": 0,
            "sampled_energy_loss": float("nan"),
        },
        {
            "step": 2,
            "gradnorm/raw_hessian_gradient_norm": raw_ratio,
            "gradnorm/energy_force_gradient_norm": 1.0,
            "median_relative_frobenius": 1.0,
        },
    ]
    (arm / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )


def test_smoke_analysis_applies_same_seed_control_rule(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "test100_access_allowed": False,
                "test100_evaluations_used": 0,
                "arms": [{"id": "control"}, {"id": "better"}, {"id": "worse"}],
                "smoke_selection": {
                    "control_arm": "control",
                    "maximum_formal_non_control_arms": 2,
                },
            }
        )
    )
    root = tmp_path / "runs"
    _write_arm(root, protocol, "control", score=5.0, raw_ratio=0.1)
    _write_arm(root, protocol, "better", score=4.0, raw_ratio=0.2)
    _write_arm(root, protocol, "worse", score=5.1, raw_ratio=0.3)
    args = type(
        "Args",
        (),
        {"protocol": protocol, "run_root": root, "output_dir": tmp_path / "analysis"},
    )()

    result = run(args)

    assert result["selected_formal_arms"] == ["better"]
    assert result["formal_submission_authorized"] is True
    assert result["test100_evaluations_used"] == 0
