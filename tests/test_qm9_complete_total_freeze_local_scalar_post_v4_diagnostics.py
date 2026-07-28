from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from scripts.qm9_complete_total_freeze_local_scalar_post_v4_diagnostics import (
    _sha256,
    build_manifest,
)


def _fixture(tmp_path: Path, *, authorized: bool = False) -> tuple[Path, Path, Path]:
    protocol = {
        "frozen_at": "2026-07-22",
        "test100_access_allowed": False,
        "test100_evaluations_used": 0,
        "inputs": {"parents": ["a", "b"]},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    run_root = tmp_path / "runs"
    run_dir = run_root / "arm"
    run_dir.mkdir(parents=True)
    summary = {
        "protocol_sha256": _sha256(protocol_path),
        "arm_id": "arm",
        "run_mode": "formal",
        "selected_parent_ids": ["a", "b"],
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    torch.save(
        {
            "protocol_sha256": _sha256(protocol_path),
            "arm_id": "arm",
            "selected_parent_ids": ["a", "b"],
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "step": 17,
            "state_dict": {},
        },
        run_dir / "best.ckpt",
    )
    analysis = {
        "protocol_sha256": _sha256(protocol_path),
        "selected_formal_arms": ["arm"],
        "expected_parent_ids": ["a", "b"],
        "arms": [
            {
                "arm_id": "arm",
                "gate_passed": authorized,
                "median_relative_frobenius": 0.6,
                "max_relative_frobenius": 0.7,
                "energy_median_ratio_to_source": 0.8,
                "force_median_ratio_to_source": 0.8,
            }
        ],
        "parent_cv_design_authorized": authorized,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    return protocol_path, analysis_path, run_root


def _add_failed_arm(
    protocol_path: Path, analysis_path: Path, run_root: Path, arm_id: str
) -> None:
    analysis = json.loads(analysis_path.read_text())
    analysis["selected_formal_arms"].append(arm_id)
    analysis["arms"].append(
        {
            "arm_id": arm_id,
            "gate_passed": False,
            "median_relative_frobenius": 0.8,
            "max_relative_frobenius": 0.9,
            "energy_median_ratio_to_source": 0.8,
            "force_median_ratio_to_source": 0.8,
        }
    )
    analysis_path.write_text(json.dumps(analysis))
    run_dir = run_root / arm_id
    run_dir.mkdir()
    summary = {
        "protocol_sha256": _sha256(protocol_path),
        "arm_id": arm_id,
        "run_mode": "formal",
        "selected_parent_ids": ["a", "b"],
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    torch.save(
        {
            "protocol_sha256": _sha256(protocol_path),
            "arm_id": arm_id,
            "selected_parent_ids": ["a", "b"],
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "step": 21,
            "state_dict": {},
        },
        run_dir / "best.ckpt",
    )


def test_freeze_manifest_binds_failed_formal_checkpoint(tmp_path: Path) -> None:
    protocol, analysis, run_root = _fixture(tmp_path)

    result = build_manifest(
        protocol_path=protocol,
        formal_analysis_path=analysis,
        run_root=run_root,
    )

    assert result["source_arm_id"] == "arm"
    assert result["source_checkpoint_step"] == 17
    assert result["jacobian"]["iterations"] == 30
    assert result["lbfgs"]["joint"]["parent_cv_authorization_allowed"] is True
    assert result["lbfgs"]["hessian_only"]["parent_cv_authorization_allowed"] is False
    assert result["vibrational_gate"]["mean_frequency_mae_cm-1_max"] == 200.0
    assert result["vibrational_gate"]["mean_mode_overlap_min"] == 0.8
    assert result["test100_accessed"] is False


def test_freeze_manifest_rejects_passing_formal(tmp_path: Path) -> None:
    protocol, analysis, run_root = _fixture(tmp_path, authorized=True)

    try:
        build_manifest(
            protocol_path=protocol,
            formal_analysis_path=analysis,
            run_root=run_root,
        )
    except ValueError as error:
        assert "not authorized" in str(error)
    else:
        raise AssertionError("passing formal must reject post-failure diagnostics")


def test_freeze_manifest_selects_best_of_multiple_failed_formals(tmp_path: Path) -> None:
    protocol, analysis, run_root = _fixture(tmp_path)
    current = json.loads(analysis.read_text())
    current["arms"][0].update(
        {
            "median_relative_frobenius": 0.6,
            "max_relative_frobenius": 0.7,
            "energy_median_ratio_to_source": 0.8,
            "force_median_ratio_to_source": 0.8,
        }
    )
    analysis.write_text(json.dumps(current))
    _add_failed_arm(protocol, analysis, run_root, "worse")

    result = build_manifest(
        protocol_path=protocol,
        formal_analysis_path=analysis,
        run_root=run_root,
    )

    assert result["source_arm_id"] == "arm"
    assert result["source_selection_rule"].startswith("minimum_preregistered")
