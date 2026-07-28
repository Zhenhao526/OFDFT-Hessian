from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts.qm9_complete_total_local_scalar_lbfgs_ceiling import (
    ceiling_objective,
    ceiling_selection_score,
    validate_audit_manifest,
)


def _terms() -> dict[str, torch.Tensor]:
    return {
        "energy_loss": torch.tensor(2.0),
        "force_loss": torch.tensor(3.0),
        "hessian_loss": torch.tensor(5.0),
        "parameter_loss": torch.tensor(7.0),
    }


def _training() -> dict[str, float]:
    return {
        "lambda_energy": 1.0,
        "lambda_force": 2.0,
        "lambda_hessian": 3.0,
        "lambda_parameter": 0.5,
    }


def test_lbfgs_joint_and_hessian_only_objectives_are_explicit() -> None:
    assert float(ceiling_objective(_terms(), _training(), "joint")) == 26.5
    assert float(ceiling_objective(_terms(), _training(), "hessian_only")) == 18.5


def test_hessian_only_never_uses_energy_force_selection_penalty() -> None:
    final = {
        "selection_score": 99.0,
        "hessian_relative_frobenius": {"median": 0.02, "max": 0.04},
    }

    assert ceiling_selection_score(final, "joint") == 99.0
    assert ceiling_selection_score(final, "hessian_only") == pytest.approx(0.06)


def test_unknown_lbfgs_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        ceiling_objective(_terms(), _training(), "other")


def test_lbfgs_audit_manifest_binds_source_and_settings(tmp_path: Path) -> None:
    source_protocol = tmp_path / "source.yaml"
    source_protocol.write_text("source\n")
    source_checkpoint = tmp_path / "source.ckpt"
    source_checkpoint.write_bytes(b"checkpoint")
    from scripts.qm9_complete_total_local_scalar_lbfgs_ceiling import _sha256

    manifest = {
        "protocol_id": "qm9_complete_total_local_scalar_post_v4_diagnostics_v1",
        "source_protocol": source_protocol.resolve().as_posix(),
        "source_protocol_sha256": _sha256(source_protocol),
        "source_arm_id": "arm",
        "source_checkpoint": source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "lbfgs": {
            "joint": {
                "iterations": 100,
                "learning_rate": 0.5,
                "history_size": 25,
                "max_evaluations_per_iteration": 5,
                "tolerance_grad": 1e-9,
                "tolerance_change": 1e-12,
                "log_interval": 5,
                "parent_cv_authorization_allowed": True,
            }
        },
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = argparse.Namespace(
        audit_manifest=manifest_path,
        source_protocol=source_protocol,
        source_arm_id="arm",
        source_checkpoint=source_checkpoint,
        objective="joint",
        iterations=100,
        learning_rate=0.5,
        history_size=25,
        max_evaluations_per_iteration=5,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        log_interval=5,
        parent_id=None,
    )

    loaded, digest = validate_audit_manifest(args)

    assert loaded == manifest
    assert digest == _sha256(manifest_path)


@pytest.mark.parametrize(
    "protocol_id",
    [
        "qm9_complete_total_bounded_equivariant_lbfgs_v1",
        "qm9_complete_total_mace_lbfgs_v1",
    ],
)
def test_stable5_scalar_manifest_requires_frozen_scope_and_validation(
    tmp_path: Path,
    protocol_id: str,
) -> None:
    source_protocol = tmp_path / "source.yaml"
    source_protocol.write_text("source\n")
    source_checkpoint = tmp_path / "source.ckpt"
    source_checkpoint.write_bytes(b"checkpoint")
    from scripts.qm9_complete_total_local_scalar_lbfgs_ceiling import _sha256

    manifest = {
        "protocol_id": protocol_id,
        "parent_scope": "stable5_fit_only",
        "validation_accessed": False,
        "source_protocol": source_protocol.resolve().as_posix(),
        "source_protocol_sha256": _sha256(source_protocol),
        "source_arm_id": "bounded",
        "source_checkpoint": source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "lbfgs": {
            "joint": {
                "iterations": 30,
                "learning_rate": 0.5,
                "history_size": 20,
                "max_evaluations_per_iteration": 3,
                "tolerance_grad": 1e-9,
                "tolerance_change": 1e-12,
                "log_interval": 5,
                "parent_cv_authorization_allowed": True,
            }
        },
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = argparse.Namespace(
        audit_manifest=manifest_path,
        source_protocol=source_protocol,
        source_arm_id="bounded",
        source_checkpoint=source_checkpoint,
        objective="joint",
        iterations=30,
        learning_rate=0.5,
        history_size=20,
        max_evaluations_per_iteration=3,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        log_interval=5,
        parent_id=None,
    )

    loaded, _ = validate_audit_manifest(args)
    assert loaded == manifest

    manifest["validation_accessed"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="freeze validation"):
        validate_audit_manifest(args)

    manifest["validation_accessed"] = False
    manifest["parent_scope"] = "train20"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stable5-only"):
        validate_audit_manifest(args)


def test_one_parent_diagnostic_requires_frozen_parent_and_cannot_use_full_scope(
    tmp_path: Path,
) -> None:
    source_protocol = tmp_path / "source.yaml"
    source_protocol.write_text("source\n")
    source_checkpoint = tmp_path / "source.ckpt"
    source_checkpoint.write_bytes(b"checkpoint")
    from scripts.qm9_complete_total_local_scalar_lbfgs_ceiling import _sha256

    manifest = {
        "protocol_id": "qm9_complete_total_bounded_equivariant_lbfgs_v1",
        "parent_scope": "stable5_one_parent_diagnostic",
        "diagnostic_parent_ids": ["mol-a"],
        "validation_accessed": False,
        "source_protocol": source_protocol.resolve().as_posix(),
        "source_protocol_sha256": _sha256(source_protocol),
        "source_arm_id": "bounded",
        "source_checkpoint": source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "lbfgs": {
            "joint": {
                "iterations": 20,
                "learning_rate": 0.5,
                "history_size": 20,
                "max_evaluations_per_iteration": 3,
                "tolerance_grad": 1e-9,
                "tolerance_change": 1e-12,
                "log_interval": 5,
                "parent_cv_authorization_allowed": False,
            }
        },
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = argparse.Namespace(
        audit_manifest=manifest_path,
        source_protocol=source_protocol,
        source_arm_id="bounded",
        source_checkpoint=source_checkpoint,
        objective="joint",
        iterations=20,
        learning_rate=0.5,
        history_size=20,
        max_evaluations_per_iteration=3,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        log_interval=5,
        parent_id="mol-a",
    )

    validate_audit_manifest(args)

    args.parent_id = "mol-b"
    with pytest.raises(ValueError, match="parent is not frozen"):
        validate_audit_manifest(args)

    args.parent_id = "mol-a"
    manifest["parent_scope"] = "stable5_fit_only"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot select one parent"):
        validate_audit_manifest(args)
