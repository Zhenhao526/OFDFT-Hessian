from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.qm9_complete_total_local_scalar_post_v4_diagnostics_analysis import run


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_post_v4_merge_requires_joint_pass_for_parent_cv(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    checkpoint = tmp_path / "source.ckpt"
    protocol.write_text("protocol")
    checkpoint.write_text("checkpoint")
    import hashlib

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    common = {
        "source_protocol": protocol.resolve().as_posix(),
        "source_protocol_sha256": sha(protocol),
        "source_checkpoint": checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": sha(checkpoint),
    }
    manifest = {
        "protocol_id": "qm9_complete_total_local_scalar_post_v4_diagnostics_v1",
        **common,
        "source_arm_id": "arm",
        "parents": ["a"],
        "jacobian": {"iterations": 30, "fd_relative_step": 3e-6},
        "lbfgs": {"joint": {"iterations": 100}, "hessian_only": {"iterations": 100}},
        "vibrational_gate": {
            "mean_frequency_mae_cm-1_max": 200.0,
            "absolute_total_imaginary_mode_count_error_max": 5,
            "mean_mode_overlap_min": 0.8,
            "required_molecule_count": 1,
        },
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    jacobian = {
        **common,
        "arm_id": "arm",
        "selected_parent_ids": ["a"],
        "iterations_requested": 30,
        "iterations_completed": 30,
        "fd_relative_step": 3e-6,
        "linearized_hessian_relative_frobenius": {"median": 0.03, "p90": 0.04, "max": 0.04},
        "fd_directional_stability": {"cosine": 1.0, "relative_difference": 1e-8},
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    jacobian_path = tmp_path / "jacobian.json"
    _write_json(jacobian_path, jacobian)
    jacobian_metrics = tmp_path / "jacobian.jsonl"
    jacobian_metrics.write_text(json.dumps({"iteration": 1, "relative_residual_norm": 0.5}) + "\n")
    final = {
        "hessian_relative_frobenius": {"median": 0.04, "p90": 0.05, "max": 0.05},
        "energy_median_ratio_to_source": 1.2,
        "force_median_ratio_to_source": 0.9,
    }
    summaries = {}
    metric_paths = {}
    for objective in ("joint", "hessian_only"):
        value = {
            **common,
            "source_arm_id": "arm",
            "selected_parent_ids": ["a"],
            "objective": objective,
            "audit_manifest_sha256": sha(manifest_path),
            "iterations": 100,
            "capacity_gate_passed": False,
            "parent_cv_authorized": False,
            "final": final,
            "test100_accessed": False,
            "test100_evaluations_used": 0,
        }
        path = tmp_path / f"{objective}.json"
        _write_json(path, value)
        summaries[objective] = path
        metrics = tmp_path / f"{objective}.jsonl"
        metrics.write_text(
            json.dumps(
                {
                    "iteration": 0,
                    "median_relative_frobenius": 0.04,
                    "max_relative_frobenius": 0.05,
                    "energy_median_ratio_to_source": 1.2,
                    "force_median_ratio_to_source": 0.9,
                }
            )
            + "\n"
        )
        metric_paths[objective] = metrics
    args = argparse.Namespace(
        manifest=manifest_path,
        jacobian_summary=jacobian_path,
        jacobian_metrics=jacobian_metrics,
        joint_summary=summaries["joint"],
        joint_metrics=metric_paths["joint"],
        hessian_summary=summaries["hessian_only"],
        hessian_metrics=metric_paths["hessian_only"],
        vibrational_summary=tmp_path / "vibrational.json",
        output_dir=tmp_path / "output",
    )
    _write_json(
        args.vibrational_summary,
        {
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "summaries": [
                {
                    "result": name,
                    "n_molecules": 1,
                    "mean_frequency_mae_cm-1": 100.0,
                    "mean_frequency_rmse_cm-1": 120.0,
                    "mean_mode_overlap": 0.9,
                    "total_pbe_imaginary_modes": 2,
                    "total_model_imaginary_modes": 3,
                }
                for name in ("original_A", "lbfgs_joint", "lbfgs_hessian_only")
            ],
        },
    )

    result = run(args)

    assert result["jacobian_five_percent_pass"] is True
    assert result["hessian_only_five_percent_pass"] is True
    assert result["joint_lbfgs_capacity_passed"] is False
    assert result["diagnosis"] == "hessian_representation_pass_but_joint_multitask_gate_failed"
    assert result["parent_cv_design_authorized"] is False
    assert result["test100_accessed"] is False
