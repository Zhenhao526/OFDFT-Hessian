from __future__ import annotations

import argparse
import hashlib
import json
from types import SimpleNamespace

import numpy as np

import scripts.qm9_complete_total_geometry_parent_cv_analysis as analysis


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analysis_uses_every_held_parent_exactly_once(tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "preregistered_variants:\n"
        "  - id: G0\n"
        "    role: diagnostic\n"
        "    train800_energy_force_replay: false\n"
        "cross_validation:\n"
        "  fold_assignment:\n"
        "    - ['0000001']\n"
        "    - ['0000002']\n"
    )
    baseline_manifest = tmp_path / "baseline.json"
    baseline_manifest.write_text("{}")
    parents = [
        SimpleNamespace(
            molecule_id=molecule_id,
            hessian=np.eye(2) * 2.0,
            pbe_hessian=np.eye(2),
            energy=2.0,
            pbe_energy=1.0,
            force=np.ones(2),
            pbe_force=np.zeros(2),
        )
        for molecule_id in ("0000001", "0000002")
    ]
    monkeypatch.setattr(
        analysis,
        "_load_parents",
        lambda _: (parents, {"source_split_sha256": "split"}),
    )

    variant_dir = tmp_path / "G0"
    for fold, molecule_id in enumerate(("0000001", "0000002")):
        fold_dir = variant_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "best.ckpt").write_bytes(b"checkpoint")
        (fold_dir / f"{molecule_id}_result.npz").write_bytes(b"result")
        (fold_dir / "training_metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": 0,
                    "median_relative_frobenius": 2.0,
                }
            )
            + "\n"
        )
        summary = {
            "test100_accessed": False,
            "test100_evaluations_used": 0,
            "parent_cv": {
                "protocol_sha256": _sha256(protocol),
                "variant_id": "G0",
                "fold_index": fold,
                "held_parent_hessian_used_for_gradient": False,
                "held_parent_hessian_used_for_checkpoint_selection": False,
            },
            "per_parent": [
                {
                    "parent_cv_role": "held",
                    "molecule_id": molecule_id,
                    "natoms": 1,
                    "relative_frobenius": 0.1 + 0.01 * fold,
                    "antisymmetric_over_symmetric_frobenius": 0.0,
                    "energy_abs_error_hartree": 0.9,
                    "baseline_energy_abs_error_hartree": 1.0,
                    "force_mae_hartree_per_bohr": 0.9,
                    "baseline_force_mae_hartree_per_bohr": 1.0,
                }
            ],
        }
        (fold_dir / "summary.json").write_text(json.dumps(summary))

    output = tmp_path / "analysis"
    result = analysis.analyze(
        argparse.Namespace(
            protocol=protocol,
            baseline_manifest=baseline_manifest,
            variant=[f"G0={variant_dir}"],
            output_dir=output,
        )
    )

    assert result["test100_evaluations_used"] == 0
    assert result["variants"][0]["fraction_relative_frobenius_at_or_below_0_15"] == 1.0
    assert result["passing_replay_variants"] == []
    held = json.loads((output / "held_result_rows.json").read_text())["rows"]
    assert [row["molecule_id"] for row in held] == ["0000001", "0000002"]
    assert (output / "geometry_parent_cv.png").is_file()
