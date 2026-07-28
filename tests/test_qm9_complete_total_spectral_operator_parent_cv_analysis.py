from __future__ import annotations

import argparse
import hashlib
import json

import pytest

from scripts.qm9_complete_total_spectral_operator_parent_cv_analysis import analyze


VARIANTS = ("A", "B")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(
        "preregistered_variants:\n"
        "  - id: A\n"
        "  - id: B\n"
    )
    return path


def _run(tmp_path, protocol, name: str, scale: float, *, test100=False):
    run_dir = tmp_path / name
    vibration_dir = run_dir / "vibrational_metrics"
    vibration_dir.mkdir(parents=True)
    rows = [
        {
            "molecule_id": molecule_id,
            "source_relative_frobenius": 2.5 + index,
            "relative_frobenius": scale * (index + 1),
        }
        for index, molecule_id in enumerate(("0000001", "0000002"))
    ]
    summary = {
        "variant_id": name,
        "test100_accessed": test100,
        "test100_evaluations_used": int(test100),
        "protocol_sha256": _sha256(protocol),
        "fold_assignment": {"0": ["0000001"], "1": ["0000002"]},
        "per_parent": rows,
        "selected_ridge": 1.0e-4,
        "parent_feature_transform": "tanh",
        "parent_feature_transform_scale": 2.0,
        "block_parent_conditioning": True,
        "max_correction_to_source": None,
        "selected_cross_parent_metrics": {
            "median_relative_frobenius": 1.5 * scale,
            "p90_relative_frobenius": 1.9 * scale,
            "max_relative_frobenius": 2.0 * scale,
            "fraction_relative_frobenius_at_or_below_0_15": 1.0,
            "max_antisymmetric_over_symmetric_frobenius": 1.0e-4,
        },
        "source_cross_parent_metrics": {
            "median_relative_frobenius": 3.0,
            "p90_relative_frobenius": 3.4,
        },
        "parent_win_fraction_vs_source": 1.0,
        "cross_parent_gate": {"passed": True},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    vibration = {
        "summaries": [
            {
                "mean_frequency_mae_cm-1": 1000.0 * scale,
                "mean_frequency_rmse_cm-1": 1200.0 * scale,
                "mean_mode_overlap": 0.8,
                "total_model_imaginary_modes": 4,
                "total_pbe_imaginary_modes": 2,
            }
        ]
    }
    (vibration_dir / "summary.json").write_text(json.dumps(vibration))
    return run_dir


def _source_vibration(tmp_path):
    path = tmp_path / "source_vibration.json"
    path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "mean_frequency_mae_cm-1": 2900.0,
                        "mean_frequency_rmse_cm-1": 3800.0,
                        "mean_mode_overlap": 0.6,
                        "total_model_imaginary_modes": 80,
                        "total_pbe_imaginary_modes": 2,
                    }
                ]
            }
        )
    )
    return path


def test_analysis_writes_preregistered_comparison(tmp_path) -> None:
    protocol = _protocol(tmp_path)
    run_a = _run(tmp_path, protocol, "A", 0.04)
    run_b = _run(tmp_path, protocol, "B", 0.05)
    output = tmp_path / "analysis"

    result = analyze(
        argparse.Namespace(
            protocol=protocol,
            run=[f"A={run_a}", f"B={run_b}"],
            source_vibrational_summary=_source_vibration(tmp_path),
            output_dir=output,
        )
    )

    assert result["passing_variants"] == ["A", "B"]
    assert result["diagnostic_best_nonpromoted_variant"] == "A"
    assert result["advancement_authorized"] is True
    assert result["test100_evaluations_used"] == 0
    assert (output / "variant_summary.csv").is_file()
    assert (output / "per_parent_all_variants.csv").is_file()
    assert (output / "parent_cv_variant_comparison.png").is_file()


def test_analysis_rejects_test100_access(tmp_path) -> None:
    protocol = _protocol(tmp_path)
    run_a = _run(tmp_path, protocol, "A", 0.04, test100=True)
    run_b = _run(tmp_path, protocol, "B", 0.05)

    with pytest.raises(ValueError, match="frozen Test100"):
        analyze(
            argparse.Namespace(
                protocol=protocol,
                run=[f"A={run_a}", f"B={run_b}"],
                source_vibrational_summary=_source_vibration(tmp_path),
                output_dir=tmp_path / "analysis",
            )
        )
