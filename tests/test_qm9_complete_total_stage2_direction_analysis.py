import csv
import json

import numpy as np
import pytest

from scripts.qm9_complete_total_stage2_direction_analysis import analyze


class Args:
    pass


def test_stage2_analysis_reports_formal_distribution_gates(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    direction_dir = tmp_path / "directions"
    direction_dir.mkdir()
    direction_manifest_rows = []
    rows = []
    for index, relative in enumerate((0.05, 0.10, 0.15, 0.25)):
        molecule_id = f"{index:07d}"
        rows.append(
            {
                "molecule_id": molecule_id,
                "natoms": 5 + index,
                "energy_abs_error_hartree": 0.001 * (index + 1),
                "force_mae_hartree_per_bohr": 0.002 * (index + 1),
                "train_hvp_relative_frobenius": relative / 2,
                "heldout_hvp_relative_frobenius": relative,
                "relative_frobenius": relative,
                "antisymmetric_over_symmetric_frobenius": 1.0e-4 * (index + 1),
            }
        )
        direction_path = direction_dir / f"{molecule_id}.npz"
        np.savez_compressed(
            direction_path,
            directions=np.eye(2),
            roles=np.asarray(["train", "heldout"]),
            kinds=np.asarray(["bond", "angle"]),
            external_basis=np.empty((2, 0)),
        )
        direction_manifest_rows.append(
            {"molecule_id": molecule_id, "direction_path": str(direction_path)}
        )
        np.savez_compressed(
            run_dir / f"{molecule_id}_result.npz",
            predicted_hessian=np.diag([1.0 + relative, 2.0 - relative]),
            pbe_hessian=np.diag([1.0, 2.0]),
        )
    direction_manifest = tmp_path / "direction_manifest.json"
    direction_manifest.write_text(json.dumps({"directions": direction_manifest_rows}))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "direction_manifest": str(direction_manifest),
                "direction_manifest_sha256": "frozen",
                "best_step": 123,
                "wall_time_s": 4.5,
                "max_rss_mb": 256.0,
                "stage2_direction_generalization_gate_passed": False,
                "per_parent": rows,
            }
        )
    )
    with (run_dir / "training_metrics.jsonl").open("w") as handle:
        for step in (0, 10):
            loss_fields = (
                {
                    "energy_loss": 1.0 / (step + 1),
                    "force_loss": 2.0 / (step + 1),
                    "hessian_loss": 3.0 / (step + 1),
                    "spectrum_loss": 4.0 / (step + 1),
                }
                if step == 0
                else {
                    "sampled_energy_loss": 1.0 / (step + 1),
                    "sampled_force_loss": 2.0 / (step + 1),
                    "sampled_hvp_loss": 3.0 / (step + 1),
                    "sampled_spectrum_loss": 4.0 / (step + 1),
                }
            )
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        **loss_fields,
                        "median_train_hvp_relative_frobenius": 0.2,
                        "median_heldout_hvp_relative_frobenius": 0.3,
                        "median_relative_frobenius": 0.25,
                        "gradient_norm/energy": 1.0,
                        "gradient_norm/force": 2.0,
                        "gradient_norm/hvp": 3.0,
                        "gradient_norm/spectrum": 4.0,
                        "pcgrad/conflict": float(step > 0),
                        "pcgrad/cosine_before": -0.1,
                        "wall_time_s": float(step),
                    }
                )
                + "\n"
            )
    args = Args()
    args.run = [f"candidate={run_dir}"]
    args.output_dir = tmp_path / "analysis"

    result = analyze(args)

    summary = result["summaries"][0]
    assert summary["best_step"] == 123
    assert summary["median_full_hessian_relative_frobenius"] == pytest.approx(0.125)
    assert summary["p80_full_hessian_relative_frobenius"] == pytest.approx(0.19)
    assert summary["fraction_full_hessian_below_0p10"] == pytest.approx(0.5)
    assert summary["fraction_full_hessian_below_0p15"] == pytest.approx(0.75)
    assert summary["fraction_full_hessian_below_0p20"] == pytest.approx(0.75)
    assert summary["max_asymmetry_ratio"] == pytest.approx(4.0e-4)
    assert summary["median_train_direction_fraction_of_internal_dimension"] == 0.5
    assert summary["median_unidentified_qhq_relative_frobenius"] > 0.8
    assert len(result["direction_kind_summaries"]) == 2
    heldout_angle = next(
        row
        for row in result["direction_kind_summaries"]
        if row["role"] == "heldout" and row["kind"] == "angle"
    )
    assert heldout_angle["direction_count"] == 4
    assert result["training_history_row_count"] == 2
    assert (args.output_dir / "training_curves.png").is_file()
    with (args.output_dir / "run_summary.csv").open() as handle:
        saved = next(csv.DictReader(handle))
    assert saved["fraction_full_hessian_below_0p15"] == "0.75"
