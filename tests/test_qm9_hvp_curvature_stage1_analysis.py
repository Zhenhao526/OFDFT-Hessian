import argparse
import csv
import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    path = Path("scripts/qm9_hvp_curvature_stage1_analysis.py")
    spec = importlib.util.spec_from_file_location("hvp_stage1_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, name: str, energy: float, force: float, hvp: float) -> None:
    run = root / name
    (run / "fixed_hvp").mkdir(parents=True)
    metrics = {"energy_mae": energy, "force_component_mae": force}
    (run / "validation_energy_force.json").write_text(json.dumps(metrics))
    (run / "train700_forgetting.json").write_text(json.dumps(metrics))
    fixed = {
        "mean_mae": hvp,
        "mean_relative_frobenius": hvp,
        "rows": [
            {
                "molecule_id": f"{index:07d}",
                "direction_index": 0,
                "corrected_vs_pbe_mae": hvp,
            }
            for index in range(1, 6)
        ],
    }
    (run / "fixed_hvp/summary.json").write_text(json.dumps(fixed))


def test_stage1_matches_each_candidate_to_same_seed_baseline(tmp_path):
    module = _load_module()
    tasks = tmp_path / "tasks.tsv"
    stage1 = tmp_path / "stage1"
    rows = []
    for seed, scale in ((11, 1.0), (22, 2.0)):
        for variant in "ABCDE":
            name = f"{variant}_{seed}"
            rows.append(
                {
                    "task_index": len(rows),
                    "variant": variant,
                    "curvature_weight": 0 if variant == "A" else 1e-4,
                    "reference_floor": 1e-2,
                    "seed": seed,
                    "max_steps": 2,
                    "run_name": name,
                }
            )
            factor = 1.0 if variant == "A" else 0.9
            _write_run(stage1, name, scale, scale * factor, scale * factor)
    with tasks.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    result = module.analyze(
        argparse.Namespace(
            protocol=Path("configs/audit/qm9_hvp_curvature_training_v2.yaml"),
            tasks=tasks,
            stage1_root=stage1,
            output_dir=tmp_path / "analysis",
            max_per_variant=2,
        )
    )

    assert result["seed_count"] == 2
    assert result["baseline_runs_by_seed"] == {"11": "A_11", "22": "A_22"}
    assert result["selected_config_by_variant"]["C"]["selection_status"] == (
        "tier1_eligible"
    )
    c_group = next(group for group in result["groups"] if group["variant"] == "C")
    assert c_group["force_improved_seeds"] == 2
    assert c_group["fixed_hvp_improved_seeds"] == 2
    assert c_group["train700_energy_gate_passed_seeds"] == 2
    assert c_group["train700_force_gate_passed_seeds"] == 2


def test_stage1_blocks_train700_forgetting_before_candidate_freeze(tmp_path):
    module = _load_module()
    tasks = tmp_path / "tasks.tsv"
    stage1 = tmp_path / "stage1"
    rows = []
    for variant in "ABCDE":
        name = f"{variant}_1"
        rows.append(
            {
                "task_index": len(rows),
                "variant": variant,
                "curvature_weight": 0 if variant == "A" else 1e-4,
                "reference_floor": 1e-2,
                "seed": 1,
                "max_steps": 2,
                "run_name": name,
            }
        )
        _write_run(stage1, name, 1.0, 1.0 if variant == "A" else 0.9, 1.0 if variant == "A" else 0.9)
    # Candidate looks better on validation/HVP but regresses unlabelled train700 force by 10%.
    (stage1 / "C_1" / "train700_forgetting.json").write_text(
        json.dumps({"energy_mae": 1.0, "force_component_mae": 1.1})
    )
    with tasks.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = module.analyze(
        argparse.Namespace(
            protocol=Path("configs/audit/qm9_hvp_curvature_training_v2.yaml"),
            tasks=tasks,
            stage1_root=stage1,
            output_dir=tmp_path / "analysis",
            max_per_variant=2,
        )
    )
    candidate = next(row for row in result["groups"] if row["variant"] == "C")
    assert candidate["train700_force_gate_passed_seeds"] == 0
    assert candidate["multiseed_tier1_gate"] is False
    assert result["selected_config_by_variant"]["C"]["selection_status"] == (
        "diagnostic_fallback_not_promoted"
    )


def test_stage1_fixed_hvp_uses_only_frozen_stable_validation_sidecars():
    evaluator = Path("scripts/slurm_qm9_hvp_curvature_stage1_eval.sbatch").read_text()
    submitter = Path("scripts/slurm_qm9_hvp_train100_postprocess.sbatch").read_text()
    protocol = yaml.safe_load(
        Path("configs/audit/qm9_hvp_curvature_training_v2.yaml").read_text()
    )
    assert "formal_val20_v3/analysis_v3/stable_sidecars" in evaluator
    assert 'afterok:${SCREEN_JOB_ID}:${VAL_POST_JOB_ID}' in submitter
    assert protocol["validation"][
        "fixed_density_hvp_stable_parents_and_directions_only"
    ] is True
    assert protocol["validation"][
        "unstable_parents_or_directions_used_in_average_selection"
    ] is False
