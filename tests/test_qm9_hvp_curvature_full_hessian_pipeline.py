import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_script(name: str):
    path = Path("scripts") / f"{name}.py"
    scripts_dir = str(path.parent.resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_full_hessian_task_freeze_has_no_diagnostic_fallback(tmp_path):
    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "frozen_candidates_for_full_hessian": [],
                "runs": [],
            }
        )
    )
    per_direction = tmp_path / "directions.csv"
    per_direction.write_text("variant,molecule_id\n")
    output_tsv = tmp_path / "tasks.tsv"
    output_json = tmp_path / "tasks.json"
    subprocess.run(
        [
            "python",
            "scripts/prepare_qm9_hvp_curvature_full_hessian.py",
            "--strict-summary",
            str(strict),
            "--strict-per-direction",
            str(per_direction),
            "--reference-dir",
            str(tmp_path),
            "--output-tsv",
            str(output_tsv),
            "--output-json",
            str(output_json),
        ],
        check=True,
    )
    assert len(output_tsv.read_text().splitlines()) == 1
    assert json.loads(output_json.read_text())["tasks"] == []


def test_full_hessian_task_freeze_pairs_candidate_with_same_seed_A(tmp_path):
    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "frozen_candidates_for_full_hessian": [
                    {"run_name": "C_2", "seed": 2, "variant": "C"}
                ],
                "runs": [
                    {
                        "run_name": "A_2",
                        "variant": "A",
                        "seed": 2,
                        "curvature_weight": 0,
                        "reference_floor": 0.01,
                    },
                    {
                        "run_name": "C_2",
                        "variant": "C",
                        "seed": 2,
                        "curvature_weight": 1e-4,
                        "reference_floor": 0.01,
                    },
                ],
            }
        )
    )
    rows = []
    refs = tmp_path / "refs"
    refs.mkdir()
    for index in range(5):
        molecule = f"{index + 1:07d}"
        np.savez_compressed(
            refs / f"pbe_hessian_{molecule}_0000000.npz", pbe_hessian=np.eye(3)
        )
        for variant, run, mae in (("A", "A_2", index + 1), ("C", "C_2", index + 0.5)):
            rows.append(
                {
                    "run_name": run,
                    "variant": variant,
                    "seed": 2,
                    "molecule_id": molecule,
                    "direction_index": 0,
                    "natoms": index + 1,
                    "strict_hvp_mae": mae,
                }
            )
    per_direction = tmp_path / "directions.csv"
    _write_csv(per_direction, rows)
    output_tsv = tmp_path / "tasks.tsv"
    output_json = tmp_path / "tasks.json"
    subprocess.run(
        [
            "python",
            "scripts/prepare_qm9_hvp_curvature_full_hessian.py",
            "--strict-summary",
            str(strict),
            "--strict-per-direction",
            str(per_direction),
            "--reference-dir",
            str(refs),
            "--output-tsv",
            str(output_tsv),
            "--output-json",
            str(output_json),
            "--parent-count",
            "5",
        ],
        check=True,
    )
    payload = json.loads(output_json.read_text())
    assert len(payload["molecules"]) == 5
    assert len(payload["tasks"]) == 10
    assert {row["run_name"] for row in payload["tasks"]} == {"A_2", "C_2"}


def test_full_hessian_analysis_records_missing_tasks_and_blocks_test100(tmp_path):
    module = _load_script("qm9_hvp_curvature_full_hessian_analysis")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "full_hessian": {
                    "force_energy_diagonal_relative_degradation_max": 0.05
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pbe_manifest": str(tmp_path / "pbe.json"),
                "molecules": [{"molecule_id": "0000001"}],
                "tasks": [
                    {
                        "task_index": 0,
                        "role": "matched_A",
                        "variant": "A",
                        "curvature_weight": 0,
                        "reference_floor": 0.01,
                        "seed": 1,
                        "run_name": "A_1",
                        "molecule_id": "0000001",
                        "natoms": 1,
                        "molecule_reason": "smallest",
                    },
                    {
                        "task_index": 1,
                        "role": "candidate",
                        "variant": "C",
                        "curvature_weight": 1e-4,
                        "reference_floor": 0.01,
                        "seed": 1,
                        "run_name": "C_1",
                        "molecule_id": "0000001",
                        "natoms": 1,
                        "molecule_reason": "smallest",
                    },
                ],
            }
        )
    )
    result = module.analyze(
        argparse.Namespace(
            protocol=protocol,
            task_manifest=manifest,
            task_root=tmp_path / "tasks",
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "analysis",
        )
    )
    assert result["failed_task_count"] == 2
    assert result["validation_promoted_candidates"] == []
    assert result["test100_allowed"] is False
    assert set(result["incomplete_runs"]) == {"A_1", "C_1"}
    assert (tmp_path / "analysis" / "full_hessian_failures.csv").is_file()


def test_full_hessian_analysis_promotes_only_complete_non_regressing_run(
    tmp_path, monkeypatch
):
    module = _load_script("qm9_hvp_curvature_full_hessian_analysis")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "full_hessian": {
                    "force_energy_diagonal_relative_degradation_max": 0.05
                }
            }
        )
    )
    tasks = []
    for index, (role, variant, run) in enumerate(
        (("matched_A", "A", "A_1"), ("candidate", "C", "C_1"))
    ):
        tasks.append(
            {
                "task_index": index,
                "role": role,
                "variant": variant,
                "curvature_weight": 0 if variant == "A" else 1e-4,
                "reference_floor": 0.01,
                "seed": 1,
                "run_name": run,
                "molecule_id": "0000001",
                "natoms": 1,
                "molecule_reason": "smallest",
            }
        )
        task_dir = tmp_path / "tasks" / f"task_{index:03d}_{run}_0000001"
        task_dir.mkdir(parents=True)
        scale = 2.0 if variant == "A" else 1.0
        (task_dir / "summary.json").write_text(
            json.dumps(
                {
                    "metric_rows": [
                        {
                            "run": run,
                            "molecule_id": "0000001",
                            "sample_id": 0,
                            "success": True,
                            "hessian_npz": str(task_dir / "hessian.npz"),
                            "mae": scale,
                            "rmse": scale,
                            "relative_fro_error": scale,
                            "antisymmetric_over_symmetric_fro": 0.01,
                            "force_vs_energy_diagonal_mae": 0.1,
                            "strict_points": 6,
                            "total_displaced_points": 6,
                        }
                    ],
                    "point_rows": [],
                    "wall_time_s": 2,
                    "max_rss_mb": 3,
                    "peak_gpu_memory_mb": 4,
                }
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pbe_manifest": str(tmp_path / "pbe.json"),
                "molecules": [{"molecule_id": "0000001"}],
                "tasks": tasks,
            }
        )
    )

    monkeypatch.setattr(
        module,
        "evaluate_vibrations",
        lambda _args: {
            "rows": [
                {
                    "run": "A_1",
                    "molecule_id": "0000001",
                    "frequency_mae_cm-1": 2.0,
                    "mean_mode_overlap": 0.8,
                    "imaginary_mode_count_error": 0,
                },
                {
                    "run": "C_1",
                    "molecule_id": "0000001",
                    "frequency_mae_cm-1": 1.0,
                    "mean_mode_overlap": 0.9,
                    "imaginary_mode_count_error": 0,
                },
            ],
            "summaries": [],
        },
    )
    result = module.analyze(
        argparse.Namespace(
            protocol=protocol,
            task_manifest=manifest,
            task_root=tmp_path / "tasks",
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "analysis",
        )
    )
    assert result["failed_task_count"] == 0
    assert result["groups"][0]["full_data_complete"] is True
    assert result["groups"][0]["full_hessian_vibration_gate"] is True
    assert result["validation_promoted_candidates"][0]["run_name"] == "C_1"
    assert result["test100_allowed"] is True
