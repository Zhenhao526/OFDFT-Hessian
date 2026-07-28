import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


def _load_script(name: str):
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_strict_task_preparation_applies_direction_mask(tmp_path, monkeypatch):
    module = _load_script("prepare_qm9_hvp_curvature_strict_tasks")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "test100_access_allowed": False,
                "strict_complete_total_hvp": {"minimum_stable_parent_count": 1},
            }
        )
    )
    runs = tmp_path / "runs.tsv"
    run_rows = [
        {
            "run_name": f"{variant}_{seed}",
            "variant": variant,
            "seed": seed,
            "curvature_weight": 0 if variant == "A" else 1e-4,
            "reference_floor": 0.01,
        }
        for variant in ("A", "C")
        for seed in (1, 2, 3)
    ]
    _write_tsv(runs, run_rows)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "groups": [
                    {
                        "variant": "C",
                        "curvature_weight": 1e-4,
                        "reference_floor": 0.01,
                        "multiseed_tier1_gate": True,
                    }
                ],
            }
        )
    )
    sidecars = tmp_path / "sidecars"
    references = tmp_path / "references"
    sidecars.mkdir()
    references.mkdir()
    np.savez_compressed(
        sidecars / "0000001.0000000.npz",
        parent_stability_eligible=np.asarray(True),
        stability_mask=np.asarray([True, False, True, True]),
    )
    np.savez_compressed(references / "pbe_hessian_0000001_0000000.npz", pbe_hessian=np.eye(3))
    output = tmp_path / "strict.tsv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--protocol",
            str(protocol),
            "--multiseed-tasks",
            str(runs),
            "--multiseed-analysis",
            str(analysis),
            "--stable-sidecar-dir",
            str(sidecars),
            "--reference-dir",
            str(references),
            "--output",
            str(output),
        ],
    )
    module.main()
    with output.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 18
    assert {int(row["direction_index"]) for row in rows} == {0, 2, 3}
    assert {row["variant"] for row in rows} == {"A", "C"}
    assert {int(row["molecule_count"]) for row in rows} == {1}
    assert {row["molecules"] for row in rows} == {"0000001"}
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["stable_parent_count_gate"] is True


def test_strict_analysis_uses_matched_seed_baselines(tmp_path):
    module = _load_script("qm9_hvp_curvature_strict_analysis")
    protocol = {
        "promotion": {
            "energy_mae_relative_degradation_max": 0.05,
            "energy_gate_required_seeds": 3,
            "force_improved_seeds_min": 2,
            "strict_hvp_improved_seeds_min": 2,
            "small_reference_hvp_rms": 0.02,
            "full_hessian_candidates_max": 2,
        },
        "strict_complete_total_hvp": {
            "fallback": {"projected_gradient_threshold": 1e-5},
            "response": {"relative_residual_threshold": 1e-8},
        },
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    strict_root = tmp_path / "strict"
    stage1 = tmp_path / "stage1"
    task_rows = []
    for seed, scale in ((1, 1.0), (2, 2.0), (3, 3.0)):
        for variant, factor in (("A", 1.0), ("C", 0.8)):
            run = f"{variant}_{seed}"
            task_rows.append(
                {
                    "task_index": len(task_rows),
                    "run_name": run,
                    "variant": variant,
                    "seed": seed,
                    "curvature_weight": 0 if variant == "A" else 1e-4,
                    "reference_floor": 0.01,
                    "direction_index": 0,
                    "molecule_count": 1,
                    "molecules": "0000001",
                }
            )
            out = strict_root / run / "direction_0"
            out.mkdir(parents=True)
            row = {
                "molecule_id": "0000001",
                "direction_kind": "bond",
                "natoms": 1,
                "strict_relaxed_vs_pbe_mae": scale * factor,
                "strict_relaxed_vs_pbe_rmse": scale * factor,
                "strict_relaxed_vs_pbe_relative_frobenius": factor,
                "strict_relaxed_max_gradient_norm": 1e-9,
                "response_relative_residual": 1e-10,
                "wall_time_s": 2,
            }
            with (out / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (out / "summary.json").write_text(
                json.dumps({"wall_time_s": 2, "max_rss_mb": 3, "peak_gpu_memory_mb": 4})
            )
            np.savez_compressed(
                out / f"{run}_0000001_0000000_hvp_arrays.npz",
                pbe_hvp=np.asarray([[1.0, 0.0, 0.0]]),
            )
            (stage1 / run).mkdir(parents=True)
            (stage1 / run / "validation_energy_force.json").write_text(
                json.dumps(
                    {
                        "energy_mae": scale,
                        "force_component_mae": scale * factor,
                    }
                )
            )
    tasks = tmp_path / "tasks.tsv"
    _write_tsv(tasks, task_rows)
    result = module.analyze(
        argparse.Namespace(
            protocol=protocol_path,
            tasks=tasks,
            strict_root=strict_root,
            stage1_root=stage1,
            output_dir=tmp_path / "analysis",
        )
    )
    c_group = next(group for group in result["groups"] if group["variant"] == "C")
    assert c_group["force_improved_seeds"] == 3
    assert c_group["strict_hvp_improved_seeds"] == 3
    assert c_group["strict_gate_passed"] is True
    assert result["frozen_candidates_for_full_hessian"][0]["variant"] == "C"


def test_strict_analysis_records_missing_isolated_task_without_promotion(tmp_path):
    module = _load_script("qm9_hvp_curvature_strict_analysis")
    protocol = {
        "promotion": {
            "energy_mae_relative_degradation_max": 0.05,
            "energy_gate_required_seeds": 3,
            "force_improved_seeds_min": 2,
            "strict_hvp_improved_seeds_min": 2,
            "small_reference_hvp_rms": 0.02,
            "full_hessian_candidates_max": 2,
        },
        "strict_complete_total_hvp": {
            "fallback": {"projected_gradient_threshold": 1e-5},
            "response": {"relative_residual_threshold": 1e-8},
        },
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    tasks = tmp_path / "tasks.tsv"
    task_rows = []
    for variant in ("A", "C"):
        task_rows.append(
            {
                "task_index": len(task_rows),
                "run_name": f"{variant}_1",
                "variant": variant,
                "seed": 1,
                "curvature_weight": 0 if variant == "A" else 1e-4,
                "reference_floor": 0.01,
                "direction_index": 0,
                "molecule_count": 1,
                "molecules": "0000001",
            }
        )
    _write_tsv(tasks, task_rows)
    strict_root = tmp_path / "strict"
    out = strict_root / "A_1" / "direction_0" / "0000001"
    out.mkdir(parents=True)
    metric = {
        "molecule_id": "0000001",
        "direction_kind": "bond",
        "natoms": 1,
        "strict_relaxed_vs_pbe_mae": 1.0,
        "strict_relaxed_vs_pbe_rmse": 1.0,
        "strict_relaxed_vs_pbe_relative_frobenius": 1.0,
        "strict_relaxed_max_gradient_norm": 1e-9,
        "response_relative_residual": 1e-10,
        "wall_time_s": 2,
    }
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric))
        writer.writeheader()
        writer.writerow(metric)
    (out / "summary.json").write_text(
        json.dumps({"wall_time_s": 2, "max_rss_mb": 3, "peak_gpu_memory_mb": 4})
    )
    np.savez_compressed(
        out / "A_1_0000001_0000000_hvp_arrays.npz", pbe_hvp=np.ones((1, 3))
    )
    stage1 = tmp_path / "stage1" / "A_1"
    stage1.mkdir(parents=True)
    (stage1 / "validation_energy_force.json").write_text(
        json.dumps({"energy_mae": 1.0, "force_component_mae": 1.0})
    )
    result = module.analyze(
        argparse.Namespace(
            protocol=protocol_path,
            tasks=tasks,
            strict_root=strict_root,
            stage1_root=tmp_path / "stage1",
            output_dir=tmp_path / "analysis",
        )
    )
    assert result["failed_task_count"] == 1
    assert result["incomplete_runs"] == ["C_1"]
    assert result["frozen_candidates_for_full_hessian"] == []
    assert (tmp_path / "analysis" / "strict_failures.csv").is_file()
