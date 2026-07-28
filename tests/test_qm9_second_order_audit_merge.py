from pathlib import Path

import pytest

from scripts.qm9_second_order_audit_merge import merge


def _payload(run: str, autograd_mae: float, fd_mae: float) -> dict:
    return {
        "compare_pbe_force_secant": True,
        "definition": "fixed",
        "device": "cuda:0",
        "directional_sample_ids": [1, 2, 3],
        "fd_displacement": 1e-3,
        "fixed_density_full_hessian": [
            {
                "run": run,
                "molecule_id": "0000001",
                "sample_id": 0,
                "success": True,
                "autograd_vs_pbe": {"mae": autograd_mae},
                "fd_vs_pbe": {"mae": fd_mae},
            }
        ],
        "hvp": [{"run": run, "molecule_id": "0000001", "success": True}],
        "hvp_eps": 1e-3,
        "limitations": ["proxy"],
        "max_rss_kb": 100 if run == "a" else 200,
        "model_dtype": "torch.float64",
        "molecules": ["0000001"],
        "reference_dir": "/reference",
        "scf_iteration": -1,
        "split": "test",
        "drop_self_edges_diagnostic": [],
        "unrolled_density_prototype": [],
    }


def test_merge_recomputes_cross_model_ranking(tmp_path: Path) -> None:
    paths = []
    for run, autograd_mae, fd_mae in (("a", 0.3, 0.2), ("b", 0.1, 0.15)):
        path = tmp_path / f"{run}.json"
        import json

        path.write_text(json.dumps(_payload(run, autograd_mae, fd_mae)))
        paths.append(path)

    result = merge(paths)
    assert len(result["fixed_density_full_hessian"]) == 2
    assert len(result["hvp"]) == 2
    assert result["max_rss_kb"] == 200
    assert result["parallel_merge"]["sum_worker_max_rss_kb"] == 300
    assert result["ranking_consistency"] == [
        {
            "molecule_id": "0000001",
            "autograd_best_by_pbe_mae": "b",
            "fd_best_by_pbe_mae": "b",
            "consistent": True,
        }
    ]


def test_merge_rejects_inconsistent_protocol(tmp_path: Path) -> None:
    import json

    first = _payload("a", 0.3, 0.2)
    second = _payload("b", 0.1, 0.15)
    second["fd_displacement"] = 3e-4
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)

    with pytest.raises(ValueError, match="inconsistent fd_displacement"):
        merge(paths)
