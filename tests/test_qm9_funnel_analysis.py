from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload():
    return {
        "runs": {
            "model": {
                "mean_mae": 0.1,
                "mean_opt_cycles": 99.0,
                "n_success": 1,
                "n_failed": 0,
            }
        },
        "optimization_rows": [
            {"run": "model", "converged": True, "cycles": 3, "used_fallback": True},
            {"run": "model", "converged": False, "cycles": 7, "used_fallback": False},
        ],
    }


def test_tier2_optimization_stats_use_per_displacement_rows():
    module = _load_script("qm9_density_relaxed_tier2_analysis")
    stats = module._optimization_stats(_payload(), "model")

    assert stats == {
        "strict_converged_points": 1,
        "optimization_points": 2,
        "strict_convergence_rate": 0.5,
        "fallback_triggers": 1,
        "mean_cycles": 5.0,
        "max_cycles": 7,
    }


def test_test100_density_summary_uses_per_displacement_rows():
    module = _load_script("qm9_test100_funnel_analysis")
    summary = module._density_summary(_payload(), "model")

    assert summary["test_relaxed_strict_points"] == 1
    assert summary["test_relaxed_optimization_points"] == 2
    assert summary["test_relaxed_mean_cycles"] == 5.0


def test_proxy_diagnostics_compare_models_per_molecule():
    module = _load_script("qm9_test100_funnel_analysis")
    density_rows = [
        {
            "run": run,
            "molecule_id": "0000001",
            "natoms": 2,
            "success": True,
            "mae": mae,
            "rmse": mae * 2,
            "relative_fro_error": mae * 3,
            "model_symmetry_max_abs_error": mae * 4,
        }
        for run, mae in (("EGF_lam1", 0.2), ("EGF_w1p0", 0.1))
    ]
    fixed_rows = [
        {
            "run": run,
            "molecule_id": "0000001",
            "success": True,
            "autograd_vs_pbe": {"mae": mae},
        }
        for run, mae in (("EGF_w0p1", 0.15), ("EGF_w1p0", 0.08))
    ]

    rows, summary = module._proxy_diagnostics(
        density_rows, fixed_rows, ["EGF_w0p1", "EGF_w1p0"], ["EGF_w1p0"]
    )
    assert len(rows) == 2
    assert summary["fixed_vs_relaxed_best_model_rank_matches"] == 1
    candidate = summary["candidates_vs_historical"]["EGF_w1p0"]
    assert candidate["improved_relaxed_hessian_mae_count"] == 1
    assert candidate["median_relaxed_hessian_mae_ratio"] == 0.5
