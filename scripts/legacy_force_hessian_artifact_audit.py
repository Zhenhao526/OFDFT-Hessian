"""Freeze the already-completed QM9 force/Hessian funnel into one boundary report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _mean(rows: list[dict], *keys: str) -> float:
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _model(report: dict, name: str) -> dict:
    matches = [row for row in report["models"] if row["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one model {name}, got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.models_root.resolve()
    validation_smoke = (
        root
        / "eval/qm9_random1000_validation_funnel_smoke/20260714_job507_ground_state"
        / "fixed_hvp/summary.json"
    )
    tier2 = (
        root
        / "eval/qm9_random1000_validation_tier2_strict_hessian/20260715_064755"
        / "analysis/tier2_model_summary.json"
    )
    test_root = root / "eval/qm9_random1000_frozen_test100/20260715_081030"
    test_summary = test_root / "analysis/test100_summary.json"
    fixed_summary = test_root / "fixed_hessian/summary.json"
    relaxed_summary = test_root / "density_relaxed/summary.json"
    candidate_manifest = test_root / "candidate_manifest.json"

    validation_smoke_data = json.loads(validation_smoke.read_text())
    tier2_data = json.loads(tier2.read_text())
    test_data = json.loads(test_summary.read_text())
    fixed_data = json.loads(fixed_summary.read_text())
    relaxed_data = json.loads(relaxed_summary.read_text())
    w1_val = _model(tier2_data, "EGF_w1p0")
    w1_test = _model(test_data, "EGF_w1p0")
    w1_checkpoint = Path(w1_test["checkpoint"])
    w1_hparams = Path(w1_test["run_dir"]) / "hparams.yaml"
    cfg = OmegaConf.load(w1_hparams)
    actual_force_weight = float(cfg.model.loss_function.force_loss.weight)
    if actual_force_weight != 1.0:
        raise RuntimeError(f"Expected actual force weight 1.0, got {actual_force_weight}")

    fixed_rows = [
        row
        for row in fixed_data["fixed_density_full_hessian"]
        if row["run"] == "EGF_w1p0"
    ]
    hvp_rows = [row for row in fixed_data["hvp"] if row["run"] == "EGF_w1p0"]
    errors = []
    if len(fixed_rows) != 100 or not all(row.get("success") for row in fixed_rows):
        errors.append("actual-weight-1 fixed-density Test100 is not 100/100 successful")
    if len(hvp_rows) != 400 or not all(row.get("success") for row in hvp_rows):
        errors.append("actual-weight-1 HVP Test100 is not 400/400 successful")
    if tier2_data.get("recommended_for_test100") != ["EGF_w3p0", "EGF_w1p0"]:
        errors.append("unexpected frozen validation candidate list")
    if test_data.get("frozen_candidates") != tier2_data.get("recommended_for_test100"):
        errors.append("Test100 candidates differ from frozen validation candidates")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "artifacts": {
            "validation_fixed_hvp_smoke": _artifact(validation_smoke),
            "validation_tier2": _artifact(tier2),
            "test100_summary": _artifact(test_summary),
            "test100_fixed_hessian": _artifact(fixed_summary),
            "test100_density_relaxed": _artifact(relaxed_summary),
            "test100_candidate_manifest": _artifact(candidate_manifest),
            "actual_weight_1_checkpoint": _artifact(w1_checkpoint),
            "actual_weight_1_hparams": _artifact(w1_hparams),
        },
        "units": {
            "energy": "Hartree",
            "force": "Hartree/Bohr",
            "hessian": "Hartree/Bohr^2",
            "coordinate": "Bohr",
        },
        "fixed_density_validation_smoke": {
            "split": validation_smoke_data["split"],
            "definition": validation_smoke_data["definition"],
            "fd_displacement_bohr": validation_smoke_data["fd_displacement"],
            "hvp_eps_bohr": validation_smoke_data["hvp_eps"],
            "full_hessian_cases": len(validation_smoke_data["fixed_density_full_hessian"]),
            "hvp_cases": len(validation_smoke_data["hvp"]),
            "full_hessian_all_finite": all(
                row["autograd_stats"]["finite"]
                and row["fd_stats"]["finite"]
                and row["autograd_vs_fd"]["finite"]
                for row in validation_smoke_data["fixed_density_full_hessian"]
            ),
            "hvp_all_finite": all(
                row["hvp_stats"]["finite"] and row["hvp_vs_fd_directional"]["finite"]
                for row in validation_smoke_data["hvp"]
            ),
            "limitations": validation_smoke_data["limitations"],
        },
        "actual_force_weight_1": {
            "actual_force_loss_weight": actual_force_weight,
            "validation_selection": {
                "definition": tier2_data["definition"],
                "energy_eligibility": tier2_data["energy_eligibility"],
                "recommended_for_test100": w1_val["recommended_for_test100"],
                "strict_converged_points": w1_val["strict_converged_points"],
                "optimization_points": w1_val["optimization_points"],
                "energy_mae": w1_val["validation_energy_mae"],
                "force_component_mae": w1_val["validation_force_component_mae"],
                "relaxed_proxy_hessian_mae": w1_val["validation_relaxed_hessian_mae"],
                "relaxed_proxy_hessian_relative_fro": w1_val[
                    "validation_relaxed_hessian_relative_fro"
                ],
            },
            "one_shot_frozen_test100": {
                "definition": test_data["definition"],
                "fixed_full_hessian_cases": len(fixed_rows),
                "hvp_cases": len(hvp_rows),
                "fixed_autograd_vs_fd_mae_mean": _mean(fixed_rows, "autograd_vs_fd", "mae"),
                "fixed_autograd_vs_fd_relative_fro_mean": _mean(
                    fixed_rows, "autograd_vs_fd", "relative_fro_error"
                ),
                "hvp_vs_own_fd_relative_fro_mean": _mean(
                    hvp_rows, "hvp_vs_fd_directional", "relative_fro_error"
                ),
                "energy_mae": w1_test["test_energy_mae"],
                "force_component_mae": w1_test["test_force_component_mae"],
                "fixed_hessian_pbe_mae": w1_test["test_fixed_hessian_pbe_mae"],
                "fixed_hessian_pbe_relative_fro": w1_test[
                    "test_fixed_hessian_pbe_relative_fro"
                ],
                "hvp_vs_pbe_force_secant_mae": w1_test[
                    "test_hvp_vs_pbe_force_secant_mae"
                ],
                "relaxed_proxy_hessian_success": w1_test[
                    "test_relaxed_hessian_success"
                ],
                "relaxed_proxy_hessian_failed": w1_test["test_relaxed_hessian_failed"],
                "relaxed_proxy_strict_points": w1_test["test_relaxed_strict_points"],
                "relaxed_proxy_optimization_points": w1_test[
                    "test_relaxed_optimization_points"
                ],
                "relaxed_proxy_hessian_mae": w1_test["test_relaxed_hessian_mae"],
                "relaxed_proxy_hessian_relative_fro": w1_test[
                    "test_relaxed_hessian_relative_fro"
                ],
                "relaxed_proxy_mean_cycles": w1_test["test_relaxed_mean_cycles"],
                "relaxed_proxy_wall_sum_s": w1_test["test_relaxed_wall_sum_s"],
                "fixed_vs_relaxed_best_model_rank_match_rate": test_data[
                    "proxy_diagnostics"
                ]["fixed_vs_relaxed_best_model_rank_match_rate"],
                "improved_relaxed_mae_molecules_vs_historical": test_data[
                    "proxy_diagnostics"
                ]["candidates_vs_historical"]["EGF_w1p0"][
                    "improved_relaxed_hessian_mae_count"
                ],
            },
        },
        "interpretation_boundaries": {
            "legacy_graphformer": (
                "The recovered QM9/QMUGS Graphformer checkpoints predict energy, density "
                "gradient and density difference; they are not force or Hessian models."
            ),
            "fixed_density": (
                "The audited coordinate autograd/Hessian/HVP differentiates learned scalar "
                "model energy at fixed density. It omits density response and coordinate "
                "derivatives of classical integrals/nuclear terms."
            ),
            "density_relaxed": (
                "The relaxed result is a finite difference of incomplete derived model force "
                "after density optimization. It is not a conservative physical total-OFDFT Hessian."
            ),
            "vibrations": (
                "No physical vibrational conclusion is authorized before a conservative total "
                "force passes relaxed-total-energy central differences and closed-loop work tests."
            ),
        },
        "raw_relaxed_summary_definition": relaxed_data["definition"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
