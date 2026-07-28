"""Assemble all immutable recovery artifacts into one machine-readable acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


RUNS = (
    "qm9_model__qm9_domain",
    "qmugs_model__qmugs_domain",
    "qm9_model__combined_domain",
    "qmugs_model__combined_domain",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()), {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _baseline_summary(root: Path) -> tuple[dict, dict]:
    summaries = {}
    artifacts = {}
    for run in RUNS:
        report, artifact = _load(root / run / "report.json")
        shard_zero, shard_zero_artifact = _load(
            root / run / "summaries" / "shard_0.json"
        )
        summaries[run] = {
            "status": report["status"],
            "rows": report["rows"],
            "partition": report["partition"],
            "scf_iterations": report["scf_iterations"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "metrics": report["metrics"],
            "groups": report["groups"],
            "quantiles": report["quantiles"],
            "cost": report["cost"],
            "outliers_jsonl_gz": report["outliers_jsonl_gz"],
            "row_artifacts": report["artifacts"],
            "execution": {
                key: shard_zero.get(key)
                for key in (
                    "batch_size",
                    "command",
                    "device",
                    "dtype",
                    "edge_policy",
                    "hostname",
                    "hparams",
                )
            },
        }
        artifacts[run] = {
            "report": artifact,
            "representative_shard_summary": shard_zero_artifact,
            "outliers": _artifact(Path(report["outliers_jsonl_gz"])),
        }
    return summaries, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.restore_root.resolve()
    paths = {
        "environment": root / "recovery_manifest_20260715.json",
        "immutability": root / "immutability_audit_20260715.json",
        "path_guard_negative": root / "path_guard_negative_20260715.json",
        "content_audit": root / "content_audit_full_20260715/report.json",
        "split_audit": root / "split_leakage_audit_with_metadata_20260715.json",
        "safe_splits": root / "group_safe_splits_20260715/report.json",
        "model_acceptance": root / "model_acceptance_20260715/report.json",
        "failure_inventory": root / "failure_inventory_20260715.json",
        "pipeline_provenance": root / "pipeline_provenance_20260715.json",
        "frozen_slurm_wrappers": root / "slurm_frozen_wrappers_20260715/report.json",
        "validation_cache_final": root / "baseline_cache_validation_20260715/report.json",
        "validation_cache_scf6": root / "baseline_cache_validation_scf6_20260715/report.json",
        "validation_freeze": root / "scientific_baseline_validation_full_20260715/frozen_models.json",
        "initial_final_only_freeze": root / "scientific_baseline_validation_20260715/frozen_models.json",
        "density_optimization": root / "density_optimization_validation_20260715/report.json",
        "test_cache_final": root / "baseline_cache_test_20260715/report.json",
        "test_cache_scf6": root / "baseline_cache_test_scf6_20260715/report.json",
        "force_hessian_boundary": root / "force_hessian_boundary_report_20260715.json",
        "total_derivative_readiness": root / "total_derivative_remote_readiness_20260715.json",
        "total_derivative_local_readiness": root / "total_derivative_local_readiness_20260715.json",
    }
    loaded = {}
    artifacts = {}
    for name, path in paths.items():
        loaded[name], artifacts[name] = _load(path)
    validation, validation_artifacts = _baseline_summary(
        root / "scientific_baseline_validation_full_20260715"
    )
    test, test_artifacts = _baseline_summary(
        root / "scientific_baseline_test_full_20260715"
    )
    artifacts["validation_runs"] = validation_artifacts
    artifacts["test_runs"] = test_artifacts
    junit_path = root / "total_derivative_unit_20260715.xml"
    artifacts["total_derivative_unit_tests"] = _artifact(junit_path)
    junit = ET.parse(junit_path).getroot()
    junit_suites = list(junit.iter("testsuite"))
    junit_summary = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in junit_suites)
        for key in ("tests", "errors", "failures", "skipped")
    }

    checks = {
        "environment_frozen": loaded["environment"].get("status") == "passed",
        "original_data_and_checkpoints_unchanged": loaded["immutability"].get("status")
        == "passed",
        "forbidden_path_hard_failure": loaded["path_guard_negative"].get("status")
        == "passed"
        and loaded["path_guard_negative"].get("expected_nonzero") is True,
        "full_content_audit": loaded["content_audit"].get("status") == "passed",
        "group_safe_splits": loaded["safe_splits"].get("status") == "passed",
        "multi_mode_model_acceptance": loaded["model_acceptance"].get("status") == "passed",
        "failed_attempts_retained": loaded["failure_inventory"].get("status")
        == "audit_complete",
        "validation_freeze_test_order": loaded["pipeline_provenance"].get("status")
        == "passed",
        "execution_wrappers_frozen": loaded["frozen_slurm_wrappers"].get("status")
        == "passed",
        "validation_caches": all(
            loaded[name].get("status") == "passed"
            for name in ("validation_cache_final", "validation_cache_scf6")
        ),
        "validation_runs": all(value["status"] == "passed" for value in validation.values()),
        "validation_dual_density_coverage": all(
            value["scf_iterations"] == [[6, -1]] for value in validation.values()
        ),
        "models_frozen_before_test": loaded["validation_freeze"].get("status") == "frozen",
        "test_caches": all(
            loaded[name].get("status") == "passed"
            for name in ("test_cache_final", "test_cache_scf6")
        ),
        "one_shot_test_runs": all(value["status"] == "passed" for value in test.values()),
        "test_dual_density_coverage": all(
            value["scf_iterations"] == [[6, -1]] for value in test.values()
        ),
        "density_optimization_representatives": loaded["density_optimization"].get("status")
        == "passed",
        "fixed_density_force_hessian_funnel": loaded["force_hessian_boundary"].get("status")
        == "passed",
        "total_derivative_plan_audited": loaded["total_derivative_readiness"].get("status")
        == "audit_complete",
        "local_derivative_components_audited": loaded[
            "total_derivative_local_readiness"
        ].get("status")
        == "audit_complete",
        "local_derivative_unit_tests": (
            junit_summary["tests"] == 25
            and junit_summary["errors"] == 0
            and junit_summary["failures"] == 0
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "completion_scope": (
            "Recovery acceptance, leakage-safe scientific baselines, representative density "
            "optimization, and force/Hessian interface preparation. This status does not claim "
            "that a conservative physical total-OFDFT force/Hessian has been implemented."
        ),
        "checks": checks,
        "artifacts": artifacts,
        "checkpoint_sha256": {
            dataset: value["checkpoint_sha256"]
            for dataset, value in loaded["model_acceptance"]["datasets"].items()
        },
        "data": {
            "content_audit": loaded["content_audit"],
            "split_audit_artifact": artifacts["split_audit"],
            "safe_splits": loaded["safe_splits"],
        },
        "model_acceptance": loaded["model_acceptance"],
        "immutability": loaded["immutability"],
        "path_policy_test": loaded["path_guard_negative"],
        "pipeline_provenance": loaded["pipeline_provenance"],
        "execution_wrappers": loaded["frozen_slurm_wrappers"],
        "failed_attempts": loaded["failure_inventory"],
        "scientific_baselines": {
            "definition": (
                "All fixed-split forward metrics cover archived SCF step 6 and the final label "
                "density. Density-role groups prevent the nontrivial difference task from being "
                "conflated with its identically-zero final-density target."
            ),
            "caches": {
                "validation_final": loaded["validation_cache_final"],
                "validation_scf6": loaded["validation_cache_scf6"],
                "test_final": loaded["test_cache_final"],
                "test_scf6": loaded["test_cache_scf6"],
            },
            "validation": validation,
            "frozen_before_test": loaded["validation_freeze"],
            "initial_final_only_freeze_provenance": loaded["initial_final_only_freeze"],
            "test": test,
        },
        "density_optimization": loaded["density_optimization"],
        "force_hessian": loaded["force_hessian_boundary"],
        "total_derivative": {
            "remote_readiness": loaded["total_derivative_readiness"],
            "local_readiness": loaded["total_derivative_local_readiness"],
            "local_unit_tests": junit_summary,
            "deployment_status": "planned_not_deployed_on_recovery_server",
        },
        "interpretation": {
            "historical_models": (
                "Recovered Graphformer checkpoints are energy/density-gradient/difference models, "
                "not force/Hessian models."
            ),
            "fixed_density": "screening proxy at fixed density, with omitted total classical/Pulay response",
            "density_relaxed": "incomplete-derived-force proxy, not a physical total-OFDFT Hessian",
            "test_policy": "validation frozen first; exactly one test baseline execution followed",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
