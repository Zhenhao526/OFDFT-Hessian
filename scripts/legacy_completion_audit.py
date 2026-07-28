"""Requirement-by-requirement completion audit for the legacy recovery goal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ROWS = {
    "validation": {
        "qm9_model__qm9_domain": 26804,
        "qmugs_model__qmugs_domain": 5064,
        "qm9_model__combined_domain": 31868,
        "qmugs_model__combined_domain": 31868,
    },
    "test": {
        "qm9_model__qm9_domain": 26810,
        "qmugs_model__qmugs_domain": 1698,
        "qm9_model__combined_domain": 28510,
        "qmugs_model__combined_domain": 28510,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_zero(value) -> bool:
    if isinstance(value, dict):
        return all(_all_zero(item) for item in value.values())
    return value == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--final-markdown", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    final = json.loads(args.final_report.read_text())
    handoff_text = args.handoff.read_text()
    checks = []

    def add(requirement: str, passed: bool, evidence) -> None:
        checks.append(
            {"requirement": requirement, "passed": bool(passed), "evidence": evidence}
        )

    add("aggregate acceptance passed", final.get("status") == "passed", final.get("errors"))
    add(
        "all aggregate gates passed",
        all(final.get("checks", {}).values()),
        final.get("checks"),
    )
    add(
        "exact Slurm execution wrappers retained",
        final["execution_wrappers"].get("status") == "passed"
        and final["checks"]["execution_wrappers_frozen"],
        final["execution_wrappers"],
    )
    add(
        "portable path hard-failure test",
        final["path_policy_test"].get("status") == "passed"
        and final["path_policy_test"].get("exit_code") != 0,
        final["path_policy_test"],
    )
    add(
        "original data, metadata, statistics, hparams and checkpoints unchanged",
        final["immutability"].get("status") == "passed",
        {
            "errors": final["immutability"].get("errors"),
            "label_sources": final["immutability"].get("label_sources"),
        },
    )
    content = final["data"]["content_audit"]
    sources = content["sources"]
    add(
        "all 158455 archives and 5446247 SCF configurations read",
        sum(item["processed"] for item in sources.values()) == 158455
        and sum(item["scf_steps"] for item in sources.values()) == 5446247,
        {
            name: {"archives": item["processed"], "scf_steps": item["scf_steps"]}
            for name, item in sources.items()
        },
    )
    add(
        "all numeric content finite with no schema/shape anomaly",
        content.get("status") == "passed"
        and all(not item.get("failures") and not item.get("anomaly_counts") for item in sources.values()),
        {name: item.get("anomaly_counts") for name, item in sources.items()},
    )
    add(
        "no exact geometry/trajectory/configuration duplicates",
        _all_zero(content["duplicate_counts"]),
        content["duplicate_counts"],
    )
    safe = final["data"]["safe_splits"]["datasets"]
    add(
        "all sidecar splits canonical-SMILES/parent group safe",
        all(item["cross_partition_identity_keys"] == 0 for item in safe.values()),
        {
            name: {
                "counts": {key: value["entries"] for key, value in item["partitions"].items()},
                "sha256": item["safe_split_pickle_sha256"],
            }
            for name, item in safe.items()
        },
    )
    acceptance = final["model_acceptance"]
    acceptance_ok = acceptance.get("status") == "passed"
    for item in acceptance["datasets"].values():
        acceptance_ok &= item["status"] == "passed" and item["samples"] == 21
        acceptance_ok &= item["basis_metadata"]["order_metadata_match"] is True
    add(
        "CPU/GPU float32/float64, repeat, batch, basis/order/electron acceptance",
        acceptance_ok,
        {
            name: {
                "samples": item["samples"],
                "status": item["status"],
                "maxima": item["maxima"],
                "basis": item["basis_metadata"],
            }
            for name, item in acceptance["datasets"].items()
        },
    )
    add(
        "trusted CPU float64 outputs frozen as golden",
        all(item.get("golden_report_sha256") for item in acceptance["datasets"].values()),
        acceptance["golden_policy"],
    )
    baseline_evidence = {}
    baselines_ok = True
    for partition, expected in EXPECTED_ROWS.items():
        baseline_evidence[partition] = {}
        for run, expected_rows in expected.items():
            item = final["scientific_baselines"][partition][run]
            roles = item["groups"]["density_role"]
            required_groups = ("source", "density_role", "n_atoms", "composition")
            passed = (
                item["status"] == "passed"
                and item["rows"] == expected_rows
                and item["scf_iterations"] == [[6, -1]]
                and all(item["groups"].get(group) for group in required_groups)
                and roles["trained_perturbed_step"]["difference"]["pooled_coefficient_mae"] > 0
                and roles["ground_state_final"]["difference"]["pooled_coefficient_mae"] == 0
                and bool(item["row_artifacts"])
                and bool(item["outliers_jsonl_gz"])
            )
            baselines_ok &= passed
            baseline_evidence[partition][run] = {
                "passed": passed,
                "rows": item["rows"],
                "scf_iterations": item["scf_iterations"],
                "density_roles": roles,
            }
    add(
        "complete grouped dual-density validation and Test baselines",
        baselines_ok,
        baseline_evidence,
    )
    add(
        "validation freeze precedes exactly one accepted Test funnel",
        final["pipeline_provenance"].get("status") == "passed"
        and final["scientific_baselines"]["frozen_before_test"].get("status") == "frozen",
        final["pipeline_provenance"],
    )
    density = final["density_optimization"]
    add(
        "representative density optimization reports success/residual/cost/failure",
        density.get("status") == "passed"
        and density["groups"]["QM9_perturbed_fock"]["strict_converged"] == 3
        and density["groups"]["QMUGSBin0_perturbed_fock"]["runtime_success"] == 3
        and density["groups"]["QMUGSBin0_perturbed_fock"]["strict_converged"] == 2,
        density["groups"],
    )
    force = final["force_hessian"]
    test100 = force["actual_force_weight_1"]["one_shot_frozen_test100"]
    add(
        "fixed-density scalar force/Hessian/HVP autograd-FD and PBE comparison",
        force.get("status") == "passed"
        and force["fixed_density_validation_smoke"]["full_hessian_all_finite"] is True
        and force["fixed_density_validation_smoke"]["hvp_all_finite"] is True
        and test100["fixed_full_hessian_cases"] == 100
        and test100["hvp_cases"] == 400
        and test100["fixed_autograd_vs_fd_relative_fro_mean"] < 1e-3,
        {
            "validation": force["fixed_density_validation_smoke"],
            "test100": test100,
        },
    )
    boundaries = force["interpretation_boundaries"]
    add(
        "force/Hessian terminology and physical boundary explicit",
        "not force or Hessian models" in boundaries["legacy_graphformer"]
        and "not a conservative physical total-OFDFT Hessian" in boundaries["density_relaxed"]
        and "No physical vibrational conclusion" in boundaries["vibrations"],
        boundaries,
    )
    total = final["total_derivative"]
    stages = total["local_readiness"]["conservative_total_ofdft_plan"]
    add(
        "conservative total-OFDFT staged implementation and unit evidence",
        total["remote_readiness"].get("status") == "audit_complete"
        and total["local_readiness"].get("status") == "audit_complete"
        and total["local_unit_tests"] == {"tests": 25, "errors": 0, "failures": 0, "skipped": 0}
        and [item["stage"] for item in stages] == list(range(6))
        and total["deployment_status"] == "planned_not_deployed_on_recovery_server",
        {"unit_tests": total["local_unit_tests"], "stages": stages},
    )
    failures = final["failed_attempts"]
    add(
        "failed tasks retained and excluded from science",
        failures.get("status") == "audit_complete"
        and all(item["used_as_scientific_result"] is False for item in failures["attempts"]),
        {"attempt_count": failures["attempt_count"], "retained_file_count": failures["retained_file_count"]},
    )
    add(
        "final machine and Markdown reports exist",
        args.final_markdown.is_file() and args.final_markdown.stat().st_size > 0,
        {
            "json": {"path": str(args.final_report.resolve()), "sha256": _sha256(args.final_report)},
            "markdown": {"path": str(args.final_markdown.resolve()), "sha256": _sha256(args.final_markdown)},
        },
    )
    add(
        "handoff updated with recovery acceptance and boundary",
        "Historical-model recovery acceptance and derivative boundary" in handoff_text
        and "final_acceptance_report_20260715.json" in handoff_text
        and "not a conservative physical total-OFDFT Hessian" in handoff_text,
        {"path": str(args.handoff.resolve()), "sha256": _sha256(args.handoff)},
    )
    failed = [item["requirement"] for item in checks if not item["passed"]]
    result = {
        "status": "passed" if not failed else "failed",
        "failed_requirements": failed,
        "checks": checks,
        "final_report": str(args.final_report.resolve()),
        "handoff": str(args.handoff.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "failed": failed}, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
