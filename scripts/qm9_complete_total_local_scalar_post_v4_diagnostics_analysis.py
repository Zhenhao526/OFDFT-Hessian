#!/usr/bin/env python3
"""Merge frozen post-v4 Jacobian and LBFGS stable5 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_common(
    summary: dict[str, Any], manifest: dict[str, Any], *, kind: str
) -> None:
    if summary.get("test100_accessed") is not False:
        raise ValueError(f"{kind} does not certify frozen Test100")
    if int(summary.get("test100_evaluations_used", -1)) != 0:
        raise ValueError(f"{kind} Test100 count is nonzero")
    if summary.get("selected_parent_ids") != manifest["parents"]:
        raise ValueError(f"{kind} stable5 parent drift")
    observed_protocol = summary.get("source_protocol", summary.get("protocol"))
    observed_protocol_sha256 = summary.get(
        "source_protocol_sha256", summary.get("protocol_sha256")
    )
    observed = {
        "source_protocol": observed_protocol,
        "source_protocol_sha256": observed_protocol_sha256,
        "source_checkpoint": summary.get("source_checkpoint"),
        "source_checkpoint_sha256": summary.get("source_checkpoint_sha256"),
    }
    for key, value in observed.items():
        if value != manifest[key]:
            raise ValueError(f"{kind} {key} drift")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _json(args.manifest)
    if manifest.get("protocol_id") != "qm9_complete_total_local_scalar_post_v4_diagnostics_v1":
        raise ValueError("unexpected diagnostic manifest")
    if manifest.get("test100_accessed") is not False:
        raise ValueError("diagnostic manifest does not freeze Test100")
    if int(manifest.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("diagnostic manifest Test100 count is nonzero")
    manifest_sha256 = _sha256(args.manifest)

    jacobian = _json(args.jacobian_summary)
    _validate_common(jacobian, manifest, kind="Jacobian audit")
    if jacobian.get("arm_id") != manifest["source_arm_id"]:
        raise ValueError("Jacobian source arm drift")
    expected_jacobian = manifest["jacobian"]
    if int(jacobian.get("iterations_requested", -1)) != int(expected_jacobian["iterations"]):
        raise ValueError("Jacobian iteration drift")
    if float(jacobian.get("fd_relative_step")) != float(expected_jacobian["fd_relative_step"]):
        raise ValueError("Jacobian FD step drift")

    lbfgs: dict[str, dict[str, Any]] = {}
    for objective, path in (
        ("joint", args.joint_summary),
        ("hessian_only", args.hessian_summary),
    ):
        summary = _json(path)
        _validate_common(summary, manifest, kind=f"{objective} LBFGS")
        if summary.get("source_arm_id") != manifest["source_arm_id"]:
            raise ValueError(f"{objective} LBFGS source arm drift")
        if summary.get("objective") != objective:
            raise ValueError(f"{objective} LBFGS objective drift")
        if summary.get("audit_manifest_sha256") != manifest_sha256:
            raise ValueError(f"{objective} LBFGS audit-manifest drift")
        if int(summary.get("iterations", -1)) != int(
            manifest["lbfgs"][objective]["iterations"]
        ):
            raise ValueError(f"{objective} LBFGS iteration drift")
        if objective == "hessian_only" and bool(summary.get("parent_cv_authorized")):
            raise ValueError("Hessian-only LBFGS improperly authorizes parent-CV")
        lbfgs[objective] = summary

    vibrational = _json(args.vibrational_summary)
    if vibrational.get("test100_accessed") is not False:
        raise ValueError("vibrational summary does not certify frozen Test100")
    if int(vibrational.get("test100_evaluations_used", -1)) != 0:
        raise ValueError("vibrational summary Test100 count is nonzero")
    vibration_rows = {
        str(row["result"]): row for row in vibrational.get("summaries", [])
    }
    for name in ("original_A", "lbfgs_joint", "lbfgs_hessian_only"):
        if name not in vibration_rows:
            raise ValueError(f"vibrational summary is missing {name}")
    vibration_gate = manifest["vibrational_gate"]
    vibration_checks: dict[str, dict[str, bool]] = {}
    for objective, result_name in (
        ("joint", "lbfgs_joint"),
        ("hessian_only", "lbfgs_hessian_only"),
    ):
        row = vibration_rows[result_name]
        checks = {
            "molecule_count": int(row["n_molecules"])
            == int(vibration_gate["required_molecule_count"]),
            "frequency_mae": float(row["mean_frequency_mae_cm-1"])
            <= float(vibration_gate["mean_frequency_mae_cm-1_max"]),
            "imaginary_mode_count": abs(
                int(row["total_model_imaginary_modes"])
                - int(row["total_pbe_imaginary_modes"])
            )
            <= int(vibration_gate["absolute_total_imaginary_mode_count_error_max"]),
            "mode_overlap": float(row["mean_mode_overlap"])
            >= float(vibration_gate["mean_mode_overlap_min"]),
        }
        vibration_checks[objective] = {**checks, "passed": all(checks.values())}

    jacobian_distribution = jacobian["linearized_hessian_relative_frobenius"]
    jacobian_pass = bool(
        float(jacobian_distribution["median"]) <= 0.05
        and float(jacobian_distribution["max"]) <= 0.05
    )
    hessian_final = lbfgs["hessian_only"]["final"]
    hessian_distribution = hessian_final["hessian_relative_frobenius"]
    hessian_pass = bool(
        float(hessian_distribution["median"]) <= 0.05
        and float(hessian_distribution["max"]) <= 0.05
    )
    joint_matrix_ef_pass = bool(
        lbfgs["joint"]["capacity_gate_passed"]
        and lbfgs["joint"]["parent_cv_authorized"]
    )
    joint_pass = bool(joint_matrix_ef_pass and vibration_checks["joint"]["passed"])
    if joint_pass:
        diagnosis = "joint_stable5_capacity_pass"
    elif joint_matrix_ef_pass:
        diagnosis = "joint_matrix_energy_force_pass_but_vibrational_gate_failed"
    elif hessian_pass:
        diagnosis = "hessian_representation_pass_but_joint_multitask_gate_failed"
    elif jacobian_pass:
        diagnosis = "local_linearized_range_pass_but_nonlinear_optimization_failed"
    else:
        diagnosis = "no_five_percent_capacity_evidence_with_preregistered_local_diagnostics"

    jacobian_curve = _jsonl(args.jacobian_metrics)
    joint_curve = _jsonl(args.joint_metrics)
    hessian_curve = _jsonl(args.hessian_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(
        [row["iteration"] for row in jacobian_curve],
        [row["relative_residual_norm"] for row in jacobian_curve],
        marker="o",
        markersize=2,
    )
    axes[0].set_title("Jacobian CGLS residual")
    axes[0].set_xlabel("CGLS iteration")
    axes[0].set_ylabel("Residual / initial")
    for objective, curve in (("joint", joint_curve), ("hessian_only", hessian_curve)):
        axes[1].plot(
            [row["iteration"] for row in curve],
            [row["median_relative_frobenius"] for row in curve],
            label=f"{objective}: median",
        )
        axes[1].plot(
            [row["iteration"] for row in curve],
            [row["max_relative_frobenius"] for row in curve],
            linestyle="--",
            label=f"{objective}: max",
        )
    axes[1].axhline(0.05, color="black", linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_title("LBFGS Hessian capacity")
    axes[1].set_xlabel("LBFGS iteration")
    axes[1].set_ylabel("Relative Frobenius")
    axes[1].legend(fontsize=7)
    axes[2].plot(
        [row["iteration"] for row in joint_curve],
        [row["energy_median_ratio_to_source"] for row in joint_curve],
        label="energy",
    )
    axes[2].plot(
        [row["iteration"] for row in joint_curve],
        [row["force_median_ratio_to_source"] for row in joint_curve],
        label="force",
    )
    axes[2].axhline(1.05, color="black", linestyle=":")
    axes[2].set_title("Joint LBFGS E/F gates")
    axes[2].set_xlabel("LBFGS iteration")
    axes[2].set_ylabel("Median error / source")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    plot_path = args.output_dir / "post_v4_diagnostics.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    rows = [
        {
            "method": "jacobian_linearized",
            **jacobian_distribution,
            "energy_ratio": None,
            "force_ratio": None,
            "five_percent_hessian_pass": jacobian_pass,
            "frequency_mae_cm-1": None,
            "imaginary_mode_count_error": None,
            "mean_mode_overlap": None,
            "vibrational_gate_passed": None,
        },
        {
            "method": "lbfgs_joint",
            **lbfgs["joint"]["final"]["hessian_relative_frobenius"],
            "energy_ratio": lbfgs["joint"]["final"]["energy_median_ratio_to_source"],
            "force_ratio": lbfgs["joint"]["final"]["force_median_ratio_to_source"],
            "five_percent_hessian_pass": bool(
                float(lbfgs["joint"]["final"]["hessian_relative_frobenius"]["median"]) <= 0.05
                and float(lbfgs["joint"]["final"]["hessian_relative_frobenius"]["max"]) <= 0.05
            ),
            "frequency_mae_cm-1": vibration_rows["lbfgs_joint"]["mean_frequency_mae_cm-1"],
            "imaginary_mode_count_error": int(vibration_rows["lbfgs_joint"]["total_model_imaginary_modes"])
            - int(vibration_rows["lbfgs_joint"]["total_pbe_imaginary_modes"]),
            "mean_mode_overlap": vibration_rows["lbfgs_joint"]["mean_mode_overlap"],
            "vibrational_gate_passed": vibration_checks["joint"]["passed"],
        },
        {
            "method": "lbfgs_hessian_only",
            **hessian_distribution,
            "energy_ratio": hessian_final["energy_median_ratio_to_source"],
            "force_ratio": hessian_final["force_median_ratio_to_source"],
            "five_percent_hessian_pass": hessian_pass,
            "frequency_mae_cm-1": vibration_rows["lbfgs_hessian_only"]["mean_frequency_mae_cm-1"],
            "imaginary_mode_count_error": int(vibration_rows["lbfgs_hessian_only"]["total_model_imaginary_modes"])
            - int(vibration_rows["lbfgs_hessian_only"]["total_pbe_imaginary_modes"]),
            "mean_mode_overlap": vibration_rows["lbfgs_hessian_only"]["mean_mode_overlap"],
            "vibrational_gate_passed": vibration_checks["hessian_only"]["passed"],
        },
    ]
    csv_path = args.output_dir / "diagnostic_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "definition": "Read-only stable5 merge of a frozen post-v4 Jacobian range audit and joint/Hessian-only LBFGS ceilings.",
        "manifest": args.manifest.resolve().as_posix(),
        "manifest_sha256": manifest_sha256,
        "source_arm_id": manifest["source_arm_id"],
        "source_checkpoint": manifest["source_checkpoint"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "jacobian_linearized_hessian_relative_frobenius": jacobian_distribution,
        "jacobian_five_percent_pass": jacobian_pass,
        "jacobian_iterations_completed": jacobian["iterations_completed"],
        "jacobian_fd_directional_stability": jacobian["fd_directional_stability"],
        "joint_lbfgs_final": lbfgs["joint"]["final"],
        "joint_lbfgs_matrix_energy_force_gate_passed": joint_matrix_ef_pass,
        "joint_lbfgs_vibrational_metrics": vibration_rows["lbfgs_joint"],
        "joint_lbfgs_vibrational_gate": vibration_checks["joint"],
        "joint_lbfgs_capacity_passed": joint_pass,
        "hessian_only_lbfgs_final": hessian_final,
        "hessian_only_five_percent_pass": hessian_pass,
        "hessian_only_vibrational_metrics": vibration_rows["lbfgs_hessian_only"],
        "hessian_only_vibrational_gate": vibration_checks["hessian_only"],
        "original_A_vibrational_metrics": vibration_rows["original_A"],
        "diagnosis": diagnosis,
        "parent_cv_design_authorized": joint_pass,
        "parent_cv_submitted": False,
        "validation_accessed": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "input_artifacts": {
            "jacobian_summary_sha256": _sha256(args.jacobian_summary),
            "joint_summary_sha256": _sha256(args.joint_summary),
            "hessian_summary_sha256": _sha256(args.hessian_summary),
            "vibrational_summary_sha256": _sha256(args.vibrational_summary),
        },
        "summary_csv": csv_path.resolve().as_posix(),
        "summary_csv_sha256": _sha256(csv_path),
        "plot": plot_path.resolve().as_posix(),
        "plot_sha256": _sha256(plot_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jacobian-summary", type=Path, required=True)
    parser.add_argument("--jacobian-metrics", type=Path, required=True)
    parser.add_argument("--joint-summary", type=Path, required=True)
    parser.add_argument("--joint-metrics", type=Path, required=True)
    parser.add_argument("--hessian-summary", type=Path, required=True)
    parser.add_argument("--hessian-metrics", type=Path, required=True)
    parser.add_argument("--vibrational-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
