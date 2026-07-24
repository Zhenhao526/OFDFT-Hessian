#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "wt-pair-ti-production-analysis-v1":
        raise ValueError(f"unsupported TI analysis schema in {path}")
    return report


def spread(values: Iterable[float]) -> float:
    numbers = list(values)
    return max(numbers) - min(numbers)


def phase_failure_reason(sample: Dict[str, Any], phase: str) -> str | None:
    if sample["phase_status"] == f"{phase}_verified":
        return None
    checks = sample.get("phase_gate_checks")
    if phase == "liquid" and isinstance(checks, dict):
        failed = {name for name, passed in checks.items() if not passed}
        if failed in (
            {"late_MSD_slope_gt_0_001_A2_per_step"},
            {"diffusion_MSD_slope_gt_0_001_A2_per_step"},
        ):
            return "liquid_diffusion_slope_gate"
    return "phase_gate"


def summarize(
    reports: List[Dict[str, Any]],
    *,
    max_integral_discard_spread_mev_per_atom: float = 1.0,
    max_window_discard_spread_mev_per_atom: float = 1.0,
    max_window_block_se_mev_per_atom: float = 1.0,
    max_window_half_drift_mev_per_atom: float = 2.0,
) -> Dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two discard-fraction reports are required")
    phases = {str(report["phase"]) for report in reports}
    natoms_values = {int(report["natoms"]) for report in reports}
    if len(phases) != 1 or len(natoms_values) != 1:
        raise ValueError("reports must describe the same phase and atom count")
    phase = phases.pop()
    natoms = natoms_values.pop()
    ordered_reports = sorted(reports, key=lambda report: float(report["discard_fraction"]))
    label_sets = [set(window["label"] for window in report["windows"]) for report in reports]
    if any(labels != label_sets[0] for labels in label_sets[1:]):
        raise ValueError("reports do not contain the same lambda windows")

    report_summary = []
    for report in ordered_reports:
        report_summary.append(
            {
                "discard_fraction": float(report["discard_fraction"]),
                "status": report["status"],
                "checks": report["checks"],
                "delta_f_wt_minus_pair_mev_per_atom": float(
                    report["delta_f_wt_minus_pair_simpson_mev_per_atom"]
                ),
                "block_standard_error_mev_per_atom": float(
                    report["block_standard_error_mev_per_atom"]
                ),
                "half_drift_mev_per_atom": float(report["half_drift_mev_per_atom"]),
                "quadrature_difference_mev_per_atom": float(
                    report["quadrature_difference_mev_per_atom"]
                ),
                "minimum_adjacent_effective_sample_fraction": float(
                    report["minimum_adjacent_effective_sample_fraction"]
                ),
                "maximum_adjacent_overlap_closure_mev_per_atom": float(
                    report["maximum_adjacent_overlap_closure_mev_per_atom"]
                ),
            }
        )

    window_summaries = []
    critical_windows = []
    for label in sorted(label_sets[0]):
        samples = [
            next(window for window in report["windows"] if window["label"] == label)
            for report in ordered_reports
        ]
        lambda_value = float(samples[0]["lambda"])
        du_means = [1000.0 * float(sample["du_mean_ev_system"]) / natoms for sample in samples]
        block_errors = [
            1000.0 * float(sample["du_block_standard_error_ev_system"]) / natoms
            for sample in samples
        ]
        half_drifts = [
            1000.0
            * abs(
                float(sample["du_second_half_ev_system"])
                - float(sample["du_first_half_ev_system"])
            )
            / natoms
            for sample in samples
        ]
        temperatures = [float(sample["temperature_mean_k"]) for sample in samples]
        target_temperature = float(ordered_reports[0]["target_temperature_K"])
        tolerance = float(ordered_reports[0]["temperature_tolerance_k"])
        minimum_distance_gate = float(ordered_reports[0]["minimum_distance_gate_angstrom"])
        reasons = []
        if any(sample["status"] != "complete" for sample in samples):
            reasons.append("incomplete_window")
        phase_reasons = {
            reason
            for sample in samples
            if (reason := phase_failure_reason(sample, phase)) is not None
        }
        reasons.extend(sorted(phase_reasons))
        if min(float(sample["minimum_pair_distance_angstrom"]) for sample in samples) < minimum_distance_gate:
            reasons.append("minimum_distance_gate")
        if max(abs(temperature - target_temperature) for temperature in temperatures) > tolerance:
            reasons.append("temperature_gate")
        if max(block_errors) > max_window_block_se_mev_per_atom:
            reasons.append("window_block_standard_error")
        if max(half_drifts) > max_window_half_drift_mev_per_atom:
            reasons.append("window_half_drift")
        if spread(du_means) > max_window_discard_spread_mev_per_atom:
            reasons.append("window_discard_spread")
        summary = {
            "label": label,
            "lambda": lambda_value,
            "minimum_max_step": min(int(sample["max_step"]) for sample in samples),
            "minimum_pair_distance_angstrom": min(
                float(sample["minimum_pair_distance_angstrom"]) for sample in samples
            ),
            "temperature_mean_range_k": [min(temperatures), max(temperatures)],
            "du_mean_by_discard_mev_per_atom": du_means,
            "du_discard_spread_mev_per_atom": spread(du_means),
            "maximum_window_block_standard_error_mev_per_atom": max(block_errors),
            "maximum_window_half_drift_mev_per_atom": max(half_drifts),
            "critical_reasons": reasons,
        }
        window_summaries.append(summary)
        if reasons:
            critical_windows.append({"label": label, "lambda": lambda_value, "reasons": reasons})

    integrals = [item["delta_f_wt_minus_pair_mev_per_atom"] for item in report_summary]
    integral_spread = spread(integrals)
    all_reports_verified = all(report["status"] == "verified" for report in ordered_reports)
    lambda_refinement_required = any(
        not report["checks"]["quadrature_gate"]
        or not report["checks"]["adjacent_overlap_gate"]
        or not report["checks"]["adjacent_overlap_closure_gate"]
        for report in ordered_reports
    )
    phase_ready = (
        all_reports_verified
        and integral_spread <= max_integral_discard_spread_mev_per_atom
        and not critical_windows
        and not lambda_refinement_required
    )
    middle = min(report_summary, key=lambda item: abs(item["discard_fraction"] - 0.5))
    return {
        "schema": "wt-ti-discard-convergence-summary-v1",
        "status": "verified" if phase_ready else "extension_or_refinement_required",
        "phase": phase,
        "natoms": natoms,
        "reports": report_summary,
        "integral_consensus_mev_per_atom": middle[
            "delta_f_wt_minus_pair_mev_per_atom"
        ],
        "integral_discard_spread_mev_per_atom": integral_spread,
        "maximum_integral_discard_spread_mev_per_atom": (
            max_integral_discard_spread_mev_per_atom
        ),
        "lambda_refinement_required": lambda_refinement_required,
        "critical_windows": critical_windows,
        "windows": window_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TI convergence across multiple discard fractions"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-integral-discard-spread", type=float, default=1.0)
    parser.add_argument("--max-window-discard-spread", type=float, default=1.0)
    parser.add_argument("--max-window-block-se", type=float, default=1.0)
    parser.add_argument("--max-window-half-drift", type=float, default=2.0)
    args = parser.parse_args()
    result = summarize(
        [load_report(path.resolve()) for path in args.reports],
        max_integral_discard_spread_mev_per_atom=args.max_integral_discard_spread,
        max_window_discard_spread_mev_per_atom=args.max_window_discard_spread,
        max_window_block_se_mev_per_atom=args.max_window_block_se,
        max_window_half_drift_mev_per_atom=args.max_window_half_drift,
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
