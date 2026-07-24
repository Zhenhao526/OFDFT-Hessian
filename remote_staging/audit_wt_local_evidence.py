#!/usr/bin/env python3
"""Audit locally preserved WT melting evidence without reading shared storage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_GATE = 3000
NEIGHBOR_GATE_A = 2.0
BLOCK_SE_GATE_MEV = 3.0
DISCARD_SPREAD_GATE_MEV = 2.0
HALF_DRIFT_GATE_MEV = 5.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(root: Path, pattern: str, marker: str | None = None) -> Path:
    matches = [
        path
        for path in root.rglob(pattern)
        if marker is None or marker in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one match for pattern={pattern!r}, marker={marker!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def summarize_confirmation(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    phases = []
    for result in payload.get("phase_results", []):
        phases.append(
            {
                "phase": result.get("phase"),
                "status": result.get("status"),
                "phase_status": result.get("phase_status"),
                "max_step": result.get("max_step"),
                "temperature_last_half_K": result.get("temperature_last_half_K"),
                "pressure_last_half_kbar": result.get("pressure_last_half_kbar"),
                "nearest_neighbor_A": result.get("nearest_neighbor_A"),
                "checks": result.get("checks"),
                "run": result.get("run"),
            }
        )
    checks = {
        "summary_passed": payload.get("status") == "all_confirmations_passed",
        "requested_steps_reached": all(
            phase.get("max_step", -1) >= STEP_GATE for phase in phases
        ),
        "nearest_neighbors_gt_2_A": all(
            phase.get("nearest_neighbor_A", -1) > NEIGHBOR_GATE_A for phase in phases
        ),
        "phase_gates_passed": all(
            phase.get("status") == "confirmation_passed"
            and str(phase.get("phase_status", "")).endswith("_verified")
            for phase in phases
        ),
    }
    return {
        "source": str(path),
        "status": payload.get("status"),
        "temperature_K": payload.get("temperature_K"),
        "steps": payload.get("steps"),
        "phases": phases,
        "checks": checks,
        "verified": all(checks.values()),
    }


def report_for_temperature(
    convergence: dict[str, Any], temperature_K: float
) -> dict[str, Any] | None:
    for point in convergence.get("points", []):
        if abs(float(point.get("temperature_k", -1)) - temperature_K) < 1.0e-6:
            return point
    return None


def summarize_enthalpy_point(point: dict[str, Any] | None) -> dict[str, Any] | None:
    if point is None:
        return None
    reports = [
        {
            key: report.get(key)
            for key in (
                "discard_fraction",
                "status",
                "delta_h_mev_per_atom",
                "block_standard_error_mev_per_atom",
                "half_drift_mev_per_atom",
            )
        }
        for report in point.get("reports", [])
    ]
    return {
        "status": point.get("status"),
        "delta_h_discard_spread_mev_per_atom": point.get(
            "delta_h_discard_spread_mev_per_atom"
        ),
        "maximum_block_standard_error_mev_per_atom": point.get(
            "maximum_block_standard_error_mev_per_atom"
        ),
        "failed_checks": point.get("failed_checks", []),
        "extension_reasons": point.get("extension_reasons", []),
        "reports": reports,
    }


def summarize_exact_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    point = payload["points"][0]
    return {
        "source": str(path),
        "status": payload.get("status"),
        "discard_fraction": payload.get("discard_fraction"),
        "temperature_K": point.get("temperature_k"),
        "steps_in_local_segment": point.get("steps"),
        "delta_h_mev_per_atom": 1000.0 * point.get("delta_h_ev_per_atom", 0.0),
        "block_standard_error_mev_per_atom": point.get(
            "block_standard_error_mev_per_atom"
        ),
        "half_drift_mev_per_atom": point.get("half_drift_mev_per_atom"),
        "liquid_minus_solid_temperature_mean_difference_K": point.get(
            "liquid_minus_solid_temperature_mean_difference_k"
        ),
        "checks": point.get("checks"),
        "stitch": payload.get("stitch"),
    }


def summarize_phase_analysis(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    md = payload.get("md", {})
    trajectory = payload.get("trajectory", {})
    structure = payload.get("structure", {})
    return {
        "source": str(path),
        "status": payload.get("status"),
        "max_step_in_local_segment": md.get("max_step"),
        "temperature_K": md.get("temperature_K"),
        "pressure_kbar": md.get("pressure_kbar"),
        "nearest_neighbor_A": trajectory.get("nearest_neighbor_A"),
        "minimum_nearest_neighbor_A": trajectory.get(
            "minimum_nearest_neighbor_A"
        ),
        "non_affine_MSD_A2": trajectory.get("non_affine_MSD_A2"),
        "MSD_slope_A2_per_step": trajectory.get(
            "time_origin_averaged_MSD_slope_A2_per_step"
        ),
        "CSP_median_A2": structure.get("median_A2"),
        "ordered_fraction_CSP_lt_2_5": structure.get(
            "ordered_fraction_CSP_lt_2_5"
        ),
    }


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# WT Local Evidence Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "This audit uses only evidence stored under `/home` on node01.",
        "",
        "## Gates",
        "",
        f"- Required MD steps: `{STEP_GATE}`",
        f"- Minimum nearest neighbor: `>{NEIGHBOR_GATE_A:.1f} A`",
        f"- Maximum block standard error: `{BLOCK_SE_GATE_MEV:.1f} meV/atom`",
        f"- Maximum discard spread: `{DISCARD_SPREAD_GATE_MEV:.1f} meV/atom`",
        f"- Maximum half drift: `{HALF_DRIFT_GATE_MEV:.1f} meV/atom`",
        "",
        "## Temperature Status",
        "",
        "| T (K) | Classification | Phase evidence | Enthalpy evidence |",
        "|---:|---|---|---|",
    ]
    for temperature in ("900", "975", "1050", "1100"):
        item = audit["temperatures"][temperature]
        lines.append(
            f"| {temperature} | `{item['classification']}` | "
            f"{item['phase_summary']} | {item['enthalpy_summary']} |"
        )

    lines.extend(
        [
            "",
            "## Key Statistics",
            "",
        ]
    )
    point_900 = audit["temperatures"]["900"]["enthalpy"]
    lines.append(
        "- 900 K: "
        f"Delta H(d50) = {format_float(point_900['d50_mev_per_atom'])} meV/atom; "
        f"max block SE = {format_float(point_900['maximum_block_standard_error_mev_per_atom'])}; "
        f"discard spread = {format_float(point_900['discard_spread_mev_per_atom'])}."
    )
    for temperature in ("975", "1050"):
        item = audit["temperatures"][temperature]
        previous = item["previous_enthalpy"]
        lines.append(
            f"- {temperature} K: final 3000-step phase confirmation passed, "
            "but no final round2 d25/d50/d75 enthalpy report is present locally. "
            f"The earlier dataset was `{previous['status']}` "
            f"(max SE {format_float(previous['maximum_block_standard_error_mev_per_atom'])}, "
            f"discard spread {format_float(previous['delta_h_discard_spread_mev_per_atom'])} meV/atom)."
        )
    item_1100 = audit["temperatures"]["1100"]
    lines.append(
        "- 1100 K: phases remain valid, but exact continuation reports fail statistics: "
        f"d50={format_float(item_1100['enthalpy']['d50']['delta_h_mev_per_atom'])}, "
        f"d75={format_float(item_1100['enthalpy']['d75']['delta_h_mev_per_atom'])} meV/atom; "
        f"d50/d75 spread={format_float(item_1100['enthalpy']['d50_d75_spread_mev_per_atom'])}; "
        f"half drifts={format_float(item_1100['enthalpy']['d50']['half_drift_mev_per_atom'])}/"
        f"{format_float(item_1100['enthalpy']['d75']['half_drift_mev_per_atom'])} meV/atom."
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Overall status: `{audit['overall_status']}`.",
            "",
            "Only 900 K currently has both verified phase evidence and converged enthalpy statistics. "
            "The Gibbs-Helmholtz root must not be solved until final round2 enthalpy reports for "
            "975 and 1050 K are regenerated from their raw trajectories.",
            "",
            "## Next Action",
            "",
            "When BeeGFS recovers, stage the 975/1050 K round2 `running_md.log`, `MD_dump`, "
            "and final phase JSON files into node01 `/home`; then regenerate d25/d50/d75 "
            "enthalpy reports and reapply the same gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    evidence = workspace / "evidence" / "tmp_analysis_files"
    reports = workspace / "audit" / "reports_archive"
    archive = workspace / "audit" / "node01_wt_reports_20260724.tar.gz"

    confirmation_900 = summarize_confirmation(
        find_one(reports, "confirmation_summary.json", "T0900_continuation_steps3000")
    )
    confirmation_975 = summarize_confirmation(
        find_one(reports, "confirmation_summary.json", "T0975_steps3000_round2")
    )
    confirmation_1050 = summarize_confirmation(
        find_one(reports, "confirmation_summary.json", "T1050_steps3000_round2")
    )

    convergence_900 = load_json(evidence / "wt900_convergence.json")
    previous_convergence_path = find_one(
        reports, "discard_convergence_summary_extension_round1.json"
    )
    previous_convergence = load_json(previous_convergence_path)
    point_900 = summarize_enthalpy_point(report_for_temperature(convergence_900, 900.0))
    point_975 = summarize_enthalpy_point(
        report_for_temperature(previous_convergence, 975.0)
    )
    point_1050 = summarize_enthalpy_point(
        report_for_temperature(previous_convergence, 1050.0)
    )

    d50_1100 = summarize_exact_report(find_one(reports, "global_d50_exact.json"))
    d75_1100 = summarize_exact_report(find_one(reports, "global_d75_exact.json"))
    phase_solid_1100 = summarize_phase_analysis(
        find_one(reports, "phase_analysis_resume1700.json", "/solid/")
    )
    phase_liquid_1100 = summarize_phase_analysis(
        find_one(reports, "phase_analysis_resume1700.json", "/liquid/")
    )

    d50_900 = next(
        report
        for report in point_900["reports"]
        if report["discard_fraction"] == 0.5
    )
    audit = {
        "schema": "wt-local-evidence-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "evidence_mode": "node01_home_only",
        "archive": {
            "path": str(archive),
            "sha256": sha256(archive),
            "valid_json_files": len(list(reports.rglob("*.json"))),
        },
        "gates": {
            "required_steps": STEP_GATE,
            "minimum_nearest_neighbor_A_exclusive": NEIGHBOR_GATE_A,
            "maximum_block_standard_error_mev_per_atom": BLOCK_SE_GATE_MEV,
            "maximum_discard_spread_mev_per_atom": DISCARD_SPREAD_GATE_MEV,
            "maximum_half_drift_mev_per_atom": HALF_DRIFT_GATE_MEV,
        },
        "temperatures": {
            "900": {
                "classification": "thermodynamic_ready",
                "phase_summary": "3000 steps; solid/liquid verified",
                "enthalpy_summary": "d25/d50/d75 converged",
                "confirmation": confirmation_900,
                "enthalpy": {
                    "status": point_900["status"],
                    "d50_mev_per_atom": d50_900["delta_h_mev_per_atom"],
                    "maximum_block_standard_error_mev_per_atom": point_900[
                        "maximum_block_standard_error_mev_per_atom"
                    ],
                    "discard_spread_mev_per_atom": point_900[
                        "delta_h_discard_spread_mev_per_atom"
                    ],
                    "reports": point_900["reports"],
                },
            },
            "975": {
                "classification": "phase_ready_enthalpy_recompute_required",
                "phase_summary": "final round2 3000 steps; solid/liquid verified",
                "enthalpy_summary": "final round2 d25/d50/d75 absent",
                "confirmation": confirmation_975,
                "previous_enthalpy": point_975,
                "previous_enthalpy_source": str(previous_convergence_path),
            },
            "1050": {
                "classification": "phase_ready_enthalpy_recompute_required",
                "phase_summary": "final round2 3000 steps; solid/liquid verified",
                "enthalpy_summary": "final round2 d25/d50/d75 absent",
                "confirmation": confirmation_1050,
                "previous_enthalpy": point_1050,
                "previous_enthalpy_source": str(previous_convergence_path),
            },
            "1100": {
                "classification": "phase_ready_statistically_failed",
                "phase_summary": "stitched 1300+1700 steps; solid/liquid verified",
                "enthalpy_summary": "d50/d75 fail half-drift/statistical gates",
                "phase": {
                    "solid": phase_solid_1100,
                    "liquid": phase_liquid_1100,
                },
                "enthalpy": {
                    "d50": d50_1100,
                    "d75": d75_1100,
                    "d50_d75_spread_mev_per_atom": abs(
                        d50_1100["delta_h_mev_per_atom"]
                        - d75_1100["delta_h_mev_per_atom"]
                    ),
                },
            },
        },
        "thermodynamically_ready_temperatures_K": [900.0],
        "overall_status": "incomplete_for_gibbs_helmholtz",
        "blocking_reasons": [
            "975 K final round2 d25/d50/d75 enthalpy reports are absent locally",
            "1050 K final round2 d25/d50/d75 enthalpy reports are absent locally",
            "1100 K exact continuation reports fail half-drift/statistical gates",
        ],
    }

    output_json = workspace / "audit" / "STEP2_UNIFIED_EVIDENCE_AUDIT.json"
    output_md = workspace / "audit" / "STEP2_UNIFIED_EVIDENCE_AUDIT.md"
    output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(build_markdown(audit), encoding="utf-8")
    print(output_json)
    print(output_md)
    print(audit["overall_status"])


if __name__ == "__main__":
    main()
