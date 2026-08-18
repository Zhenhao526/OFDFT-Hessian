#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    reports: list[Path],
    phase_analysis: Path,
    expected_phase: str,
    target_kedf: str,
    maximum_discard_spread: float,
) -> dict:
    if len(reports) != 3:
        raise ValueError("exactly three discard reports are required")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    phases = json.loads(phase_analysis.read_text(encoding="utf-8"))
    natoms = {int(document["natoms"]) for document in documents}
    temperatures = {float(document["target_temperature_k"]) for document in documents}
    references = {str(document["reference_label"]) for document in documents}
    if len(natoms) != 1 or len(temperatures) != 1 or len(references) != 1:
        raise ValueError("discard reports do not describe one common TI leg")
    integrals = [
        float(document["delta_f_pair_minus_reference_simpson_mev_per_atom"])
        for document in documents
    ]
    spread = max(integrals) - min(integrals)
    checks = {
        "three_discard_reports_verified": all(
            document.get("status") == "verified" for document in documents
        ),
        "discard_spread_within_gate": spread <= maximum_discard_spread,
        "target_phase_verified": phases.get("status") == f"{expected_phase}_verified",
        "target_kedf_declared": target_kedf == "wt",
    }
    return {
        "schema": "kedf-classical-reference-leg-summary-v1",
        "status": "verified" if all(checks.values()) else "needs_extension",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_kedf": target_kedf,
        "expected_phase": expected_phase,
        "reference_label": references.pop(),
        "natoms": natoms.pop(),
        "temperature_k": temperatures.pop(),
        "checks": checks,
        "discard_integrals_mev_per_atom": integrals,
        "discard_spread_mev_per_atom": spread,
        "integral_discard_spread_mev_per_atom": spread,
        "maximum_discard_spread_mev_per_atom": maximum_discard_spread,
        "maximum_block_standard_error_mev_per_atom": max(
            float(document["block_standard_error_mev_per_atom"])
            for document in documents
        ),
        "maximum_half_drift_mev_per_atom": max(
            float(document["half_drift_mev_per_atom"]) for document in documents
        ),
        "maximum_quadrature_difference_mev_per_atom": max(
            float(document["quadrature_difference_mev_per_atom"])
            for document in documents
        ),
        "minimum_adjacent_effective_sample_fraction": min(
            float(document["minimum_adjacent_effective_sample_fraction"])
            for document in documents
        ),
        "maximum_adjacent_closure_mev_per_atom": max(
            float(document["maximum_adjacent_closure_mev_per_atom"])
            for document in documents
        ),
        "provenance": {
            "reports": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in reports
            ],
            "phase_analysis": {
                "path": str(phase_analysis.resolve()),
                "sha256": sha256(phase_analysis),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--phase-analysis", type=Path, required=True)
    parser.add_argument("--expected-phase", choices=("solid", "liquid"), required=True)
    parser.add_argument("--target-kedf", required=True)
    parser.add_argument("--maximum-discard-spread", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.reports,
        args.phase_analysis,
        args.expected_phase,
        args.target_kedf,
        args.maximum_discard_spread,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "verified":
        raise SystemExit("classical reference TI convergence gate failed")


if __name__ == "__main__":
    main()
