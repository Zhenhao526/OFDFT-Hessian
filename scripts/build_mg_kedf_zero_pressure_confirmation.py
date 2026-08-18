#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase(run: Path, expected_phase: str, target_kedf: str) -> dict:
    gate_path = run / "physical_gate.json"
    manifest_path = run / "mg_phase_md_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if gate.get("status") != "physical_verified":
        raise ValueError(f"{expected_phase} physical gate did not pass")
    if gate.get("method") != target_kedf or manifest.get("method") != target_kedf:
        raise ValueError(f"{expected_phase} method does not match {target_kedf}")
    if gate.get("expected_phase") != expected_phase:
        raise ValueError(f"expected {expected_phase}, got {gate.get('expected_phase')}")
    if gate.get("hcp_phase_status") != f"{expected_phase}_verified":
        raise ValueError(f"{expected_phase} hcp phase gate did not pass")
    if not all(bool(value) for value in gate.get("checks", {}).values()):
        raise ValueError(f"{expected_phase} physical checks are incomplete")
    temperature = float(manifest["temperature_K"])
    if abs(float(gate["temperature_last_half_K"]["mean"]) - temperature) > 20.0:
        raise ValueError(f"{expected_phase} temperature gate did not pass")
    pressure = gate["pressure_last_half_kbar"]
    if abs(float(pressure["mean"])) > 2.5:
        raise ValueError(f"{expected_phase} pressure gate did not pass")
    return {
        "phase": expected_phase,
        "status": "confirmation_passed",
        "run": str(run.resolve()),
        "volume_per_atom_A3": float(manifest["volume_per_atom_A3"]),
        "temperature_last_half_K": gate["temperature_last_half_K"],
        "pressure_last_half_kbar": pressure,
        "minimum_nearest_neighbor_A": float(gate["minimum_nearest_neighbor_A"]),
        "phase_status": gate["hcp_phase_status"],
        "physical_gate_sha256": sha256(gate_path),
        "run_manifest_sha256": sha256(manifest_path),
    }


def build(solid_run: Path, liquid_run: Path, target_kedf: str) -> dict:
    phases = [
        load_phase(solid_run, "solid", target_kedf),
        load_phase(liquid_run, "liquid", target_kedf),
    ]
    temperatures = {
        float(
            json.loads((run / "mg_phase_md_manifest.json").read_text(encoding="utf-8"))[
                "temperature_K"
            ]
        )
        for run in (solid_run, liquid_run)
    }
    if len(temperatures) != 1:
        raise ValueError("solid and liquid confirmations use different temperatures")
    return {
        "schema": "kedf-zero-pressure-confirmation-merged-v1",
        "status": "all_confirmations_passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "element": "Mg",
        "target_kedf": target_kedf,
        "temperature_K": temperatures.pop(),
        "phase_results": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solid-run", type=Path, required=True)
    parser.add_argument("--liquid-run", type=Path, required=True)
    parser.add_argument("--target-kedf", choices=("wt", "xwm", "lkt"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.solid_run, args.liquid_run, args.target_kedf)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = args.out / "zero_pressure_confirmation_summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "SHA256SUMS").write_text(
        f"{sha256(summary)}  zero_pressure_confirmation_summary.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
