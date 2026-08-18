#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mpn_melting.free_energy import exponential_free_energy_difference


def analyze_bridge(
    windows: List[Path],
    *,
    du_key: str,
    discard_fraction: float,
    minimum_overlap_ess: float,
    maximum_overlap_closure_mev_per_atom: float,
) -> Dict[str, Any]:
    loaded = []
    natoms_values = set()
    temperatures = set()
    for window in windows:
        rows = [
            json.loads(line)
            for line in (window / "trajectory.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        summary = json.loads((window / "summary.json").read_text(encoding="utf-8"))
        start = int(len(rows) * discard_fraction)
        production = [float(row[du_key]) for row in rows[start:]]
        if not production:
            raise ValueError(f"no production samples in {window}")
        loaded.append(
            {
                "lambda": float(rows[0]["lambda"]),
                "path": str(window.resolve()),
                "total_samples": len(rows),
                "production_samples": len(production),
                "production_du": production,
            }
        )
        natoms_values.add(int(summary["natoms"]))
        temperatures.add(float(summary["target_temperature_k"]))

    if len(natoms_values) != 1:
        raise ValueError(f"inconsistent atom counts: {sorted(natoms_values)}")
    if len(temperatures) != 1:
        raise ValueError(f"inconsistent temperatures: {sorted(temperatures)}")
    loaded.sort(key=lambda item: item["lambda"])
    if len(loaded) < 3:
        raise ValueError("a local overlap bridge requires at least three windows")

    natoms = natoms_values.pop()
    temperature = temperatures.pop()
    pairs = []
    for left, right in zip(loaded, loaded[1:]):
        delta_lambda = right["lambda"] - left["lambda"]
        forward = exponential_free_energy_difference(
            [delta_lambda * value * natoms for value in left["production_du"]],
            temperature,
        )
        reverse = exponential_free_energy_difference(
            [-delta_lambda * value * natoms for value in right["production_du"]],
            temperature,
        )
        pairs.append(
            {
                "lambda_left": left["lambda"],
                "lambda_right": right["lambda"],
                "forward_delta_f_mev_per_atom": 1000.0
                * forward["delta_f_ev"]
                / natoms,
                "reverse_delta_f_mev_per_atom": 1000.0
                * reverse["delta_f_ev"]
                / natoms,
                "closure_mev_per_atom": 1000.0
                * abs(forward["delta_f_ev"] + reverse["delta_f_ev"])
                / natoms,
                "forward_effective_sample_fraction": forward[
                    "effective_sample_fraction"
                ],
                "reverse_effective_sample_fraction": reverse[
                    "effective_sample_fraction"
                ],
            }
        )

    minimum_ess = min(
        min(
            pair["forward_effective_sample_fraction"],
            pair["reverse_effective_sample_fraction"],
        )
        for pair in pairs
    )
    maximum_closure = max(pair["closure_mev_per_atom"] for pair in pairs)
    checks = {
        "adjacent_overlap_ess": minimum_ess >= minimum_overlap_ess,
        "adjacent_overlap_closure": (
            maximum_closure <= maximum_overlap_closure_mev_per_atom
        ),
    }
    for item in loaded:
        del item["production_du"]
    return {
        "schema": "mpn-local-overlap-bridge-analysis-v1",
        "status": "verified" if all(checks.values()) else "overlap_gate_failed",
        "discard_fraction": discard_fraction,
        "natoms": natoms,
        "target_temperature_k": temperature,
        "windows": loaded,
        "adjacent_overlap": pairs,
        "minimum_adjacent_effective_sample_fraction": minimum_ess,
        "maximum_adjacent_closure_mev_per_atom": maximum_closure,
        "checks": checks,
        "gates": {
            "minimum_adjacent_effective_sample_fraction": minimum_overlap_ess,
            "maximum_adjacent_closure_mev_per_atom": (
                maximum_overlap_closure_mev_per_atom
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a local TI overlap bridge")
    parser.add_argument("--windows", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--du-key", required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-overlap-ess", type=float, default=0.05)
    parser.add_argument("--max-overlap-closure", type=float, default=2.0)
    args = parser.parse_args()

    result = analyze_bridge(
        args.windows,
        du_key=args.du_key,
        discard_fraction=args.discard_fraction,
        minimum_overlap_ess=args.minimum_overlap_ess,
        maximum_overlap_closure_mev_per_atom=args.max_overlap_closure,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
