#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from mpn_melting.free_energy import KB_EV_PER_K, exponential_free_energy_difference
from run_pair_reference_md import evaluate_model


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose direct free-energy perturbation overlap")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model_document = json.loads(args.model.read_text(encoding="utf-8"))
    model = model_document["model"]
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"delta": [], "temperature": [], "natoms": []}
    )
    with args.dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            positions = torch.tensor(record["positions_angstrom"], dtype=torch.float64)
            lattice = torch.tensor(record["lattice_angstrom"], dtype=torch.float64)
            reference_energy = float(evaluate_model(positions, lattice, model)[0])
            delta = reference_energy - float(record["potential_energy_ev"])
            phase = str(record["phase"])
            grouped[phase]["delta"].append(delta)
            grouped[phase]["temperature"].append(float(record["temperature_k"]))
            grouped[phase]["natoms"].append(float(record["natoms"]))

    phases: Dict[str, object] = {}
    all_passed = True
    for phase, values in sorted(grouped.items()):
        deltas = values["delta"]
        temperature = mean(values["temperature"])
        natoms = int(values["natoms"][0])
        fep = exponential_free_energy_difference(deltas, temperature)
        beta_sigma = standard_deviation(deltas) / (KB_EV_PER_K * temperature)
        gate = fep["effective_sample_fraction"] >= 0.2 and beta_sigma <= 2.0
        all_passed = all_passed and gate
        phases[phase] = {
            "frames": len(deltas),
            "natoms": natoms,
            "temperature_mean_k": temperature,
            "delta_u_ref_minus_mpn_mean_ev": mean(deltas),
            "delta_u_ref_minus_mpn_std_ev": standard_deviation(deltas),
            "delta_u_std_mev_per_atom": standard_deviation(deltas) * 1000.0 / natoms,
            "beta_sigma_delta_u": beta_sigma,
            "reverse_fep_ref_minus_mpn_delta_f_ev": fep["delta_f_ev"],
            "reverse_fep_delta_f_mev_per_atom": fep["delta_f_ev"] * 1000.0 / natoms,
            "effective_sample_size": fep["effective_sample_size"],
            "effective_sample_fraction": fep["effective_sample_fraction"],
            "single_step_fep_gate_passed": gate,
        }
    result = {
        "schema": "mpn-reference-overlap-diagnostic-v1",
        "dataset": str(args.dataset.resolve()),
        "model": str(args.model.resolve()),
        "gate_thresholds": {"effective_sample_fraction_min": 0.2, "beta_sigma_delta_u_max": 2.0},
        "phases": phases,
        "single_step_fep_gate_passed": all_passed,
        "recommendation": (
            "direct Zwanzig correction is statistically plausible"
            if all_passed
            else "use thermodynamic integration with intermediate lambda states"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
