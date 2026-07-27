#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from mpn_melting.suf_reference import SUFParameters, evaluate_suf, suf_reduced_density
from run_pair_reference_md import evaluate_model


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def standard_deviation(values: List[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def lattice_volume(lattice: torch.Tensor) -> float:
    return abs(float(torch.linalg.det(lattice)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a scaled Uhlenbeck-Ford reference for a fitted liquid pair model"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--sigma-min", type=float, default=0.8)
    parser.add_argument("--sigma-max", type=float, default=1.8)
    parser.add_argument("--sigma-count", type=int, default=51)
    parser.add_argument("--cutoff-sigma", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if args.sigma_count < 2:
        parser.error("--sigma-count must be at least two")
    if args.sigma_max <= args.sigma_min:
        parser.error("--sigma-max must exceed --sigma-min")
    torch.set_num_threads(args.threads)

    model_document = json.loads(args.model.read_text(encoding="utf-8"))
    if not model_document.get("reference_gate_passed"):
        raise ValueError("pair model has not passed the static reference gate")
    model = model_document["model"]
    frames = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not frames:
        raise ValueError("dataset contains no frames")

    prepared: List[Dict[str, object]] = []
    for record in frames:
        positions = torch.tensor(record["positions_angstrom"], dtype=torch.float64)
        lattice = torch.tensor(record["lattice_angstrom"], dtype=torch.float64)
        pair_energy, pair_forces, nearest = evaluate_model(positions, lattice, model)
        prepared.append(
            {
                "positions": positions,
                "lattice": lattice,
                "pair_energy": float(pair_energy),
                "pair_forces": pair_forces,
                "natoms": positions.shape[0],
                "volume": lattice_volume(lattice),
                "nearest": float(nearest),
            }
        )

    scan: List[Dict[str, float]] = []
    for index in range(args.sigma_count):
        sigma = args.sigma_min + index * (args.sigma_max - args.sigma_min) / (args.sigma_count - 1)
        parameters = SUFParameters(
            p=args.p,
            sigma_angstrom=sigma,
            temperature_k=args.temperature,
            cutoff_sigma=args.cutoff_sigma,
        )
        energy_differences: List[float] = []
        force_squared_errors: List[float] = []
        for frame in prepared:
            suf_energy, suf_forces, _ = evaluate_suf(
                frame["positions"], frame["lattice"], parameters
            )
            energy_differences.append(
                (frame["pair_energy"] - float(suf_energy)) / frame["natoms"]
            )
            force_difference = frame["pair_forces"] - suf_forces
            force_squared_errors.extend((force_difference * force_difference).flatten().tolist())
        energy_std = standard_deviation(energy_differences)
        force_rmse = math.sqrt(mean(force_squared_errors))
        scan.append(
            {
                "sigma_angstrom": sigma,
                "delta_u_pair_minus_suf_mean_ev_per_atom": mean(energy_differences),
                "delta_u_pair_minus_suf_std_mev_per_atom": 1000.0 * energy_std,
                "force_difference_rmse_ev_per_angstrom": force_rmse,
            }
        )

    best = min(scan, key=lambda row: row["delta_u_pair_minus_suf_std_mev_per_atom"])
    number_densities = [frame["natoms"] / frame["volume"] for frame in prepared]
    density = mean(number_densities)
    sigma = best["sigma_angstrom"]
    result = {
        "schema": "mpn-suf-reference-fit-v1",
        "target_kedf": model_document.get("target_kedf"),
        "dataset": str(args.dataset.resolve()),
        "pair_model": str(args.model.resolve()),
        "frames": len(prepared),
        "natoms": prepared[0]["natoms"],
        "temperature_k": args.temperature,
        "p": args.p,
        "cutoff_sigma": args.cutoff_sigma,
        "number_density_angstrom_minus3": density,
        "selected_sigma_angstrom": sigma,
        "selected_reduced_density_x": suf_reduced_density(density, sigma),
        "minimum_dataset_neighbor_angstrom": min(frame["nearest"] for frame in prepared),
        "selection_metric": "minimum standard deviation of U_pair-U_sUF per atom",
        "selected": best,
        "scan": scan,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
