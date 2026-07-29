#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from mpn_melting.suf_reference import SUFParameters, evaluate_suf
from run_pair_reference_md import (
    ACCELERATION_FACTOR,
    AL_MASS_AMU,
    KB_EV_PER_K,
    evaluate_model,
    kinetic_energy,
    random_velocities,
    remove_center_of_mass_velocity,
    temperature,
    wrap_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVT sampling for sUF-to-fitted-pair thermodynamic integration"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--suf-p", type=int, default=50)
    parser.add_argument("--suf-sigma", type=float, required=True)
    parser.add_argument("--suf-cutoff-sigma", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--gamma-per-fs", type=float, default=0.02)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--store-positions", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.lambda_value <= 1.0:
        parser.error("--lambda-value must be between zero and one")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    model_document = json.loads(args.model.read_text(encoding="utf-8"))
    if not model_document.get("reference_gate_passed"):
        raise ValueError("pair model has not passed the static reference gate")
    model = model_document["model"]
    restart = json.loads(args.restart.read_text(encoding="utf-8"))
    positions = torch.tensor(
        restart["positions_angstrom"], dtype=torch.float64, device=device
    )
    lattice = torch.tensor(
        restart["lattice_angstrom"], dtype=torch.float64, device=device
    )
    velocities = random_velocities(
        positions.shape[0], args.temperature, args.seed, device
    )
    initial_positions = positions.clone()
    unwrapped = positions.clone()
    suf_parameters = SUFParameters(
        p=args.suf_p,
        sigma_angstrom=args.suf_sigma,
        temperature_k=args.temperature,
        cutoff_sigma=args.suf_cutoff_sigma,
    )

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_energy, pair_forces, pair_nearest = evaluate_model(positions, lattice, model)
        suf_energy, suf_forces, suf_nearest = evaluate_suf(
            positions, lattice, suf_parameters
        )
        mixed_energy = (
            args.lambda_value * pair_energy
            + (1.0 - args.lambda_value) * suf_energy
        )
        mixed_forces = (
            args.lambda_value * pair_forces
            + (1.0 - args.lambda_value) * suf_forces
        )
        return (
            mixed_energy,
            mixed_forces,
            pair_energy,
            suf_energy,
            torch.minimum(pair_nearest, suf_nearest),
        )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    acceleration_scale = ACCELERATION_FACTOR / AL_MASS_AMU
    thermostat_decay = math.exp(-args.gamma_per_fs * args.dt_fs)
    thermostat_sigma = math.sqrt(
        (1.0 - thermostat_decay**2)
        * KB_EV_PER_K
        * args.temperature
        * ACCELERATION_FACTOR
        / AL_MASS_AMU
    )
    energy, forces, pair_energy, suf_energy, nearest = evaluate()
    args.out.mkdir(parents=True, exist_ok=False)
    trajectory_path = args.out / "trajectory.jsonl"
    minimum_distance = float(nearest)
    with trajectory_path.open("w", encoding="utf-8") as trajectory:
        for step in range(args.steps + 1):
            if step % args.sample_every == 0:
                displacement = unwrapped - initial_positions
                record = {
                    "step": step,
                    "time_fs": step * args.dt_fs,
                    "lambda": args.lambda_value,
                    "temperature_k": temperature(velocities),
                    "potential_energy_ev_per_atom": float(energy) / positions.shape[0],
                    "kinetic_energy_ev_per_atom": kinetic_energy(velocities) / positions.shape[0],
                    "pair_energy_ev_per_atom": float(pair_energy) / positions.shape[0],
                    "suf_energy_ev_per_atom": float(suf_energy) / positions.shape[0],
                    "du_pair_minus_suf_ev_per_atom": float(pair_energy - suf_energy)
                    / positions.shape[0],
                    "nearest_neighbor_angstrom": float(nearest),
                    "msd_angstrom2": float(
                        (displacement * displacement).sum(dim=1).mean()
                    ),
                }
                if args.store_positions:
                    record["positions_angstrom"] = positions.tolist()
                trajectory.write(json.dumps(record, separators=(",", ":")) + "\n")
            if step == args.steps:
                break
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            displacement = 0.5 * args.dt_fs * velocities
            unwrapped += displacement
            positions = wrap_positions(positions + displacement, lattice)
            velocities = thermostat_decay * velocities + thermostat_sigma * torch.randn(
                velocities.shape,
                generator=generator,
                dtype=torch.float64,
                device=device,
            )
            displacement = 0.5 * args.dt_fs * velocities
            unwrapped += displacement
            positions = wrap_positions(positions + displacement, lattice)
            energy, forces, pair_energy, suf_energy, nearest = evaluate()
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            if (step + 1) % 10 == 0:
                velocities = remove_center_of_mass_velocity(velocities)
            minimum_distance = min(minimum_distance, float(nearest))
            if float(nearest) < 1.5 or not torch.isfinite(energy):
                raise RuntimeError(
                    f"sUF-pair TI MD became unstable at step {step + 1}: "
                    f"rmin={float(nearest)}"
                )

    checkpoint = {
        "schema": "mpn-suf-pair-ti-checkpoint-v1",
        "model": str(args.model.resolve()),
        "lambda": args.lambda_value,
        "temperature_k": args.temperature,
        "target_temperature_k": args.temperature,
        "suf_p": args.suf_p,
        "suf_sigma_angstrom": args.suf_sigma,
        "steps": args.steps,
        "dt_fs": args.dt_fs,
        "positions_angstrom": positions.tolist(),
        "velocities_angstrom_per_fs": velocities.tolist(),
        "lattice_angstrom": lattice.tolist(),
    }
    (args.out / "checkpoint.json").write_text(
        json.dumps(checkpoint, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    samples = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = {
        "schema": "mpn-suf-pair-ti-summary-v1",
        "natoms": positions.shape[0],
        "lambda": args.lambda_value,
        "target_temperature_k": args.temperature,
        "steps": args.steps,
        "samples": len(samples),
        "minimum_distance_angstrom": minimum_distance,
        "temperature_mean_k": sum(row["temperature_k"] for row in samples)
        / len(samples),
        "temperature_last_k": samples[-1]["temperature_k"],
        "du_mean_ev_per_atom": sum(
            row["du_pair_minus_suf_ev_per_atom"] for row in samples
        )
        / len(samples),
        "msd_last_angstrom2": samples[-1]["msd_angstrom2"],
        "stable": minimum_distance >= 1.5,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
