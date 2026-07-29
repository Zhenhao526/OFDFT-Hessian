#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

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
        description="COM-constrained Einstein-crystal-to-pair TI sampling"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--einstein-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--gamma-per-fs", type=float, default=0.02)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--store-positions", action="store_true")
    parser.add_argument("--minimum-distance", type=float, default=1.3)
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
    reference = json.loads(args.einstein_reference.read_text(encoding="utf-8"))
    positions = torch.tensor(
        restart["positions_angstrom"], dtype=torch.float64, device=device
    )
    lattice = torch.tensor(
        restart["lattice_angstrom"], dtype=torch.float64, device=device
    )
    anchors = torch.tensor(
        reference["anchor_positions_angstrom"], dtype=torch.float64, device=device
    )
    spring = float(reference["spring_constant_ev_per_angstrom2"])
    if positions.shape != anchors.shape:
        raise ValueError("restart and Einstein reference atom counts differ")
    velocities = random_velocities(
        positions.shape[0], args.temperature, args.seed, device
    )
    inverse = torch.linalg.inv(lattice)

    def harmonic() -> tuple[torch.Tensor, torch.Tensor]:
        fractional = (positions - anchors) @ inverse
        fractional -= torch.round(fractional)
        displacement = fractional @ lattice
        return 0.5 * spring * (displacement * displacement).sum(), -spring * displacement

    def constrain_center_of_mass() -> None:
        nonlocal positions, velocities
        fractional = (positions - anchors) @ inverse
        fractional -= torch.round(fractional)
        displacement = fractional @ lattice
        positions = wrap_positions(positions - displacement.mean(dim=0), lattice)
        velocities = remove_center_of_mass_velocity(velocities)

    constrain_center_of_mass()

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_energy, pair_forces, nearest = evaluate_model(positions, lattice, model)
        harmonic_energy, harmonic_forces = harmonic()
        mixed_energy = (
            args.lambda_value * pair_energy
            + (1.0 - args.lambda_value) * harmonic_energy
        )
        mixed_forces = (
            args.lambda_value * pair_forces
            + (1.0 - args.lambda_value) * harmonic_forces
        )
        return mixed_energy, mixed_forces, pair_energy, harmonic_energy, nearest

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
    energy, forces, pair_energy, harmonic_energy, nearest = evaluate()
    args.out.mkdir(parents=True, exist_ok=False)
    trajectory_path = args.out / "trajectory.jsonl"
    minimum_distance = float(nearest)
    with trajectory_path.open("w", encoding="utf-8") as trajectory:
        for step in range(args.steps + 1):
            if step % args.sample_every == 0:
                harmonic_energy, _ = harmonic()
                fractional = (positions - anchors) @ inverse
                fractional -= torch.round(fractional)
                site_displacement = fractional @ lattice
                record = {
                    "step": step,
                    "time_fs": step * args.dt_fs,
                    "lambda": args.lambda_value,
                    "temperature_k": temperature(velocities),
                    "potential_energy_ev_per_atom": float(energy) / positions.shape[0],
                    "pair_energy_ev_per_atom": float(pair_energy) / positions.shape[0],
                    "harmonic_energy_ev_per_atom": float(harmonic_energy)
                    / positions.shape[0],
                    "du_pair_minus_harmonic_ev_per_atom": float(
                        pair_energy - harmonic_energy
                    )
                    / positions.shape[0],
                    "kinetic_energy_ev_per_atom": kinetic_energy(velocities)
                    / positions.shape[0],
                    "nearest_neighbor_angstrom": float(nearest),
                    "msd_angstrom2": float(
                        (site_displacement * site_displacement).sum(dim=1).mean()
                    ),
                }
                if args.store_positions:
                    record["positions_angstrom"] = positions.tolist()
                trajectory.write(json.dumps(record, separators=(",", ":")) + "\n")
            if step == args.steps:
                break
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            positions = wrap_positions(positions + 0.5 * args.dt_fs * velocities, lattice)
            velocities = thermostat_decay * velocities + thermostat_sigma * torch.randn(
                velocities.shape,
                generator=generator,
                dtype=torch.float64,
                device=device,
            )
            positions = wrap_positions(positions + 0.5 * args.dt_fs * velocities, lattice)
            constrain_center_of_mass()
            energy, forces, pair_energy, harmonic_energy, nearest = evaluate()
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            constrain_center_of_mass()
            minimum_distance = min(minimum_distance, float(nearest))
            if float(nearest) < args.minimum_distance or not torch.isfinite(energy):
                raise RuntimeError(
                    f"Einstein-pair TI MD became unstable at step {step + 1}: "
                    f"rmin={float(nearest)}"
                )

    checkpoint = {
        "schema": "mpn-einstein-pair-ti-checkpoint-v1",
        "model": str(args.model.resolve()),
        "lambda": args.lambda_value,
        "target_temperature_k": args.temperature,
        "dt_fs": args.dt_fs,
        "steps": args.steps,
        "positions_angstrom": positions.tolist(),
        "velocities_angstrom_per_fs": velocities.tolist(),
        "lattice_angstrom": lattice.tolist(),
    }
    (args.out / "checkpoint.json").write_text(
        json.dumps(checkpoint, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "mpn-einstein-pair-ti-summary-v1",
        "natoms": positions.shape[0],
        "lambda": args.lambda_value,
        "target_temperature_k": args.temperature,
        "spring_constant_ev_per_angstrom2": spring,
        "steps": args.steps,
        "minimum_distance_angstrom": minimum_distance,
        "temperature_last_k": temperature(velocities),
        "stable": minimum_distance >= args.minimum_distance,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
