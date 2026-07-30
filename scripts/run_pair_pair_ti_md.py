#!/usr/bin/env python3
"""Classical NVT sampling for thermodynamic integration between pair models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts.run_pair_reference_md import (
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


def load_pair_document(
    path: Path,
    *,
    target_kedf: str | None,
    phase: str | None,
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("reference_gate_passed"):
        raise ValueError(f"pair model has not passed its static gate: {path}")
    if not document.get("short_range_guard_passed"):
        raise ValueError(f"pair model has not passed its short-range gate: {path}")
    if target_kedf is not None and document.get("target_kedf") != target_kedf:
        raise ValueError(f"pair model has the wrong target KEDF: {path}")
    document_phase = document.get("phase", document.get("reference_phase"))
    if phase is not None and document_phase != phase:
        raise ValueError(f"pair model has the wrong phase: {path}")
    return document


def evaluate_pair_interpolation(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    reference_model: dict[str, Any],
    target_model: dict[str, Any],
    lambda_value: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    reference_energy, reference_forces, reference_nearest = evaluate_model(
        positions, lattice, reference_model
    )
    target_energy, target_forces, target_nearest = evaluate_model(
        positions, lattice, target_model
    )
    mixed_energy = (
        (1.0 - lambda_value) * reference_energy
        + lambda_value * target_energy
    )
    mixed_forces = (
        (1.0 - lambda_value) * reference_forces
        + lambda_value * target_forces
    )
    return (
        mixed_energy,
        mixed_forces,
        reference_energy,
        target_energy,
        torch.minimum(reference_nearest, target_nearest),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVT sampling for pair-reference thermodynamic integration"
    )
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--target-kedf")
    parser.add_argument("--phase", choices=("solid", "liquid"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--gamma-per-fs", type=float, default=0.02)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-distance", type=float, default=2.0)
    parser.add_argument("--store-positions", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.lambda_value <= 1.0:
        parser.error("--lambda-value must be between zero and one")
    if args.minimum_distance <= 0.0:
        parser.error("--minimum-distance must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    reference_document = load_pair_document(
        args.reference_model,
        target_kedf=args.target_kedf,
        phase=args.phase,
    )
    target_document = load_pair_document(
        args.target_model,
        target_kedf=args.target_kedf,
        phase=args.phase,
    )
    reference_model = reference_document["model"]
    target_model = target_document["model"]
    restart = json.loads(args.restart.read_text(encoding="utf-8"))
    positions = torch.tensor(
        restart["positions_angstrom"], dtype=torch.float64, device=device
    )
    lattice = torch.tensor(
        restart["lattice_angstrom"], dtype=torch.float64, device=device
    )
    if restart.get("velocities_angstrom_per_fs") is None:
        velocities = random_velocities(
            positions.shape[0], args.temperature, args.seed, device
        )
        source_velocities_preserved = False
    else:
        velocities = torch.tensor(
            restart["velocities_angstrom_per_fs"],
            dtype=torch.float64,
            device=device,
        )
        velocities = remove_center_of_mass_velocity(velocities)
        velocities *= math.sqrt(args.temperature / temperature(velocities))
        source_velocities_preserved = True
    initial_positions = positions.clone()
    unwrapped = positions.clone()

    def evaluate():
        return evaluate_pair_interpolation(
            positions,
            lattice,
            reference_model,
            target_model,
            args.lambda_value,
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
    energy, forces, reference_energy, target_energy, nearest = evaluate()
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
                    "potential_energy_ev_per_atom": float(energy)
                    / positions.shape[0],
                    "kinetic_energy_ev_per_atom": kinetic_energy(velocities)
                    / positions.shape[0],
                    "reference_energy_ev_per_atom": float(reference_energy)
                    / positions.shape[0],
                    "target_energy_ev_per_atom": float(target_energy)
                    / positions.shape[0],
                    "du_target_minus_reference_ev_per_atom": float(
                        target_energy - reference_energy
                    )
                    / positions.shape[0],
                    "nearest_neighbor_angstrom": float(nearest),
                    "msd_angstrom2": float(
                        (displacement * displacement).sum(dim=1).mean()
                    ),
                }
                if args.store_positions:
                    record["positions_angstrom"] = positions.tolist()
                trajectory.write(
                    json.dumps(record, separators=(",", ":")) + "\n"
                )
            if step == args.steps:
                break
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            displacement = 0.5 * args.dt_fs * velocities
            unwrapped += displacement
            positions = wrap_positions(positions + displacement, lattice)
            velocities = (
                thermostat_decay * velocities
                + thermostat_sigma
                * torch.randn(
                    velocities.shape,
                    generator=generator,
                    dtype=torch.float64,
                    device=device,
                )
            )
            displacement = 0.5 * args.dt_fs * velocities
            unwrapped += displacement
            positions = wrap_positions(positions + displacement, lattice)
            energy, forces, reference_energy, target_energy, nearest = evaluate()
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            if (step + 1) % 10 == 0:
                velocities = remove_center_of_mass_velocity(velocities)
            minimum_distance = min(minimum_distance, float(nearest))
            if (
                float(nearest) < args.minimum_distance
                or not torch.isfinite(energy)
                or not torch.isfinite(forces).all()
            ):
                raise RuntimeError(
                    f"pair-pair TI became unstable at step {step + 1}: "
                    f"rmin={float(nearest)}"
                )

    checkpoint = {
        "schema": "mpn-pair-pair-ti-checkpoint-v1",
        "reference_model": str(args.reference_model.resolve()),
        "target_model": str(args.target_model.resolve()),
        "target_kedf": args.target_kedf,
        "phase": args.phase,
        "lambda": args.lambda_value,
        "temperature_k": args.temperature,
        "target_temperature_k": args.temperature,
        "steps": args.steps,
        "dt_fs": args.dt_fs,
        "source_velocities_preserved": source_velocities_preserved,
        "positions_angstrom": positions.tolist(),
        "velocities_angstrom_per_fs": velocities.tolist(),
        "lattice_angstrom": lattice.tolist(),
    }
    (args.out / "checkpoint.json").write_text(
        json.dumps(checkpoint, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    samples = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = {
        "schema": "mpn-pair-pair-ti-summary-v1",
        "natoms": positions.shape[0],
        "reference_model": str(args.reference_model.resolve()),
        "target_model": str(args.target_model.resolve()),
        "target_kedf": args.target_kedf,
        "phase": args.phase,
        "lambda": args.lambda_value,
        "target_temperature_k": args.temperature,
        "steps": args.steps,
        "samples": len(samples),
        "minimum_distance_angstrom": minimum_distance,
        "temperature_mean_k": sum(row["temperature_k"] for row in samples)
        / len(samples),
        "temperature_last_k": samples[-1]["temperature_k"],
        "du_mean_ev_per_atom": sum(
            row["du_target_minus_reference_ev_per_atom"] for row in samples
        )
        / len(samples),
        "msd_last_angstrom2": samples[-1]["msd_angstrom2"],
        "source_velocities_preserved": source_velocities_preserved,
        "stable": minimum_distance >= args.minimum_distance,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
