#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch


KB_EV_PER_K = 8.617333262145e-5
ACCELERATION_FACTOR = 0.009648533219  # A/fs^2 from (eV/A)/amu
AL_MASS_AMU = 26.9815385


def fcc_structure(
    cells: int, lattice_constant: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    basis = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    positions: List[List[float]] = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for bx, by, bz in basis:
                    positions.append(
                        [
                            (i + bx) * lattice_constant,
                            (j + by) * lattice_constant,
                            (k + bz) * lattice_constant,
                        ]
                    )
    length = cells * lattice_constant
    lattice = torch.diag(
        torch.tensor([length, length, length], dtype=torch.float64, device=device)
    )
    return torch.tensor(positions, dtype=torch.float64, device=device), lattice


def wrap_positions(positions: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    fractional = positions @ torch.linalg.inv(lattice)
    return (fractional - torch.floor(fractional)) @ lattice


def evaluate_model(
    positions: torch.Tensor, lattice: torch.Tensor, model: Dict[str, object]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.tensor(
        model["centers_angstrom"], dtype=torch.float64, device=positions.device
    )
    coefficients = torch.tensor(
        model["coefficients_ev"], dtype=torch.float64, device=positions.device
    )
    sigma = float(model["sigma_angstrom"])
    cutoff = float(model["cutoff_angstrom"])
    core = model["repulsive_core"]
    core_amplitude = float(core["amplitude_ev"])
    core_cutoff = float(core["cutoff_angstrom"])
    core_power = int(core["power"])

    fractional = positions @ torch.linalg.inv(lattice)
    differences = fractional.unsqueeze(0) - fractional.unsqueeze(1)
    differences -= torch.round(differences)
    displacements = differences @ lattice
    distances = torch.linalg.vector_norm(displacements, dim=2)
    valid = (distances > 0.0) & (distances < cutoff)
    safe = torch.where(valid, distances, torch.ones_like(distances))
    directions = displacements / safe.unsqueeze(2)
    directions *= valid.unsqueeze(2)

    delta = safe.unsqueeze(2) - centers.reshape(1, 1, -1)
    gaussian = torch.exp(-0.5 * (delta / sigma) ** 2)
    angle = math.pi * safe / cutoff
    cutoff_value = 0.5 * (torch.cos(angle) + 1.0)
    cutoff_derivative = -0.5 * math.pi / cutoff * torch.sin(angle)
    basis = gaussian * cutoff_value.unsqueeze(2) * valid.unsqueeze(2)
    derivative = (
        -delta / sigma**2 * gaussian * cutoff_value.unsqueeze(2)
        + gaussian * cutoff_derivative.unsqueeze(2)
    ) * valid.unsqueeze(2)
    pair_energy = basis @ coefficients[1:]
    pair_derivative = derivative @ coefficients[1:]

    core_valid = (distances > 0.0) & (distances < core_cutoff)
    reduced = torch.clamp(1.0 - distances / core_cutoff, min=0.0)
    pair_energy += core_amplitude * reduced**core_power * core_valid
    pair_derivative += (
        -core_amplitude
        * core_power
        / core_cutoff
        * reduced ** (core_power - 1)
        * core_valid
    )
    energy = coefficients[0] * positions.shape[0] + 0.5 * pair_energy.sum()
    forces = (pair_derivative.unsqueeze(2) * directions).sum(dim=1)
    masked_distances = torch.where(distances > 0.0, distances, torch.full_like(distances, float("inf")))
    return energy, forces, masked_distances.min()


def remove_center_of_mass_velocity(velocities: torch.Tensor) -> torch.Tensor:
    return velocities - velocities.mean(dim=0, keepdim=True)


def temperature(velocities: torch.Tensor, mass_amu: float = AL_MASS_AMU) -> float:
    kinetic = 0.5 * mass_amu / ACCELERATION_FACTOR * float((velocities * velocities).sum())
    dof = 3 * velocities.shape[0] - 3
    return 2.0 * kinetic / (dof * KB_EV_PER_K)


def kinetic_energy(velocities: torch.Tensor, mass_amu: float = AL_MASS_AMU) -> float:
    return 0.5 * mass_amu / ACCELERATION_FACTOR * float((velocities * velocities).sum())


def random_velocities(
    natoms: int,
    target_temperature: float,
    seed: int,
    device: torch.device,
    mass_amu: float = AL_MASS_AMU,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    sigma = math.sqrt(KB_EV_PER_K * target_temperature * ACCELERATION_FACTOR / mass_amu)
    velocities = torch.randn(
        (natoms, 3), generator=generator, dtype=torch.float64, device=device
    ) * sigma
    velocities = remove_center_of_mass_velocity(velocities)
    velocities *= math.sqrt(target_temperature / temperature(velocities, mass_amu))
    return velocities


def load_dataset_frame(path: Path, index: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"dataset has no frames: {path}")
    try:
        record = records[index]
    except IndexError as error:
        raise ValueError(f"frame index {index} is out of range for {path}") from error
    positions = torch.tensor(
        record["positions_angstrom"], dtype=torch.float64, device=device
    )
    lattice = torch.tensor(
        record["lattice_angstrom"], dtype=torch.float64, device=device
    )
    return positions, lattice


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-system NVT stability pilot for a pair reference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--dataset-frame", type=Path)
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--lattice-constant", type=float, default=4.05)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--mass-amu", type=float, default=AL_MASS_AMU)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--gamma-per-fs", type=float, default=0.02)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--store-positions", action="store_true")
    args = parser.parse_args()

    if args.mass_amu <= 0.0:
        parser.error("--mass-amu must be positive")

    if args.restart and args.dataset_frame:
        parser.error("--restart and --dataset-frame are mutually exclusive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    model_document = json.loads(args.model.read_text(encoding="utf-8"))
    if not model_document.get("reference_gate_passed"):
        raise ValueError("model has not passed the static reference gate")
    model = model_document["model"]
    if args.restart:
        restart = json.loads(args.restart.read_text(encoding="utf-8"))
        positions = torch.tensor(
            restart["positions_angstrom"], dtype=torch.float64, device=device
        )
        lattice = torch.tensor(
            restart["lattice_angstrom"], dtype=torch.float64, device=device
        )
        velocities = torch.tensor(
            restart["velocities_angstrom_per_fs"], dtype=torch.float64, device=device
        )
        velocities *= math.sqrt(
            args.temperature / temperature(velocities, args.mass_amu)
        )
    elif args.dataset_frame:
        positions, lattice = load_dataset_frame(args.dataset_frame, args.frame_index, device)
        velocities = random_velocities(
            positions.shape[0], args.temperature, args.seed, device, args.mass_amu
        )
    else:
        positions, lattice = fcc_structure(args.cells, args.lattice_constant, device)
        velocities = random_velocities(
            positions.shape[0], args.temperature, args.seed, device, args.mass_amu
        )
    initial_positions = positions.clone()
    unwrapped = positions.clone()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    acceleration_scale = ACCELERATION_FACTOR / args.mass_amu
    thermostat_decay = math.exp(-args.gamma_per_fs * args.dt_fs)
    thermostat_sigma = math.sqrt(
        (1.0 - thermostat_decay**2)
        * KB_EV_PER_K
        * args.temperature
        * ACCELERATION_FACTOR
        / args.mass_amu
    )
    energy, forces, nearest = evaluate_model(positions, lattice, model)
    args.out.mkdir(parents=True, exist_ok=False)
    trajectory_path = args.out / "trajectory.jsonl"
    minimum_distance = float(nearest)
    with trajectory_path.open("w", encoding="utf-8") as trajectory:
        for step in range(args.steps + 1):
            if step % args.sample_every == 0:
                displacement = unwrapped - initial_positions
                msd = float((displacement * displacement).sum(dim=1).mean())
                record = {
                    "step": step,
                    "time_fs": step * args.dt_fs,
                    "temperature_k": temperature(velocities, args.mass_amu),
                    "potential_energy_ev_per_atom": float(energy) / positions.shape[0],
                    "kinetic_energy_ev_per_atom": kinetic_energy(
                        velocities, args.mass_amu
                    ) / positions.shape[0],
                    "nearest_neighbor_angstrom": float(nearest),
                    "msd_angstrom2": msd,
                }
                if args.store_positions:
                    record["positions_angstrom"] = positions.tolist()
                trajectory.write(json.dumps(record, separators=(",", ":")) + "\n")
            if step == args.steps:
                break
            half_kick = 0.5 * args.dt_fs * acceleration_scale * forces
            velocities += half_kick
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
            energy, forces, nearest = evaluate_model(positions, lattice, model)
            velocities += 0.5 * args.dt_fs * acceleration_scale * forces
            if (step + 1) % 10 == 0:
                velocities = remove_center_of_mass_velocity(velocities)
            minimum_distance = min(minimum_distance, float(nearest))
            if float(nearest) < 1.5 or not torch.isfinite(energy):
                raise RuntimeError(f"reference MD became unstable at step {step + 1}: rmin={float(nearest)}")

    checkpoint = {
        "schema": "mpn-pair-reference-md-checkpoint-v1",
        "model": str(args.model.resolve()),
        "target_temperature_k": args.temperature,
        "mass_amu": args.mass_amu,
        "device": str(device),
        "steps": args.steps,
        "dt_fs": args.dt_fs,
        "positions_angstrom": positions.tolist(),
        "velocities_angstrom_per_fs": velocities.tolist(),
        "lattice_angstrom": lattice.tolist(),
    }
    (args.out / "checkpoint.json").write_text(
        json.dumps(checkpoint, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    samples = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    summary = {
        "schema": "mpn-pair-reference-md-summary-v1",
        "natoms": positions.shape[0],
        "target_temperature_k": args.temperature,
        "mass_amu": args.mass_amu,
        "steps": args.steps,
        "minimum_distance_angstrom": minimum_distance,
        "temperature_mean_k": sum(row["temperature_k"] for row in samples) / len(samples),
        "temperature_last_k": samples[-1]["temperature_k"],
        "potential_energy_initial_ev_per_atom": samples[0]["potential_energy_ev_per_atom"],
        "potential_energy_last_ev_per_atom": samples[-1]["potential_energy_ev_per_atom"],
        "msd_last_angstrom2": samples[-1]["msd_angstrom2"],
        "stable": minimum_distance >= 1.5,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
