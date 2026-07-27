#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from mpn_melting.pair_reference import evenly_spaced_centers


def load_records(path: Path, max_per_phase: int) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grouped[str(record["phase"])].append(record)
    selected: List[Dict[str, object]] = []
    for phase in sorted(grouped):
        records = grouped[phase]
        if max_per_phase > 0 and len(records) > max_per_phase:
            indices = {
                round(index * (len(records) - 1) / (max_per_phase - 1))
                for index in range(max_per_phase)
            }
            records = [records[index] for index in sorted(indices)]
        selected.extend(records)
    return selected


def split_records(records: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["phase"])].append(record)
    train: List[Dict[str, object]] = []
    validation: List[Dict[str, object]] = []
    for phase in sorted(grouped):
        phase_records = grouped[phase]
        if len(phase_records) < 3:
            raise ValueError(f"phase {phase} needs at least three selected frames")
        validation_indices = {
            index for index in range(len(phase_records)) if (index + 1) % 3 == 0
        }
        validation_indices.add(len(phase_records) - 1)
        for index, record in enumerate(phase_records):
            (validation if index in validation_indices else train).append(record)
    return train, validation


def frame_features(
    record: Dict[str, object],
    centers: torch.Tensor,
    sigma: float,
    cutoff: float,
    core_amplitude: float,
    core_cutoff: float,
    core_power: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lattice = torch.tensor(record["lattice_angstrom"], dtype=torch.float64)
    positions = torch.tensor(record["positions_angstrom"], dtype=torch.float64)
    fractional = positions @ torch.linalg.inv(lattice)
    differences = fractional.unsqueeze(0) - fractional.unsqueeze(1)
    differences -= torch.round(differences)
    displacements = differences @ lattice
    distances = torch.linalg.vector_norm(displacements, dim=2)
    valid = (distances > 0.0) & (distances < cutoff)
    safe_distances = torch.where(valid, distances, torch.ones_like(distances))

    delta = safe_distances.unsqueeze(2) - centers.reshape(1, 1, -1)
    gaussian = torch.exp(-0.5 * (delta / sigma) ** 2)
    angle = math.pi * safe_distances / cutoff
    cutoff_value = 0.5 * (torch.cos(angle) + 1.0)
    cutoff_derivative = -0.5 * math.pi / cutoff * torch.sin(angle)
    basis = gaussian * cutoff_value.unsqueeze(2)
    derivative = (
        -delta / (sigma**2) * gaussian * cutoff_value.unsqueeze(2)
        + gaussian * cutoff_derivative.unsqueeze(2)
    )
    basis *= valid.unsqueeze(2)
    derivative *= valid.unsqueeze(2)

    energy_features = 0.5 * basis.sum(dim=(0, 1)) / positions.shape[0]
    directions = displacements / safe_distances.unsqueeze(2)
    directions *= valid.unsqueeze(2)
    force_features = torch.einsum("ijk,ijc->ick", derivative, directions).reshape(-1, centers.numel())
    core_valid = (distances > 0.0) & (distances < core_cutoff)
    core_reduced = torch.clamp(1.0 - distances / core_cutoff, min=0.0)
    core_pair = core_amplitude * core_reduced**core_power * core_valid
    core_derivative = (
        -core_amplitude
        * core_power
        / core_cutoff
        * core_reduced ** (core_power - 1)
        * core_valid
    )
    core_energy = 0.5 * core_pair.sum() / positions.shape[0]
    core_force = (core_derivative.unsqueeze(2) * directions).sum(dim=1).reshape(-1)
    return energy_features, force_features, core_energy, core_force


def target_tensors(record: Dict[str, object]) -> Tuple[float, torch.Tensor]:
    natoms = int(record["natoms"])
    energy_per_atom = float(record["potential_energy_ev"]) / natoms
    forces = torch.tensor(record["forces_ev_per_angstrom"], dtype=torch.float64).reshape(-1)
    return energy_per_atom, forces


def fit_coefficients(
    records: Sequence[Dict[str, object]],
    centers: torch.Tensor,
    sigma: float,
    cutoff: float,
    energy_scale: float,
    force_scale: float,
    ridge: float,
    core_amplitude: float,
    core_cutoff: float,
    core_power: int,
) -> torch.Tensor:
    size = centers.numel() + 1
    energy_normal = torch.zeros((size, size), dtype=torch.float64)
    energy_rhs = torch.zeros(size, dtype=torch.float64)
    force_normal = torch.zeros((size, size), dtype=torch.float64)
    force_rhs = torch.zeros(size, dtype=torch.float64)
    energy_count = 0
    force_count = 0
    for record in records:
        energy_features, force_features, core_energy, core_force = frame_features(
            record, centers, sigma, cutoff, core_amplitude, core_cutoff, core_power
        )
        energy_target, force_targets = target_tensors(record)
        energy_target -= float(core_energy)
        force_targets = force_targets - core_force
        energy_row = torch.cat((torch.ones(1, dtype=torch.float64), energy_features))
        energy_normal += torch.outer(energy_row, energy_row)
        energy_rhs += energy_row * energy_target
        force_normal[1:, 1:] += force_features.T @ force_features
        force_rhs[1:] += force_features.T @ force_targets
        energy_count += 1
        force_count += force_targets.numel()

    normal = energy_normal / (energy_count * energy_scale**2)
    normal += force_normal / (force_count * force_scale**2)
    rhs = energy_rhs / (energy_count * energy_scale**2)
    rhs += force_rhs / (force_count * force_scale**2)
    regularization = ridge * torch.diag(normal).mean().clamp_min(1.0)
    normal += torch.eye(size, dtype=torch.float64) * regularization
    return torch.linalg.solve(normal, rhs)


def evaluate(
    records: Sequence[Dict[str, object]],
    coefficients: torch.Tensor,
    centers: torch.Tensor,
    sigma: float,
    cutoff: float,
    core_amplitude: float,
    core_cutoff: float,
    core_power: int,
) -> Dict[str, object]:
    energy_errors: List[float] = []
    force_squared_error = 0.0
    force_count = 0
    phase_errors: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        energy_features, force_features, core_energy, core_force = frame_features(
            record, centers, sigma, cutoff, core_amplitude, core_cutoff, core_power
        )
        energy_target, force_targets = target_tensors(record)
        energy_prediction = coefficients[0] + energy_features @ coefficients[1:] + core_energy
        energy_error = float(energy_prediction) - energy_target
        energy_errors.append(energy_error)
        phase_errors[str(record["phase"])].append(energy_error)
        force_error = force_features @ coefficients[1:] + core_force - force_targets
        force_squared_error += float(force_error @ force_error)
        force_count += force_targets.numel()
    return {
        "frames": len(records),
        "energy_rmse_ev_per_atom": math.sqrt(sum(value * value for value in energy_errors) / len(energy_errors)),
        "energy_bias_ev_per_atom": sum(energy_errors) / len(energy_errors),
        "force_rmse_ev_per_angstrom": math.sqrt(force_squared_error / force_count),
        "phase_energy_bias_ev_per_atom": {
            phase: sum(values) / len(values) for phase, values in sorted(phase_errors.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and gate a radial pair reference against MPN data")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-phase", type=int, default=6)
    parser.add_argument("--basis-count", type=int, default=17)
    parser.add_argument("--basis-min", type=float, default=2.0)
    parser.add_argument("--basis-max", type=float, default=6.0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--cutoff", type=float, default=6.5)
    parser.add_argument("--energy-scale", type=float, default=0.02)
    parser.add_argument("--force-scale", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--core-amplitude", type=float, default=500.0)
    parser.add_argument("--core-cutoff", type=float, default=2.2)
    parser.add_argument("--core-power", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    records = load_records(args.dataset, args.max_per_phase)
    dataset_manifest_path = args.dataset.parent / "manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.exists()
        else {}
    )
    target_kedf = str(dataset_manifest.get("target_kedf", "")).lower() or None
    train, validation = split_records(records)
    centers_list = evenly_spaced_centers(args.basis_min, args.basis_max, args.basis_count)
    centers = torch.tensor(centers_list, dtype=torch.float64)
    coefficients = fit_coefficients(
        train,
        centers,
        args.sigma,
        args.cutoff,
        args.energy_scale,
        args.force_scale,
        args.ridge,
        args.core_amplitude,
        args.core_cutoff,
        args.core_power,
    )
    train_metrics = evaluate(
        train,
        coefficients,
        centers,
        args.sigma,
        args.cutoff,
        args.core_amplitude,
        args.core_cutoff,
        args.core_power,
    )
    validation_metrics = evaluate(
        validation,
        coefficients,
        centers,
        args.sigma,
        args.cutoff,
        args.core_amplitude,
        args.core_cutoff,
        args.core_power,
    )
    phase_biases = list(validation_metrics["phase_energy_bias_ev_per_atom"].values())
    phase_bias_gap = max(phase_biases) - min(phase_biases)
    overlap_gate = (
        validation_metrics["energy_rmse_ev_per_atom"] <= 0.02
        and validation_metrics["force_rmse_ev_per_angstrom"] <= 0.25
        and phase_bias_gap <= 0.03
    )
    def pair_value_and_derivative(distance: float) -> Tuple[float, float]:
        delta = centers - distance
        gaussian = torch.exp(-0.5 * (delta / args.sigma) ** 2)
        angle = math.pi * distance / args.cutoff
        cutoff_value = 0.5 * (math.cos(angle) + 1.0) if distance < args.cutoff else 0.0
        cutoff_derivative = (
            -0.5 * math.pi / args.cutoff * math.sin(angle) if distance < args.cutoff else 0.0
        )
        basis = gaussian * cutoff_value
        derivative = (
            (delta / args.sigma**2) * gaussian * cutoff_value
            + gaussian * cutoff_derivative
        )
        reduced = max(0.0, 1.0 - distance / args.core_cutoff)
        core_value = args.core_amplitude * reduced**args.core_power
        core_derivative = (
            -args.core_amplitude
            * args.core_power
            / args.core_cutoff
            * reduced ** (args.core_power - 1)
            if distance < args.core_cutoff
            else 0.0
        )
        return (
            float(basis @ coefficients[1:]) + core_value,
            float(derivative @ coefficients[1:]) + core_derivative,
        )

    short_value, short_derivative = pair_value_and_derivative(1.5)
    sampled_value, _ = pair_value_and_derivative(2.0)
    short_range_guard = short_value > sampled_value + 1.0 and short_derivative < 0.0
    result = {
        "schema": "mpn-radial-pair-reference-v1",
        "target_kedf": target_kedf,
        "purpose": "overlap diagnostic only; not approved for TI unless the validation gate passes",
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest": (
            str(dataset_manifest_path.resolve())
            if dataset_manifest_path.exists()
            else None
        ),
        "selected_frames": len(records),
        "train_frames": len(train),
        "validation_frames": len(validation),
        "model": {
            "centers_angstrom": centers_list,
            "sigma_angstrom": args.sigma,
            "cutoff_angstrom": args.cutoff,
            "coefficients_ev": coefficients.tolist(),
            "repulsive_core": {
                "amplitude_ev": args.core_amplitude,
                "cutoff_angstrom": args.core_cutoff,
                "power": args.core_power,
            },
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_phase_bias_gap_ev_per_atom": phase_bias_gap,
        "gate_thresholds": {
            "energy_rmse_ev_per_atom": 0.02,
            "force_rmse_ev_per_angstrom": 0.25,
            "phase_energy_bias_gap_ev_per_atom": 0.03,
        },
        "short_range_diagnostic": {
            "u_1p5_ev": short_value,
            "du_dr_1p5_ev_per_angstrom": short_derivative,
            "u_2p0_ev": sampled_value,
        },
        "overlap_gate_passed": overlap_gate,
        "short_range_guard_passed": short_range_guard,
        "reference_gate_passed": overlap_gate and short_range_guard,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
