from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


KB_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class SUFParameters:
    p: int
    sigma_angstrom: float
    temperature_k: float
    cutoff_sigma: float = 5.0

    @property
    def cutoff_angstrom(self) -> float:
        return self.cutoff_sigma * self.sigma_angstrom


def evaluate_suf(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    parameters: SUFParameters,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the scaled Uhlenbeck-Ford pair potential and forces.

    The dimensionless pair potential is
    beta * u(r) = -p * log(1 - exp(-(r / sigma)^2)).
    A five-sigma cutoff omits a tail below roughly 1e-9 kBT per pair.
    """
    if parameters.p <= 0:
        raise ValueError("p must be positive")
    if parameters.sigma_angstrom <= 0.0:
        raise ValueError("sigma must be positive")
    if parameters.temperature_k <= 0.0:
        raise ValueError("temperature must be positive")

    fractional = positions @ torch.linalg.inv(lattice)
    differences = fractional.unsqueeze(0) - fractional.unsqueeze(1)
    differences -= torch.round(differences)
    displacements = differences @ lattice
    distances = torch.linalg.vector_norm(displacements, dim=2)
    cutoff = min(
        parameters.cutoff_angstrom,
        0.5 * float(torch.linalg.vector_norm(lattice, dim=1).min()),
    )
    valid = (distances > 0.0) & (distances < cutoff)
    safe = torch.where(valid, distances, torch.ones_like(distances))
    directions = displacements / safe.unsqueeze(2)
    directions *= valid.unsqueeze(2)

    reduced_squared = (safe / parameters.sigma_angstrom) ** 2
    exponential = torch.exp(-reduced_squared)
    denominator = -torch.expm1(-reduced_squared)
    beta_pair_energy = -parameters.p * torch.log(denominator)
    beta_pair_derivative = (
        -2.0
        * parameters.p
        * safe
        / parameters.sigma_angstrom**2
        * exponential
        / denominator
    )
    thermal_energy = KB_EV_PER_K * parameters.temperature_k
    pair_energy = thermal_energy * beta_pair_energy * valid
    pair_derivative = thermal_energy * beta_pair_derivative * valid
    energy = 0.5 * pair_energy.sum()
    forces = (pair_derivative.unsqueeze(2) * directions).sum(dim=1)
    masked = torch.where(
        distances > 0.0,
        distances,
        torch.full_like(distances, float("inf")),
    )
    return energy, forces, masked.min()


def suf_reduced_density(number_density_angstrom3: float, sigma_angstrom: float) -> float:
    if number_density_angstrom3 <= 0.0:
        raise ValueError("number density must be positive")
    if sigma_angstrom <= 0.0:
        raise ValueError("sigma must be positive")
    return 0.5 * (math.pi * sigma_angstrom**2) ** 1.5 * number_density_angstrom3
