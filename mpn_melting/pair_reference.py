from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

from mpn_melting.trajectory import invert_3x3, lattice_volume, matmul_row


KB_EV_PER_K = 8.617333262145e-5
EV_PER_ANGSTROM3_TO_KBAR = 1602.176634


@dataclass(frozen=True)
class RadialBasis:
    centers_angstrom: Sequence[float]
    sigma_angstrom: float
    cutoff_angstrom: float

    def values_and_derivatives(self, distance_angstrom: float) -> Tuple[List[float], List[float]]:
        if distance_angstrom <= 0.0 or distance_angstrom >= self.cutoff_angstrom:
            zeros = [0.0] * len(self.centers_angstrom)
            return zeros, zeros.copy()
        angle = math.pi * distance_angstrom / self.cutoff_angstrom
        cutoff = 0.5 * (math.cos(angle) + 1.0)
        cutoff_derivative = -0.5 * math.pi / self.cutoff_angstrom * math.sin(angle)
        values: List[float] = []
        derivatives: List[float] = []
        for center in self.centers_angstrom:
            delta = distance_angstrom - center
            gaussian = math.exp(-0.5 * (delta / self.sigma_angstrom) ** 2)
            gaussian_derivative = -delta / (self.sigma_angstrom**2) * gaussian
            values.append(gaussian * cutoff)
            derivatives.append(gaussian_derivative * cutoff + gaussian * cutoff_derivative)
        return values, derivatives


@dataclass(frozen=True)
class SmoothRepulsiveCore:
    amplitude_ev: float = 500.0
    cutoff_angstrom: float = 2.2
    power: int = 4

    def value_and_derivative(self, distance_angstrom: float) -> Tuple[float, float]:
        if distance_angstrom <= 0.0:
            raise ValueError("distance must be positive")
        if distance_angstrom >= self.cutoff_angstrom:
            return 0.0, 0.0
        reduced = 1.0 - distance_angstrom / self.cutoff_angstrom
        value = self.amplitude_ev * reduced**self.power
        derivative = (
            -self.amplitude_ev
            * self.power
            / self.cutoff_angstrom
            * reduced ** (self.power - 1)
        )
        return value, derivative


@dataclass(frozen=True)
class PairVirialResult:
    potential_energy_ev: float
    configurational_virial_ev: float
    nearest_neighbor_angstrom: float


def evaluate_pair_virial(
    positions_angstrom: Sequence[Sequence[float]],
    lattice_angstrom: Sequence[Sequence[float]],
    model: Mapping[str, object],
) -> PairVirialResult:
    """Evaluate the fitted pair energy and scalar configurational virial."""
    centers = [float(value) for value in model["centers_angstrom"]]
    coefficients = [float(value) for value in model["coefficients_ev"]]
    if len(coefficients) != len(centers) + 1:
        raise ValueError("pair model must contain one constant and one coefficient per center")
    basis = RadialBasis(
        centers_angstrom=centers,
        sigma_angstrom=float(model["sigma_angstrom"]),
        cutoff_angstrom=float(model["cutoff_angstrom"]),
    )
    core_document = model["repulsive_core"]
    if not isinstance(core_document, Mapping):
        raise ValueError("repulsive_core must be a mapping")
    core = SmoothRepulsiveCore(
        amplitude_ev=float(core_document["amplitude_ev"]),
        cutoff_angstrom=float(core_document["cutoff_angstrom"]),
        power=int(core_document["power"]),
    )
    lattice = tuple(tuple(float(value) for value in row) for row in lattice_angstrom)
    inverse = invert_3x3(lattice)
    fractional = [matmul_row(position, inverse) for position in positions_angstrom]
    energy = coefficients[0] * len(fractional)
    virial = 0.0
    nearest = math.inf
    for i, left in enumerate(fractional):
        for right in fractional[i + 1 :]:
            delta_fractional = tuple(
                (right[axis] - left[axis]) - round(right[axis] - left[axis])
                for axis in range(3)
            )
            displacement = matmul_row(delta_fractional, lattice)
            distance = math.sqrt(sum(value * value for value in displacement))
            nearest = min(nearest, distance)
            values, derivatives = basis.values_and_derivatives(distance)
            pair_energy = sum(
                coefficient * value
                for coefficient, value in zip(coefficients[1:], values)
            )
            pair_derivative = sum(
                coefficient * derivative
                for coefficient, derivative in zip(coefficients[1:], derivatives)
            )
            core_energy, core_derivative = core.value_and_derivative(distance)
            energy += pair_energy + core_energy
            virial -= distance * (pair_derivative + core_derivative)
    return PairVirialResult(
        potential_energy_ev=energy,
        configurational_virial_ev=virial,
        nearest_neighbor_angstrom=nearest,
    )


def pair_pressure_kbar(
    result: PairVirialResult,
    natoms: int,
    temperature_k: float,
    lattice_angstrom: Sequence[Sequence[float]],
) -> float:
    volume = lattice_volume(tuple(tuple(float(value) for value in row) for row in lattice_angstrom))
    pressure_ev_per_angstrom3 = (
        natoms * KB_EV_PER_K * temperature_k
        + result.configurational_virial_ev / 3.0
    ) / volume
    return pressure_ev_per_angstrom3 * EV_PER_ANGSTROM3_TO_KBAR


def evenly_spaced_centers(start: float, stop: float, count: int) -> List[float]:
    if count < 2:
        raise ValueError("count must be at least two")
    if stop <= start:
        raise ValueError("stop must exceed start")
    spacing = (stop - start) / (count - 1)
    return [start + index * spacing for index in range(count)]
