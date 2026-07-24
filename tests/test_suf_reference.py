from __future__ import annotations

import math

import torch

from mpn_melting.suf_reference import SUFParameters, evaluate_suf, suf_reduced_density


def test_suf_force_matches_finite_difference() -> None:
    positions = torch.tensor([[1.0, 1.0, 1.0], [3.2, 1.0, 1.0]], dtype=torch.float64)
    lattice = torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))
    parameters = SUFParameters(p=50, sigma_angstrom=1.2, temperature_k=900.0)
    _, forces, nearest = evaluate_suf(positions, lattice, parameters)
    step = 1.0e-5
    plus = positions.clone()
    minus = positions.clone()
    plus[0, 0] += step
    minus[0, 0] -= step
    energy_plus = float(evaluate_suf(plus, lattice, parameters)[0])
    energy_minus = float(evaluate_suf(minus, lattice, parameters)[0])
    finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step)
    assert math.isclose(float(nearest), 2.2, rel_tol=1.0e-12)
    assert math.isclose(float(forces[0, 0]), finite_difference_force, rel_tol=1.0e-8)
    assert math.isclose(float(forces[0, 0]), -float(forces[1, 0]), rel_tol=1.0e-12)


def test_suf_reduced_density() -> None:
    expected = 0.5 * (math.pi * 1.2**2) ** 1.5 * 0.055
    assert math.isclose(suf_reduced_density(0.055, 1.2), expected)
