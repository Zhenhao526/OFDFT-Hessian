import torch

from scripts.run_pair_reference_md import random_velocities, temperature


def test_random_velocities_respect_explicit_magnesium_mass():
    mass_amu = 24.305
    velocities = random_velocities(
        128, 900.0, 1234, torch.device("cpu"), mass_amu
    )

    assert abs(temperature(velocities, mass_amu) - 900.0) < 1.0e-9


def test_default_mass_remains_backward_compatible():
    velocities = random_velocities(32, 800.0, 4321, torch.device("cpu"))

    assert abs(temperature(velocities) - 800.0) < 1.0e-9
