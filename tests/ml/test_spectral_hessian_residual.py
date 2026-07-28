from __future__ import annotations

import torch

from mldft.ml.models.components.spectral_hessian_residual import (
    anchored_cartesian_quadratic_energy,
    equivariant_block_operator_basis,
    spectral_operator_basis,
    symmetric_hvp_interpolant,
)


def test_spectral_basis_is_orthogonally_equivariant() -> None:
    generator = torch.Generator().manual_seed(67)
    raw = torch.randn(9, 9, dtype=torch.float64, generator=generator)
    source = 0.5 * (raw + raw.T)
    external = torch.linalg.qr(
        torch.randn(9, 3, dtype=torch.float64, generator=generator), mode="reduced"
    ).Q
    rotation = torch.linalg.qr(
        torch.randn(9, 9, dtype=torch.float64, generator=generator), mode="complete"
    ).Q
    basis, names, _, _ = spectral_operator_basis(source, external)
    transformed, transformed_names, _, _ = spectral_operator_basis(
        rotation @ source @ rotation.T, rotation @ external
    )
    expected = torch.einsum("ij,mjk,lk->mil", rotation, basis, rotation)

    assert names == transformed_names
    assert torch.allclose(transformed, expected, atol=1e-10, rtol=1e-10)


def test_anchored_spectral_scalar_owns_exact_energy_force_and_hessian() -> None:
    generator = torch.Generator().manual_seed(71)
    reference = torch.randn(4, 3, dtype=torch.float64, generator=generator)
    raw = torch.randn(12, 12, dtype=torch.float64, generator=generator)
    correction = 0.5 * (raw + raw.T)
    geometry = reference.detach().requires_grad_(True)
    energy = anchored_cartesian_quadratic_energy(geometry, reference, correction)
    force = -torch.autograd.grad(energy, geometry, create_graph=True)[0]
    hessian = torch.func.hessian(
        lambda value: anchored_cartesian_quadratic_energy(value, reference, correction)
    )(reference).reshape(12, 12)

    assert float(energy) == 0.0
    assert torch.count_nonzero(force) == 0
    assert torch.allclose(hessian, correction, atol=1e-12, rtol=1e-12)


def test_block_operator_basis_is_rotation_equivariant_and_internal() -> None:
    generator = torch.Generator().manual_seed(73)
    atomic_numbers = torch.tensor([1, 6, 8], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.1, 0.3, -0.1], [3.2, 1.7, 0.4]],
        dtype=torch.float64,
    )
    raw = torch.randn(9, 9, dtype=torch.float64, generator=generator)
    source = 0.5 * (raw + raw.T)
    external = torch.linalg.qr(
        torch.randn(9, 3, dtype=torch.float64, generator=generator), mode="reduced"
    ).Q
    rotation = torch.linalg.qr(
        torch.randn(3, 3, dtype=torch.float64, generator=generator), mode="complete"
    ).Q
    coordinate_rotation = torch.block_diag(rotation, rotation, rotation)
    basis, names = equivariant_block_operator_basis(
        source, atomic_numbers, positions, external
    )
    transformed, transformed_names = equivariant_block_operator_basis(
        coordinate_rotation @ source @ coordinate_rotation.T,
        atomic_numbers,
        positions @ rotation.T,
        coordinate_rotation @ external,
    )
    expected = torch.einsum(
        "ij,mjk,lk->mil", coordinate_rotation, basis, coordinate_rotation
    )

    assert names == transformed_names
    assert torch.allclose(transformed, expected, atol=1e-9, rtol=1e-9)
    assert torch.max(torch.abs(torch.einsum("mij,jk->mik", basis, external))) < 1e-9


def test_symmetric_hvp_interpolant_is_exact_symmetric_and_equivariant() -> None:
    generator = torch.Generator().manual_seed(79)
    raw = torch.randn(12, 12, dtype=torch.float64, generator=generator)
    reference = 0.5 * (raw + raw.T)
    directions = torch.linalg.qr(
        torch.randn(12, 7, dtype=torch.float64, generator=generator), mode="reduced"
    ).Q.T
    target = directions @ reference.T
    interpolant = symmetric_hvp_interpolant(directions, target)
    rotation = torch.linalg.qr(
        torch.randn(12, 12, dtype=torch.float64, generator=generator), mode="complete"
    ).Q
    transformed = symmetric_hvp_interpolant(
        directions @ rotation.T,
        target @ rotation.T,
    )

    assert torch.max(torch.abs(interpolant - interpolant.T)) < 1e-12
    assert torch.allclose(directions @ interpolant.T, target, atol=1e-11, rtol=1e-10)
    assert torch.allclose(
        transformed,
        rotation @ interpolant @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )
