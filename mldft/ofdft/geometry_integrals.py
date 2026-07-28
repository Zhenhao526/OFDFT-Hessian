"""Geometry-dependent auxiliary-basis integrals for conservative OFDFT derivatives.

PySCF/libcint is not a torch-autograd library. This module makes that boundary explicit: a
provider returns integral values together with nuclear-coordinate derivatives. The finite-
difference provider is deliberately a slow reference implementation that includes both operator
and moving atom-centred basis (Pulay) contributions. It is intended to validate total forces and
future analytic integral derivative implementations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pyscf import gto

from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_normalization_vector,
    get_nuclear_attraction_vector,
    get_overlap_coordinate_derivatives,
    get_overlap_matrix,
)


@dataclass(frozen=True)
class GeometryIntegralValues:
    """Auxiliary-basis integral values at one nuclear geometry."""

    normalization: np.ndarray
    overlap: np.ndarray
    coulomb: np.ndarray
    nuclear_attraction: np.ndarray
    nuclear_repulsion: float


@dataclass(frozen=True)
class GeometryIntegralDerivatives:
    """First derivatives with leading flattened nuclear-coordinate dimension."""

    normalization: np.ndarray
    overlap: np.ndarray
    coulomb: np.ndarray
    nuclear_attraction: np.ndarray
    nuclear_repulsion: np.ndarray


@dataclass(frozen=True)
class GeometryIntegralBundle:
    """Integral values and complete numerical first nuclear derivatives."""

    values: GeometryIntegralValues
    derivatives: GeometryIntegralDerivatives
    positions_bohr: np.ndarray
    derivative_step_bohr: float
    directional_second_derivatives: GeometryIntegralDerivatives | None = None
    direction: np.ndarray | None = None
    directional_second_step_bohr: float | None = None


class FiniteDifferencePySCFIntegralProvider:
    """Reference PySCF integral provider including moving-basis/Pulay derivatives.

    The same atom-centred auxiliary basis is rebuilt at each displaced geometry. Consequently the
    central differences include derivatives of basis centres, nuclear operators, Coulomb
    integrals, overlap/natural-reparametrization inputs, and the electron-number constraint.
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray | torch.Tensor,
        basis: str | dict[str, Any],
        charge: int = 0,
        spin: int | None = None,
        derivative_step_bohr: float = 1e-4,
        analytic_overlap_derivative: bool = True,
        derivative_workers: int = 1,
    ) -> None:
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        if atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must be one-dimensional")
        if derivative_step_bohr <= 0:
            raise ValueError("derivative_step_bohr must be positive")
        self.atomic_numbers = atomic_numbers
        self.basis = basis
        self.charge = int(charge)
        nelectron = int(np.sum(atomic_numbers) - charge)
        self.spin = nelectron % 2 if spin is None else int(spin)
        self.derivative_step_bohr = float(derivative_step_bohr)
        self.analytic_overlap_derivative = bool(analytic_overlap_derivative)
        if derivative_workers <= 0:
            raise ValueError("derivative_workers must be positive")
        self.derivative_workers = int(derivative_workers)

    def build_molecule(self, positions_bohr: np.ndarray) -> gto.Mole:
        positions_bohr = np.asarray(positions_bohr, dtype=np.float64)
        expected_shape = (self.atomic_numbers.size, 3)
        if positions_bohr.shape != expected_shape:
            raise ValueError(
                f"positions_bohr must have shape {expected_shape}, got {positions_bohr.shape}"
            )
        atoms = [
            (int(charge), tuple(float(value) for value in position))
            for charge, position in zip(self.atomic_numbers, positions_bohr)
        ]
        return gto.M(
            atom=atoms,
            unit="Bohr",
            charge=self.charge,
            spin=self.spin,
            basis=self.basis,
            verbose=0,
        )

    def evaluate(self, positions_bohr: np.ndarray) -> GeometryIntegralValues:
        mol = self.build_molecule(positions_bohr)
        return GeometryIntegralValues(
            normalization=np.asarray(get_normalization_vector(mol), dtype=np.float64),
            overlap=np.asarray(get_overlap_matrix(mol), dtype=np.float64),
            coulomb=np.asarray(get_coulomb_matrix(mol), dtype=np.float64),
            nuclear_attraction=np.asarray(
                get_nuclear_attraction_vector(mol), dtype=np.float64
            ),
            nuclear_repulsion=float(mol.energy_nuc()),
        )

    def _evaluate_displaced_derivative_fields(
        self, positions_bohr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Evaluate only fields that require numerical moving-basis derivatives."""
        mol = self.build_molecule(positions_bohr)
        overlap = (
            None
            if self.analytic_overlap_derivative
            else np.asarray(get_overlap_matrix(mol), dtype=np.float64)
        )
        return (
            np.asarray(get_coulomb_matrix(mol), dtype=np.float64),
            np.asarray(get_nuclear_attraction_vector(mol), dtype=np.float64),
            overlap,
        )

    def _nuclear_repulsion_derivative(
        self, positions_bohr: np.ndarray
    ) -> np.ndarray:
        derivative = np.zeros_like(positions_bohr, dtype=np.float64)
        charges = self.atomic_numbers.astype(np.float64)
        for atom_i in range(self.atomic_numbers.size):
            for atom_j in range(atom_i + 1, self.atomic_numbers.size):
                displacement = positions_bohr[atom_i] - positions_bohr[atom_j]
                distance = np.linalg.norm(displacement)
                pair_derivative = (
                    -charges[atom_i]
                    * charges[atom_j]
                    * displacement
                    / distance**3
                )
                derivative[atom_i] += pair_derivative
                derivative[atom_j] -= pair_derivative
        return derivative.reshape(-1)

    def evaluate_with_derivatives(
        self, positions_bohr: np.ndarray
    ) -> GeometryIntegralBundle:
        positions_bohr = np.asarray(positions_bohr, dtype=np.float64)
        values = self.evaluate(positions_bohr)
        n_coord = positions_bohr.size
        coulomb_derivatives: list[np.ndarray] = []
        attraction_derivatives: list[np.ndarray] = []
        overlap_derivatives: list[np.ndarray] = []
        step = self.derivative_step_bohr
        flat = positions_bohr.reshape(-1)
        def coordinate_derivative(
            coordinate: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
            plus = flat.copy()
            minus = flat.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            plus_fields = self._evaluate_displaced_derivative_fields(
                plus.reshape(positions_bohr.shape)
            )
            minus_fields = self._evaluate_displaced_derivative_fields(
                minus.reshape(positions_bohr.shape)
            )
            overlap_derivative = None
            if not self.analytic_overlap_derivative:
                assert plus_fields[2] is not None and minus_fields[2] is not None
                overlap_derivative = (plus_fields[2] - minus_fields[2]) / (2.0 * step)
            return (
                (plus_fields[0] - minus_fields[0]) / (2.0 * step),
                (plus_fields[1] - minus_fields[1]) / (2.0 * step),
                overlap_derivative,
            )

        if self.derivative_workers == 1:
            coordinate_results = map(coordinate_derivative, range(n_coord))
            for coulomb, attraction, overlap in coordinate_results:
                coulomb_derivatives.append(coulomb)
                attraction_derivatives.append(attraction)
                if overlap is not None:
                    overlap_derivatives.append(overlap)
        else:
            with ThreadPoolExecutor(max_workers=self.derivative_workers) as executor:
                coordinate_results = executor.map(coordinate_derivative, range(n_coord))
                for coulomb, attraction, overlap in coordinate_results:
                    coulomb_derivatives.append(coulomb)
                    attraction_derivatives.append(attraction)
                    if overlap is not None:
                        overlap_derivatives.append(overlap)

        overlap_derivative = (
            get_overlap_coordinate_derivatives(self.build_molecule(positions_bohr))
            if self.analytic_overlap_derivative
            else np.asarray(overlap_derivatives, dtype=np.float64)
        )
        derivatives = GeometryIntegralDerivatives(
            # Integrals of normalized atom-centred auxiliary functions do not change when their
            # centres move. Keeping this exact also stabilizes the KKT constraint.
            normalization=np.zeros((n_coord, values.normalization.size), dtype=np.float64),
            overlap=overlap_derivative,
            coulomb=np.asarray(coulomb_derivatives, dtype=np.float64),
            nuclear_attraction=np.asarray(attraction_derivatives, dtype=np.float64),
            nuclear_repulsion=self._nuclear_repulsion_derivative(positions_bohr),
        )
        return GeometryIntegralBundle(
            values=values,
            derivatives=derivatives,
            positions_bohr=positions_bohr.copy(),
            derivative_step_bohr=step,
        )

    def evaluate_with_directional_second_derivatives(
        self,
        positions_bohr: np.ndarray,
        direction: np.ndarray | torch.Tensor,
        *,
        directional_step_bohr: float | None = None,
    ) -> GeometryIntegralBundle:
        """Return first derivatives and their derivative along one coordinate direction.

        PySCF/libcint does not participate in torch autograd.  The directional
        second derivatives are therefore evaluated numerically from first-
        derivative bundles, while the density response and model derivatives
        remain analytic.  This requires one central density optimization and no
        density optimization at displaced geometries.
        """
        positions_bohr = np.asarray(positions_bohr, dtype=np.float64)
        direction_np = np.asarray(
            direction.detach().cpu() if isinstance(direction, torch.Tensor) else direction,
            dtype=np.float64,
        )
        if direction_np.shape != positions_bohr.shape:
            raise ValueError(
                f"direction must have shape {positions_bohr.shape}, got {direction_np.shape}"
            )
        if not np.all(np.isfinite(direction_np)):
            raise ValueError("direction must be finite")
        if float(np.linalg.norm(direction_np)) == 0.0:
            raise ValueError("direction must be nonzero")
        step = (
            self.derivative_step_bohr
            if directional_step_bohr is None
            else float(directional_step_bohr)
        )
        if step <= 0:
            raise ValueError("directional_step_bohr must be positive")

        central = self.evaluate_with_derivatives(positions_bohr)
        plus = self.evaluate_with_derivatives(positions_bohr + step * direction_np)
        minus = self.evaluate_with_derivatives(positions_bohr - step * direction_np)

        def difference(field: str) -> np.ndarray:
            return (
                np.asarray(getattr(plus.derivatives, field), dtype=np.float64)
                - np.asarray(getattr(minus.derivatives, field), dtype=np.float64)
            ) / (2.0 * step)

        return GeometryIntegralBundle(
            values=central.values,
            derivatives=central.derivatives,
            positions_bohr=central.positions_bohr,
            derivative_step_bohr=central.derivative_step_bohr,
            directional_second_derivatives=GeometryIntegralDerivatives(
                normalization=difference("normalization"),
                overlap=difference("overlap"),
                coulomb=difference("coulomb"),
                nuclear_attraction=difference("nuclear_attraction"),
                nuclear_repulsion=difference("nuclear_repulsion"),
            ),
            direction=direction_np.copy(),
            directional_second_step_bohr=step,
        )


def linearized_integral_tensor(
    value: np.ndarray | float,
    coordinate_derivative: np.ndarray,
    positions: torch.Tensor,
    *,
    directional_second_derivative: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> torch.Tensor:
    """Attach an externally evaluated value and first derivative to a coordinate graph.

    The returned tensor is exact to first order at ``positions.detach()``. Rebuilding the bundle
    at each force/Hessian geometry supplies the local derivative there; finite differences of
    those complete forces therefore include second geometry derivatives without differentiating
    through PySCF itself.
    """
    dtype = positions.dtype
    device = positions.device
    value_tensor = torch.as_tensor(value, dtype=dtype, device=device)
    derivative_tensor = torch.as_tensor(coordinate_derivative, dtype=dtype, device=device)
    displacement = positions.reshape(-1) - positions.detach().reshape(-1)
    result = value_tensor + torch.tensordot(
        displacement, derivative_tensor, dims=([0], [0])
    )
    if directional_second_derivative is None and direction is None:
        return result
    if directional_second_derivative is None or direction is None:
        raise ValueError(
            "directional_second_derivative and direction must be supplied together"
        )
    direction_tensor = torch.as_tensor(
        direction, dtype=dtype, device=device
    ).reshape(-1)
    if direction_tensor.shape != displacement.shape:
        raise ValueError(
            "direction must have the same flattened coordinate shape as positions"
        )
    norm_squared = torch.dot(direction_tensor, direction_tensor)
    if float(norm_squared.detach().cpu()) == 0.0:
        raise ValueError("direction must be nonzero")
    directional_second_tensor = torch.as_tensor(
        directional_second_derivative, dtype=dtype, device=device
    )
    if directional_second_tensor.shape[0] != displacement.numel():
        raise ValueError(
            "directional_second_derivative must have a leading coordinate dimension"
        )

    # This minimum-rank quadratic extension leaves the value and first
    # derivative unchanged at the expansion point and has exactly
    # ``d/dR(grad_R I) @ direction`` as its Hessian-vector product there.
    alpha = torch.dot(displacement, direction_tensor) / norm_squared
    displacement_contraction = torch.tensordot(
        displacement, directional_second_tensor, dims=([0], [0])
    )
    direction_contraction = torch.tensordot(
        direction_tensor, directional_second_tensor, dims=([0], [0])
    )
    return (
        result
        + alpha * displacement_contraction
        - 0.5 * alpha.square() * direction_contraction
    )


def integral_tensor_from_bundle(
    value: np.ndarray | float,
    coordinate_derivative: np.ndarray,
    positions: torch.Tensor,
    bundle: GeometryIntegralBundle,
    *,
    directional_second_derivative: np.ndarray | None = None,
) -> torch.Tensor:
    """Attach the first-order or directionally second-order bundle graph."""
    if bundle.directional_second_derivatives is None:
        return linearized_integral_tensor(value, coordinate_derivative, positions)
    if bundle.direction is None or directional_second_derivative is None:
        raise ValueError("directional integral bundle is incomplete")
    return linearized_integral_tensor(
        value,
        coordinate_derivative,
        positions,
        directional_second_derivative=directional_second_derivative,
        direction=bundle.direction,
    )


def classical_energy_from_bundle(
    coeffs: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bundle: GeometryIntegralBundle,
) -> torch.Tensor:
    """Assemble differentiable Hartree, external, and nuclear energies from a bundle."""
    from mldft.ofdft.functional_factory import (
        hartree_energy_tensor,
        nuclear_attraction_energy_tensor,
        nuclear_repulsion_energy_tensor,
    )

    directional = bundle.directional_second_derivatives
    coulomb = integral_tensor_from_bundle(
        bundle.values.coulomb,
        bundle.derivatives.coulomb,
        positions,
        bundle,
        directional_second_derivative=(
            None if directional is None else directional.coulomb
        ),
    )
    attraction = integral_tensor_from_bundle(
        bundle.values.nuclear_attraction,
        bundle.derivatives.nuclear_attraction,
        positions,
        bundle,
        directional_second_derivative=(
            None if directional is None else directional.nuclear_attraction
        ),
    )
    return (
        hartree_energy_tensor(coeffs, coulomb)
        + nuclear_attraction_energy_tensor(coeffs, attraction)
        + nuclear_repulsion_energy_tensor(positions, atomic_numbers)
    )
