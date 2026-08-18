"""Torch-autograd Gaussian integrals for geometry derivatives.

The scalar OFDFT model, its density response, and the geometry integrals must
share one automatic-differentiation system.  This module uses the molecular
integral layer from DQC, whose libcint calls are exposed as
``torch.autograd.Function`` primitives with higher-order backward rules.  No
JAX dependency or nuclear-coordinate finite difference is permitted here.

Only the small DQC molecular-integral subset is loaded.  The dependency lives
in a pinned, isolated overlay selected by ``MLDFT_DQC_OVERLAY`` so importing it
cannot mutate the established training environment.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyscf import gto

from mldft.ofdft.basis_integrals import FLAT_GAUSSIAN_CORRECTION


TORCH_INTEGRAL_BACKEND = "torch_autograd_dqc_libcint"
TORCH_DIRECTIONAL_SECOND_BACKEND = "torch_autograd_double_backward"
_DQC_COMMIT = "0fe821fc92cb3457fb14f6dff0c223641c514ddb"
_DQCLIBS_WHEEL_SHA256 = (
    "912a0103f2157a89a1117f8b2a432a9fd32ceb020e1cdcb89948ae4fac2da083"
)


@dataclass(frozen=True)
class TorchGeometryIntegralValues:
    """Geometry-dependent tensors that remain connected to ``positions``."""

    normalization: torch.Tensor
    overlap: torch.Tensor
    coulomb: torch.Tensor
    nuclear_attraction: torch.Tensor
    derivative_backend: str = TORCH_INTEGRAL_BACKEND
    directional_second_backend: str = TORCH_DIRECTIONAL_SECOND_BACKEND


_DQC_MODULES: tuple[Any, Any, Any, Any, Any] | None = None
_DQC_IMPORT_LOCK = threading.Lock()


def _install_namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if path.resolve() not in existing_paths:
            raise RuntimeError(
                f"refusing an already imported foreign {name!r} package: "
                f"{existing_paths}"
            )
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


def _load_dqc_integral_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Load the pinned DQC integral subset without importing its full QC stack."""
    global _DQC_MODULES
    with _DQC_IMPORT_LOCK:
        if _DQC_MODULES is not None:
            return _DQC_MODULES

        overlay_value = os.environ.get("MLDFT_DQC_OVERLAY")
        if not overlay_value:
            raise RuntimeError(
                "torch_autograd_dqc requires MLDFT_DQC_OVERLAY; JAX and "
                "finite-difference fallback are forbidden"
            )
        overlay = Path(overlay_value).expanduser().resolve()
        source_root = overlay / "src" / "dqc" / "dqc"
        site_packages = overlay / "site-packages"
        provenance = overlay / "PROVENANCE.txt"
        required = (
            source_root / "hamilton" / "intor" / "molintor.py",
            source_root / "hamilton" / "intor" / "lcintwrap.py",
            site_packages / "dqclibs" / "__init__.py",
            provenance,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "incomplete pinned DQC torch-integral overlay: " + ", ".join(missing)
            )
        provenance_text = provenance.read_text()
        required_provenance = (
            f"dqc_git_commit={_DQC_COMMIT}",
            f"dqclibs_wheel_sha256={_DQCLIBS_WHEEL_SHA256}",
            "autodiff_owner=torch",
            "jax_forbidden=true",
        )
        if any(item not in provenance_text for item in required_provenance):
            raise RuntimeError("DQC torch-integral overlay provenance mismatch")

        sys.path[:0] = [str(site_packages), str(overlay / "src" / "dqc")]
        _install_namespace("dqc", source_root)
        _install_namespace("dqc.hamilton", source_root / "hamilton")
        _install_namespace("dqc.hamilton.intor", source_root / "hamilton" / "intor")

        from dqc.hamilton.intor.lcintwrap import LibcintWrapper
        from dqc.hamilton.intor.molintor import coul2c, nuclattr, overlap
        from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis

        _DQC_MODULES = (
            AtomCGTOBasis,
            CGTOBasis,
            LibcintWrapper,
            overlap,
            (coul2c, nuclattr),
        )
        return _DQC_MODULES


class TorchAutogradLibcintIntegralProvider:
    """Build DQC/libcint tensors whose first and second derivatives are Torch-owned."""

    backend_name = TORCH_INTEGRAL_BACKEND
    directional_second_backend = TORCH_DIRECTIONAL_SECOND_BACKEND

    def __init__(
        self,
        atomic_numbers: np.ndarray | torch.Tensor,
        basis: str | dict[str, Any],
        charge: int = 0,
        derivative_step_bohr: float = 0.0,
        derivative_workers: int = 1,
    ) -> None:
        if derivative_step_bohr != 0.0:
            raise ValueError(
                "torch_autograd_dqc requires derivative_step_bohr=0; "
                "coordinate finite differences are forbidden"
            )
        if derivative_workers <= 0:
            raise ValueError("derivative_workers must be positive")
        numbers = np.asarray(
            atomic_numbers.detach().cpu()
            if isinstance(atomic_numbers, torch.Tensor)
            else atomic_numbers,
            dtype=np.int64,
        )
        if numbers.ndim != 1 or numbers.size == 0:
            raise ValueError("atomic_numbers must be a nonempty one-dimensional array")
        self.atomic_numbers = numbers
        self.basis = basis
        self.charge = int(charge)
        self.derivative_step_bohr = 0.0
        self.derivative_workers = int(derivative_workers)
        self._basis_by_atomic_number = self._canonical_basis_by_atomic_number()

    def _canonical_basis_by_atomic_number(self) -> dict[int, list[Any]]:
        """Let PySCF parse the frozen basis, but never differentiate through it."""
        result: dict[int, list[Any]] = {}
        for atomic_number in np.unique(self.atomic_numbers):
            z = int(atomic_number)
            atom = gto.M(
                atom=[(z, (0.0, 0.0, 0.0))],
                unit="Bohr",
                charge=0,
                spin=z % 2,
                basis=self.basis,
                verbose=0,
            )
            result[z] = atom._basis[atom.atom_symbol(0)]
        return result

    @staticmethod
    def _dqc_shells(
        shell_data: list[Any],
        *,
        dtype: torch.dtype,
        device: torch.device,
        cgto_basis: Any,
    ) -> list[Any]:
        shells: list[Any] = []
        for shell in shell_data:
            if not isinstance(shell, (list, tuple)) or len(shell) < 2:
                raise ValueError(f"unsupported PySCF basis shell {shell!r}")
            angular_momentum = int(shell[0])
            primitive_rows = list(shell[1:])
            if primitive_rows and isinstance(primitive_rows[0], (int, np.integer)):
                raise ValueError("spinor/kappa PySCF shells are not supported")
            if not primitive_rows or any(
                not isinstance(row, (list, tuple)) or len(row) < 2
                for row in primitive_rows
            ):
                raise ValueError(f"malformed PySCF primitive rows in shell {shell!r}")
            contraction_count = len(primitive_rows[0]) - 1
            if any(len(row) != contraction_count + 1 for row in primitive_rows):
                raise ValueError("inconsistent contraction columns in PySCF shell")
            exponents = torch.tensor(
                [float(row[0]) for row in primitive_rows],
                dtype=dtype,
                device=device,
            )
            for contraction in range(contraction_count):
                coefficients = torch.tensor(
                    [float(row[contraction + 1]) for row in primitive_rows],
                    dtype=dtype,
                    device=device,
                )
                shells.append(
                    cgto_basis(
                        angular_momentum,
                        exponents.clone(),
                        coefficients,
                        normalized=False,
                    )
                )
        return shells

    def evaluate(self, positions_bohr: torch.Tensor) -> TorchGeometryIntegralValues:
        if positions_bohr.shape != (self.atomic_numbers.size, 3):
            raise ValueError(
                "positions_bohr must have shape "
                f"{(self.atomic_numbers.size, 3)}, got {tuple(positions_bohr.shape)}"
            )
        if positions_bohr.device.type != "cpu":
            raise ValueError("DQC/libcint geometry integrals currently require CPU positions")
        if not positions_bohr.is_floating_point():
            raise ValueError("positions_bohr must have a floating dtype")
        if not bool(torch.isfinite(positions_bohr).all()):
            raise ValueError("positions_bohr must be finite")

        atom_basis_type, cgto_basis_type, wrapper_type, overlap, pair = (
            _load_dqc_integral_modules()
        )
        coul2c, nuclattr = pair
        atombases = []
        for atom_index, atomic_number in enumerate(self.atomic_numbers):
            z = int(atomic_number)
            atombases.append(
                atom_basis_type(
                    atomz=z,
                    bases=self._dqc_shells(
                        self._basis_by_atomic_number[z],
                        dtype=positions_bohr.dtype,
                        device=positions_bohr.device,
                        cgto_basis=cgto_basis_type,
                    ),
                    pos=positions_bohr[atom_index],
                )
            )
        auxiliary = wrapper_type(atombases, spherical=True)
        flat = wrapper_type(
            [
                atom_basis_type(
                    atomz=0,
                    bases=[
                        cgto_basis_type(
                            0,
                            torch.tensor(
                                [0.0],
                                dtype=positions_bohr.dtype,
                                device=positions_bohr.device,
                            ),
                            torch.tensor(
                                [1.0],
                                dtype=positions_bohr.dtype,
                                device=positions_bohr.device,
                            ),
                            normalized=True,
                        )
                    ],
                    pos=positions_bohr[0],
                )
            ],
            spherical=True,
        )
        auxiliary, flat = wrapper_type.concatenate(auxiliary, flat)
        values = TorchGeometryIntegralValues(
            normalization=(
                overlap(auxiliary, other=flat).squeeze(-1)
                * FLAT_GAUSSIAN_CORRECTION
            ),
            overlap=overlap(auxiliary),
            coulomb=coul2c(auxiliary),
            nuclear_attraction=(
                nuclattr(auxiliary, other=flat).squeeze(-1)
                * FLAT_GAUSSIAN_CORRECTION
            ),
        )
        tensors = (
            values.normalization,
            values.overlap,
            values.coulomb,
            values.nuclear_attraction,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("torch-autograd geometry integrals are non-finite")
        return values
