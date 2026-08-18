"""Small density-aware, Euclidean-covariant Cartesian Hessian readout.

The head predicts atom-pair and atom-diagonal 3x3 blocks from invariant
geometry/density features.  The assembled matrix is symmetrized and projected
onto the molecular internal subspace, so symmetry and rigid-motion null modes
hold by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


QM9_ELEMENTS = (1, 6, 7, 8, 9)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _outer(first: Tensor, second: Tensor) -> Tensor:
    return first[:, :, None] * second[:, None, :]


class StructuredDensityHessianHead(nn.Module):
    """Predict a variable-size molecular Hessian with exact structural constraints.

    Density inputs are per-atom invariant summaries of the converged auxiliary
    density coefficients.  Setting ``use_density=False`` produces the matched
    geometry-only ablation.
    """

    pair_basis_count = 8
    node_basis_count = 3

    def __init__(
        self,
        density_feature_dim: int,
        *,
        hidden_dim: int = 64,
        radial_centers_bohr: Sequence[float] = (
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
        ),
        radial_width_bohr: float = 0.75,
        environment_scale_bohr: float = 4.0,
        use_density: bool = True,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        if density_feature_dim <= 0:
            raise ValueError("density_feature_dim must be positive")
        if hidden_dim <= 0 or radial_width_bohr <= 0.0 or environment_scale_bohr <= 0.0:
            raise ValueError("invalid structured-head hyperparameters")
        centers = torch.as_tensor(tuple(radial_centers_bohr), dtype=torch.float64)
        if centers.ndim != 1 or centers.numel() == 0:
            raise ValueError("radial centers must be a nonempty sequence")
        self.density_feature_dim = int(density_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.radial_width_bohr = float(radial_width_bohr)
        self.environment_scale_bohr = float(environment_scale_bohr)
        self.use_density = bool(use_density)
        self.register_buffer("radial_centers_bohr", centers)

        pair_type_count = len(QM9_ELEMENTS) * (len(QM9_ELEMENTS) + 1) // 2
        pair_environment_feature_count = 8
        pair_density_feature_count = (
            3 * self.density_feature_dim if self.use_density else 0
        )
        pair_input_dim = (
            pair_type_count
            + centers.numel()
            + pair_environment_feature_count
            + pair_density_feature_count
        )
        node_environment_feature_count = 4
        node_density_feature_count = self.density_feature_dim if self.use_density else 0
        node_input_dim = (
            len(QM9_ELEMENTS)
            + node_environment_feature_count
            + node_density_feature_count
        )
        self.pair_mlp = _mlp(
            pair_input_dim, self.hidden_dim, self.pair_basis_count
        )
        self.node_mlp = _mlp(
            node_input_dim, self.hidden_dim, self.node_basis_count
        )
        if zero_initialize:
            for readout in (self.pair_mlp[-1], self.node_mlp[-1]):
                nn.init.zeros_(readout.weight)
                nn.init.zeros_(readout.bias)

    @staticmethod
    def _pair_type_features(atomic_numbers: Tensor, first: Tensor, second: Tensor) -> Tensor:
        pair_types = [
            (left, right)
            for left_index, left in enumerate(QM9_ELEMENTS)
            for right in QM9_ELEMENTS[left_index:]
        ]
        left = atomic_numbers[first]
        right = atomic_numbers[second]
        low = torch.minimum(left, right)
        high = torch.maximum(left, right)
        return torch.stack(
            [
                ((low == pair_left) & (high == pair_right)).to(
                    dtype=torch.get_default_dtype()
                )
                for pair_left, pair_right in pair_types
            ],
            dim=1,
        )

    @staticmethod
    def _node_type_features(atomic_numbers: Tensor, dtype: torch.dtype) -> Tensor:
        return torch.stack(
            [(atomic_numbers == element).to(dtype=dtype) for element in QM9_ELEMENTS],
            dim=1,
        )

    def forward(
        self,
        atomic_numbers: Tensor,
        positions_bohr: Tensor,
        density_features: Tensor,
        internal_projector: Tensor,
    ) -> Tensor:
        positions = torch.as_tensor(positions_bohr)
        atomic_numbers = torch.as_tensor(
            atomic_numbers, dtype=torch.long, device=positions.device
        )
        density = torch.as_tensor(
            density_features, dtype=positions.dtype, device=positions.device
        )
        projector = torch.as_tensor(
            internal_projector, dtype=positions.dtype, device=positions.device
        )
        atom_count = int(atomic_numbers.numel())
        coordinate_count = 3 * atom_count
        if positions.shape != (atom_count, 3):
            raise ValueError("positions and atomic numbers disagree")
        if density.shape != (atom_count, self.density_feature_dim):
            raise ValueError("density feature shape mismatch")
        if projector.shape != (coordinate_count, coordinate_count):
            raise ValueError("internal projector shape mismatch")
        if not set(int(value) for value in atomic_numbers.tolist()).issubset(
            QM9_ELEMENTS
        ):
            raise ValueError("unsupported element in structured Hessian head")

        first, second = torch.triu_indices(
            atom_count, atom_count, offset=1, device=positions.device
        )
        vectors = positions[second] - positions[first]
        distances = torch.linalg.vector_norm(vectors, dim=1).clamp_min(
            torch.finfo(positions.dtype).eps
        )
        unit = vectors / distances[:, None]
        unit_outer = _outer(unit, unit)
        weights = torch.exp(-distances / self.environment_scale_bohr)

        environment_vector = positions.new_zeros((atom_count, 3))
        environment_vector = environment_vector.index_add(
            0, first, weights[:, None] * unit
        )
        environment_vector = environment_vector.index_add(
            0, second, -weights[:, None] * unit
        )
        environment_matrix = positions.new_zeros((atom_count, 3, 3))
        environment_matrix = environment_matrix.index_add(
            0, first, weights[:, None, None] * unit_outer
        )
        environment_matrix = environment_matrix.index_add(
            0, second, weights[:, None, None] * unit_outer
        )

        matrix_first = environment_matrix[first]
        matrix_second = environment_matrix[second]
        vector_first = environment_vector[first]
        vector_second = environment_vector[second]
        matrix_sum = 0.5 * (matrix_first + matrix_second)
        matrix_difference = matrix_second - matrix_first
        vector_sum = vector_first + vector_second
        vector_difference = vector_second - vector_first
        matrix_unit = torch.einsum("pi,pij,pj->p", unit, matrix_sum, unit)
        matrix_norm_first = torch.linalg.matrix_norm(matrix_first)
        matrix_norm_second = torch.linalg.matrix_norm(matrix_second)
        vector_norm_first = torch.linalg.vector_norm(vector_first, dim=1)
        vector_norm_second = torch.linalg.vector_norm(vector_second, dim=1)
        pair_environment = torch.stack(
            (
                torch.diagonal(matrix_sum, dim1=1, dim2=2).sum(dim=1),
                matrix_unit,
                matrix_norm_first + matrix_norm_second,
                torch.abs(matrix_norm_first - matrix_norm_second),
                vector_norm_first + vector_norm_second,
                torch.abs(vector_norm_first - vector_norm_second),
                torch.sum(vector_first * vector_second, dim=1),
                torch.sum(unit * vector_difference, dim=1),
            ),
            dim=1,
        )
        radial = torch.exp(
            -0.5
            * (
                (
                    distances[:, None]
                    - self.radial_centers_bohr.to(
                        dtype=positions.dtype, device=positions.device
                    )[None, :]
                )
                / self.radial_width_bohr
            )
            ** 2
        )
        pair_features = [
            self._pair_type_features(atomic_numbers, first, second).to(
                dtype=positions.dtype, device=positions.device
            ),
            radial,
            pair_environment,
        ]
        if self.use_density:
            density_first = density[first]
            density_second = density[second]
            pair_features.extend(
                (
                    density_first + density_second,
                    torch.abs(density_first - density_second),
                    density_first * density_second,
                )
            )
        pair_coefficients = self.pair_mlp(torch.cat(pair_features, dim=1))

        identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
        vector_outer = _outer(vector_difference, vector_difference)
        symmetric_unit_vector = 0.5 * (
            _outer(unit, vector_difference) + _outer(vector_difference, unit)
        )
        antisymmetric_unit_vector_sum = 0.5 * (
            _outer(unit, vector_sum) - _outer(vector_sum, unit)
        )
        matrix_times_unit = torch.einsum("pij,pj->pi", matrix_sum, unit)
        difference_times_unit = torch.einsum(
            "pij,pj->pi", matrix_difference, unit
        )
        symmetric_unit_matrix = 0.5 * (
            _outer(unit, matrix_times_unit) + _outer(matrix_times_unit, unit)
        )
        antisymmetric_unit_matrix_difference = 0.5 * (
            _outer(unit, difference_times_unit)
            - _outer(difference_times_unit, unit)
        )
        pair_bases = torch.stack(
            (
                identity.expand(first.numel(), -1, -1),
                unit_outer,
                matrix_sum,
                0.5
                * (
                    torch.einsum("pij,pjk->pik", unit_outer, matrix_sum)
                    + torch.einsum("pij,pjk->pik", matrix_sum, unit_outer)
                ),
                vector_outer,
                symmetric_unit_vector,
                antisymmetric_unit_vector_sum,
                antisymmetric_unit_matrix_difference,
            ),
            dim=1,
        )
        pair_blocks = torch.einsum(
            "pb,pbij->pij", pair_coefficients, pair_bases
        )

        matrix_trace = torch.diagonal(
            environment_matrix, dim1=1, dim2=2
        ).sum(dim=1)
        vector_norm = torch.linalg.vector_norm(environment_vector, dim=1)
        node_environment = torch.stack(
            (
                matrix_trace,
                torch.linalg.matrix_norm(environment_matrix),
                vector_norm,
                torch.einsum(
                    "ni,nij,nj->n",
                    environment_vector,
                    environment_matrix,
                    environment_vector,
                ),
            ),
            dim=1,
        )
        node_features = [
            self._node_type_features(atomic_numbers, positions.dtype).to(
                device=positions.device
            ),
            node_environment,
        ]
        if self.use_density:
            node_features.append(density)
        node_coefficients = self.node_mlp(torch.cat(node_features, dim=1))
        node_bases = torch.stack(
            (
                identity.expand(atom_count, -1, -1),
                environment_matrix,
                _outer(environment_vector, environment_vector),
            ),
            dim=1,
        )
        node_blocks = torch.einsum(
            "nb,nbij->nij", node_coefficients, node_bases
        )

        coordinate_offsets = torch.arange(3, device=positions.device)
        pair_rows = (
            3 * first[:, None, None] + coordinate_offsets[None, :, None]
        ).expand(-1, 3, 3)
        pair_columns = (
            3 * second[:, None, None] + coordinate_offsets[None, None, :]
        ).expand(-1, 3, 3)
        raw = positions.new_zeros((coordinate_count, coordinate_count))
        raw = raw.index_put(
            (pair_rows.reshape(-1), pair_columns.reshape(-1)),
            pair_blocks.reshape(-1),
            accumulate=True,
        )
        raw = raw.index_put(
            (pair_columns.reshape(-1), pair_rows.reshape(-1)),
            pair_blocks.reshape(-1),
            accumulate=True,
        )
        atom_indices = torch.arange(atom_count, device=positions.device)
        node_rows = (
            3 * atom_indices[:, None, None] + coordinate_offsets[None, :, None]
        ).expand(-1, 3, 3)
        node_columns = (
            3 * atom_indices[:, None, None] + coordinate_offsets[None, None, :]
        ).expand(-1, 3, 3)
        raw = raw.index_put(
            (node_rows.reshape(-1), node_columns.reshape(-1)),
            node_blocks.reshape(-1),
            accumulate=True,
        )
        symmetric = 0.5 * (raw + raw.transpose(0, 1))
        result = projector @ symmetric @ projector
        return 0.5 * (result + result.transpose(0, 1))
