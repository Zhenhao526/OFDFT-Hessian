"""Conservative invariant three-body scalar geometry residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from mldft.ml.data.components.of_data import OFData, Representation


@dataclass(frozen=True)
class TripletGroup:
    element_key: tuple[int, int, int]
    center_indices: tuple[int, ...]
    first_indices: tuple[int, ...]
    second_indices: tuple[int, ...]
    equal_neighbor_elements: bool


def build_triplet_groups(atomic_numbers: np.ndarray) -> list[TripletGroup]:
    """Group atom-centered triplets by central and unordered neighbor elements."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    grouped: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for center in range(atomic_numbers.size):
        neighbors = [index for index in range(atomic_numbers.size) if index != center]
        for first_offset, first in enumerate(neighbors):
            for second in neighbors[first_offset + 1 :]:
                ordered = sorted(
                    (first, second), key=lambda index: (int(atomic_numbers[index]), index)
                )
                first_ordered, second_ordered = ordered
                key = (
                    int(atomic_numbers[center]),
                    int(atomic_numbers[first_ordered]),
                    int(atomic_numbers[second_ordered]),
                )
                grouped.setdefault(key, []).append(
                    (center, first_ordered, second_ordered)
                )
    return [
        TripletGroup(
            element_key=key,
            center_indices=tuple(row[0] for row in rows),
            first_indices=tuple(row[1] for row in rows),
            second_indices=tuple(row[2] for row in rows),
            equal_neighbor_elements=key[1] == key[2],
        )
        for key, rows in sorted(grouped.items())
    ]


def _legendre_values(cosine: torch.Tensor, maximum_order: int) -> torch.Tensor:
    values = [torch.ones_like(cosine)]
    if maximum_order >= 1:
        values.append(cosine)
    for order in range(2, maximum_order + 1):
        values.append(
            ((2 * order - 1) * cosine * values[-1] - (order - 1) * values[-2])
            / order
        )
    return torch.stack(values, dim=1)


def make_three_body_feature_function(
    groups: list[TripletGroup],
    centers_bohr: np.ndarray,
    sigma_bohr: float,
    angular_order: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], list[tuple[int, ...]]]:
    """Construct an invariant three-body scalar feature vector."""
    if sigma_bohr <= 0 or angular_order < 0:
        raise ValueError("sigma must be positive and angular_order must be non-negative")
    centers_numpy = np.asarray(centers_bohr, dtype=np.float64)
    if centers_numpy.ndim != 1 or centers_numpy.size == 0:
        raise ValueError("centers must be a non-empty vector")

    feature_keys: list[tuple[int, ...]] = []
    center_pairs_by_group: list[list[tuple[int, int]]] = []
    for group in groups:
        if group.equal_neighbor_elements:
            center_pairs = [
                (first, second)
                for first in range(centers_numpy.size)
                for second in range(first, centers_numpy.size)
            ]
        else:
            center_pairs = [
                (first, second)
                for first in range(centers_numpy.size)
                for second in range(centers_numpy.size)
            ]
        center_pairs_by_group.append(center_pairs)
        for first, second in center_pairs:
            for order in range(angular_order + 1):
                feature_keys.append((*group.element_key, first, second, order))

    def features(positions_bohr: torch.Tensor) -> torch.Tensor:
        centers = torch.as_tensor(
            centers_numpy, dtype=positions_bohr.dtype, device=positions_bohr.device
        )
        output = []
        for group, center_pairs in zip(groups, center_pairs_by_group, strict=True):
            center = positions_bohr[list(group.center_indices)]
            first_vector = positions_bohr[list(group.first_indices)] - center
            second_vector = positions_bohr[list(group.second_indices)] - center
            first_distance = torch.linalg.vector_norm(first_vector, dim=1)
            second_distance = torch.linalg.vector_norm(second_vector, dim=1)
            cosine = torch.sum(first_vector * second_vector, dim=1) / (
                first_distance * second_distance
            )
            angular = _legendre_values(cosine, angular_order)
            first_rbf = torch.exp(
                -0.5 * ((first_distance[:, None] - centers[None, :]) / sigma_bohr) ** 2
            )
            second_rbf = torch.exp(
                -0.5 * ((second_distance[:, None] - centers[None, :]) / sigma_bohr) ** 2
            )
            radial_columns = []
            for first_center, second_center in center_pairs:
                radial = first_rbf[:, first_center] * second_rbf[:, second_center]
                if group.equal_neighbor_elements and first_center != second_center:
                    radial = radial + (
                        first_rbf[:, second_center] * second_rbf[:, first_center]
                    )
                radial_columns.append(radial)
            radial_matrix = torch.stack(radial_columns, dim=1)
            output.append(
                torch.einsum("tp,tl->pl", radial_matrix, angular).reshape(-1)
            )
        return torch.cat(output)

    return features, feature_keys


def all_three_body_feature_keys(
    elements: Iterable[int], center_count: int, angular_order: int
) -> list[tuple[int, ...]]:
    """Enumerate the fixed QM9-compatible coefficient key space."""
    elements = tuple(sorted(int(element) for element in elements))
    keys = []
    for center_element in elements:
        for first_offset, first_element in enumerate(elements):
            for second_element in elements[first_offset:]:
                if first_element == second_element:
                    center_pairs = [
                        (first, second)
                        for first in range(center_count)
                        for second in range(first, center_count)
                    ]
                else:
                    center_pairs = [
                        (first, second)
                        for first in range(center_count)
                        for second in range(center_count)
                    ]
                for first_center, second_center in center_pairs:
                    for order in range(angular_order + 1):
                        keys.append(
                            (
                                center_element,
                                first_element,
                                second_element,
                                first_center,
                                second_center,
                                order,
                            )
                        )
    return keys


class ThreeBodyGeometryResidual(torch.nn.Module):
    """A density-independent scalar residual; forces remain energy derivatives."""

    target_key = "geometry_residual"

    def __init__(
        self,
        center_min_bohr: float = 0.5,
        center_max_bohr: float = 8.0,
        center_count: int = 4,
        sigma_bohr: float = 0.5,
        angular_order: int = 3,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
    ) -> None:
        super().__init__()
        if center_count <= 0:
            raise ValueError("center_count must be positive")
        self.center_min_bohr = float(center_min_bohr)
        self.center_max_bohr = float(center_max_bohr)
        self.center_count = int(center_count)
        self.sigma_bohr = float(sigma_bohr)
        self.angular_order = int(angular_order)
        self.elements = tuple(sorted(int(element) for element in elements))
        self.feature_keys = all_three_body_feature_keys(
            self.elements, self.center_count, self.angular_order
        )
        self.feature_key_to_index = {
            key: index for index, key in enumerate(self.feature_keys)
        }
        element_to_index = {
            element: index for index, element in enumerate(self.elements)
        }
        feature_index_lookup = torch.full(
            (
                len(self.elements),
                len(self.elements),
                len(self.elements),
                self.center_count,
                self.center_count,
                self.angular_order + 1,
            ),
            -1,
            dtype=torch.long,
        )
        for feature_index, key in enumerate(self.feature_keys):
            center_element, first_element, second_element, first, second, order = key
            lookup_key = (
                element_to_index[center_element],
                element_to_index[first_element],
                element_to_index[second_element],
                first,
                second,
                order,
            )
            feature_index_lookup[lookup_key] = feature_index
            if first_element == second_element:
                symmetric_key = (*lookup_key[:3], second, first, order)
                feature_index_lookup[symmetric_key] = feature_index
        for first_index in range(len(self.elements)):
            for second_index in range(first_index, len(self.elements)):
                if bool(
                    torch.any(
                        feature_index_lookup[
                            :, first_index, second_index
                        ]
                        < 0
                    )
                ):
                    raise AssertionError("three-body feature lookup is incomplete")
        self.register_buffer(
            "element_values", torch.as_tensor(self.elements, dtype=torch.long)
        )
        self.register_buffer("feature_index_lookup", feature_index_lookup)
        self.coefficients = torch.nn.Parameter(
            torch.zeros(len(self.feature_keys), dtype=torch.float64)
        )
        self.__name__ = self.target_key

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficients.dtype

    @property
    def centers_bohr(self) -> np.ndarray:
        return np.linspace(
            self.center_min_bohr, self.center_max_bohr, self.center_count
        )

    def load_sparse_coefficients(
        self, feature_keys: np.ndarray, coefficients: np.ndarray
    ) -> None:
        feature_keys = np.asarray(feature_keys, dtype=np.int64)
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if feature_keys.shape != (coefficients.size, 6):
            raise ValueError("feature keys must have shape (n_coefficients, 6)")
        values = torch.zeros_like(self.coefficients)
        for key_row, value in zip(feature_keys, coefficients, strict=True):
            key = tuple(int(item) for item in key_row)
            if key not in self.feature_key_to_index:
                raise ValueError(f"unsupported three-body feature key: {key}")
            values[self.feature_key_to_index[key]] = float(value)
        with torch.no_grad():
            self.coefficients.copy_(values)

    def forward_energy(self, sample: OFData) -> torch.Tensor:
        positions = sample.pos
        atomic_numbers = torch.as_tensor(
            sample.atomic_numbers, dtype=torch.long, device=positions.device
        )
        batch_value = getattr(sample, "batch", None)
        batch = (
            torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=positions.device)
            if batch_value is None
            else torch.as_tensor(batch_value, dtype=torch.long, device=positions.device)
        )
        num_graphs = int(getattr(sample, "num_graphs", 1))
        atom_count = positions.shape[0]
        pair_first, pair_second = torch.triu_indices(
            atom_count, atom_count, offset=1, device=positions.device
        )
        centers = torch.arange(atom_count, device=positions.device)[:, None].expand(
            atom_count, pair_first.numel()
        ).reshape(-1)
        neighbor_first = pair_first.repeat(atom_count)
        neighbor_second = pair_second.repeat(atom_count)
        valid = (
            (centers != neighbor_first)
            & (centers != neighbor_second)
            & (batch[centers] == batch[neighbor_first])
            & (batch[centers] == batch[neighbor_second])
        )
        centers = centers[valid]
        neighbor_first = neighbor_first[valid]
        neighbor_second = neighbor_second[valid]
        if centers.numel() == 0:
            zero = positions.sum() * 0.0 + self.coefficients.sum() * 0.0
            return zero.expand(num_graphs)

        swap = atomic_numbers[neighbor_first] > atomic_numbers[neighbor_second]
        ordered_first = torch.where(swap, neighbor_second, neighbor_first)
        ordered_second = torch.where(swap, neighbor_first, neighbor_second)
        first_vector = positions[ordered_first] - positions[centers]
        second_vector = positions[ordered_second] - positions[centers]
        first_distance = torch.linalg.vector_norm(first_vector, dim=1)
        second_distance = torch.linalg.vector_norm(second_vector, dim=1)
        cosine = torch.sum(first_vector * second_vector, dim=1) / (
            first_distance * second_distance
        )
        angular = _legendre_values(cosine, self.angular_order)
        radial_centers = torch.linspace(
            self.center_min_bohr,
            self.center_max_bohr,
            self.center_count,
            dtype=positions.dtype,
            device=positions.device,
        )
        first_rbf = torch.exp(
            -0.5
            * ((first_distance[:, None] - radial_centers[None, :]) / self.sigma_bohr)
            ** 2
        )
        second_rbf = torch.exp(
            -0.5
            * ((second_distance[:, None] - radial_centers[None, :]) / self.sigma_bohr)
            ** 2
        )
        radial = first_rbf[:, :, None] * second_rbf[:, None, :]
        matches = atomic_numbers[:, None] == self.element_values[None, :]
        element_indices = torch.argmax(matches.to(torch.long), dim=1)
        coefficient_indices = self.feature_index_lookup[
            element_indices[centers],
            element_indices[ordered_first],
            element_indices[ordered_second],
        ]
        triplet_energy = torch.sum(
            radial[:, :, :, None]
            * angular[:, None, None, :]
            * self.coefficients[coefficient_indices],
            dim=(1, 2, 3),
        )
        energies = torch.zeros(
            num_graphs, dtype=positions.dtype, device=positions.device
        )
        return energies.index_add(0, batch[centers], triplet_energy)

    def forward_predictions(
        self,
        sample: OFData,
        compute_density_gradients: bool = False,
        compute_forces: bool = False,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor | None]:
        energy = self.forward_energy(sample)
        force = None
        if compute_forces:
            force = -torch.autograd.grad(
                energy.sum(), sample.pos, create_graph=self.training, retain_graph=self.training
            )[0]
        return energy, None, None, force

    def sample_forward(self, sample: OFData) -> OFData:
        energy = self.forward_energy(sample)
        sample.add_item("pred_energy", energy, Representation.SCALAR)
        sample.add_item(
            "pred_gradient", torch.zeros_like(sample.coeffs), Representation.GRADIENT
        )
        return sample
