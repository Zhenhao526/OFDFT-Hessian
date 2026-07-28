"""Smooth local angular scalar residual for conservative E/F/Hessian fitting."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology
from mldft.ml.models.components.local_message_passing_residual import _smooth_activation


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


class _LocalAngularNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        element_count: int,
        hidden_size: int,
        radial_size: int,
        angular_order: int,
        cutoff_bohr: float,
        seed: int,
        activation: str,
        softplus_beta: float,
        radial_feature_scale: float,
        angular_feature_scale: float,
    ) -> None:
        super().__init__()
        self.element_count = int(element_count)
        self.radial_size = int(radial_size)
        self.angular_order = int(angular_order)
        self.cutoff_bohr = float(cutoff_bohr)
        self.radial_feature_scale = float(radial_feature_scale)
        self.angular_feature_scale = float(angular_feature_scale)
        self.neighbor_pair_count = element_count * (element_count + 1) // 2
        angular_size = (
            self.neighbor_pair_count
            * radial_size
            * radial_size
            * (angular_order + 1)
        )
        input_size = element_count + element_count * radial_size + angular_size
        self.atomic_energy = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
            torch.nn.Linear(hidden_size, 1, dtype=torch.float64),
        )
        centers = torch.linspace(0.5, cutoff_bohr, radial_size, dtype=torch.float64)
        spacing = (cutoff_bohr - 0.5) / max(radial_size - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer(
            "radial_inverse_width_squared",
            torch.tensor(1.0 / max(spacing, 1e-6) ** 2, dtype=torch.float64),
        )
        pair_lookup = torch.empty((element_count, element_count), dtype=torch.long)
        pair_index = 0
        for first in range(element_count):
            for second in range(first, element_count):
                pair_lookup[first, second] = pair_index
                pair_lookup[second, first] = pair_index
                pair_index += 1
        self.register_buffer("neighbor_pair_lookup", pair_lookup)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                scale = math.sqrt(2.0 / (module.in_features + module.out_features))
                module.weight.copy_(
                    scale
                    * torch.randn(
                        module.weight.shape,
                        dtype=module.weight.dtype,
                        generator=generator,
                    )
                )
                module.bias.zero_()

    def _cutoff(self, distance: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp(distance / self.cutoff_bohr, min=0.0, max=1.0)
        value = (
            1.0
            - 35.0 * scaled**4
            + 84.0 * scaled**5
            - 70.0 * scaled**6
            + 20.0 * scaled**7
        )
        return torch.where(distance < self.cutoff_bohr, value, torch.zeros_like(value))

    def _radial(self, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cutoff = self._cutoff(distance)
        radial = torch.exp(
            -self.radial_inverse_width_squared
            * (distance[:, None] - self.radial_centers[None, :]) ** 2
        )
        return radial * cutoff[:, None], cutoff

    def atomic_features(
        self,
        positions_bohr: torch.Tensor,
        element_index: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        device = positions_bohr.device
        atom_count = positions_bohr.shape[0]

        pair_first = topology.pair_first.to(device=device)
        pair_second = topology.pair_second.to(device=device)
        pair_vector = positions_bohr[pair_second] - positions_bohr[pair_first]
        pair_distance = torch.linalg.vector_norm(pair_vector, dim=1)
        pair_radial, _ = self._radial(pair_distance)
        center = torch.cat((pair_first, pair_second), dim=0)
        neighbor = torch.cat((pair_second, pair_first), dim=0)
        directed_radial = torch.cat((pair_radial, pair_radial), dim=0)
        radial_index = center * self.element_count + element_index[neighbor]
        radial_features = torch.zeros(
            (atom_count * self.element_count, self.radial_size),
            dtype=positions_bohr.dtype,
            device=device,
        ).index_add(0, radial_index, directed_radial)
        radial_features = radial_features.reshape(atom_count, -1)
        radial_features = radial_features / self.radial_feature_scale

        triplet_center = topology.triplet_center.to(device=device)
        triplet_first = topology.triplet_first.to(device=device)
        triplet_second = topology.triplet_second.to(device=device)
        first_vector = positions_bohr[triplet_first] - positions_bohr[triplet_center]
        second_vector = positions_bohr[triplet_second] - positions_bohr[triplet_center]
        first_distance = torch.linalg.vector_norm(first_vector, dim=1)
        second_distance = torch.linalg.vector_norm(second_vector, dim=1)
        first_radial, _ = self._radial(first_distance)
        second_radial, _ = self._radial(second_distance)
        cosine = torch.sum(first_vector * second_vector, dim=1) / (
            first_distance * second_distance
        )
        angular = _legendre_values(cosine, self.angular_order)
        radial_outer = first_radial[:, :, None] * second_radial[:, None, :]
        equal_neighbor_element = (
            element_index[triplet_first] == element_index[triplet_second]
        )
        symmetric_outer = 0.5 * (radial_outer + radial_outer.transpose(1, 2))
        radial_outer = torch.where(
            equal_neighbor_element[:, None, None], symmetric_outer, radial_outer
        )
        angular_rows = (
            radial_outer[:, :, :, None]
            * angular[:, None, None, :]
        ).reshape(triplet_center.numel(), -1)
        neighbor_pair_type = self.neighbor_pair_lookup[
            element_index[triplet_first], element_index[triplet_second]
        ]
        angular_index = triplet_center * self.neighbor_pair_count + neighbor_pair_type
        angular_features = torch.zeros(
            (atom_count * self.neighbor_pair_count, angular_rows.shape[1]),
            dtype=positions_bohr.dtype,
            device=device,
        ).index_add(0, angular_index, angular_rows)
        angular_features = angular_features.reshape(atom_count, -1)
        angular_features = angular_features / self.angular_feature_scale

        central_element = F.one_hot(
            element_index, num_classes=self.element_count
        ).to(dtype=positions_bohr.dtype)
        return torch.cat((central_element, radial_features, angular_features), dim=1)

    def forward(
        self,
        positions_bohr: torch.Tensor,
        element_index: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        features = self.atomic_features(positions_bohr, element_index, topology)
        return self.atomic_energy(features).sum()


class LocalAngularScalarResidual(torch.nn.Module):
    """A local-additive angular scalar with exact zero-function initialization."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_size: int = 128,
        radial_size: int = 6,
        angular_order: int = 3,
        cutoff_bohr: float = 8.0,
        seed: int = 20260727,
        activation: str = "tanh",
        softplus_beta: float = 1.0,
        radial_feature_scale: float = 4.0,
        angular_feature_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or radial_size <= 1 or angular_order < 0:
            raise ValueError("hidden_size/radial_size/angular_order are invalid")
        if cutoff_bohr <= 0.5:
            raise ValueError("cutoff_bohr must exceed the first radial center")
        if radial_feature_scale <= 0.0 or angular_feature_scale <= 0.0:
            raise ValueError("feature scales must be positive")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.register_buffer("element_lookup", self._build_element_lookup(self.elements))
        settings = dict(
            element_count=len(self.elements),
            hidden_size=hidden_size,
            radial_size=radial_size,
            angular_order=angular_order,
            cutoff_bohr=cutoff_bohr,
            seed=seed,
            activation=activation,
            softplus_beta=softplus_beta,
            radial_feature_scale=radial_feature_scale,
            angular_feature_scale=angular_feature_scale,
        )
        self.network = _LocalAngularNetwork(**settings)
        self.initial_network = copy.deepcopy(self.network)
        self.initial_network.requires_grad_(False)

    @staticmethod
    def _build_element_lookup(elements: tuple[int, ...]) -> torch.Tensor:
        lookup = torch.full((max(elements) + 1,), -1, dtype=torch.long)
        for index, atomic_number in enumerate(elements):
            lookup[atomic_number] = index
        return lookup

    def _element_index(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        values = atomic_numbers.to(device=self.element_lookup.device, dtype=torch.long)
        if bool(torch.any(values < 0)) or bool(
            torch.any(values >= self.element_lookup.numel())
        ):
            raise ValueError("unsupported atomic number")
        result = self.element_lookup[values]
        if bool(torch.any(result < 0)):
            raise ValueError("unsupported atomic number")
        return result

    def forward_energy(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        element_index = self._element_index(atomic_numbers)
        return self.network(positions_bohr, element_index, topology) - self.initial_network(
            positions_bohr, element_index, topology
        )

    def forward_anchored_energy(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
        reference_positions_bohr: torch.Tensor,
    ) -> torch.Tensor:
        reference = reference_positions_bohr.detach().clone().requires_grad_(True)
        reference_energy = self.forward_energy(reference, atomic_numbers, topology)
        reference_gradient = torch.autograd.grad(
            reference_energy, reference, create_graph=True
        )[0]
        displacement = positions_bohr - reference
        return (
            self.forward_energy(positions_bohr, atomic_numbers, topology)
            - reference_energy
            - torch.sum(reference_gradient * displacement)
        )

    def energy_force_hessian(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
        *,
        create_parameter_graph: bool = True,
        reference_positions_bohr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = positions_bohr
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        energy = (
            self.forward_energy(positions, atomic_numbers, topology)
            if reference_positions_bohr is None
            else self.forward_anchored_energy(
                positions, atomic_numbers, topology, reference_positions_bohr
            )
        )
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        flat_gradient = gradient.reshape(-1)
        rows = []
        for index, value in enumerate(flat_gradient):
            rows.append(
                torch.autograd.grad(
                    value,
                    positions,
                    create_graph=create_parameter_graph,
                    retain_graph=(
                        create_parameter_graph or index + 1 < flat_gradient.numel()
                    ),
                )[0].reshape(-1)
            )
        return energy, -gradient, torch.stack(rows)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.network.parameters() if parameter.requires_grad]
