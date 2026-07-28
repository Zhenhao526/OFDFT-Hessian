"""Bounded scalar/vector message-passing energy for smooth Hessian learning."""

from __future__ import annotations

import copy
import math

import torch

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology


class _EquivariantInteraction(torch.nn.Module):
    def __init__(self, hidden_size: int, radial_size: int) -> None:
        super().__init__()
        self.radial_filter = torch.nn.Sequential(
            torch.nn.Linear(radial_size, hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 4 * hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
        )
        self.vector_mix = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.float64
        )
        self.intra_atom = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 2 * hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
        )

    @staticmethod
    def _bounded_vectors(vectors: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(1.0 + torch.mean(vectors * vectors, dim=(1, 2)))
        return vectors / scale[:, None, None]

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        unit: torch.Tensor,
        radial: torch.Tensor,
        cutoff: torch.Tensor,
        residual_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filters = self.radial_filter(radial) * cutoff[:, None]
        scalar_scalar, vector_scalar, scalar_vector, vector_vector = filters.chunk(
            4, dim=1
        )
        projected_vector = torch.sum(
            vector[source] * unit[:, None, :], dim=2
        )
        scalar_message = (
            scalar_scalar * scalar[source] + vector_scalar * projected_vector
        )
        vector_message = (
            scalar_vector[:, :, None] * scalar[source, :, None] * unit[:, None, :]
            + vector_vector[:, :, None] * vector[source]
        )
        scalar_aggregate = torch.zeros_like(scalar).index_add(
            0, destination, scalar_message
        )
        vector_aggregate = torch.zeros_like(vector).index_add(
            0, destination, vector_message
        )
        scalar = torch.tanh(scalar + residual_scale * scalar_aggregate)
        vector = self._bounded_vectors(vector + residual_scale * vector_aggregate)

        mixed_vector = torch.einsum(
            "ncd,oc->nod", vector, self.vector_mix.weight
        )
        vector_squared = torch.sum(mixed_vector * mixed_vector, dim=2)
        scalar_update, vector_gate = self.intra_atom(
            torch.cat((scalar, vector_squared), dim=1)
        ).chunk(2, dim=1)
        scalar = torch.tanh(scalar + residual_scale * scalar_update)
        vector = self._bounded_vectors(
            vector + residual_scale * vector_gate[:, :, None] * mixed_vector
        )
        return scalar, vector


class _BoundedEquivariantNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        element_count: int,
        hidden_size: int,
        radial_size: int,
        interaction_layers: int,
        cutoff_bohr: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.radial_size = int(radial_size)
        self.cutoff_bohr = float(cutoff_bohr)
        self.embedding = torch.nn.Embedding(
            element_count, hidden_size, dtype=torch.float64
        )
        self.interactions = torch.nn.ModuleList(
            [
                _EquivariantInteraction(hidden_size, radial_size)
                for _ in range(interaction_layers)
            ]
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, dtype=torch.float64),
        )
        centers = torch.linspace(0.5, cutoff_bohr, radial_size, dtype=torch.float64)
        spacing = (cutoff_bohr - 0.5) / max(radial_size - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer(
            "radial_inverse_width_squared",
            torch.tensor(1.0 / max(spacing, 1e-6) ** 2, dtype=torch.float64),
        )
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.ndim == 1:
                    parameter.zero_()
                    continue
                scale = math.sqrt(2.0 / max(parameter.shape[-1] + parameter.shape[0], 1))
                parameter.copy_(
                    scale
                    * torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        generator=generator,
                    )
                )

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

    def atomic_scalar_features(
        self,
        positions_bohr: torch.Tensor,
        element_index: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        device = positions_bohr.device
        first = topology.pair_first.to(device=device)
        second = topology.pair_second.to(device=device)
        source = torch.cat((first, second), dim=0)
        destination = torch.cat((second, first), dim=0)
        pair_vector = positions_bohr[second] - positions_bohr[first]
        pair_distance = torch.linalg.vector_norm(pair_vector, dim=1)
        directed_vector = torch.cat((pair_vector, -pair_vector), dim=0)
        distance = torch.cat((pair_distance, pair_distance), dim=0)
        unit = directed_vector / distance[:, None]
        cutoff = self._cutoff(distance)
        radial = torch.exp(
            -self.radial_inverse_width_squared
            * (distance[:, None] - self.radial_centers[None, :]) ** 2
        )
        radial = radial * cutoff[:, None]

        scalar = torch.tanh(self.embedding(element_index.to(device=device)))
        vector = torch.zeros(
            scalar.shape[0],
            self.hidden_size,
            3,
            dtype=positions_bohr.dtype,
            device=device,
        )
        states = []
        residual_scale = 1.0 / math.sqrt(max(len(self.interactions), 1))
        for interaction in self.interactions:
            scalar, vector = interaction(
                scalar,
                vector,
                source,
                destination,
                unit,
                radial,
                cutoff,
                residual_scale,
            )
            states.append(scalar)
        return torch.cat(states, dim=1)

    def forward(
        self,
        positions_bohr: torch.Tensor,
        element_index: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        features = self.atomic_scalar_features(positions_bohr, element_index, topology)
        final = features[:, -self.hidden_size :]
        return self.readout(final).sum()


class LocalEquivariantScalarResidual(torch.nn.Module):
    """Bounded PaiNN-style local scalar with all derivatives owned by its energy."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_size: int = 64,
        radial_size: int = 16,
        interaction_layers: int = 3,
        cutoff_bohr: float = 8.0,
        seed: int = 20260730,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or radial_size <= 1 or interaction_layers <= 0:
            raise ValueError("hidden/radial/layer dimensions must be positive")
        if cutoff_bohr <= 0.5:
            raise ValueError("cutoff must exceed the first radial center")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.register_buffer("element_lookup", self._build_element_lookup(self.elements))
        self.network = _BoundedEquivariantNetwork(
            element_count=len(self.elements),
            hidden_size=hidden_size,
            radial_size=radial_size,
            interaction_layers=interaction_layers,
            cutoff_bohr=cutoff_bohr,
            seed=seed,
        )
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
        indices = self.element_lookup[values]
        if bool(torch.any(indices < 0)):
            raise ValueError("unsupported atomic number")
        return indices

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
                positions,
                atomic_numbers,
                topology,
                reference_positions_bohr,
            )
        )
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        rows = []
        flat_gradient = gradient.reshape(-1)
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
