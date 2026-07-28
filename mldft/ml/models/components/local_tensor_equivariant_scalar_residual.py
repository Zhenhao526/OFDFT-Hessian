"""Second-order-safe e3nn scalar energy with explicit quadrupole channels."""

from __future__ import annotations

import copy
import math

import torch
from e3nn import o3

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology


class _TensorProductInteraction(torch.nn.Module):
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge: o3.Irreps,
        radial_size: int,
        radial_hidden_size: int,
    ) -> None:
        super().__init__()
        self.self_connection = o3.Linear(irreps_node, irreps_node)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_node,
            irreps_edge,
            irreps_node,
            shared_weights=False,
        )
        self.radial_filter = torch.nn.Sequential(
            torch.nn.Linear(radial_size, radial_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(radial_hidden_size, self.tensor_product.weight_numel),
            torch.nn.Tanh(),
        )

    def forward(
        self,
        node: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        spherical_harmonics: torch.Tensor,
        radial: torch.Tensor,
        cutoff: torch.Tensor,
        residual_scale: float,
    ) -> torch.Tensor:
        weights = self.radial_filter(radial) * cutoff[:, None]
        messages = self.tensor_product(
            node[source], spherical_harmonics, weights
        )
        aggregate = torch.zeros_like(node).index_add(0, destination, messages)
        updated = self.self_connection(node) + residual_scale * aggregate
        invariant_scale = torch.sqrt(
            1.0 + torch.mean(updated * updated, dim=1, keepdim=True)
        )
        return updated / invariant_scale


class _TensorEquivariantNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        element_count: int,
        scalar_channels: int,
        vector_channels: int,
        tensor_channels: int,
        radial_size: int,
        radial_hidden_size: int,
        interaction_layers: int,
        cutoff_bohr: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.scalar_channels = int(scalar_channels)
        self.radial_size = int(radial_size)
        self.cutoff_bohr = float(cutoff_bohr)
        self.irreps_node = o3.Irreps(
            f"{scalar_channels}x0e + {vector_channels}x1o + {tensor_channels}x2e"
        )
        self.irreps_edge = o3.Irreps.spherical_harmonics(2)
        self.embedding = torch.nn.Embedding(element_count, scalar_channels)
        self.interactions = torch.nn.ModuleList(
            [
                _TensorProductInteraction(
                    self.irreps_node,
                    self.irreps_edge,
                    radial_size,
                    radial_hidden_size,
                )
                for _ in range(interaction_layers)
            ]
        )
        self.scalar_projection = o3.Linear(
            self.irreps_node, o3.Irreps(f"{scalar_channels}x0e")
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(scalar_channels, scalar_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(scalar_channels, 1),
        )
        centers = torch.linspace(0.5, cutoff_bohr, radial_size)
        spacing = (cutoff_bohr - 0.5) / max(radial_size - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer(
            "radial_inverse_width_squared",
            torch.tensor(1.0 / max(spacing, 1e-6) ** 2),
        )
        self._initialize(seed)
        self.to(dtype=torch.float64)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.ndim == 1:
                    parameter.zero_()
                    continue
                scale = math.sqrt(
                    2.0 / max(parameter.shape[-1] + parameter.shape[0], 1)
                )
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

    def forward(
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
        spherical_harmonics = o3.spherical_harmonics(
            self.irreps_edge,
            unit,
            normalize=False,
            normalization="component",
        )
        cutoff = self._cutoff(distance)
        radial = torch.exp(
            -self.radial_inverse_width_squared
            * (distance[:, None] - self.radial_centers[None, :]) ** 2
        )
        radial = radial * cutoff[:, None]

        scalar = torch.tanh(self.embedding(element_index.to(device=device)))
        nonscalar = torch.zeros(
            scalar.shape[0],
            self.irreps_node.dim - self.scalar_channels,
            dtype=positions_bohr.dtype,
            device=device,
        )
        node = torch.cat((scalar, nonscalar), dim=1)
        residual_scale = 1.0 / math.sqrt(max(len(self.interactions), 1))
        for interaction in self.interactions:
            node = interaction(
                node,
                source,
                destination,
                spherical_harmonics,
                radial,
                cutoff,
                residual_scale,
            )
        atomic_scalar = self.scalar_projection(node)
        return self.readout(atomic_scalar).sum()


class LocalTensorEquivariantScalarResidual(torch.nn.Module):
    """Invariant scalar energy with l=0, l=1, and l=2 message channels."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        scalar_channels: int = 32,
        vector_channels: int = 16,
        tensor_channels: int = 8,
        radial_size: int = 30,
        radial_hidden_size: int = 64,
        interaction_layers: int = 2,
        cutoff_bohr: float = 20.0,
        output_scale: float = 1.0,
        seed: int = 20260731,
    ) -> None:
        super().__init__()
        dimensions = (
            scalar_channels,
            vector_channels,
            tensor_channels,
            radial_size,
            radial_hidden_size,
            interaction_layers,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("all channel, radial, and layer dimensions must be positive")
        if radial_size <= 1 or cutoff_bohr <= 0.5:
            raise ValueError("radial size/cutoff do not define a valid basis")
        if output_scale <= 0.0:
            raise ValueError("output scale must be positive")
        self.output_scale = float(output_scale)
        self.elements = tuple(sorted(int(value) for value in elements))
        self.register_buffer("element_lookup", self._build_element_lookup(self.elements))
        self.network = _TensorEquivariantNetwork(
            element_count=len(self.elements),
            scalar_channels=scalar_channels,
            vector_channels=vector_channels,
            tensor_channels=tensor_channels,
            radial_size=radial_size,
            radial_hidden_size=radial_hidden_size,
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
        return self.output_scale * (
            self.network(positions_bohr, element_index, topology)
            - self.initial_network(positions_bohr, element_index, topology)
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
        flat_gradient = gradient.reshape(-1)
        rows = [
            torch.autograd.grad(
                value,
                positions,
                create_graph=create_parameter_graph,
                retain_graph=(create_parameter_graph or index + 1 < flat_gradient.numel()),
            )[0].reshape(-1)
            for index, value in enumerate(flat_gradient)
        ]
        return energy, -gradient, torch.stack(rows)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.network.parameters() if parameter.requires_grad]
