"""Smooth invariant message-passing scalar residuals for E/F/Hessian fitting."""

from __future__ import annotations

import copy
import math

import torch

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology


def _smooth_activation(name: str, *, softplus_beta: float) -> torch.nn.Module:
    if name == "softplus":
        if softplus_beta <= 0.0:
            raise ValueError("softplus_beta must be positive")
        return torch.nn.Softplus(beta=softplus_beta)
    if name == "silu":
        return torch.nn.SiLU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"unsupported smooth activation: {name}")


class _SmoothMessageLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        radial_size: int,
        *,
        activation: str,
        softplus_beta: float,
    ) -> None:
        super().__init__()
        self.message = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + radial_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
        )
        self.update = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
        )


class _InvariantMessageNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        element_count: int,
        hidden_size: int,
        radial_size: int,
        message_layers: int,
        cutoff_bohr: float,
        seed: int,
        activation: str,
        softplus_beta: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cutoff_bohr = float(cutoff_bohr)
        self.embedding = torch.nn.Embedding(
            element_count, hidden_size, dtype=torch.float64
        )
        self.layers = torch.nn.ModuleList(
            [
                _SmoothMessageLayer(
                    hidden_size,
                    radial_size,
                    activation=activation,
                    softplus_beta=softplus_beta,
                )
                for _ in range(message_layers)
            ]
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            _smooth_activation(activation, softplus_beta=softplus_beta),
            torch.nn.Linear(hidden_size, 1, dtype=torch.float64),
        )
        centers = torch.linspace(0.0, cutoff_bohr, radial_size, dtype=torch.float64)
        spacing = cutoff_bohr / max(radial_size - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer(
            "radial_inverse_width_squared",
            torch.tensor(1.0 / max(spacing, 1e-6) ** 2, dtype=torch.float64),
        )
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.embedding.weight.copy_(
                0.2
                * torch.randn(
                    self.embedding.weight.shape,
                    dtype=self.embedding.weight.dtype,
                    generator=generator,
                )
            )
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
        distance = torch.cat((pair_distance, pair_distance), dim=0)
        cutoff = self._cutoff(distance)
        radial = torch.exp(
            -self.radial_inverse_width_squared
            * (distance[:, None] - self.radial_centers[None, :]) ** 2
        )
        radial = radial * cutoff[:, None]

        node_state = self.embedding(element_index.to(device=device))
        residual_scale = 1.0 / math.sqrt(max(len(self.layers), 1))
        for layer in self.layers:
            message = layer.message(torch.cat((node_state[source], radial), dim=1))
            message = message * cutoff[:, None]
            aggregate = torch.zeros_like(node_state).index_add(
                0, destination, message
            )
            update = layer.update(torch.cat((node_state, aggregate), dim=1))
            node_state = node_state + residual_scale * update
        return self.readout(node_state).sum()


class LocalMessagePassingResidual(torch.nn.Module):
    """A smooth invariant local scalar with exact zero-function initialization.

    The trainable network is paired with a frozen copy of its initial function. Their
    difference is exactly zero at initialization while gradients still reach every
    trainable layer. Forces and Hessians are derivatives of this scalar; no force head
    is present.
    """

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_size: int = 96,
        radial_size: int = 16,
        message_layers: int = 3,
        cutoff_bohr: float = 8.0,
        seed: int = 20260722,
        activation: str = "softplus",
        softplus_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or radial_size <= 1 or message_layers <= 0:
            raise ValueError("hidden_size, radial_size, and message_layers must be positive")
        if cutoff_bohr <= 0.0:
            raise ValueError("cutoff_bohr must be positive")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.register_buffer(
            "element_lookup",
            self._build_element_lookup(self.elements),
        )
        self.network = _InvariantMessageNetwork(
            element_count=len(self.elements),
            hidden_size=hidden_size,
            radial_size=radial_size,
            message_layers=message_layers,
            cutoff_bohr=cutoff_bohr,
            seed=seed,
            activation=activation,
            softplus_beta=softplus_beta,
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
        atomic_numbers = atomic_numbers.to(
            device=self.element_lookup.device, dtype=torch.long
        )
        if bool(torch.any(atomic_numbers < 0)) or bool(
            torch.any(atomic_numbers >= self.element_lookup.numel())
        ):
            raise ValueError("unsupported atomic number")
        result = self.element_lookup[atomic_numbers]
        if bool(torch.any(result < 0)):
            missing = torch.unique(atomic_numbers[result < 0]).tolist()
            raise ValueError(f"unsupported atomic numbers: {missing}")
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
                positions,
                atomic_numbers,
                topology,
                reference_positions_bohr,
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
