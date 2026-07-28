"""Smooth local-additive geometry residuals for conservative E/F/Hessian fitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, product

import torch


_COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class BodyOrderTopology:
    """Fixed element/type topology for one molecular geometry."""

    pair_first: torch.Tensor
    pair_second: torch.Tensor
    pair_type: torch.Tensor
    triplet_center: torch.Tensor
    triplet_first: torch.Tensor
    triplet_second: torch.Tensor
    triplet_type: torch.Tensor
    torsion_first: torch.Tensor
    torsion_second: torch.Tensor
    torsion_third: torch.Tensor
    torsion_fourth: torch.Tensor
    torsion_type: torch.Tensor


def _element_index(atomic_numbers: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    matches = atomic_numbers[:, None] == elements[None, :]
    if not bool(torch.all(matches.any(dim=1))):
        missing = torch.unique(atomic_numbers[~matches.any(dim=1)]).tolist()
        raise ValueError(f"unsupported atomic numbers: {missing}")
    return torch.argmax(matches.to(torch.long), dim=1)


def build_body_order_topology(
    atomic_numbers: torch.Tensor,
    positions_bohr: torch.Tensor | None = None,
    *,
    elements: tuple[int, ...] = (1, 6, 7, 8, 9),
    bond_scale: float = 1.25,
) -> BodyOrderTopology:
    """Build pair, triplet, and optional bonded-chain indices without self interactions."""
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long, device="cpu")
    if bond_scale <= 0.0:
        raise ValueError("bond_scale must be positive")
    element_values = torch.as_tensor(sorted(elements), dtype=torch.long)
    element_indices = _element_index(atomic_numbers, element_values)
    atom_count = int(atomic_numbers.numel())

    pair_types = list(combinations_with_replacement(range(len(elements)), 2))
    pair_lookup = {key: index for index, key in enumerate(pair_types)}
    pair_rows: list[tuple[int, int, int]] = []
    for first in range(atom_count):
        for second in range(first + 1, atom_count):
            key = tuple(
                sorted((int(element_indices[first]), int(element_indices[second])))
            )
            pair_rows.append((first, second, pair_lookup[key]))

    triplet_types = [
        (center, first, second)
        for center in range(len(elements))
        for first, second in pair_types
    ]
    triplet_lookup = {key: index for index, key in enumerate(triplet_types)}
    triplet_rows: list[tuple[int, int, int, int]] = []
    for center in range(atom_count):
        neighbors = [index for index in range(atom_count) if index != center]
        for first_offset, first in enumerate(neighbors):
            for second in neighbors[first_offset + 1 :]:
                ordered = sorted(
                    (first, second),
                    key=lambda index: (int(element_indices[index]), index),
                )
                first_ordered, second_ordered = ordered
                key = (
                    int(element_indices[center]),
                    int(element_indices[first_ordered]),
                    int(element_indices[second_ordered]),
                )
                triplet_rows.append(
                    (
                        center,
                        first_ordered,
                        second_ordered,
                        triplet_lookup[key],
                    )
                )

    torsion_types = sorted(
        {
            min(key, tuple(reversed(key)))
            for key in product(range(len(elements)), repeat=4)
        }
    )
    torsion_lookup = {key: index for index, key in enumerate(torsion_types)}
    torsion_rows: list[tuple[int, int, int, int, int]] = []
    if positions_bohr is not None:
        positions = torch.as_tensor(
            positions_bohr, dtype=torch.float64, device="cpu"
        )
        if positions.shape != (atom_count, 3):
            raise ValueError("positions_bohr must have shape (n_atoms, 3)")
        bonds: list[tuple[int, int]] = []
        neighbors = [[] for _ in range(atom_count)]
        for first in range(atom_count):
            for second in range(first + 1, atom_count):
                radius = bond_scale * (
                    _COVALENT_RADII_ANGSTROM[int(atomic_numbers[first])]
                    + _COVALENT_RADII_ANGSTROM[int(atomic_numbers[second])]
                ) * _ANGSTROM_TO_BOHR
                if float(torch.linalg.vector_norm(positions[first] - positions[second])) <= radius:
                    bonds.append((first, second))
                    neighbors[first].append(second)
                    neighbors[second].append(first)
        seen_chains: set[tuple[int, int, int, int]] = set()
        for second, third in bonds:
            for first in neighbors[second]:
                if first == third:
                    continue
                for fourth in neighbors[third]:
                    if fourth in (second, first):
                        continue
                    chain = (first, second, third, fourth)
                    key = tuple(int(element_indices[index]) for index in chain)
                    reverse_chain = tuple(reversed(chain))
                    reverse_key = tuple(reversed(key))
                    if reverse_key < key or (
                        reverse_key == key and reverse_chain < chain
                    ):
                        chain = reverse_chain
                        key = reverse_key
                    if chain in seen_chains:
                        continue
                    seen_chains.add(chain)
                    torsion_rows.append((*chain, torsion_lookup[key]))

    def column(rows: list[tuple[int, ...]], index: int) -> torch.Tensor:
        return torch.as_tensor([row[index] for row in rows], dtype=torch.long)

    return BodyOrderTopology(
        pair_first=column(pair_rows, 0),
        pair_second=column(pair_rows, 1),
        pair_type=column(pair_rows, 2),
        triplet_center=column(triplet_rows, 0),
        triplet_first=column(triplet_rows, 1),
        triplet_second=column(triplet_rows, 2),
        triplet_type=column(triplet_rows, 3),
        torsion_first=column(torsion_rows, 0),
        torsion_second=column(torsion_rows, 1),
        torsion_third=column(torsion_rows, 2),
        torsion_fourth=column(torsion_rows, 3),
        torsion_type=column(torsion_rows, 4),
    )


class TypedCancelingPairMLP(torch.nn.Module):
    """Type-specific Softplus differences with permanently paired outputs."""

    def __init__(
        self,
        type_count: int,
        input_size: int,
        hidden_size: int,
        *,
        seed: int,
        canceling_output: float = 1e-3,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or hidden_size % 2:
            raise ValueError("hidden_size must be a positive even integer")
        if type_count <= 0 or input_size <= 0:
            raise ValueError("type_count and input_size must be positive")
        if canceling_output <= 0.0:
            raise ValueError("canceling_output must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pair_count = hidden_size // 2
        incoming = torch.randn(
            type_count,
            pair_count,
            input_size,
            dtype=torch.float64,
            generator=generator,
        ) / input_size**0.5
        bias = torch.randn(
            type_count,
            pair_count,
            dtype=torch.float64,
            generator=generator,
        ) * 0.1
        self.weight = torch.nn.Parameter(torch.repeat_interleave(incoming, 2, dim=1))
        self.bias = torch.nn.Parameter(torch.repeat_interleave(bias, 2, dim=1))
        self.output_magnitude = torch.nn.Parameter(
            torch.full(
                (type_count, pair_count), canceling_output, dtype=torch.float64
            )
        )

    def forward(self, features: torch.Tensor, type_index: torch.Tensor) -> torch.Tensor:
        weight = self.weight[type_index]
        bias = self.bias[type_index]
        activation = torch.einsum("nid,nd->ni", weight, features) + bias
        paired = torch.nn.functional.softplus(activation).reshape(
            activation.shape[0], self.output_magnitude.shape[1], 2
        )
        return torch.sum(
            self.output_magnitude[type_index] * (paired[:, :, 0] - paired[:, :, 1]),
            dim=1,
        )


class LocalBodyOrderResidual(torch.nn.Module):
    """Pair-plus-triplet local scalar whose force and Hessian come from autograd."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        pair_hidden_size: int = 32,
        triplet_hidden_size: int = 64,
        torsion_hidden_size: int = 64,
        seed: int = 20260721,
        canceling_output: float = 1e-3,
        cutoff_bohr: float = 8.0,
        distance_center_bohr: float = 3.0,
        distance_scale_bohr: float = 2.0,
    ) -> None:
        super().__init__()
        if cutoff_bohr <= 0.0 or distance_scale_bohr <= 0.0:
            raise ValueError("cutoff_bohr and distance_scale_bohr must be positive")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.cutoff_bohr = float(cutoff_bohr)
        self.distance_center_bohr = float(distance_center_bohr)
        self.distance_scale_bohr = float(distance_scale_bohr)
        pair_type_count = len(self.elements) * (len(self.elements) + 1) // 2
        triplet_type_count = len(self.elements) * pair_type_count
        torsion_type_count = len(
            {
                min(key, tuple(reversed(key)))
                for key in product(range(len(self.elements)), repeat=4)
            }
        )
        self.pair = TypedCancelingPairMLP(
            pair_type_count,
            1,
            pair_hidden_size,
            seed=seed,
            canceling_output=canceling_output,
        )
        self.triplet = TypedCancelingPairMLP(
            triplet_type_count,
            3,
            triplet_hidden_size,
            seed=seed + 1,
            canceling_output=canceling_output,
        )
        self.torsion = TypedCancelingPairMLP(
            torsion_type_count,
            6,
            torsion_hidden_size,
            seed=seed + 2,
            canceling_output=canceling_output,
        )

    def _cutoff(self, distance: torch.Tensor) -> torch.Tensor:
        """Compact C3 cutoff, including a zero third derivative at the boundary."""
        scaled = torch.clamp(distance / self.cutoff_bohr, min=0.0, max=1.0)
        value = (
            1.0
            - 35.0 * scaled**4
            + 84.0 * scaled**5
            - 70.0 * scaled**6
            + 20.0 * scaled**7
        )
        return torch.where(distance < self.cutoff_bohr, value, torch.zeros_like(value))

    def forward_energy(
        self, positions_bohr: torch.Tensor, topology: BodyOrderTopology
    ) -> torch.Tensor:
        device = positions_bohr.device
        pair_first = topology.pair_first.to(device=device)
        pair_second = topology.pair_second.to(device=device)
        pair_type = topology.pair_type.to(device=device)
        pair_vector = positions_bohr[pair_second] - positions_bohr[pair_first]
        pair_distance = torch.linalg.vector_norm(pair_vector, dim=1)
        pair_features = (
            (pair_distance - self.distance_center_bohr) / self.distance_scale_bohr
        )[:, None]

        center = topology.triplet_center.to(device=device)
        first = topology.triplet_first.to(device=device)
        second = topology.triplet_second.to(device=device)
        triplet_type = topology.triplet_type.to(device=device)
        first_vector = positions_bohr[first] - positions_bohr[center]
        second_vector = positions_bohr[second] - positions_bohr[center]
        first_distance = torch.linalg.vector_norm(first_vector, dim=1)
        second_distance = torch.linalg.vector_norm(second_vector, dim=1)
        cosine = torch.sum(first_vector * second_vector, dim=1) / (
            first_distance * second_distance
        )
        triplet_features = torch.stack(
            (
                (first_distance - self.distance_center_bohr)
                / self.distance_scale_bohr,
                (second_distance - self.distance_center_bohr)
                / self.distance_scale_bohr,
                cosine,
            ),
            dim=1,
        )
        pair_energy = self.pair(pair_features, pair_type) * self._cutoff(pair_distance)
        triplet_energy = self.triplet(triplet_features, triplet_type) * (
            self._cutoff(first_distance) * self._cutoff(second_distance)
        )

        torsion_first = topology.torsion_first.to(device=device)
        torsion_second = topology.torsion_second.to(device=device)
        torsion_third = topology.torsion_third.to(device=device)
        torsion_fourth = topology.torsion_fourth.to(device=device)
        torsion_type = topology.torsion_type.to(device=device)
        first_bond = positions_bohr[torsion_second] - positions_bohr[torsion_first]
        central_bond = positions_bohr[torsion_third] - positions_bohr[torsion_second]
        third_bond = positions_bohr[torsion_fourth] - positions_bohr[torsion_third]
        first_bond_distance = torch.linalg.vector_norm(first_bond, dim=1)
        central_bond_distance = torch.linalg.vector_norm(central_bond, dim=1)
        third_bond_distance = torch.linalg.vector_norm(third_bond, dim=1)
        first_angle_cosine = torch.sum(-first_bond * central_bond, dim=1) / (
            first_bond_distance * central_bond_distance
        )
        second_angle_cosine = torch.sum(-central_bond * third_bond, dim=1) / (
            central_bond_distance * third_bond_distance
        )
        first_normal = torch.linalg.cross(first_bond, central_bond, dim=1)
        second_normal = torch.linalg.cross(central_bond, third_bond, dim=1)
        first_normal_squared = torch.sum(first_normal * first_normal, dim=1)
        second_normal_squared = torch.sum(second_normal * second_normal, dim=1)
        normal_floor = 64.0 * torch.finfo(positions_bohr.dtype).eps
        valid_torsion = (first_normal_squared > normal_floor**2) & (
            second_normal_squared > normal_floor**2
        )
        safe_first_squared = torch.where(
            valid_torsion, first_normal_squared, torch.ones_like(first_normal_squared)
        )
        safe_second_squared = torch.where(
            valid_torsion,
            second_normal_squared,
            torch.ones_like(second_normal_squared),
        )
        torsion_cosine_value = torch.sum(first_normal * second_normal, dim=1) / (
            torch.sqrt(safe_first_squared) * torch.sqrt(safe_second_squared)
        )
        torsion_cosine = torch.where(
            valid_torsion, torsion_cosine_value, torch.zeros_like(torsion_cosine_value)
        )
        torsion_features = torch.stack(
            (
                (first_bond_distance - self.distance_center_bohr)
                / self.distance_scale_bohr,
                (central_bond_distance - self.distance_center_bohr)
                / self.distance_scale_bohr,
                (third_bond_distance - self.distance_center_bohr)
                / self.distance_scale_bohr,
                first_angle_cosine,
                second_angle_cosine,
                torsion_cosine,
            ),
            dim=1,
        )
        torsion_energy = self.torsion(torsion_features, torsion_type) * (
            self._cutoff(first_bond_distance)
            * self._cutoff(central_bond_distance)
            * self._cutoff(third_bond_distance)
        )
        return pair_energy.sum() + triplet_energy.sum() + torsion_energy.sum()

    def forward_anchored_energy(
        self,
        positions_bohr: torch.Tensor,
        topology: BodyOrderTopology,
        reference_positions_bohr: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the value and first-order Taylor term at a fixed geometry."""
        reference = (
            reference_positions_bohr.detach().clone().requires_grad_(True)
        )
        reference_energy = self.forward_energy(reference, topology)
        reference_gradient = torch.autograd.grad(
            reference_energy, reference, create_graph=True
        )[0]
        displacement = positions_bohr - reference
        return (
            self.forward_energy(positions_bohr, topology)
            - reference_energy
            - torch.sum(reference_gradient * displacement)
        )

    def energy_force_hessian(
        self,
        positions_bohr: torch.Tensor,
        topology: BodyOrderTopology,
        *,
        create_parameter_graph: bool = True,
        reference_positions_bohr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = positions_bohr
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        energy = (
            self.forward_energy(positions, topology)
            if reference_positions_bohr is None
            else self.forward_anchored_energy(
                positions, topology, reference_positions_bohr
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
                        create_parameter_graph
                        or index + 1 < flat_gradient.numel()
                    ),
                )[0].reshape(-1)
            )
        hessian = torch.stack(rows)
        return energy, -gradient, hessian

    def energy_force_hvp(
        self,
        positions_bohr: torch.Tensor,
        topology: BodyOrderTopology,
        directions: torch.Tensor,
        *,
        create_parameter_graph: bool = True,
        reference_positions_bohr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate scalar energy, force, and energy-Hessian vector products."""
        positions = positions_bohr
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
        directions = torch.as_tensor(
            directions, dtype=positions.dtype, device=positions.device
        )
        if directions.ndim == positions.ndim:
            directions = directions[None, ...]
        if directions.ndim != positions.ndim + 1 or directions.shape[1:] != positions.shape:
            raise ValueError("directions must have shape (n_directions, n_atoms, 3)")
        energy = (
            self.forward_energy(positions, topology)
            if reference_positions_bohr is None
            else self.forward_anchored_energy(
                positions, topology, reference_positions_bohr
            )
        )
        gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
        products = []
        for index, direction in enumerate(directions):
            products.append(
                torch.autograd.grad(
                    torch.sum(gradient * direction),
                    positions,
                    create_graph=create_parameter_graph,
                    retain_graph=(
                        create_parameter_graph or index + 1 < directions.shape[0]
                    ),
                )[0]
            )
        return energy, -gradient, torch.stack(products)
