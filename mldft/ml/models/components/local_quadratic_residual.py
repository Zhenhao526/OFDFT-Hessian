"""Reference-anchored local internal-coordinate quadratic scalar features."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch


_COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class InternalCoordinateSpec:
    kind: str
    atoms: tuple[int, ...]
    element_key: tuple[int, ...]
    environment_key: tuple[tuple[int, ...], ...]
    parity: int


@dataclass(frozen=True)
class QuadraticTerm:
    key: str
    first: int
    second: int
    scale: float


def build_internal_coordinate_specs(
    atomic_numbers: torch.Tensor,
    positions_bohr: torch.Tensor,
    *,
    bond_scale: float = 1.25,
) -> list[InternalCoordinateSpec]:
    """Build bonded distance, angle-cosine, and parity-safe torsion coordinates."""
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long, device="cpu")
    positions = torch.as_tensor(positions_bohr, dtype=torch.float64, device="cpu")
    atom_count = int(atomic_numbers.numel())
    if positions.shape != (atom_count, 3):
        raise ValueError("positions_bohr must have shape (n_atoms, 3)")
    if bond_scale <= 0.0:
        raise ValueError("bond_scale must be positive")
    unsupported = sorted(set(atomic_numbers.tolist()) - set(_COVALENT_RADII_ANGSTROM))
    if unsupported:
        raise ValueError(f"unsupported atomic numbers: {unsupported}")

    bonds: list[tuple[int, int]] = []
    neighbors = [[] for _ in range(atom_count)]
    for first in range(atom_count):
        for second in range(first + 1, atom_count):
            cutoff = bond_scale * (
                _COVALENT_RADII_ANGSTROM[int(atomic_numbers[first])]
                + _COVALENT_RADII_ANGSTROM[int(atomic_numbers[second])]
            ) * _ANGSTROM_TO_BOHR
            distance = float(torch.linalg.vector_norm(positions[first] - positions[second]))
            if distance <= cutoff:
                bonds.append((first, second))
                neighbors[first].append(second)
                neighbors[second].append(first)

    specs: list[InternalCoordinateSpec] = []
    environment_signatures = [
        (
            int(atomic_numbers[index]),
            *sorted(int(atomic_numbers[neighbor]) for neighbor in neighbors[index]),
        )
        for index in range(atom_count)
    ]
    for first, second in bonds:
        ordered_atoms = sorted(
            (first, second),
            key=lambda index: (int(atomic_numbers[index]), environment_signatures[index]),
        )
        specs.append(
            InternalCoordinateSpec(
                kind="bond",
                atoms=(first, second),
                element_key=tuple(
                    sorted((int(atomic_numbers[first]), int(atomic_numbers[second])))
                ),
                environment_key=tuple(
                    environment_signatures[index] for index in ordered_atoms
                ),
                parity=1,
            )
        )
    for center in range(atom_count):
        ordered_neighbors = sorted(neighbors[center])
        for first_offset, first in enumerate(ordered_neighbors):
            for second in ordered_neighbors[first_offset + 1 :]:
                outer_elements = tuple(
                    sorted((int(atomic_numbers[first]), int(atomic_numbers[second])))
                )
                outer_environments = tuple(
                    sorted(
                        (
                            environment_signatures[first],
                            environment_signatures[second],
                        )
                    )
                )
                specs.append(
                    InternalCoordinateSpec(
                        kind="angle_cos",
                        atoms=(first, center, second),
                        element_key=(int(atomic_numbers[center]), *outer_elements),
                        environment_key=(
                            environment_signatures[center],
                            *outer_environments,
                        ),
                        parity=1,
                    )
                )

    seen: set[tuple[int, int, int, int]] = set()
    for second, third in bonds:
        for first in neighbors[second]:
            if first == third:
                continue
            for fourth in neighbors[third]:
                if fourth in (first, second):
                    continue
                chain = (first, second, third, fourth)
                element_key = tuple(int(atomic_numbers[index]) for index in chain)
                reverse_chain = tuple(reversed(chain))
                reverse_key = tuple(reversed(element_key))
                if reverse_key < element_key or (
                    reverse_key == element_key and reverse_chain < chain
                ):
                    chain = reverse_chain
                    element_key = reverse_key
                if chain in seen:
                    continue
                seen.add(chain)
                specs.append(
                    InternalCoordinateSpec(
                        kind="torsion_cos",
                        atoms=chain,
                        element_key=element_key,
                        environment_key=tuple(
                            environment_signatures[index] for index in chain
                        ),
                        parity=1,
                    )
                )
                specs.append(
                    InternalCoordinateSpec(
                        kind="torsion_sin",
                        atoms=chain,
                        element_key=element_key,
                        environment_key=tuple(
                            environment_signatures[index] for index in chain
                        ),
                        parity=-1,
                    )
                )
    return specs


def evaluate_internal_coordinates(
    positions_bohr: torch.Tensor,
    specs: list[InternalCoordinateSpec],
) -> torch.Tensor:
    """Evaluate smooth scalar internal coordinates in the fixed reference topology."""
    values = []
    tiny = 64.0 * torch.finfo(positions_bohr.dtype).eps
    for spec in specs:
        if spec.kind == "bond":
            first, second = spec.atoms
            values.append(
                torch.linalg.vector_norm(positions_bohr[second] - positions_bohr[first])
            )
            continue
        if spec.kind == "angle_cos":
            first, center, second = spec.atoms
            first_vector = positions_bohr[first] - positions_bohr[center]
            second_vector = positions_bohr[second] - positions_bohr[center]
            values.append(
                torch.dot(first_vector, second_vector)
                / (
                    torch.linalg.vector_norm(first_vector)
                    * torch.linalg.vector_norm(second_vector)
                )
            )
            continue
        first, second, third, fourth = spec.atoms
        first_bond = positions_bohr[second] - positions_bohr[first]
        central_bond = positions_bohr[third] - positions_bohr[second]
        third_bond = positions_bohr[fourth] - positions_bohr[third]
        first_normal = torch.linalg.cross(first_bond, central_bond)
        second_normal = torch.linalg.cross(central_bond, third_bond)
        first_squared = torch.dot(first_normal, first_normal)
        second_squared = torch.dot(second_normal, second_normal)
        valid = (first_squared > tiny**2) & (second_squared > tiny**2)
        safe_first = torch.where(valid, first_squared, torch.ones_like(first_squared))
        safe_second = torch.where(valid, second_squared, torch.ones_like(second_squared))
        denominator = torch.sqrt(safe_first * safe_second)
        if spec.kind == "torsion_cos":
            raw = torch.dot(first_normal, second_normal) / denominator
        elif spec.kind == "torsion_sin":
            central_unit = central_bond / torch.linalg.vector_norm(central_bond)
            raw = torch.dot(
                torch.linalg.cross(first_normal, second_normal), central_unit
            ) / denominator
        else:
            raise ValueError(f"unsupported coordinate kind: {spec.kind}")
        values.append(torch.where(valid, raw, torch.zeros_like(raw)))
    if not values:
        raise ValueError("topology produced no internal coordinates")
    return torch.stack(values)


def internal_coordinate_values_and_jacobian(
    positions_bohr: torch.Tensor,
    specs: list[InternalCoordinateSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.as_tensor(positions_bohr, dtype=torch.float64)
    function = lambda value: evaluate_internal_coordinates(value, specs)
    values = function(positions)
    jacobian = torch.func.jacrev(function)(positions).reshape(values.numel(), -1)
    return values, jacobian


def _legendre_values(value: float, order: int = 4) -> list[float]:
    result = [1.0]
    if order == 0:
        return result
    result.append(value)
    for degree in range(2, order + 1):
        result.append(
            ((2 * degree - 1) * value * result[-1] - (degree - 1) * result[-2])
            / degree
        )
    return result


def _diagonal_basis(spec: InternalCoordinateSpec, value: float) -> list[tuple[int, float]]:
    if spec.kind == "bond":
        centers = torch.linspace(1.2, 3.4, 7, dtype=torch.float64)
        radial = torch.exp(-0.5 * ((torch.tensor(value) - centers) / 0.4) ** 2)
        return [(index, float(item)) for index, item in enumerate(radial)] + [
            (len(centers), 1.0)
        ]
    return list(enumerate(_legendre_values(value, order=4)))


def build_quadratic_terms(
    specs: list[InternalCoordinateSpec],
    values: torch.Tensor,
    *,
    include_cross_terms: bool,
    cross_all_pairs: bool = False,
    use_environment_types: bool = False,
) -> list[QuadraticTerm]:
    """Build shared local force-constant feature keys at one reference geometry."""
    terms = []
    type_labels = []
    for spec in specs:
        label: tuple[object, ...] = (spec.kind, *spec.element_key)
        if use_environment_types:
            label = (*label, spec.environment_key)
        type_labels.append(label)
    for index, (spec, value) in enumerate(zip(specs, values.tolist(), strict=True)):
        for basis_index, scale in _diagonal_basis(spec, float(value)):
            key = json.dumps(
                ["diagonal", *type_labels[index], basis_index], separators=(",", ":")
            )
            terms.append(
                QuadraticTerm(key=key, first=index, second=index, scale=scale)
            )
    if include_cross_terms:
        for first in range(len(specs)):
            first_atoms = set(specs[first].atoms)
            for second in range(first + 1, len(specs)):
                if specs[first].parity * specs[second].parity < 0:
                    continue
                overlap = len(first_atoms.intersection(specs[second].atoms))
                if overlap == 0 and not cross_all_pairs:
                    continue
                labels = sorted((type_labels[first], type_labels[second]))
                key = json.dumps(
                    ["cross", overlap, labels[0], labels[1]], separators=(",", ":")
                )
                terms.append(
                    QuadraticTerm(key=key, first=first, second=second, scale=1.0)
                )
    return terms


def assemble_quadratic_hessian(
    jacobian: torch.Tensor,
    terms: list[QuadraticTerm],
    coefficients: dict[str, float | torch.Tensor],
) -> torch.Tensor:
    coordinate_count = jacobian.shape[1]
    result = torch.zeros(
        (coordinate_count, coordinate_count),
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    for term in terms:
        coefficient = coefficients.get(term.key, 0.0)
        if term.first == term.second:
            basis = torch.outer(jacobian[term.first], jacobian[term.first])
        else:
            basis = torch.outer(jacobian[term.first], jacobian[term.second])
            basis = basis + basis.T
        result = result + coefficient * term.scale * basis
    return result


def anchored_quadratic_energy(
    positions_bohr: torch.Tensor,
    reference_values: torch.Tensor,
    specs: list[InternalCoordinateSpec],
    terms: list[QuadraticTerm],
    coefficients: dict[str, float | torch.Tensor],
) -> torch.Tensor:
    """Evaluate the scalar whose reference Hessian is the assembled feature model."""
    displacement = evaluate_internal_coordinates(positions_bohr, specs) - reference_values
    energy = positions_bohr.new_zeros(())
    for term in terms:
        coefficient = coefficients.get(term.key, 0.0)
        product = displacement[term.first] * displacement[term.second]
        factor = 0.5 if term.first == term.second else 1.0
        energy = energy + factor * coefficient * term.scale * product
    return energy


def coordinate_feature_matrix(
    specs: list[InternalCoordinateSpec],
    values: torch.Tensor,
    jacobian: torch.Tensor,
) -> torch.Tensor:
    """Encode continuous local chemistry for shared force-constant prediction."""
    elements = (1, 6, 7, 8, 9)
    element_index = {value: index for index, value in enumerate(elements)}
    kind_index = {
        "bond": 0,
        "angle_cos": 1,
        "torsion_cos": 2,
        "torsion_sin": 3,
    }
    rows = []
    for index, spec in enumerate(specs):
        kind = torch.zeros(4, dtype=torch.float64)
        kind[kind_index[spec.kind]] = 1.0
        composition = torch.zeros(len(elements), dtype=torch.float64)
        slots = torch.zeros(4, len(elements), dtype=torch.float64)
        for slot, atomic_number in enumerate(spec.element_key):
            composition[element_index[atomic_number]] += 1.0
            slots[slot, element_index[atomic_number]] = 1.0
        neighbor_composition = torch.zeros(len(elements), dtype=torch.float64)
        coordination = []
        for environment in spec.environment_key:
            coordination.append(float(len(environment) - 1))
            for atomic_number in environment[1:]:
                neighbor_composition[element_index[atomic_number]] += 1.0
        coordination_tensor = torch.zeros(4, dtype=torch.float64)
        coordination_tensor[: len(coordination)] = torch.tensor(
            coordination, dtype=torch.float64
        )
        raw_value = values[index]
        scaled_value = (
            (raw_value - 2.3) / 1.1 if spec.kind == "bond" else raw_value
        )
        continuous = torch.stack(
            (
                scaled_value,
                scaled_value.square(),
                torch.linalg.vector_norm(jacobian[index]) / 3.0,
                torch.tensor(float(len(spec.atoms)) / 4.0, dtype=torch.float64),
            )
        )
        rows.append(
            torch.cat(
                (
                    kind,
                    composition / 4.0,
                    slots.reshape(-1),
                    neighbor_composition / 12.0,
                    coordination_tensor / 4.0,
                    continuous,
                )
            )
        )
    return torch.stack(rows).to(device=values.device)


def build_cross_coordinate_pairs(
    specs: list[InternalCoordinateSpec],
    *,
    local_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_rows = []
    second_rows = []
    overlap_rows = []
    for first in range(len(specs)):
        first_atoms = set(specs[first].atoms)
        for second in range(first + 1, len(specs)):
            if specs[first].parity * specs[second].parity < 0:
                continue
            overlap = len(first_atoms.intersection(specs[second].atoms))
            if local_only and overlap == 0:
                continue
            first_rows.append(first)
            second_rows.append(second)
            overlap_rows.append(overlap)
    return (
        torch.as_tensor(first_rows, dtype=torch.long),
        torch.as_tensor(second_rows, dtype=torch.long),
        torch.as_tensor(overlap_rows, dtype=torch.float64),
    )


def cross_coordinate_feature_matrix(
    coordinate_features: torch.Tensor,
    jacobian: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    overlap: torch.Tensor,
    specs: list[InternalCoordinateSpec],
) -> torch.Tensor:
    """Encode an unordered, parity-even pair of internal coordinates."""
    first = first.to(device=coordinate_features.device)
    second = second.to(device=coordinate_features.device)
    overlap = overlap.to(dtype=coordinate_features.dtype, device=coordinate_features.device)
    summed = coordinate_features[first] + coordinate_features[second]
    difference = torch.abs(coordinate_features[first] - coordinate_features[second])
    first_jacobian = jacobian[first]
    second_jacobian = jacobian[second]
    cosine = torch.sum(first_jacobian * second_jacobian, dim=1) / (
        torch.linalg.vector_norm(first_jacobian, dim=1)
        * torch.linalg.vector_norm(second_jacobian, dim=1)
    ).clamp_min(torch.finfo(coordinate_features.dtype).tiny)
    odd_pair = torch.as_tensor(
        [
            float(specs[int(a)].parity < 0 and specs[int(b)].parity < 0)
            for a, b in zip(first.cpu(), second.cpu(), strict=True)
        ],
        dtype=coordinate_features.dtype,
        device=coordinate_features.device,
    )
    extra = torch.stack((overlap / 4.0, cosine, odd_pair), dim=1)
    return torch.cat((summed, difference, extra), dim=1)


def global_cross_coordinate_feature_matrix(
    coordinate_features: torch.Tensor,
    jacobian: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    overlap: torch.Tensor,
    specs: list[InternalCoordinateSpec],
    positions_bohr: torch.Tensor,
) -> torch.Tensor:
    """Add rotation/translation-invariant geometry for nonlocal coordinate pairs."""
    base = cross_coordinate_feature_matrix(
        coordinate_features,
        jacobian,
        first,
        second,
        overlap,
        specs,
    )
    positions = torch.as_tensor(
        positions_bohr,
        dtype=coordinate_features.dtype,
        device=coordinate_features.device,
    )
    atom_count = int(positions.shape[0])
    if positions.shape != (atom_count, 3):
        raise ValueError("positions_bohr must have shape (n_atoms, 3)")
    if jacobian.shape[1] != atom_count * 3:
        raise ValueError("jacobian Cartesian dimension does not match positions")

    first_device = first.to(device=coordinate_features.device)
    second_device = second.to(device=coordinate_features.device)
    gradients = jacobian.reshape(jacobian.shape[0], atom_count, 3)
    moments = torch.einsum("nai,naj->nij", gradients, gradients)
    moment_norm = torch.linalg.matrix_norm(moments, dim=(-2, -1)).clamp_min(
        torch.finfo(coordinate_features.dtype).tiny
    )
    moments = moments / moment_norm[:, None, None]
    centers = torch.stack(
        [
            torch.mean(
                positions[
                    torch.as_tensor(
                        spec.atoms,
                        dtype=torch.long,
                        device=positions.device,
                    )
                ],
                dim=0,
            )
            for spec in specs
        ]
    )
    separation = centers[first_device] - centers[second_device]
    center_distance = torch.linalg.vector_norm(separation, dim=1)
    safe_distance = center_distance.clamp_min(
        torch.finfo(coordinate_features.dtype).eps
    )
    unit = separation / safe_distance[:, None]

    first_moment = moments[first_device]
    second_moment = moments[second_device]
    moment_overlap = torch.einsum("nij,nji->n", first_moment, second_moment)
    first_longitudinal = torch.einsum(
        "ni,nij,nj->n", unit, first_moment, unit
    )
    second_longitudinal = torch.einsum(
        "ni,nij,nj->n", unit, second_moment, unit
    )
    mixed_moment = 0.5 * (
        first_moment @ second_moment + second_moment @ first_moment
    )
    mixed_longitudinal = torch.einsum(
        "ni,nij,nj->n", unit, mixed_moment, unit
    )

    minimum_distances = []
    for raw_first, raw_second in zip(first.tolist(), second.tolist(), strict=True):
        first_atoms = torch.as_tensor(
            specs[raw_first].atoms, dtype=torch.long, device=positions.device
        )
        second_atoms = torch.as_tensor(
            specs[raw_second].atoms, dtype=torch.long, device=positions.device
        )
        pair_distances = torch.cdist(
            positions[first_atoms], positions[second_atoms]
        )
        minimum_distances.append(torch.min(pair_distances))
    minimum_distance = torch.stack(minimum_distances)

    radial_centers = torch.linspace(
        0.0,
        10.0,
        6,
        dtype=coordinate_features.dtype,
        device=coordinate_features.device,
    )
    center_radial = torch.exp(
        -0.5 * ((center_distance[:, None] - radial_centers[None, :]) / 2.0) ** 2
    )
    geometric = torch.cat(
        (
            center_radial,
            torch.stack(
                (
                    center_distance / 10.0,
                    minimum_distance / 10.0,
                    moment_overlap,
                    first_longitudinal + second_longitudinal,
                    torch.abs(first_longitudinal - second_longitudinal),
                    mixed_longitudinal,
                ),
                dim=1,
            ),
        ),
        dim=1,
    )
    return torch.cat((base, geometric), dim=1)


class LocalQuadraticCoefficientNetwork(torch.nn.Module):
    """Shared smooth coefficient maps for an anchored quadratic scalar."""

    def __init__(
        self,
        coordinate_feature_count: int,
        cross_feature_count: int,
        *,
        hidden_size: int = 128,
        seed: int = 20260721,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if coordinate_feature_count <= 0 or cross_feature_count <= 0 or hidden_size <= 0:
            raise ValueError("feature counts and hidden size must be positive")
        torch.manual_seed(seed)
        self.diagonal = self._network(coordinate_feature_count, hidden_size)
        self.cross = self._network(cross_feature_count, hidden_size)
        if zero_output:
            with torch.no_grad():
                self.diagonal[-1].weight.zero_()
                self.diagonal[-1].bias.zero_()
                self.cross[-1].weight.zero_()
                self.cross[-1].bias.zero_()

    @staticmethod
    def _network(input_size: int, hidden_size: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, dtype=torch.float64),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_size, 1, dtype=torch.float64),
        )

    def forward(
        self,
        coordinate_features: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.diagonal(coordinate_features).reshape(-1),
            self.cross(cross_features).reshape(-1),
        )


def network_anchored_quadratic_energy(
    positions_bohr: torch.Tensor,
    reference_values: torch.Tensor,
    specs: list[InternalCoordinateSpec],
    diagonal_coefficients: torch.Tensor,
    cross_coefficients: torch.Tensor,
    cross_first: torch.Tensor,
    cross_second: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the scalar whose reference Hessian is assembled below."""
    displacement = evaluate_internal_coordinates(positions_bohr, specs) - reference_values
    first = cross_first.to(device=displacement.device)
    second = cross_second.to(device=displacement.device)
    diagonal = 0.5 * torch.sum(diagonal_coefficients * displacement.square())
    cross = torch.sum(
        cross_coefficients * displacement[first] * displacement[second]
    )
    return diagonal + cross


def assemble_network_hessian(
    jacobian: torch.Tensor,
    diagonal_coefficients: torch.Tensor,
    cross_coefficients: torch.Tensor,
    cross_first: torch.Tensor,
    cross_second: torch.Tensor,
) -> torch.Tensor:
    """Assemble the exact reference Hessian of the network-owned quadratic scalar."""
    result = (jacobian.T * diagonal_coefficients[None, :]) @ jacobian
    first = cross_first.to(device=jacobian.device)
    second = cross_second.to(device=jacobian.device)
    first_jacobian = jacobian[first]
    second_jacobian = jacobian[second]
    result = result + (first_jacobian.T * cross_coefficients[None, :]) @ second_jacobian
    result = result + (second_jacobian.T * cross_coefficients[None, :]) @ first_jacobian
    return result


def network_hvp(
    jacobian: torch.Tensor,
    directions: torch.Tensor,
    diagonal_coefficients: torch.Tensor,
    cross_coefficients: torch.Tensor,
    cross_first: torch.Tensor,
    cross_second: torch.Tensor,
) -> torch.Tensor:
    projection = directions @ jacobian.T
    result = (projection * diagonal_coefficients[None, :]) @ jacobian
    first = cross_first.to(device=jacobian.device)
    second = cross_second.to(device=jacobian.device)
    result = result + (
        projection[:, second] * cross_coefficients[None, :]
    ) @ jacobian[first]
    result = result + (
        projection[:, first] * cross_coefficients[None, :]
    ) @ jacobian[second]
    return result
