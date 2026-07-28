"""Reference-local equivariant quadratic scalar for curvature-capacity audits."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as functional

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology


class _EquivariantCurvatureNetwork(torch.nn.Module):
    """Predict a covariant Cartesian curvature matrix from one reference geometry."""

    def __init__(
        self,
        *,
        element_count: int,
        hidden_size: int,
        radial_size: int,
        cutoff_bohr: float,
        coefficient_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.element_count = int(element_count)
        self.radial_size = int(radial_size)
        self.cutoff_bohr = float(cutoff_bohr)
        self.coefficient_scale = float(coefficient_scale)
        atom_input_size = element_count + element_count * radial_size
        self.atom_encoder = self._mlp(atom_input_size, hidden_size, hidden_size)
        self.pair_readout = self._mlp(
            2 * hidden_size + radial_size + 27,
            hidden_size,
            10,
        )
        self.diagonal_readout = self._mlp(hidden_size + 3, hidden_size, 3)
        centers = torch.linspace(0.5, cutoff_bohr, radial_size, dtype=torch.float64)
        spacing = (cutoff_bohr - 0.5) / max(radial_size - 1, 1)
        self.register_buffer("radial_centers", centers)
        self.register_buffer(
            "radial_inverse_width_squared",
            torch.tensor(1.0 / max(spacing, 1e-6) ** 2, dtype=torch.float64),
        )
        self._initialize(seed)

    @staticmethod
    def _mlp(input_size: int, hidden_size: int, output_size: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, output_size, dtype=torch.float64),
        )

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

    @staticmethod
    def _bounded_frame(frame: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(1.0 + torch.mean(frame * frame, dim=(1, 2)))
        return frame / scale[:, None, None]

    @staticmethod
    def _internal_projector(positions_bohr: torch.Tensor) -> torch.Tensor:
        atom_count = positions_bohr.shape[0]
        coordinate_count = 3 * atom_count
        centered = positions_bohr - torch.mean(positions_bohr, dim=0, keepdim=True)
        translations = []
        rotations = []
        identity3 = torch.eye(
            3, dtype=positions_bohr.dtype, device=positions_bohr.device
        )
        for axis in range(3):
            translation = torch.zeros_like(positions_bohr)
            translation[:, axis] = 1.0
            translations.append(translation.reshape(-1))
            rotations.append(
                torch.linalg.cross(
                    identity3[axis][None, :].expand_as(centered), centered, dim=1
                ).reshape(-1)
            )
        rigid = torch.stack((*translations, *rotations), dim=1)
        left, singular, _ = torch.linalg.svd(rigid, full_matrices=False)
        tolerance = (
            max(rigid.shape)
            * torch.finfo(rigid.dtype).eps
            * torch.max(singular).clamp_min(torch.finfo(rigid.dtype).tiny)
        )
        rank = int(torch.sum(singular > tolerance).detach().cpu())
        external = left[:, :rank]
        identity = torch.eye(
            coordinate_count, dtype=positions_bohr.dtype, device=positions_bohr.device
        )
        return identity - external @ external.T

    def forward(
        self,
        positions_bohr: torch.Tensor,
        element_index: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        device = positions_bohr.device
        first = topology.pair_first.to(device=device)
        second = topology.pair_second.to(device=device)
        pair_vector = positions_bohr[second] - positions_bohr[first]
        distance = torch.linalg.vector_norm(pair_vector, dim=1)
        unit = pair_vector / distance[:, None]
        cutoff = self._cutoff(distance)
        radial = torch.exp(
            -self.radial_inverse_width_squared
            * (distance[:, None] - self.radial_centers[None, :]) ** 2
        ) * cutoff[:, None]

        source = torch.cat((first, second), dim=0)
        destination = torch.cat((second, first), dim=0)
        directed_radial = torch.cat((radial, radial), dim=0)
        neighbor_type = element_index[source]
        neighbor_radial = (
            functional.one_hot(neighbor_type, num_classes=self.element_count)
            .to(dtype=positions_bohr.dtype)[:, :, None]
            * directed_radial[:, None, :]
        ).reshape(source.numel(), -1)
        environment = torch.zeros(
            positions_bohr.shape[0],
            self.element_count * self.radial_size,
            dtype=positions_bohr.dtype,
            device=device,
        ).index_add(0, destination, neighbor_radial)
        atom_input = torch.cat(
            (
                functional.one_hot(element_index, num_classes=self.element_count).to(
                    dtype=positions_bohr.dtype
                ),
                environment,
            ),
            dim=1,
        )
        atom_hidden = self.atom_encoder(atom_input)

        pair_outer = unit[:, :, None] * unit[:, None, :]
        moment_contribution = (
            torch.exp(-distance / 4.0) * cutoff
        )[:, None, None] * pair_outer
        moment = torch.zeros(
            positions_bohr.shape[0],
            3,
            3,
            dtype=positions_bohr.dtype,
            device=device,
        ).index_add(
            0,
            torch.cat((first, second), dim=0),
            torch.cat((moment_contribution, moment_contribution), dim=0),
        )
        trace = torch.diagonal(moment, dim1=1, dim2=2).sum(dim=1)
        moment = moment / (1.0 + trace[:, None, None])
        moment_squared = moment @ moment

        first_frame = self._bounded_frame(
            torch.stack(
                (
                    unit,
                    torch.einsum("pij,pj->pi", moment[first], unit),
                    torch.einsum("pij,pj->pi", moment_squared[first], unit),
                ),
                dim=2,
            )
        )
        second_frame = self._bounded_frame(
            torch.stack(
                (
                    unit,
                    torch.einsum("pij,pj->pi", moment[second], unit),
                    torch.einsum("pij,pj->pi", moment_squared[second], unit),
                ),
                dim=2,
            )
        )
        first_gram = first_frame.transpose(1, 2) @ first_frame
        second_gram = second_frame.transpose(1, 2) @ second_frame
        cross_gram = first_frame.transpose(1, 2) @ second_frame
        forward_input = torch.cat(
            (
                atom_hidden[first],
                atom_hidden[second],
                radial,
                first_gram.reshape(first.numel(), -1),
                second_gram.reshape(first.numel(), -1),
                cross_gram.reshape(first.numel(), -1),
            ),
            dim=1,
        )
        reverse_input = torch.cat(
            (
                atom_hidden[second],
                atom_hidden[first],
                radial,
                second_gram.reshape(first.numel(), -1),
                first_gram.reshape(first.numel(), -1),
                cross_gram.transpose(1, 2).reshape(first.numel(), -1),
            ),
            dim=1,
        )
        forward_coefficients = self.coefficient_scale * torch.tanh(
            self.pair_readout(forward_input)
        )
        reverse_coefficients = self.coefficient_scale * torch.tanh(
            self.pair_readout(reverse_input)
        )
        coefficient_matrix = 0.5 * (
            forward_coefficients[:, :9].reshape(-1, 3, 3)
            + reverse_coefficients[:, :9].reshape(-1, 3, 3).transpose(1, 2)
        )
        pair_block = first_frame @ coefficient_matrix @ second_frame.transpose(1, 2)
        isotropic = 0.5 * (
            forward_coefficients[:, 9] + reverse_coefficients[:, 9]
        )
        pair_block = pair_block + isotropic[:, None, None] * torch.eye(
            3, dtype=positions_bohr.dtype, device=device
        )[None, :, :]

        diagonal_invariants = torch.stack(
            (
                trace,
                torch.sum(moment * moment, dim=(1, 2)),
                torch.linalg.det(moment),
            ),
            dim=1,
        )
        diagonal_coefficients = self.coefficient_scale * torch.tanh(
            self.diagonal_readout(
                torch.cat((atom_hidden, diagonal_invariants), dim=1)
            )
        )
        identity3 = torch.eye(3, dtype=positions_bohr.dtype, device=device)
        diagonal_block = (
            diagonal_coefficients[:, 0, None, None] * identity3[None, :, :]
            + diagonal_coefficients[:, 1, None, None] * moment
            + diagonal_coefficients[:, 2, None, None] * moment_squared
        )

        zero = torch.zeros(3, 3, dtype=positions_bohr.dtype, device=device)
        blocks = [
            [zero for _ in range(positions_bohr.shape[0])]
            for _ in range(positions_bohr.shape[0])
        ]
        for atom in range(positions_bohr.shape[0]):
            blocks[atom][atom] = diagonal_block[atom]
        for pair_index in range(first.numel()):
            first_atom = int(first[pair_index])
            second_atom = int(second[pair_index])
            blocks[first_atom][second_atom] = pair_block[pair_index]
            blocks[second_atom][first_atom] = pair_block[pair_index].T
        raw = torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)
        raw = 0.5 * (raw + raw.T)
        projector = self._internal_projector(positions_bohr)
        projected = projector @ raw @ projector
        return 0.5 * (projected + projected.T)


class LocalEquivariantQuadraticScalarResidual(torch.nn.Module):
    """Local Taylor scalar whose curvature is a shared equivariant block network."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_size: int = 64,
        radial_size: int = 20,
        cutoff_bohr: float = 20.0,
        coefficient_scale: float = 1.0,
        seed: int = 20260731,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or radial_size <= 1:
            raise ValueError("hidden and radial sizes must be positive")
        if cutoff_bohr <= 0.5 or coefficient_scale <= 0.0:
            raise ValueError("cutoff and coefficient scale must be positive")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.register_buffer("element_lookup", self._build_element_lookup(self.elements))
        self.network = _EquivariantCurvatureNetwork(
            element_count=len(self.elements),
            hidden_size=hidden_size,
            radial_size=radial_size,
            cutoff_bohr=cutoff_bohr,
            coefficient_scale=coefficient_scale,
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

    def hessian_correction(
        self,
        reference_positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        element_index = self._element_index(atomic_numbers)
        active = self.network(reference_positions_bohr, element_index, topology)
        with torch.no_grad():
            initial = self.initial_network(
                reference_positions_bohr, element_index, topology
            )
        return active - initial

    def forward_anchored_energy(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
        reference_positions_bohr: torch.Tensor,
    ) -> torch.Tensor:
        reference = reference_positions_bohr.detach().clone()
        correction = self.hessian_correction(reference, atomic_numbers, topology)
        displacement = (positions_bohr - reference).reshape(-1)
        return 0.5 * displacement @ correction @ displacement

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
        reference = positions.detach() if reference_positions_bohr is None else reference_positions_bohr
        energy = self.forward_anchored_energy(
            positions, atomic_numbers, topology, reference
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
