"""Linear scalar-energy readout over frozen MACE invariant atom features."""

from __future__ import annotations

import math

import torch

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology
from mldft.ml.models.components.local_mace_scalar_residual import (
    LocalMACEScalarResidual,
)


class LocalMACEInvariantReadoutResidual(torch.nn.Module):
    """Reference-local scalar with a frozen MACE backbone and linear coefficients."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_channels: int = 16,
        mlp_channels: int = 16,
        max_ell: int = 2,
        correlation: int = 3,
        num_interactions: int = 2,
        num_bessel: int = 8,
        cutoff_polynomial_order: int = 5,
        cutoff_bohr: float = 20.0,
        avg_num_neighbors: float = 16.0,
        feature_mode: str = "scalar",
        random_feature_width: int = 512,
        random_feature_input_scale: float = 1.0,
        output_scale: float = 1.0,
        seed: int = 20260804,
    ) -> None:
        super().__init__()
        if random_feature_width <= 0:
            raise ValueError("random feature width must be positive")
        if output_scale <= 0.0:
            raise ValueError("output scale must be positive")
        if random_feature_input_scale <= 0.0:
            raise ValueError("random feature input scale must be positive")
        if feature_mode not in {"scalar", "power_spectrum"}:
            raise ValueError("unsupported MACE invariant feature mode")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.output_scale = float(output_scale)
        self.feature_mode = feature_mode
        self.random_feature_input_scale = float(random_feature_input_scale)
        self.backbone = LocalMACEScalarResidual(
            elements=self.elements,
            hidden_channels=hidden_channels,
            mlp_channels=mlp_channels,
            max_ell=max_ell,
            correlation=correlation,
            num_interactions=num_interactions,
            num_bessel=num_bessel,
            cutoff_polynomial_order=cutoff_polynomial_order,
            cutoff_bohr=cutoff_bohr,
            avg_num_neighbors=avg_num_neighbors,
            output_scale=1.0,
            seed=seed,
        )
        self.backbone.requires_grad_(False)
        raw_feature_width = (
            hidden_channels * num_interactions
            if feature_mode == "scalar"
            else self.backbone.invariant_feature_width
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        projection = torch.randn(
            raw_feature_width,
            random_feature_width,
            dtype=torch.float64,
            generator=generator,
        ) / math.sqrt(float(raw_feature_width))
        bias = 2.0 * torch.rand(
            random_feature_width, dtype=torch.float64, generator=generator
        ) - 1.0
        self.register_buffer("random_projection", projection)
        self.register_buffer("random_bias", bias)
        self.readout = torch.nn.Parameter(
            torch.zeros(
                len(self.elements),
                raw_feature_width + random_feature_width,
                dtype=torch.float64,
            )
        )

    def load_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.backbone.load_state_dict(state_dict)
        self.backbone.requires_grad_(False)

    def atom_features(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        raw_features = (
            self.backbone.forward_node_scalar_features(
                positions_bohr, atomic_numbers, topology
            )
            if self.feature_mode == "scalar"
            else self.backbone.forward_node_invariant_features(
                positions_bohr, atomic_numbers, topology
            )
        )
        nonlinear = torch.tanh(
            self.random_feature_input_scale
            * (raw_features @ self.random_projection)
            + self.random_bias
        )
        return torch.cat((raw_features, nonlinear), dim=1)

    def forward_energy(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        features = self.atom_features(positions_bohr, atomic_numbers, topology)
        element_index = self.backbone._element_index(atomic_numbers)
        coefficients = self.readout[element_index]
        return self.output_scale * torch.sum(coefficients * features)

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
        return [self.readout]
