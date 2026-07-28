"""Optional MACE scalar-energy adapter for strict curvature-capacity audits."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as functional

from mldft.ml.models.components.local_body_order_residual import BodyOrderTopology


class LocalMACEScalarResidual(torch.nn.Module):
    """Reference-anchored residual owned by a vendored MACE scalar energy."""

    def __init__(
        self,
        *,
        elements: tuple[int, ...] = (1, 6, 7, 8, 9),
        hidden_channels: int = 32,
        mlp_channels: int = 16,
        max_ell: int = 2,
        correlation: int = 3,
        num_interactions: int = 2,
        num_bessel: int = 8,
        cutoff_polynomial_order: int = 5,
        cutoff_bohr: float = 20.0,
        avg_num_neighbors: float = 16.0,
        output_scale: float = 1.0,
        seed: int = 20260801,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or mlp_channels <= 0:
            raise ValueError("MACE channel counts must be positive")
        if max_ell < 1 or correlation < 1 or num_interactions < 1:
            raise ValueError("MACE body-order settings must be positive")
        if num_bessel <= 1 or cutoff_polynomial_order <= 0 or cutoff_bohr <= 0.5:
            raise ValueError("invalid MACE radial settings")
        if output_scale <= 0.0:
            raise ValueError("output_scale must be positive")
        self.elements = tuple(sorted(int(value) for value in elements))
        self.output_scale = float(output_scale)
        self.hidden_channels = int(hidden_channels)
        self.max_ell = int(max_ell)
        self.num_interactions = int(num_interactions)
        self.register_buffer("element_lookup", self._build_element_lookup(self.elements))
        self.network = self._build_mace(
            hidden_channels=hidden_channels,
            mlp_channels=mlp_channels,
            max_ell=max_ell,
            correlation=correlation,
            num_interactions=num_interactions,
            num_bessel=num_bessel,
            cutoff_polynomial_order=cutoff_polynomial_order,
            cutoff_bohr=cutoff_bohr,
            avg_num_neighbors=avg_num_neighbors,
            seed=seed,
        ).to(dtype=torch.float64)
        self.initial_network = copy.deepcopy(self.network)
        self.initial_network.requires_grad_(False)

    def _build_mace(
        self,
        *,
        hidden_channels: int,
        mlp_channels: int,
        max_ell: int,
        correlation: int,
        num_interactions: int,
        num_bessel: int,
        cutoff_polynomial_order: int,
        cutoff_bohr: float,
        avg_num_neighbors: float,
        seed: int,
    ) -> torch.nn.Module:
        try:
            from e3nn import o3
            from mace.modules import (
                MACE,
                RealAgnosticInteractionBlock,
                RealAgnosticResidualInteractionBlock,
            )
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "LocalMACEScalarResidual requires the isolated mace-torch vendor path"
            ) from error
        hidden_irreps = o3.Irreps(
            " + ".join(
                f"{hidden_channels}x{ell}{'e' if ell % 2 == 0 else 'o'}"
                for ell in range(max_ell + 1)
            )
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return MACE(
                r_max=cutoff_bohr,
                num_bessel=num_bessel,
                num_polynomial_cutoff=cutoff_polynomial_order,
                max_ell=max_ell,
                interaction_cls=RealAgnosticResidualInteractionBlock,
                interaction_cls_first=RealAgnosticInteractionBlock,
                num_interactions=num_interactions,
                num_elements=len(self.elements),
                hidden_irreps=hidden_irreps,
                MLP_irreps=o3.Irreps(f"{mlp_channels}x0e"),
                atomic_energies=np.zeros(len(self.elements), dtype=np.float64),
                avg_num_neighbors=avg_num_neighbors,
                atomic_numbers=list(self.elements),
                correlation=correlation,
                gate=functional.silu,
                pair_repulsion=False,
                radial_MLP=[64, 64, 64],
                heads=["Default"],
            )

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

    def _graph(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> dict[str, torch.Tensor]:
        device = positions_bohr.device
        first = topology.pair_first.to(device=device)
        second = topology.pair_second.to(device=device)
        edge_index = torch.stack(
            (torch.cat((first, second)), torch.cat((second, first))), dim=0
        )
        element_index = self._element_index(atomic_numbers)
        node_attrs = functional.one_hot(
            element_index, num_classes=len(self.elements)
        ).to(dtype=positions_bohr.dtype)
        edge_count = edge_index.shape[1]
        atom_count = positions_bohr.shape[0]
        return {
            "positions": positions_bohr,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": torch.zeros(
                edge_count, 3, dtype=positions_bohr.dtype, device=device
            ),
            "unit_shifts": torch.zeros(
                edge_count, 3, dtype=positions_bohr.dtype, device=device
            ),
            "cell": torch.zeros(1, 3, 3, dtype=positions_bohr.dtype, device=device),
            "batch": torch.zeros(atom_count, dtype=torch.long, device=device),
            "ptr": torch.tensor([0, atom_count], dtype=torch.long, device=device),
        }

    def _network_energy(
        self,
        network: torch.nn.Module,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        output = network(
            self._graph(positions_bohr, atomic_numbers, topology),
            training=True,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_hessian=False,
        )
        return output["energy"].sum()

    def forward_node_scalar_features(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        """Return the l=0 channels from every MACE interaction layer."""
        node_features = self._forward_node_features(
            positions_bohr, atomic_numbers, topology
        )
        full_block_width = self._full_irreps_block_width()
        expected = self._expected_node_feature_width()
        if node_features.shape[1] != expected:
            raise RuntimeError(
                f"unexpected MACE node feature width: {node_features.shape[1]} != {expected}"
            )
        scalar_blocks = [
            node_features[
                :, interaction * full_block_width : interaction * full_block_width
                + self.hidden_channels
            ]
            for interaction in range(self.num_interactions - 1)
        ]
        final_start = (self.num_interactions - 1) * full_block_width
        scalar_blocks.append(
            node_features[:, final_start : final_start + self.hidden_channels]
        )
        return torch.cat(scalar_blocks, dim=1)

    @property
    def invariant_feature_width(self) -> int:
        cross_channels = self.hidden_channels * (self.hidden_channels + 1) // 2
        return (
            self.num_interactions * self.hidden_channels
            + (self.num_interactions - 1) * self.max_ell * cross_channels
        )

    def forward_node_invariant_features(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        """Return scalar channels plus invariant Gram features of l>0 channels."""
        node_features = self._forward_node_features(
            positions_bohr, atomic_numbers, topology
        )
        full_block_width = self._full_irreps_block_width()
        expected = self._expected_node_feature_width()
        if node_features.shape[1] != expected:
            raise RuntimeError(
                f"unexpected MACE node feature width: {node_features.shape[1]} != {expected}"
            )
        upper = torch.triu_indices(
            self.hidden_channels,
            self.hidden_channels,
            device=node_features.device,
        )
        invariants = []
        for interaction in range(self.num_interactions - 1):
            cursor = interaction * full_block_width
            invariants.append(
                node_features[:, cursor : cursor + self.hidden_channels]
            )
            cursor += self.hidden_channels
            for ell in range(1, self.max_ell + 1):
                irrep_dimension = 2 * ell + 1
                width = self.hidden_channels * irrep_dimension
                equivariant = node_features[:, cursor : cursor + width].reshape(
                    node_features.shape[0], self.hidden_channels, irrep_dimension
                )
                gram = torch.einsum(
                    "ncm,ndm->ncd", equivariant, equivariant
                ) / math.sqrt(float(irrep_dimension))
                invariants.append(gram[:, upper[0], upper[1]])
                cursor += width
        final_start = (self.num_interactions - 1) * full_block_width
        invariants.append(
            node_features[:, final_start : final_start + self.hidden_channels]
        )
        result = torch.cat(invariants, dim=1)
        if result.shape[1] != self.invariant_feature_width:
            raise RuntimeError("MACE invariant feature width mismatch")
        return result

    def _forward_node_features(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        output = self.network(
            self._graph(positions_bohr, atomic_numbers, topology),
            training=True,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_hessian=False,
        )
        return output["node_feats"]

    def _full_irreps_block_width(self) -> int:
        return self.hidden_channels * sum(
            2 * ell + 1 for ell in range(self.max_ell + 1)
        )

    def _expected_node_feature_width(self) -> int:
        full_block_width = self._full_irreps_block_width()
        # MACE's default keep_last_layer_irreps=False retains the full irreps
        # for intermediate products and only scalars for the final product.
        return (
            (self.num_interactions - 1) * full_block_width
            + self.hidden_channels
        )

    def forward_energy(
        self,
        positions_bohr: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topology: BodyOrderTopology,
    ) -> torch.Tensor:
        return self.output_scale * (
            self._network_energy(
                self.network, positions_bohr, atomic_numbers, topology
            )
            - self._network_energy(
                self.initial_network, positions_bohr, atomic_numbers, topology
            )
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
