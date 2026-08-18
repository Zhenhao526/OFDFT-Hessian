"""Equivariant force attention readout for frozen Graphformer atom states."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class GraphformerStructuredForceAttentionHead(nn.Module):
    """Predict molecular forces with exact rigid-motion constraints.

    Final invariant Graphformer atom states are contextualized by a
    distance-biased multi-head attention layer. Symmetric atom-pair features
    produce one scalar coefficient per undirected pair. Multiplying by the
    pair direction and adding equal-and-opposite contributions gives a
    rotation-covariant force field with zero net force and zero net torque.
    """

    def __init__(
        self,
        node_feature_dim: int = 768,
        *,
        attention_dim: int = 128,
        attention_heads: int = 8,
        latent_dim: int = 32,
        pair_hidden_dim: int = 64,
        radial_centers_bohr: Sequence[float] = (
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
        ),
        radial_width_bohr: float = 0.75,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or attention_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if attention_heads <= 0 or attention_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide attention_dim")
        if latent_dim <= 0 or pair_hidden_dim <= 0:
            raise ValueError("readout dimensions must be positive")
        if radial_width_bohr <= 0.0:
            raise ValueError("radial width must be positive")
        centers = torch.as_tensor(tuple(radial_centers_bohr), dtype=torch.float64)
        if centers.ndim != 1 or centers.numel() == 0:
            raise ValueError("radial centers must be a nonempty sequence")

        self.node_feature_dim = int(node_feature_dim)
        self.attention_dim = int(attention_dim)
        self.attention_heads = int(attention_heads)
        self.channels_per_head = self.attention_dim // self.attention_heads
        self.latent_dim = int(latent_dim)
        self.radial_width_bohr = float(radial_width_bohr)
        self.register_buffer("radial_centers_bohr", centers)

        self.input_norm = nn.LayerNorm(self.node_feature_dim)
        self.qkv_projection = nn.Linear(
            self.node_feature_dim, 3 * self.attention_dim
        )
        self.skip_projection = nn.Linear(
            self.node_feature_dim, self.attention_dim
        )
        self.distance_bias = nn.Linear(
            centers.numel(), self.attention_heads
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(2 * self.attention_dim, self.attention_dim),
            nn.SiLU(),
            nn.Linear(self.attention_dim, self.latent_dim),
            nn.Tanh(),
        )
        pair_input_dim = 3 * self.latent_dim + centers.numel()
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, 1),
        )
        if zero_initialize:
            nn.init.zeros_(self.pair_mlp[-1].weight)
            nn.init.zeros_(self.pair_mlp[-1].bias)

    def _radial(self, distances: Tensor) -> Tensor:
        centers = self.radial_centers_bohr.to(
            dtype=distances.dtype, device=distances.device
        )
        return torch.exp(
            -0.5
            * (
                (distances[..., None] - centers)
                / self.radial_width_bohr
            )
            ** 2
        )

    def attention_latents(
        self, node_features: Tensor, positions_bohr: Tensor
    ) -> Tensor:
        return self.attention_latents_batch(
            torch.as_tensor(node_features)[None, ...],
            torch.as_tensor(positions_bohr)[None, ...],
        )[0]

    def attention_latents_batch(
        self, node_features: Tensor, positions_bohr: Tensor
    ) -> Tensor:
        features = torch.as_tensor(node_features)
        positions = torch.as_tensor(
            positions_bohr, dtype=features.dtype, device=features.device
        )
        if features.ndim != 3:
            raise ValueError("batched Graphformer features must be rank 3")
        batch_size, atom_count, feature_dim = features.shape
        if feature_dim != self.node_feature_dim:
            raise ValueError("Graphformer node feature shape mismatch")
        if positions.shape != (batch_size, atom_count, 3):
            raise ValueError("position shape mismatch")

        normalized = self.input_norm(features)
        query, key, value = self.qkv_projection(normalized).chunk(3, dim=-1)
        query = query.reshape(
            batch_size,
            atom_count,
            self.attention_heads,
            self.channels_per_head,
        )
        key = key.reshape(
            batch_size,
            atom_count,
            self.attention_heads,
            self.channels_per_head,
        )
        value = value.reshape(
            batch_size,
            atom_count,
            self.attention_heads,
            self.channels_per_head,
        )
        distances = torch.linalg.vector_norm(
            positions[:, :, None, :] - positions[:, None, :, :], dim=-1
        )
        logits = torch.einsum("bihd,bjhd->bijh", query, key)
        logits = logits / math.sqrt(self.channels_per_head)
        logits = logits + self.distance_bias(self._radial(distances))
        if atom_count > 1:
            diagonal = torch.eye(
                atom_count, dtype=torch.bool, device=features.device
            )
            logits = logits.masked_fill(
                diagonal[None, :, :, None], -torch.inf
            )
            attention = torch.softmax(logits, dim=2)
        else:
            attention = torch.ones_like(logits)
        context = torch.einsum(
            "bijh,bjhd->bihd", attention, value
        ).reshape(
            batch_size, atom_count, self.attention_dim
        )
        skip = self.skip_projection(normalized)
        return self.latent_mlp(torch.cat((context, skip), dim=-1))

    def forward(
        self, node_features: Tensor, positions_bohr: Tensor
    ) -> Tensor:
        return self.forward_batch(
            torch.as_tensor(node_features)[None, ...],
            torch.as_tensor(positions_bohr)[None, ...],
        )[0]

    def forward_batch(
        self, node_features: Tensor, positions_bohr: Tensor
    ) -> Tensor:
        features = torch.as_tensor(node_features)
        positions = torch.as_tensor(
            positions_bohr, dtype=features.dtype, device=features.device
        )
        if features.ndim != 3:
            raise ValueError("batched Graphformer features must be rank 3")
        batch_size, atom_count, _ = features.shape
        if positions.shape != (batch_size, atom_count, 3):
            raise ValueError("batched position shape mismatch")
        if atom_count < 2:
            return positions.new_zeros((batch_size, atom_count, 3))
        latents = self.attention_latents_batch(features, positions)
        first, second = torch.triu_indices(
            atom_count, atom_count, offset=1, device=positions.device
        )
        vectors = positions[:, second] - positions[:, first]
        distances = torch.linalg.vector_norm(vectors, dim=2).clamp_min(
            torch.finfo(positions.dtype).eps
        )
        directions = vectors / distances[:, :, None]
        latent_first = latents[:, first]
        latent_second = latents[:, second]
        pair_features = torch.cat(
            (
                latent_first + latent_second,
                torch.abs(latent_first - latent_second),
                latent_first * latent_second,
                self._radial(distances),
            ),
            dim=-1,
        )
        coefficients = self.pair_mlp(pair_features)
        pair_forces = coefficients * directions
        forces = positions.new_zeros((batch_size, atom_count, 3))
        forces = forces.index_add(1, first, pair_forces)
        forces = forces.index_add(1, second, -pair_forces)
        return forces
