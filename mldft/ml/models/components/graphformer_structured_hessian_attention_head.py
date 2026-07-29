"""Structured Hessian attention readout for frozen Graphformer node states."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mldft.ml.models.components.structured_density_hessian_head import (
    StructuredDensityHessianHead,
)


class GraphformerStructuredHessianAttentionHead(nn.Module):
    """Read a covariant Hessian from invariant Graphformer atom states.

    A distance-biased multi-head attention layer first converts the final
    Graphformer atom states into compact, context-aware atom latents.  The
    existing structured block readout then maps those latents and geometry to a
    symmetric Cartesian Hessian in the molecular internal subspace.

    The Graphformer backbone is deliberately not owned by this module.  It can
    therefore be restored from an old checkpoint unchanged and evaluated under
    ``torch.no_grad()`` while this readout alone is trained.
    """

    def __init__(
        self,
        node_feature_dim: int = 768,
        *,
        attention_dim: int = 128,
        attention_heads: int = 8,
        latent_dim: int = 16,
        structured_hidden_dim: int = 64,
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
        environment_scale_bohr: float = 4.0,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or attention_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if attention_heads <= 0 or attention_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide attention_dim")
        if latent_dim <= 0 or radial_width_bohr <= 0.0:
            raise ValueError("invalid attention-head hyperparameters")
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
            centers.numel(), self.attention_heads, bias=True
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(2 * self.attention_dim, self.attention_dim),
            nn.SiLU(),
            nn.Linear(self.attention_dim, self.latent_dim),
            nn.Tanh(),
        )
        self.structured_readout = StructuredDensityHessianHead(
            density_feature_dim=self.latent_dim,
            hidden_dim=structured_hidden_dim,
            radial_centers_bohr=radial_centers_bohr,
            radial_width_bohr=radial_width_bohr,
            environment_scale_bohr=environment_scale_bohr,
            use_density=True,
            zero_initialize=zero_initialize,
        )

    def attention_latents(
        self, node_features: Tensor, positions_bohr: Tensor
    ) -> Tensor:
        """Return permutation-equivariant, rotation-invariant atom latents."""

        features = torch.as_tensor(node_features)
        positions = torch.as_tensor(
            positions_bohr, dtype=features.dtype, device=features.device
        )
        atom_count = int(features.shape[0])
        if features.shape != (atom_count, self.node_feature_dim):
            raise ValueError("Graphformer node feature shape mismatch")
        if positions.shape != (atom_count, 3):
            raise ValueError("position shape mismatch")

        normalized = self.input_norm(features)
        query, key, value = self.qkv_projection(normalized).chunk(3, dim=-1)
        query = query.reshape(
            atom_count, self.attention_heads, self.channels_per_head
        )
        key = key.reshape(
            atom_count, self.attention_heads, self.channels_per_head
        )
        value = value.reshape(
            atom_count, self.attention_heads, self.channels_per_head
        )

        displacement = positions[:, None, :] - positions[None, :, :]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        centers = self.radial_centers_bohr.to(
            dtype=features.dtype, device=features.device
        )
        radial = torch.exp(
            -0.5
            * (
                (distances[:, :, None] - centers[None, None, :])
                / self.radial_width_bohr
            )
            ** 2
        )
        logits = torch.einsum("ihd,jhd->ijh", query, key)
        logits = logits / math.sqrt(self.channels_per_head)
        logits = logits + self.distance_bias(radial)
        if atom_count > 1:
            diagonal = torch.eye(
                atom_count, dtype=torch.bool, device=features.device
            )
            logits = logits.masked_fill(diagonal[:, :, None], -torch.inf)
            attention = torch.softmax(logits, dim=1)
        else:
            attention = torch.ones_like(logits)
        context = torch.einsum("ijh,jhd->ihd", attention, value).reshape(
            atom_count, self.attention_dim
        )
        skip = self.skip_projection(normalized)
        return self.latent_mlp(torch.cat((context, skip), dim=-1))

    def forward(
        self,
        atomic_numbers: Tensor,
        positions_bohr: Tensor,
        node_features: Tensor,
        internal_projector: Tensor,
    ) -> Tensor:
        latents = self.attention_latents(node_features, positions_bohr)
        return self.structured_readout(
            atomic_numbers,
            positions_bohr,
            latents,
            internal_projector,
        )
