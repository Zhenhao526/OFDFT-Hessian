"""Module containing the loss functions for the ML-DFT model."""
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.sample_weighers import SampleWeigher


def project_gradient(gradient: torch.Tensor, batch_or_sample: OFData) -> torch.Tensor:
    r"""Calculates the projected gradient.

    Note that when using non-orthogonal transformations we need the dual vector of :math:`w`, written as :math:`w^*`,
    since it will transform different to :math:`w^T`. For orthogonal transformations we have :math:`w^* = w^T`.
    We get rid of the matrix multiplications by rewriting the equation as follows:

    .. math::

        \left( \boldsymbol{I} - \frac{\boldsymbol{w}\boldsymbol{w}^T}{\boldsymbol{w}^T \boldsymbol{w}} \right)
        \nabla_\boldsymbol{p} T =
        \nabla_\boldsymbol{p} T - \boldsymbol{w}
        \frac{\boldsymbol{w}^T \nabla_\boldsymbol{p} T}
        {{\boldsymbol{w}^T \boldsymbol{w}}}

    Args:
        gradient (torch.Tensor): The predicted gradients
        batch_or_sample (OFData): The OFData object (can be a batch or a single sample) containing the basis integrals.

    Returns:
        torch.Tensor: The projected gradient
    """
    if isinstance(batch_or_sample, Batch):
        coeffs_batch = batch_or_sample.coeffs_batch
    else:
        coeffs_batch = None

    w_p = global_add_pool(batch_or_sample.dual_basis_integrals * gradient, coeffs_batch)
    w_w = global_add_pool(
        batch_or_sample.dual_basis_integrals * batch_or_sample.dual_basis_integrals, coeffs_batch
    )
    factor = w_p / w_w

    if isinstance(batch_or_sample, Batch):
        projected_gradient = (
            gradient
            - batch_or_sample.dual_basis_integrals
            * torch.repeat_interleave(factor, torch.bincount(batch_or_sample.coeffs_batch))
        )
    else:
        projected_gradient = gradient - batch_or_sample.dual_basis_integrals * factor

    return projected_gradient


def project_gradient_difference(pred_gradients: torch.Tensor, batch: OFData) -> torch.Tensor:
    r"""Calculates the projected gradient difference and the absolute projected gradient error.

    We get rid of the matrix multiplications by rewriting the equation as follows:

    .. math::

        \left( \boldsymbol{I} - \frac{\boldsymbol{w}\boldsymbol{w}^T}{\boldsymbol{w}^T \boldsymbol{w}} \right)
        \left(\nabla_\boldsymbol{p} T_{\mathrm{pred}} - \nabla_\boldsymbol{p} T \right) =
        \left(\nabla_\boldsymbol{p} T_{\mathrm{pred}}-\nabla_\boldsymbol{p} T\right) - \boldsymbol{w}
        \frac{\boldsymbol{w}^T \left(\nabla_\boldsymbol{p} T_{\text{pred}}-\nabla_\boldsymbol{p} T\right)}
        {{\boldsymbol{w}^T \boldsymbol{w}}}

    Args:
        pred_gradients (torch.Tensor): The predicted gradients
        batch (Batch): The batch object containing the target gradients and the basis integrals

    Returns:
        Unreduced tensor of projected differences
    """
    diff_gradients_batch = pred_gradients - batch.gradient_label
    return project_gradient(diff_gradients_batch, batch)


class SingleLossFunction(nn.Module):
    """Base class for loss functions that compute a single loss value."""

    def __init__(
        self,
        loss_function: nn.Module = None,
        sample_weigher: SampleWeigher = None,
        reduction: str = "mean",
    ):
        """Initialize the LossFunction by setting the loss function, weighing and reduction.

        Args:
            loss_function (nn.Module, optional): Loss function to be used. Defaults to nn.L1Loss().
                **Important** The loss function should not apply any reduction, as this is handled by the LossFunction
                after weighting the loss.
            sample_weigher (SampleWeigher, optional): Sample weigher to be used. Defaults to None.
            reduction (str, optional): Reduction type to be used. Defaults to "mean".
        """
        super().__init__()
        if loss_function is None:
            loss_function = nn.L1Loss(reduction="none")
        self.loss_function = loss_function
        self.sample_weigher = sample_weigher
        self.reduction = reduction

    def forward(self, batch: OFData, **kwargs) -> Tensor:
        """Get the per-sample losses and apply the sample weights, as defined by the
        sample_weigher.

        Args:
            batch (OFData): The batch object, used in the loss calculation and for the sample weights
            **kwargs: Additional arguments to be passed to the loss function

        Returns:
            Tensor: The scalar loss
        """
        loss = self.get_loss(batch, **kwargs)
        if self.sample_weigher is None:
            weighted_loss = loss
        else:
            weighted_loss = self.weigh_loss(batch, loss)
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Function that applies weights to the loss.

        Has to be implemented by subclasses.
        """
        raise NotImplementedError

    def get_loss(self, batch: OFData, **kwargs) -> Tensor:
        """Function that computes the loss.

        Has to be implemented by subclasses.
        """
        raise NotImplementedError


class PerSampleWeightedPerSampleLossFunction(SingleLossFunction):
    """Base class for loss functions that compute a loss value per sample."""

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Applies weights to the loss for each sample."""
        weights = self.sample_weigher.get_weights(batch)
        assert loss.shape == weights.shape, f"{loss.shape} != {weights.shape}"
        return loss * weights


class PerSampleWeightedPerCoeffLossFunction(SingleLossFunction):
    """Base class for loss functions that compute a loss value per basis function."""

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Applies weights to the loss for each basis function."""
        weights = self.sample_weigher.get_weights(batch)
        weights = weights[batch.coeffs_batch]
        assert loss.shape == weights.shape, f"{loss.shape} != {weights.shape}"
        return (loss * weights).mean()


class EnergyLoss(PerSampleWeightedPerSampleLossFunction):
    """Calculates the loss between the predicted energy and the target energy."""

    def get_loss(self, batch: OFData, pred_energy: Tensor, **_) -> Tensor:
        """Computes the loss between the predicted energy and target energy.

        Args:
            batch (Batch): Batch object containing the target energy
            pred_energy (Tensor): Tensor containing the predicted energy

        Returns:
            Tensor: the loss for each sample in the batch
        """
        return self.loss_function(pred_energy, batch.energy_label)


class EnergyGradientLoss(PerSampleWeightedPerCoeffLossFunction):
    """Calculates the projected loss between the predicted gradients and the target gradients."""

    def get_loss(self, batch: Batch, projected_gradient_difference: Tensor, **_) -> Tensor:
        """Computes the loss to the projected difference of the predicted and ground truth
        gradient. Since the input is already the difference to the ground truth, the label is set
        to zero in the loss function call.

        Args:
            batch (Batch): Batch object containing the target energy
            projected_gradient_difference (Tensor): the difference of the predicted and real gradient, after projection

        Returns:
            Tensor: loss function of the input
        """
        return self.loss_function(
            projected_gradient_difference, torch.zeros_like(projected_gradient_difference)
        )


class CoefficientLoss(PerSampleWeightedPerCoeffLossFunction):
    """Calculates the loss between the predicted difference of coefficients and the target
    difference of coefficients."""

    def get_loss(self, batch: Batch, pred_diff: Tensor, **_) -> Tensor:
        """Returns the loss between the predicted difference of coefficients and the target
        difference of coefficients from the batch object.

        Args:
            batch (Batch): Batch object containing the target difference of coefficients
            pred_diff (Tensor): the predicted difference of the ground state coefficients and the current coefficients

        Returns:
            Tensor: loss between the predicted difference of coefficients and the target difference of coefficients,
                per basis function
        """
        return self.loss_function(pred_diff, batch.coeffs - batch.ground_state_coeffs)


class ForceLoss(SingleLossFunction):
    """Calculates the loss between autograd forces and reference nuclear forces."""

    def get_loss(self, batch: Batch, pred_forces: Tensor | None, **_) -> Tensor:
        """Return one force loss value per atom."""
        if pred_forces is None:
            raise ValueError("pred_forces is required for ForceLoss.")
        if not hasattr(batch, "force_label"):
            raise AttributeError("ForceLoss requires batch.force_label.")
        loss = self.loss_function(pred_forces, batch.force_label)
        if loss.ndim == 1:
            return loss
        return loss.flatten(start_dim=1).mean(dim=1)

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Apply optional per-molecule sample weights to per-atom force losses."""
        weights = self.sample_weigher.get_weights(batch)
        atom_batch = getattr(batch, "atomic_numbers_batch", None)
        if atom_batch is None:
            atom_batch = getattr(batch, "batch", None)
        if atom_batch is None:
            raise AttributeError(
                "ForceLoss with a sample_weigher requires atom-to-molecule batch indices."
            )
        assert loss.shape == atom_batch.shape, f"{loss.shape} != {atom_batch.shape}"
        return loss * weights[atom_batch]


class DirectionalHVPLoss(SingleLossFunction):
    """Normalized direct Hessian-vector-product loss for labelled graphs.

    ``pred_hvp`` and ``batch.hvp_label`` are energy-Hessian products in Cartesian
    coordinates. The per-graph component error is averaged over ``3N`` components, divided by
    the sampled direction norm and by a reference RMS scale. This makes the loss invariant to
    direction rescaling and prevents large molecules from dominating solely through atom count.
    """

    def __init__(
        self,
        reference_scale_floor: float = 1e-8,
        max_normalized_loss: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.sample_weigher is not None:
            raise ValueError("DirectionalHVPLoss does not support a sample weigher")
        if reference_scale_floor <= 0:
            raise ValueError("reference_scale_floor must be positive")
        if max_normalized_loss is not None and max_normalized_loss <= 0:
            raise ValueError("max_normalized_loss must be positive")
        self.reference_scale_floor = float(reference_scale_floor)
        self.max_normalized_loss = (
            None if max_normalized_loss is None else float(max_normalized_loss)
        )

    def get_loss(
        self,
        batch: Batch,
        pred_hvp: Tensor | None,
        hvp_active_mask: Tensor | None = None,
        **_,
    ) -> Tensor:
        if pred_hvp is None:
            raise ValueError("pred_hvp is required for DirectionalHVPLoss")
        required = (
            "hvp_label",
            "hvp_label_mask",
            "hvp_direction_norm",
            "hvp_reference_scale",
        )
        missing = [key for key in required if getattr(batch, key, None) is None]
        if missing:
            raise AttributeError(f"DirectionalHVPLoss requires fields {missing}")
        atom_batch = getattr(batch, "atomic_numbers_batch", None)
        if atom_batch is None:
            atom_batch = getattr(batch, "batch", None)
        if atom_batch is None:
            raise AttributeError("DirectionalHVPLoss requires atom-to-graph indices")

        component_loss = self.loss_function(pred_hvp, batch.hvp_label)
        if component_loss.ndim == 2:
            component_loss = component_loss.mean(dim=1)
        elif component_loss.ndim != 1:
            component_loss = component_loss.reshape(component_loss.shape[0], -1).mean(dim=1)
        graph_count = int(batch.batch_size)
        per_graph_sum = torch.zeros(
            graph_count, dtype=component_loss.dtype, device=component_loss.device
        )
        per_graph_sum.scatter_add_(0, atom_batch, component_loss)
        atom_count = torch.bincount(atom_batch, minlength=graph_count).clamp_min(1)
        per_graph = per_graph_sum / atom_count
        direction_norm = batch.hvp_direction_norm.reshape(-1).to(per_graph).clamp_min(
            torch.finfo(per_graph.dtype).tiny
        )
        reference_scale = batch.hvp_reference_scale.reshape(-1).to(per_graph).clamp_min(
            self.reference_scale_floor
        )
        mask = (
            batch.hvp_label_mask if hvp_active_mask is None else hvp_active_mask
        ).reshape(-1).bool()
        if mask.numel() != graph_count:
            raise ValueError("hvp_label_mask must contain one value per graph")
        if not bool(mask.any()):
            return pred_hvp.sum().reshape(1) * 0.0
        normalized = (per_graph / (direction_norm * reference_scale))[mask]
        if self.max_normalized_loss is not None:
            cap = torch.as_tensor(
                self.max_normalized_loss,
                dtype=normalized.dtype,
                device=normalized.device,
            )
            normalized = cap * normalized / (cap + normalized)
        return normalized


class PairEnergySecantLoss(SingleLossFunction):
    """Match directional ``kin_plus_xc`` energy secants for exact geometry pairs.

    The loss uses scalar energy labels and predictions only. It therefore preserves an explicitly
    conservative learned-energy objective and does not reinterpret a total PBE force as the
    derivative of the learned ``kin_plus_xc`` contribution.
    """

    def __init__(self, strict_pairs: bool = True, **kwargs):
        super().__init__(**kwargs)
        if self.sample_weigher is not None:
            raise ValueError("PairEnergySecantLoss does not support a per-sample weigher")
        self.strict_pairs = bool(strict_pairs)

    @staticmethod
    def _graph_positions(batch: Batch, graph_index: int) -> Tensor:
        ptr = getattr(batch, "atomic_numbers_ptr", None)
        if ptr is None:
            ptr = getattr(batch, "ptr", None)
        if ptr is None:
            raise AttributeError("PairEnergySecantLoss requires graph atom pointers")
        start = int(ptr[graph_index])
        stop = int(ptr[graph_index + 1])
        return batch.pos[start:stop]

    def get_loss(self, batch: Batch, pred_energy: Tensor, **_) -> Tensor:
        required = (
            "source_molecule_id",
            "perturbation_pair_id",
            "perturbation_pair_sign",
            "paired_perturbations",
        )
        missing = [key for key in required if getattr(batch, key, None) is None]
        if missing:
            raise AttributeError(f"PairEnergySecantLoss requires metadata fields {missing}")

        pred = pred_energy.reshape(-1)
        labels = batch.energy_label.reshape(-1).to(pred)
        has_label = batch.has_energy_label.reshape(-1).bool()
        source = torch.as_tensor(batch.source_molecule_id).reshape(-1)
        pair_id = torch.as_tensor(batch.perturbation_pair_id).reshape(-1)
        sign = torch.as_tensor(batch.perturbation_pair_sign).reshape(-1)
        is_pair = torch.as_tensor(batch.paired_perturbations).reshape(-1).bool()
        if not all(value.numel() == pred.numel() for value in (source, pair_id, sign, is_pair)):
            raise ValueError("Pair metadata must contain one value per graph")

        groups: dict[tuple[int, int], dict[int, int]] = {}
        for index in range(pred.numel()):
            if not bool(is_pair[index]) or not bool(has_label[index]):
                continue
            key = (int(source[index]), int(pair_id[index]))
            pair_sign = int(sign[index])
            if pair_sign not in (-1, 1):
                continue
            if pair_sign in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate sign {pair_sign} for pair {key}")
            groups[key][pair_sign] = index

        losses = []
        incomplete = []
        for key, indices in groups.items():
            if set(indices) != {-1, 1}:
                incomplete.append(key)
                continue
            minus_index = indices[-1]
            plus_index = indices[1]
            plus_positions = self._graph_positions(batch, plus_index)
            minus_positions = self._graph_positions(batch, minus_index)
            if plus_positions.shape != minus_positions.shape:
                raise ValueError(f"Geometry shape mismatch for pair {key}")
            separation = torch.linalg.vector_norm(plus_positions - minus_positions)
            if not bool(separation > 0):
                raise ValueError(f"Zero geometry separation for pair {key}")
            predicted_secant = (pred[plus_index] - pred[minus_index]) / separation
            target_secant = (labels[plus_index] - labels[minus_index]) / separation
            losses.append(self.loss_function(predicted_secant, target_secant))

        if incomplete and self.strict_pairs:
            raise ValueError(f"Incomplete geometry pairs in batch: {incomplete[:5]}")
        if not losses:
            if self.strict_pairs:
                raise ValueError("No complete labelled geometry pair in batch")
            return pred.sum().reshape(1) * 0.0
        return torch.stack([loss.reshape(()) for loss in losses])


class PairRelaxedForceSecantLoss(SingleLossFunction):
    """Match PBE complete-total relaxed-force secants for exact geometry pairs.

    Both predicted endpoint forces are derivatives of the learned scalar energy. Reference
    endpoint forces are the complete PBE forces evaluated after the KS density has converged.
    This is a force-secant curvature surrogate; it is not an implicit density-relaxed OFDFT HVP.
    """

    def __init__(
        self,
        strict_pairs: bool = True,
        reference_scale_floor: float = 1e-2,
        max_normalized_loss: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.sample_weigher is not None:
            raise ValueError("PairRelaxedForceSecantLoss does not support a sample weigher")
        if reference_scale_floor <= 0:
            raise ValueError("reference_scale_floor must be positive")
        if max_normalized_loss is not None and max_normalized_loss <= 0:
            raise ValueError("max_normalized_loss must be positive")
        self.strict_pairs = bool(strict_pairs)
        self.reference_scale_floor = float(reference_scale_floor)
        self.max_normalized_loss = (
            None if max_normalized_loss is None else float(max_normalized_loss)
        )

    @staticmethod
    def _atom_slice(batch: Batch, graph_index: int) -> slice:
        ptr = getattr(batch, "atomic_numbers_ptr", None)
        if ptr is None:
            ptr = getattr(batch, "ptr", None)
        if ptr is None:
            raise AttributeError("PairRelaxedForceSecantLoss requires graph atom pointers")
        return slice(int(ptr[graph_index]), int(ptr[graph_index + 1]))

    def get_loss(self, batch: Batch, pred_forces: Tensor | None, **_) -> Tensor:
        if pred_forces is None:
            raise ValueError("pred_forces is required for PairRelaxedForceSecantLoss")
        if not hasattr(batch, "force_label"):
            raise AttributeError("PairRelaxedForceSecantLoss requires batch.force_label")
        required = (
            "source_molecule_id",
            "perturbation_pair_id",
            "perturbation_pair_sign",
            "paired_perturbations",
        )
        missing = [key for key in required if getattr(batch, key, None) is None]
        if missing:
            raise AttributeError(f"PairRelaxedForceSecantLoss requires fields {missing}")

        graph_count = int(batch.batch_size)
        source = torch.as_tensor(batch.source_molecule_id).reshape(-1)
        pair_id = torch.as_tensor(batch.perturbation_pair_id).reshape(-1)
        sign = torch.as_tensor(batch.perturbation_pair_sign).reshape(-1)
        is_pair = torch.as_tensor(batch.paired_perturbations).reshape(-1).bool()
        has_label = batch.has_energy_label.reshape(-1).bool()
        if not all(
            value.numel() == graph_count
            for value in (source, pair_id, sign, is_pair, has_label)
        ):
            raise ValueError("Pair metadata must contain one value per graph")

        groups: dict[tuple[int, int], dict[int, int]] = {}
        for index in range(graph_count):
            if not bool(is_pair[index]) or not bool(has_label[index]):
                continue
            pair_sign = int(sign[index])
            if pair_sign not in (-1, 1):
                continue
            key = (int(source[index]), int(pair_id[index]))
            if pair_sign in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate sign {pair_sign} for pair {key}")
            groups[key][pair_sign] = index

        losses = []
        incomplete = []
        for key, indices in groups.items():
            if set(indices) != {-1, 1}:
                incomplete.append(key)
                continue
            minus_index, plus_index = indices[-1], indices[1]
            minus_slice = self._atom_slice(batch, minus_index)
            plus_slice = self._atom_slice(batch, plus_index)
            minus_positions = batch.pos[minus_slice]
            plus_positions = batch.pos[plus_slice]
            if minus_positions.shape != plus_positions.shape:
                raise ValueError(f"Geometry shape mismatch for pair {key}")
            separation = torch.linalg.vector_norm(plus_positions - minus_positions)
            if not bool(separation > 0):
                raise ValueError(f"Zero geometry separation for pair {key}")
            predicted = (
                pred_forces[plus_slice] - pred_forces[minus_slice]
            ) / separation
            reference = (
                batch.force_label[plus_slice] - batch.force_label[minus_slice]
            ) / separation
            component_loss = self.loss_function(predicted, reference)
            component_mean = component_loss.reshape(-1).mean()
            reference_scale = torch.sqrt(torch.mean(reference.square())).clamp_min(
                self.reference_scale_floor
            )
            normalized = component_mean / reference_scale
            if self.max_normalized_loss is not None:
                cap = torch.as_tensor(
                    self.max_normalized_loss,
                    dtype=normalized.dtype,
                    device=normalized.device,
                )
                normalized = cap * normalized / (cap + normalized)
            losses.append(normalized)

        if incomplete and self.strict_pairs:
            raise ValueError(f"Incomplete geometry pairs in batch: {incomplete[:5]}")
        if not losses:
            if self.strict_pairs:
                raise ValueError("No complete labelled geometry pair in batch")
            return pred_forces.sum().reshape(1) * 0.0
        return torch.stack(losses)


class WeightedLoss(nn.Module):
    """Module used to combine multiple losses with different weights.

    The forward pass does not return a single scalar loss, but rather two dictionaries containing
    the weights and the losses for each individual component.
    """

    def __init__(self, **kwargs: Mapping[str, float | nn.Module]):
        """Initialize the WeightedLoss object by passing a mapping of str to loss function and
        weight.

        Args:
            **kwargs: Mapping of loss names to the corresponding loss functions and weights. The names will be used
                for logging purposes. The values should be dictionaries with the keys "loss" and "weight", the former
                containing the (nn.Module) loss function and the latter the (scalar) weight.
        """
        super().__init__()
        assert all("loss" in v and "weight" in v and len(v) == 2 for v in kwargs.values())
        self.loss_module_dict = nn.ModuleDict({k: v["loss"] for k, v in kwargs.items()})
        self.weight_dict = {k: v["weight"] for k, v in kwargs.items()}

    def forward(self, batch: OFData, **kwargs) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The weighted sum of the energy loss, the gradient loss and the coefficient loss.

        Loss = energy_weight * energy_loss + gradient_weight * gradient_loss
        + coefficient_weight * coefficient_loss.

        Args:
            batch (Batch): Batch object containing the target energy and target gradients
            **kwargs: Additional arguments to be passed to the loss functions

        Returns:
            dict[str, Tensor]: Dictionary containing the weights for each loss component
            dict[str, Tensor]: Dictionary containing the losses for each loss component
        """

        loss_dict = {}
        for key, loss in self.loss_module_dict.items():
            loss_dict[key] = loss(batch, **kwargs)

        return self.weight_dict, loss_dict


class FullLoss(nn.Module):
    """Previous version of the WeightedLoss module, which is no longer supported.

    Included to make loading old checkpoints possible.
    """

    def __init__(self):
        """Just raises an error."""
        raise NotImplementedError("FullLoss is no longer supported, use WeightedLoss instead.)")
