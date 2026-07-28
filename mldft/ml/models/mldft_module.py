"""Lightning module for the neural network."""

import itertools
from contextlib import nullcontext
from time import perf_counter
from typing import Any, Dict, Tuple

import torch
import torch_geometric
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.data import Batch
from torchmetrics import MeanMetric, MinMetric

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.components.sample_weighers import GroundStateOnlySampleWeigher
from mldft.ml.models.components.training_metrics import (
    MAEEnergy,
    MAEGradient,
    MAEInitialGuess,
)
from mldft.utils import RankedLogger
from mldft.utils.log_utils.logging_mixin import locate_logging_mixins


def get_layers(model: torch.nn.Module):
    children = list(model.children())
    return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]


class MLDFTLitModule(LightningModule):
    """The MLDFTLitModule class is a LightningModule that is used to wrap the neural network in the
    pytorch lightning framework."""

    def __init__(
        self,
        net: torch.nn.Module,
        basis_info: BasisInfo,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        target_key: str,
        loss_function: torch.nn.Module,
        compile: bool,
        validation_loss_function: torch.nn.Module | None = None,
        variational: bool = True,
        force_supervision: bool = False,
        metric_interval: int = 10,
        logging_mixin_interval: int | None = None,
        show_logging_mixins_in_progress_bar: bool = False,
        profile_timing: bool = False,
        hvp_batch_fraction: float = 1.0,
        loss_gradient_norm_interval: int | None = None,
        loss_gradient_norm_max_logs: int = 5,
        hvp_start_step: int = 0,
        hvp_ramp_steps: int = 0,
        hvp_update_interval: int = 1,
        hvp_update_phase: int = 0,
        loss_balance_mode: str = "none",
        hvp_gradnorm_target_ratio: float = 0.25,
        gradnorm_multiplier_min: float = 0.05,
        gradnorm_multiplier_max: float = 5.0,
    ) -> None:
        """Initializes the MLDFTLitModule object.

        Args:
            net (torch.nn.Module): the neural network
            basis_info (BasisInfo): the basis info object used for the dataset. Added here for logging purposes.
            optimizer (torch.optim.Optimizer): the optimizer to use for training
            scheduler (torch.optim.lr_scheduler): the learning rate scheduler to use for training
            loss_function (torch.nn.Module): the loss function to use for training
            validation_loss_function (torch.nn.Module | None): optional validation/test loss. If
                None, the training loss is reused.
            target_key (str): the name of the target key. Added here to easily determine which
                targets were used for training afterward.
            compile (bool): whether to compile the model with :func:`torch.compile`
            variational (bool): whether the model is variational or not. If True, the model is assumed to predict
                two outputs, the energy and the coefficient difference to the ground state (for proj minao).
                If False, the model is assumed to predict the gradient directly, in a non-variational manner, so
                three outputs are expected: The energy, the gradient and the coefficient difference.
            force_supervision (bool): whether to additionally compute nuclear forces from the scalar
                predicted energy as ``-dE/dR`` and pass them to the configured loss function.
            metric_interval (int): the interval (in steps) at which the metrics are calculated and logged.
            logging_mixin_interval (int | None): the interval (in steps) at which the logging mixins are called.
                Defaults to None, which means that the logging mixins are not called.
            show_logging_mixins_in_progress_bar (bool): whether to show the values logged using the logging mixins in
                the progress bar. Defaults to False.
            profile_timing (bool): whether to collect coarse forward/autograd timing for throughput calibration.
            hvp_batch_fraction: Maximum deterministic fraction of training batches on which
                direct HVP autograd is enabled. A batch without an HVP-labelled graph remains
                inactive, so the realized fraction can be lower.
            loss_gradient_norm_interval: Optional step interval for measuring each weighted loss
                component's parameter-gradient norm. This is intentionally sparse because it
                requires one extra reverse pass per component.
            loss_gradient_norm_max_logs: Maximum number of gradient-scale snapshots per run.
            hvp_start_step: Optimizer step before which HVP loss is disabled.
            hvp_ramp_steps: Linear HVP-weight ramp length after ``hvp_start_step``.
            hvp_update_interval: Apply HVP loss on one phase of this many optimizer steps.
            hvp_update_phase: Active phase in ``[0, hvp_update_interval)``.
            loss_balance_mode: ``none`` or sparse GradNorm-like HVP/force balancing.
            hvp_gradnorm_target_ratio: Target weighted HVP gradient norm divided by force norm.
            gradnorm_multiplier_min: Minimum adaptive HVP weight multiplier.
            gradnorm_multiplier_max: Maximum adaptive HVP weight multiplier.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # logger=false means that the hyperparameters will not be logged
        self.save_hyperparameters(logger=False, ignore=["validation_loss_function"])

        # save the neural network
        self.net = net
        self.target_key = target_key
        self.variational = variational
        self.force_supervision = force_supervision
        self._last_forward_timing: dict[str, float] = {}
        if not 0.0 < hvp_batch_fraction <= 1.0:
            raise ValueError("hvp_batch_fraction must be in (0, 1]")
        self.hvp_batch_fraction = float(hvp_batch_fraction)
        self.loss_gradient_norm_interval = loss_gradient_norm_interval
        self.loss_gradient_norm_max_logs = int(loss_gradient_norm_max_logs)
        self._loss_gradient_norm_logs = 0
        self._last_loss_gradient_norm_step: int | None = None
        if hvp_start_step < 0 or hvp_ramp_steps < 0:
            raise ValueError("HVP start/ramp steps must be non-negative")
        if hvp_update_interval <= 0:
            raise ValueError("hvp_update_interval must be positive")
        if hvp_update_phase < 0 or hvp_update_phase >= hvp_update_interval:
            raise ValueError("hvp_update_phase must be within the update interval")
        if loss_balance_mode not in {"none", "gradnorm"}:
            raise ValueError("loss_balance_mode must be 'none' or 'gradnorm'")
        if hvp_gradnorm_target_ratio <= 0:
            raise ValueError("hvp_gradnorm_target_ratio must be positive")
        if not 0 < gradnorm_multiplier_min <= gradnorm_multiplier_max:
            raise ValueError("Invalid GradNorm multiplier bounds")
        self.hvp_start_step = int(hvp_start_step)
        self.hvp_ramp_steps = int(hvp_ramp_steps)
        self.hvp_update_interval = int(hvp_update_interval)
        self.hvp_update_phase = int(hvp_update_phase)
        self.loss_balance_mode = loss_balance_mode
        self.hvp_gradnorm_target_ratio = float(hvp_gradnorm_target_ratio)
        self.gradnorm_multiplier_min = float(gradnorm_multiplier_min)
        self.gradnorm_multiplier_max = float(gradnorm_multiplier_max)
        self._hvp_gradnorm_multiplier = 1.0

        # define loss function
        self.loss_function = loss_function
        self.validation_loss_function = validation_loss_function
        self.console_logger = RankedLogger(__name__, rank_zero_only=True)

        # metric objects for calculating and averaging accuracy across batches

        # The metric objects have to be initialized in this way, otherwise Lighting doesn't recognize them
        # Dicts of the metrics to iterate through when logging
        def create_metric_module_dict(split: str):
            """Creates a dictionary of metrics for a given split."""
            return torch.nn.ModuleDict(
                {
                    f"{split}_metrics/mae_energy": MAEEnergy(mode="per molecule"),
                    f"{split}_metrics_ground_state/mae_energy_ground_state": MAEEnergy(
                        mode="per molecule", sample_weigher=GroundStateOnlySampleWeigher()
                    ),
                    f"{split}_metrics_per_electron/mae_energy_per_electron": MAEEnergy(
                        mode="per electron"
                    ),
                    f"{split}_metrics/mae_gradient": MAEGradient(mode="per molecule"),
                    f"{split}_metrics_ground_state/mae_gradient_ground_state": MAEGradient(
                        mode="per molecule", sample_weigher=GroundStateOnlySampleWeigher()
                    ),
                    f"{split}_metrics_per_electron/mae_gradient_per_electron": MAEGradient(
                        mode="per electron"
                    ),
                    f"{split}_metrics/mae_initial_guess": MAEInitialGuess(mode="per molecule"),
                    f"{split}_metrics_per_electron/mae_initial_guess_per_electron": MAEInitialGuess(
                        mode="per electron"
                    ),
                }
            )

        self.train_metrics = create_metric_module_dict("train")
        self.val_metrics = create_metric_module_dict("val")
        self.test_metrics = create_metric_module_dict("test")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for each validation metric, also track the best value so far
        self.val_metrics_best = torch.nn.ModuleDict(
            {
                f"val_metrics_best/{key.split('/')[-1]}": MinMetric()
                for key in self.val_metrics.keys()
            }
        )

        # logging mixins
        self.logging_mixin_interval = logging_mixin_interval
        self.use_logging_mixins = self.logging_mixin_interval is not None
        self.val_logging_mixin_interval = (
            self.logging_mixin_interval // 5 if self.use_logging_mixins else None
        )
        if self.use_logging_mixins:
            self.logging_mixin_dict = locate_logging_mixins(self.net)
        else:
            self.logging_mixin_dict = {}
        self.show_logging_mixins_in_progress_bar = show_logging_mixins_in_progress_bar

        self.basis_info = basis_info

    def _active_loss_weight(
        self, loss_key: str, loss_function: torch.nn.Module | None = None
    ) -> float:
        """Return the configured scalar weight for a loss component."""
        loss_function = loss_function or self.loss_function
        weight_dict = getattr(loss_function, "weight_dict", {})
        weight = weight_dict.get(loss_key, 0.0)
        try:
            return float(weight)
        except (TypeError, ValueError):
            return 1.0

    def _loss_component_active(
        self, loss_key: str, loss_function: torch.nn.Module | None = None
    ) -> bool:
        """Return whether a configured loss component should contribute gradients."""
        loss_function = loss_function or self.loss_function
        loss_modules = getattr(loss_function, "loss_module_dict", {})
        return loss_key in loss_modules and self._active_loss_weight(loss_key, loss_function) != 0.0

    def _any_non_gradient_loss_active(self) -> bool:
        """Return whether outputs other than density gradients are used by active losses."""
        loss_modules = getattr(self.loss_function, "loss_module_dict", {})
        return any(
            key != "gradient_loss" and self._active_loss_weight(key) != 0.0
            for key in loss_modules.keys()
        )

    def _any_non_force_loss_active(self) -> bool:
        """Return whether outputs other than forces are used by active losses."""
        loss_modules = getattr(self.loss_function, "loss_module_dict", {})
        return any(
            key != "force_loss" and self._active_loss_weight(key) != 0.0
            for key in loss_modules.keys()
        )

    def _compute_directional_hvp(
        self,
        batch: Batch,
        pred_forces: torch.Tensor | None,
        batch_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Differentiate scalar-derived forces once more along stored directions."""
        if not self._loss_component_active("hvp_loss"):
            return None, None
        if pred_forces is None:
            raise RuntimeError("HVP supervision requires scalar-derived forces")
        direction = getattr(batch, "hvp_direction", None)
        if direction is None:
            raise AttributeError("HVP supervision requires batch.hvp_direction")
        if direction.shape != pred_forces.shape:
            raise ValueError(
                f"HVP direction shape {direction.shape} != force shape {pred_forces.shape}"
            )
        labelled = batch.hvp_label_mask.reshape(-1).bool()
        optimizer_step = int(self.trainer.global_step)
        schedule_step = optimizer_step - self.hvp_start_step
        schedule_active = (
            schedule_step >= 0
            and schedule_step % self.hvp_update_interval == self.hvp_update_phase
        )
        period = max(1, round(1.0 / self.hvp_batch_fraction))
        selected = (
            schedule_active
            and (
                self.hvp_batch_fraction >= 1.0
                or (optimizer_step + int(batch_idx)) % period == 0
            )
        )
        active = labelled if selected else torch.zeros_like(labelled)
        self.log(
            "train_hvp/active_batch",
            active.any().to(dtype=pred_forces.dtype),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.batch_size,
        )
        self.log(
            "train_hvp/active_graph_fraction",
            active.to(dtype=pred_forces.dtype).mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.batch_size,
        )
        if not bool(active.any()):
            # Preserve a differentiable zero while avoiding second-order autograd.
            return pred_forces * 0.0, active
        # F=-dE/dR, therefore -d(F.v)/dR is the energy-Hessian product H.v.
        return (
            -torch.autograd.grad(
                torch.sum(pred_forces * direction),
                batch.pos,
                create_graph=self.net.training,
                retain_graph=True,
            )[0],
            active,
        )

    def _hvp_weight_scale(self) -> float:
        step = int(self.trainer.global_step) - self.hvp_start_step
        if step < 0:
            return 0.0
        if self.hvp_ramp_steps == 0:
            return 1.0
        return min(1.0, float(step + 1) / float(self.hvp_ramp_steps))

    def _effective_loss_weights(
        self,
        weight_dict: dict[str, float],
        hvp_active_mask: torch.Tensor | None,
    ) -> dict[str, float]:
        effective = dict(weight_dict)
        if "hvp_loss" in effective:
            active = hvp_active_mask is not None and bool(hvp_active_mask.any())
            effective["hvp_loss"] = float(effective["hvp_loss"]) * (
                self._hvp_weight_scale() * self._hvp_gradnorm_multiplier
                if active
                else 0.0
            )
        return effective

    def _maybe_log_loss_gradient_norms(
        self,
        batch: Batch,
        weight_dict: dict[str, float],
        loss_dict: dict[str, torch.Tensor],
        hvp_active_mask: torch.Tensor | None,
    ) -> None:
        interval = self.loss_gradient_norm_interval
        if interval is None or interval <= 0:
            return
        if self._loss_gradient_norm_logs >= self.loss_gradient_norm_max_logs:
            return
        if self.trainer.global_step % interval != 0:
            return
        if self._last_loss_gradient_norm_step == self.trainer.global_step:
            return
        if self._loss_component_active("hvp_loss") and (
            hvp_active_mask is None or not bool(hvp_active_mask.any())
        ):
            return
        parameters = [parameter for parameter in self.net.parameters() if parameter.requires_grad]
        component_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        component_norms: dict[str, torch.Tensor] = {}
        for key, component in loss_dict.items():
            if float(weight_dict[key]) == 0.0:
                continue
            weighted = weight_dict[key] * component
            gradients = torch.autograd.grad(
                weighted,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared = torch.zeros((), dtype=component.dtype, device=component.device)
            for gradient in gradients:
                if gradient is not None:
                    squared = squared + torch.sum(gradient.detach() ** 2)
            squared = self._distributed_mean(squared)
            norm = torch.sqrt(squared)
            component_gradients[key] = tuple(
                None if gradient is None else gradient.detach() for gradient in gradients
            )
            component_norms[key] = norm
            self.log(
                f"train_gradient_norm/{key}",
                norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        for left_key, right_key in itertools.combinations(component_gradients, 2):
            dot = torch.zeros((), dtype=batch.pos.dtype, device=batch.pos.device)
            for left, right in zip(
                component_gradients[left_key], component_gradients[right_key]
            ):
                if left is not None and right is not None:
                    dot = dot + torch.sum(left * right)
            dot = self._distributed_mean(dot)
            denominator = component_norms[left_key] * component_norms[right_key]
            cosine = dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny)
            self.log(
                f"train_gradient_cosine/{left_key}_vs_{right_key}",
                cosine,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if (
            self.loss_balance_mode == "gradnorm"
            and "hvp_loss" in component_norms
            and "force_loss" in component_norms
        ):
            ratio = float(
                component_norms["hvp_loss"]
                / component_norms["force_loss"].clamp_min(
                    torch.finfo(component_norms["force_loss"].dtype).tiny
                )
            )
            if ratio > 0.0 and bool(torch.isfinite(component_norms["hvp_loss"])):
                proposed = self._hvp_gradnorm_multiplier * (
                    self.hvp_gradnorm_target_ratio / ratio
                )
                self._hvp_gradnorm_multiplier = min(
                    self.gradnorm_multiplier_max,
                    max(self.gradnorm_multiplier_min, proposed),
                )
            self.log(
                "train_hvp/gradnorm_multiplier",
                self._hvp_gradnorm_multiplier,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self._loss_gradient_norm_logs += 1
        self._last_loss_gradient_norm_step = int(self.trainer.global_step)

    def _profile_timing_enabled(self) -> bool:
        """Return whether coarse training timing should be collected."""
        return bool(getattr(self.hparams, "profile_timing", False))

    @staticmethod
    def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
            result /= torch.distributed.get_world_size()
        return result

    def _eval_loss_function(self) -> torch.nn.Module:
        """Return the validation/test loss function."""
        return self.validation_loss_function or self.loss_function

    @staticmethod
    def _sync_cuda_for_timing(batch: Batch) -> None:
        """Synchronize the current CUDA stream when timing GPU work."""
        pos = getattr(batch, "pos", None)
        if pos is not None and getattr(pos, "is_cuda", False):
            torch.cuda.synchronize(pos.device)

    def _timed_section_start(self, batch: Batch) -> float:
        if self._profile_timing_enabled():
            self._sync_cuda_for_timing(batch)
        return perf_counter()

    def _timed_section_end(self, batch: Batch, start: float) -> float:
        if self._profile_timing_enabled():
            self._sync_cuda_for_timing(batch)
        return perf_counter() - start

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()
        for metric_best in self.val_metrics_best.values():
            metric_best.reset()

    def activate_logging_mixins(self) -> None:
        """Activates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.activate_logging()

    def deactivate_logging_mixins(self) -> None:
        """Deactivates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.deactivate_logging()

    def forward_predictions(
        self,
        batch: Batch,
        compute_density_gradients: bool = True,
        compute_forces: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Applies the forward pass of the model to the batch, and computes the energy gradients
        via backprop in the variational case (otherwise they are calculated directly).

        Args:
            batch (Batch): Batch object containing the data

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: A tuple
            containing predicted energy, predicted density gradients, predicted coefficient
            differences and optional predicted nuclear forces.
        """
        timing: dict[str, float] = {}
        profile_timing = self._profile_timing_enabled()
        if compute_forces is None:
            compute_forces = self.force_supervision
        grad_context = torch.enable_grad() if self.variational or compute_forces else nullcontext()
        with grad_context:
            if compute_forces:
                batch.pos.requires_grad_(True)
            if not self.variational:
                # model predicts energy, gradient and initial guess directly
                batch.coeffs.requires_grad_(True)  # for atomref image logging
                start = self._timed_section_start(batch)
                pred_energy, pred_gradients, pred_diff = self.net(batch)
                if profile_timing:
                    timing["net_forward_s"] = self._timed_section_end(batch, start)
            else:
                # set the flag for the gradients to be calculated
                batch.coeffs.requires_grad_(True)
                # calculate the forward pass
                start = self._timed_section_start(batch)
                pred_energy, pred_diff = self.net(batch)
                if profile_timing:
                    timing["net_forward_s"] = self._timed_section_end(batch, start)

                # calculate the gradients from the predicted energy with autograd
                # The density-gradient loss needs create_graph during training. This is separate
                # from the force path. EG runs never enable force-specific higher-order autograd.
                gradient_create_graph = (
                    compute_density_gradients
                    and self.net.training
                    and self._loss_component_active("gradient_loss")
                )
                gradient_retain_graph = compute_forces or (
                    self.net.training and self._any_non_gradient_loss_active()
                )
                if compute_density_gradients:
                    start = self._timed_section_start(batch)
                    pred_gradients = torch.autograd.grad(
                        pred_energy.sum(),
                        batch.coeffs,
                        create_graph=gradient_create_graph,
                        retain_graph=gradient_retain_graph,
                    )[0]
                    if profile_timing:
                        timing["density_gradient_autograd_s"] = self._timed_section_end(
                            batch, start
                        )
                else:
                    pred_gradients = None

            pred_forces = None
            if compute_forces:
                force_create_graph = self.net.training and self._loss_component_active(
                    "force_loss"
                )
                force_retain_graph = force_create_graph and self._any_non_force_loss_active()
                start = self._timed_section_start(batch)
                pred_forces = -torch.autograd.grad(
                    pred_energy.sum(),
                    batch.pos,
                    create_graph=force_create_graph,
                    retain_graph=force_retain_graph,
                )[0]
                if profile_timing:
                    timing["force_autograd_s"] = self._timed_section_end(batch, start)

        self._last_forward_timing = timing
        return pred_energy, pred_gradients, pred_diff, pred_forces

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the historical three-output model prediction tuple."""
        pred_energy, pred_gradients, pred_diff, _ = self.forward_predictions(batch)
        return pred_energy, pred_gradients, pred_diff

    def sample_forward(self, sample: OFData) -> OFData:
        """Applies the forward pass of the model to the batch.

        Args:
            sample: OFData object.

        Returns:
            OFData: The batch with the model predictions added
        """
        pred_energy, pred_gradients, pred_diff = self.forward(sample)
        sample.add_item("pred_energy", pred_energy, Representation.SCALAR)
        sample.add_item("pred_gradient", pred_gradients, Representation.GRADIENT)
        sample.add_item("pred_diff", pred_diff, Representation.VECTOR)
        return sample

    @property
    def tensorboard_logger(self):
        """Get the tensorboard logger from the trainer."""
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
        return tb_logger

    def training_step(self, batch: Batch, batch_idx: int = 0) -> dict:
        """Performs a single training step on a batch of data from the training set.

        Args:
            batch (Batch): A batch of data

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        compute_metrics_this_step = (
            self.hparams.metric_interval > 0
            and self.trainer.global_step % self.hparams.metric_interval == 0
        )
        use_logging_mixins_this_step = (
            self.use_logging_mixins and self.trainer.global_step % self.logging_mixin_interval == 0
        )

        # activate logging mixins if steps matches logging_mixin_interval
        if use_logging_mixins_this_step:
            self.activate_logging_mixins()

        pred_energy, pred_gradients, pred_diff, pred_forces = self.forward_predictions(batch)
        hvp_start = self._timed_section_start(batch)
        pred_hvp, hvp_active_mask = self._compute_directional_hvp(
            batch, pred_forces, batch_idx
        )
        if pred_hvp is not None and self._profile_timing_enabled():
            self._last_forward_timing["hvp_autograd_s"] = self._timed_section_end(
                batch, hvp_start
            )

        # calculate and log the losses
        projected_gradient_difference = project_gradient_difference(pred_gradients, batch)
        weight_dict, loss_dict = self.loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
            pred_forces=pred_forces,
            pred_hvp=pred_hvp,
            hvp_active_mask=hvp_active_mask,
        )
        effective_weight_dict = self._effective_loss_weights(
            weight_dict, hvp_active_mask
        )
        self._maybe_log_loss_gradient_norms(
            batch, effective_weight_dict, loss_dict, hvp_active_mask
        )
        loss = sum(
            effective_weight_dict[key] * loss_dict[key] for key in loss_dict.keys()
        )
        if "hvp_loss" in effective_weight_dict:
            self.log(
                "train_hvp/effective_weight",
                effective_weight_dict["hvp_loss"],
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self.log(
            "train_loss/total",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.train_loss(loss)
        log_dict = {f"train_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch.batch_size
        )

        # update and log every metric
        if compute_metrics_this_step:
            for metric in self.train_metrics.values():
                metric(batch, pred_energy, projected_gradient_difference, pred_diff)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True)

        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_train_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
            # For some reason, the below would significantly slow down training, even for iterations where it is not
            # called.
            # I think it's because lightning does not support logging at times other than on_step and on_epoch, i.e.
            # if you call log(..., on_step=True) once, the values will be logged at every step
            # Hence, we log directly to tensorboard (see above)
            # self.log_dict(
            #     mixin_log_dict,
            #     on_step=True,
            #     prog_bar=self.show_logging_mixins_in_progress_bar,
            #     batch_size=batch.batch_size
            # )
            self.deactivate_logging_mixins()
        return dict(
            loss=loss,
            model_outputs=dict(
                pred_energy=pred_energy,
                pred_gradients=pred_gradients,
                pred_diff=pred_diff,
                pred_forces=pred_forces,
            ),
            projected_gradient_difference=projected_gradient_difference,
        )

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the train step
        # self.console_logger.info(f"Train loss: {self.train_loss.compute()}")

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        super().on_validation_epoch_start()
        self.activate_logging_mixins()  # log values from logging mixins at every step during validation

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        """Performs a single validation step on a batch of data from the validation set.

        Args:
            batch (Batch): A batch of data
            batch_idx (int): The index of the batch

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        compute_metrics_this_step = (
            self.hparams.metric_interval > 0 and batch_idx % self.hparams.metric_interval == 0
        )
        use_logging_mixins_this_step = (
            self.use_logging_mixins
            and self.trainer.global_step % self.val_logging_mixin_interval == 0
        )

        loss_function = self._eval_loss_function()
        need_density_gradients = compute_metrics_this_step or self._loss_component_active(
            "gradient_loss", loss_function
        )
        need_forces = self.force_supervision and self._loss_component_active(
            "force_loss", loss_function
        )
        pred_energy, pred_gradients, pred_diff, pred_forces = self.forward_predictions(
            batch,
            compute_density_gradients=need_density_gradients,
            compute_forces=need_forces,
        )
        # calculate the loss
        projected_gradient_difference = (
            project_gradient_difference(pred_gradients, batch)
            if pred_gradients is not None
            else None
        )
        weight_dict, loss_dict = loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
            pred_forces=pred_forces,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"val_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )
        # log losses
        self.val_loss(loss)
        self.log(
            "val_loss/total",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            prog_bar=True,
            sync_dist=True,
        )

        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_val_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
        # update and log metrics
        if compute_metrics_this_step:
            if pred_gradients is None or projected_gradient_difference is None:
                raise RuntimeError("Validation metrics require density gradients.")
            for metric in self.val_metrics.values():
                metric(batch, pred_energy, projected_gradient_difference, pred_diff)
            self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return dict(
            loss=loss,
            model_outputs=dict(
                pred_energy=pred_energy,
                pred_gradients=pred_gradients,
                pred_diff=pred_diff,
                pred_forces=pred_forces,
            ),
            projected_gradient_difference=projected_gradient_difference,
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        if self.hparams.metric_interval <= 0:
            self.deactivate_logging_mixins()
            return
        for metric, (metric_best_key, metric_best) in zip(
            self.val_metrics.values(), self.val_metrics_best.items()
        ):
            scalar = metric.compute()  # get current val acc
            metric_best.update(scalar)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(metric_best_key, metric_best.compute(), sync_dist=True, prog_bar=False)

        # deactivate logging mixins after validation
        self.deactivate_logging_mixins()

    def test_step(self, batch: Batch) -> None:
        """Performs a single test step on a batch of data from the test set.

        Args:
            batch (Batch): A batch of data
        """
        loss_function = self._eval_loss_function()
        need_density_gradients = self._loss_component_active("gradient_loss", loss_function)
        need_forces = self.force_supervision and self._loss_component_active(
            "force_loss", loss_function
        )
        pred_energy, pred_gradients, pred_diff, pred_forces = self.forward_predictions(
            batch,
            compute_density_gradients=need_density_gradients,
            compute_forces=need_forces,
        )
        # calculate the loss
        projected_gradient_difference = (
            project_gradient_difference(pred_gradients, batch)
            if pred_gradients is not None
            else None
        )
        weight_dict, loss_dict = loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
            pred_forces=pred_forces,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"test_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )
        self.test_loss(loss)

        self.log(
            "test_loss/total",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )
        # update and log every metric
        if self.hparams.metric_interval > 0:
            if pred_gradients is None or projected_gradient_difference is None:
                raise RuntimeError("Test metrics require density gradients.")
            for metric in self.test_metrics.values():
                metric(batch, pred_energy, projected_gradient_difference, pred_diff)
            self.log_dict(
                self.test_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the test step
        # self.console_logger.info(f"Test loss: {self.test_loss.compute()}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"val"`, `"test"`, or `"predict"`.
        """
        # Doesn't work for double backward and doesn't provide a speed up currently.
        if self.hparams.compile:
            self.net = torch_geometric.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            Dict[str, Any]: A dict containing the configured optimizers and learning-rate schedulers
                to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
