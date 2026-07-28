from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    EnergyGradientLoss,
    EnergyLoss,
    ForceLoss,
    DirectionalHVPLoss,
    WeightedLoss,
)
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("k_neighbors", [3, 5])
def test_mldft_module(dummy_basis_info, dummy_dataset_torch, batch_size, k_neighbors):
    """Tests the model_step function of the MLDFTLitModule class."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    net = ToyNet(dummy_basis_info, k_neighbors)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )

    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
    )

    test = model.forward(batch)

    assert test[0].shape == batch.energy_label.shape
    assert test[1].shape == batch.gradient_label.shape
    assert test[2].shape == batch.coeffs.shape


@set_default_torch_dtype(torch.float64)
def test_mldft_module_force_supervision_predicts_forces(dummy_basis_info, dummy_dataset_torch):
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.force_label = torch.zeros_like(batch.pos)

    net = ToyNet(dummy_basis_info, k_neighbors=3)
    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=None,
        target_key="kin",
        compile=False,
        basis_info=dummy_basis_info,
        force_supervision=True,
    )

    pred_energy, pred_gradients, pred_diff, pred_forces = model.forward_predictions(batch)

    assert pred_energy.shape == batch.energy_label.shape
    assert pred_gradients.shape == batch.gradient_label.shape
    assert pred_diff.shape == batch.coeffs.shape
    assert pred_forces.shape == batch.force_label.shape


@set_default_torch_dtype(torch.float64)
def test_mldft_module_force_supervision_profiles_force_autograd(
    dummy_basis_info, dummy_dataset_torch
):
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.force_label = torch.zeros_like(batch.pos)

    net = ToyNet(dummy_basis_info, k_neighbors=3)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.1, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.8, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        force_loss=dict(weight=0.1, loss=ForceLoss(loss_function=nn.L1Loss(reduction="none"))),
    )
    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=False,
        basis_info=dummy_basis_info,
        force_supervision=True,
        profile_timing=True,
    )

    _, _, _, pred_forces = model.forward_predictions(batch)

    assert pred_forces is not None
    assert pred_forces.shape == batch.force_label.shape
    assert "net_forward_s" in model._last_forward_timing
    assert "density_gradient_autograd_s" in model._last_forward_timing
    assert "force_autograd_s" in model._last_forward_timing


@set_default_torch_dtype(torch.float64)
def test_mldft_module_eg_does_not_compute_forces(dummy_basis_info, dummy_dataset_torch):
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    net = ToyNet(dummy_basis_info, k_neighbors=3)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.1, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.8, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )
    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=False,
        basis_info=dummy_basis_info,
        force_supervision=False,
        profile_timing=True,
    )

    _, _, _, pred_forces = model.forward_predictions(batch)

    assert pred_forces is None
    assert "force_autograd_s" not in model._last_forward_timing


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("global_step, expected_active", [(2, False), (4, True)])
def test_mldft_module_hvp_schedule_keeps_backward_finite(
    dummy_basis_info, dummy_dataset_torch, global_step, expected_active
):
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.force_label = torch.zeros_like(batch.pos)
    batch.hvp_direction = torch.randn_like(batch.pos)
    batch.hvp_label = torch.zeros_like(batch.pos)
    batch.hvp_label_mask = torch.tensor([True, False])
    batch.hvp_direction_norm = torch.tensor(
        [torch.linalg.vector_norm(batch.hvp_direction[batch.batch == index]) for index in range(2)]
    )
    batch.hvp_reference_scale = torch.ones(2, dtype=batch.pos.dtype)

    net = ToyNet(dummy_basis_info, k_neighbors=3)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.1, loss=EnergyLoss()),
        force_loss=dict(weight=1.0, loss=ForceLoss()),
        hvp_loss=dict(weight=1.0e-4, loss=DirectionalHVPLoss()),
    )
    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=False,
        basis_info=dummy_basis_info,
        force_supervision=True,
        hvp_start_step=0,
        hvp_update_interval=4,
        hvp_update_phase=0,
    )
    model._trainer = SimpleNamespace(global_step=global_step)
    model.log = lambda *args, **kwargs: None

    pred_energy, _, _, pred_forces = model.forward_predictions(
        batch, compute_density_gradients=False
    )
    pred_hvp, active = model._compute_directional_hvp(batch, pred_forces, batch_idx=0)
    assert bool(active.any()) is expected_active
    objective = pred_energy.mean() + pred_forces.square().mean() + pred_hvp.square().mean()
    objective.backward()
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
