import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    DirectionalHVPLoss,
    EnergyGradientLoss,
    EnergyLoss,
    ForceLoss,
    PairEnergySecantLoss,
    PairRelaxedForceSecantLoss,
    WeightedLoss,
    project_gradient_difference,
)


@pytest.mark.parametrize("batch_size", [4, 16])
def test_loss_functions_shape(batch_size):
    """Tests the shape of the output of the loss function and checks if the loss function is zero
    for the same input and target."""

    # Define a dataset to be used in the dataloader

    dataset = []

    for i in range(100):
        # define random length coefficients
        length = int(torch.randint(1, 20, (1,)))

        kin_energy = torch.rand(1)
        has_energy_label = torch.randint(1, (1,), dtype=torch.bool)
        kin_gradients = torch.randn((length,))
        basis_integrals = torch.randn((length,))
        n_atom = int(torch.randint(2, 8, (1,)))
        force_label = torch.randn((n_atom, 3))
        atomic_numbers = torch.ones(n_atom, dtype=torch.long)

        ground_state = torch.randn((length,))
        coeffs = ground_state + 1

        data = Data(
            energy_label=kin_energy,
            has_energy_label=has_energy_label,
            gradient_label=kin_gradients,
            basis_integrals=basis_integrals,
            dual_basis_integrals=basis_integrals,
            ground_state_coeffs=ground_state,
            coeffs=coeffs,
            force_label=force_label,
            atomic_numbers=atomic_numbers,
        )

        dataset.append(data)

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, follow_batch=["coeffs", "atomic_numbers"])

    # create Loss function
    module = WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        force_loss=dict(weight=0.3, loss=ForceLoss(loss_function=nn.L1Loss(reduction="none"))),
    )

    # check for 10 batches
    for batch in dataloader:
        # calculate the loss
        projected_gradient_difference = project_gradient_difference(batch.gradient_label, batch)
        weight_dict, loss_dict = module.forward(
            batch,
            pred_energy=batch.energy_label,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=batch.coeffs - batch.ground_state_coeffs,
            pred_forces=batch.force_label,
        )
        assert weight_dict.keys() == loss_dict.keys()
        for loss in loss_dict.values():
            assert loss == 0.0
            assert loss.shape == torch.Size([])
        total_loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        # check if the loss is zero and if the shape is correct
        assert total_loss == 0.0
        assert total_loss.shape == torch.Size([])


def test_pair_energy_secant_loss_matches_labelled_directional_secant():
    minus = Data(
        pos=torch.tensor([[-0.1, 0.0, 0.0]], dtype=torch.float64),
        energy_label=torch.tensor([1.0], dtype=torch.float64),
        has_energy_label=torch.tensor([True]),
        source_molecule_id=torch.tensor([7]),
        perturbation_pair_id=torch.tensor([3]),
        perturbation_pair_sign=torch.tensor([-1]),
        paired_perturbations=torch.tensor([True]),
    )
    plus = Data(
        pos=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
        energy_label=torch.tensor([3.0], dtype=torch.float64),
        has_energy_label=torch.tensor([True]),
        source_molecule_id=torch.tensor([7]),
        perturbation_pair_id=torch.tensor([3]),
        perturbation_pair_sign=torch.tensor([1]),
        paired_perturbations=torch.tensor([True]),
    )
    batch = next(iter(DataLoader([minus, plus], batch_size=2)))
    prediction = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)
    loss = PairEnergySecantLoss(loss_function=nn.L1Loss(reduction="none"))(
        batch, pred_energy=prediction
    )

    # Label and prediction directional secants are 10 and 5 Ha/Bohr, respectively.
    torch.testing.assert_close(loss, torch.tensor(5.0, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.tensor([5.0, -5.0], dtype=torch.float64)
    )


def test_pair_energy_secant_loss_rejects_incomplete_pair():
    sample = Data(
        pos=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
        energy_label=torch.tensor([3.0], dtype=torch.float64),
        has_energy_label=torch.tensor([True]),
        source_molecule_id=torch.tensor([7]),
        perturbation_pair_id=torch.tensor([3]),
        perturbation_pair_sign=torch.tensor([1]),
        paired_perturbations=torch.tensor([True]),
    )
    batch = next(iter(DataLoader([sample], batch_size=1)))
    with pytest.raises(ValueError, match="Incomplete geometry pairs"):
        PairEnergySecantLoss()(batch, pred_energy=torch.tensor([2.5]))


def test_directional_hvp_loss_normalizes_direction_scale_reference_and_atom_count():
    labelled = Data(
        atomic_numbers=torch.ones(2, dtype=torch.long),
        hvp_label=torch.zeros(2, 3, dtype=torch.float64),
        hvp_label_mask=torch.tensor([True]),
        hvp_direction_norm=torch.tensor([2.0], dtype=torch.float64),
        hvp_reference_scale=torch.tensor([4.0], dtype=torch.float64),
    )
    unlabelled = Data(
        atomic_numbers=torch.ones(3, dtype=torch.long),
        hvp_label=torch.zeros(3, 3, dtype=torch.float64),
        hvp_label_mask=torch.tensor([False]),
        hvp_direction_norm=torch.tensor([1.0], dtype=torch.float64),
        hvp_reference_scale=torch.tensor([1.0], dtype=torch.float64),
    )
    batch = next(
        iter(
            DataLoader(
                [labelled, unlabelled],
                batch_size=2,
                follow_batch=["atomic_numbers"],
            )
        )
    )
    prediction = torch.full((5, 3), 8.0, dtype=torch.float64, requires_grad=True)
    loss = DirectionalHVPLoss(loss_function=nn.L1Loss(reduction="none"))(
        batch, pred_hvp=prediction
    )
    torch.testing.assert_close(loss, torch.tensor(1.0, dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[2:]) == 0


def test_direct_hvp_loss_has_finite_third_order_parameter_gradient():
    theta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[0.5, -0.7, 1.1]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[0.2, 0.3, -0.4]], dtype=torch.float64)
    energy = theta * torch.sum(positions**3)
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    predicted_hvp = -torch.autograd.grad(
        torch.sum(force * direction), positions, create_graph=True
    )[0]
    loss = torch.mean(torch.abs(predicted_hvp))
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad)
    assert theta.grad.abs() > 0


def test_directional_hvp_loss_respects_batch_activation_mask():
    sample = Data(
        atomic_numbers=torch.ones(2, dtype=torch.long),
        hvp_label=torch.ones(2, 3, dtype=torch.float64),
        hvp_label_mask=torch.tensor([True]),
        hvp_direction_norm=torch.tensor([1.0], dtype=torch.float64),
        hvp_reference_scale=torch.tensor([1.0], dtype=torch.float64),
    )
    batch = next(
        iter(DataLoader([sample], batch_size=1, follow_batch=["atomic_numbers"]))
    )
    prediction = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    loss = DirectionalHVPLoss()(
        batch, pred_hvp=prediction, hvp_active_mask=torch.tensor([False])
    )
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64))
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 0


def test_directional_hvp_loss_smoothly_caps_pathological_graph_contribution():
    sample = Data(
        atomic_numbers=torch.ones(2, dtype=torch.long),
        hvp_label=torch.zeros(2, 3, dtype=torch.float64),
        hvp_label_mask=torch.tensor([True]),
        hvp_direction_norm=torch.tensor([1.0], dtype=torch.float64),
        hvp_reference_scale=torch.tensor([1.0], dtype=torch.float64),
    )
    batch = next(
        iter(DataLoader([sample], batch_size=1, follow_batch=["atomic_numbers"]))
    )
    prediction = torch.full((2, 3), 100.0, dtype=torch.float64, requires_grad=True)
    loss = DirectionalHVPLoss(max_normalized_loss=2.0)(batch, pred_hvp=prediction)

    assert 0.0 < float(loss) < 2.0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_relaxed_force_secant_loss_matches_complete_pair_and_backpropagates():
    minus = Data(
        pos=torch.tensor([[-0.1, 0.0, 0.0]], dtype=torch.float64),
        atomic_numbers=torch.ones(1, dtype=torch.long),
        force_label=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        energy_label=torch.tensor([0.0], dtype=torch.float64),
        has_energy_label=torch.tensor([True]),
        source_molecule_id=torch.tensor([7]),
        perturbation_pair_id=torch.tensor([3]),
        perturbation_pair_sign=torch.tensor([-1]),
        paired_perturbations=torch.tensor([True]),
    )
    plus = Data(
        pos=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
        atomic_numbers=torch.ones(1, dtype=torch.long),
        force_label=torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64),
        energy_label=torch.tensor([0.0], dtype=torch.float64),
        has_energy_label=torch.tensor([True]),
        source_molecule_id=torch.tensor([7]),
        perturbation_pair_id=torch.tensor([3]),
        perturbation_pair_sign=torch.tensor([1]),
        paired_perturbations=torch.tensor([True]),
    )
    batch = next(
        iter(DataLoader([minus, plus], batch_size=2, follow_batch=["atomic_numbers"]))
    )
    prediction = torch.tensor(
        [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = PairRelaxedForceSecantLoss(reference_scale_floor=1.0)(
        batch, pred_forces=prediction
    )

    torch.testing.assert_close(
        loss, torch.tensor(1.0 / (12.0**0.5), dtype=torch.float64)
    )
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 2


if __name__ == "__main__":
    test_loss_functions_shape()
