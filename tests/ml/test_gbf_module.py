import pytest
import torch
from torch import Tensor

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.gbf_module import GBFModule, GaussianLayer


def _full_hessian_from_scalar(scalar: Tensor, pos: Tensor) -> Tensor:
    grad_pos = torch.autograd.grad(scalar, pos, create_graph=True, retain_graph=True)[0]
    rows = []
    for component in grad_pos.reshape(-1):
        row = torch.autograd.grad(component, pos, retain_graph=True)[0]
        rows.append(row.reshape(-1))
    return torch.stack(rows)


@pytest.mark.parametrize(
    "positions, edge_index, expected",
    [
        (
            Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 2.0]]),
            Tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).int(),
            Tensor(
                [
                    [
                        0.24197073,
                        0.31944802,
                        0.37738323,
                        0.39894229,
                        0.37738323,
                        0.31944802,
                        0.24197073,
                        0.16401009,
                        0.09947712,
                        0.05399097,
                    ],
                    [
                        0.24197073,
                        0.31944802,
                        0.37738323,
                        0.39894229,
                        0.37738323,
                        0.31944802,
                        0.24197073,
                        0.16401009,
                        0.09947712,
                        0.05399097,
                    ],
                    [
                        0.00443185,
                        0.01139598,
                        0.02622189,
                        0.05399097,
                        0.09947714,
                        0.16401006,
                        0.24197073,
                        0.31944799,
                        0.37738323,
                        0.39894229,
                    ],
                    [
                        0.03274718,
                        0.06527553,
                        0.11643190,
                        0.18583974,
                        0.26543015,
                        0.33923998,
                        0.38797957,
                        0.39705965,
                        0.36361989,
                        0.29797831,
                    ],
                ]
            ),
        ),
    ],
)
def test_gbf_module_test_cases(positions: Tensor, edge_index: Tensor, expected: Tensor) -> None:
    """Tests the forward pass of the GBFModule with some test cases.

    Args:
        positions (Tensor): positions of the atoms
        edge_index (Tensor): edge index of the graph
        expected (Tensor): expected output of the forward pass
    """
    module = GBFModule(normalized=True)
    assert torch.allclose(module(OFData(pos=positions, edge_index=edge_index)), expected)


@pytest.mark.parametrize(
    "num_gaussians, normalized",
    [
        (5, True),
        (10, False),
        (20, True),
    ],
)
def test_gbf_module_shapes(num_gaussians: int, normalized: bool) -> None:
    """Test the shape of the output of the GBFModule.

    Args:
        num_gaussians (int): Number of gaussians to be used
        normalized (bool): Normalization flag of the gaussians
    """
    module = GBFModule(num_gaussians=num_gaussians, normalized=normalized)

    for i in range(10):
        num_atoms = int(torch.randint(1, 100, (1,)).item())
        num_edges = int(torch.randint(1, 100, (1,)).item())
        pos = torch.randn(num_atoms, 3)
        edge_index = torch.randint(0, num_atoms, (2, num_edges))
        edge_attr = module(OFData(pos=pos, edge_index=edge_index))
        assert edge_attr.shape == (num_edges, num_gaussians)


@pytest.mark.parametrize(
    "num_gaussians, integrated_area, scale",
    [
        (5, Tensor([0.5, 0.773373, 0.933193, 0.987776, 0.998650]), torch.ones(5)),
        (5, Tensor([0.5, 0.933193, 0.99865, 0.999997, 1]), torch.ones(5) * 0.5),
    ],
)
def test_gbf_normalisation(num_gaussians: int, integrated_area: Tensor, scale: Tensor) -> None:
    """Test the normalisation of the gaussians.

    Args:
        num_gaussians (int): Number of gaussians to be used
        integrated_area (Tensor): Integrated area of the gaussians in the boundaries [0, infintiy]
        scale (Tensor): Scale parameter of the gaussians
    """

    module = GBFModule(num_gaussians=num_gaussians, normalized=True)

    module.scale = torch.nn.Parameter(scale)

    num_atoms = 1000000

    # create a linear space of positions to get a linear distribution of distances to the first atom
    # Here 10 is a placeholder for the infinite integration boundary
    positions_x = torch.linspace(0, 10, num_atoms)
    positions_y = torch.zeros(num_atoms)
    positions_z = torch.zeros(num_atoms)

    # stack the positions into one tensor
    positions = torch.stack([positions_x, positions_y, positions_z], dim=1)

    # calculate the edge index where every node is only connected to the first atom
    edge_index_1 = torch.zeros(num_atoms)
    edge_index_2 = torch.linspace(0, num_atoms - 1, num_atoms)
    edge_index = torch.stack([edge_index_1, edge_index_2], dim=0).long()

    # calculate the output of the module
    output = module.forward(OFData(pos=positions, edge_index=edge_index))

    # test if the integrated area of the gaussians is  equal to the given integrated_area
    for i in range(num_gaussians):
        assert torch.allclose(10 / num_atoms * torch.sum(output[:, i]), integrated_area[i])


@pytest.mark.parametrize("module_cls", [GBFModule])
def test_gbf_module_self_loop_second_order_autograd_is_finite(module_cls) -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [-0.4, 0.8, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(pos.shape[0]), torch.arange(pos.shape[0]), indexing="ij")
    ).reshape(2, -1)
    module = module_cls(num_gaussians=8, normalized=True).to(torch.float64)

    edge_attr = module(OFData(pos=pos, edge_index=edge_index))
    hessian = _full_hessian_from_scalar(edge_attr.sum(), pos)

    assert torch.isfinite(hessian).all()


def test_gaussian_layer_self_loop_second_order_autograd_is_finite(dummy_basis_info) -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [-0.4, 0.8, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(pos.shape[0]), torch.arange(pos.shape[0]), indexing="ij")
    ).reshape(2, -1)
    sample = OFData(
        pos=pos,
        edge_index=edge_index,
        atom_ind=torch.tensor([0, 1, 2], dtype=torch.long),
    )
    module = GaussianLayer(dummy_basis_info, num_gaussians=8, normalized=True).to(torch.float64)

    edge_attr, length = module(sample)
    hessian = _full_hessian_from_scalar(edge_attr.sum() + length.sum(), pos)

    assert torch.isfinite(hessian).all()


def test_gaussian_layer_non_self_edges_match_filtered_graph(dummy_basis_info) -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [-0.4, 0.8, 0.3]],
        dtype=torch.float64,
    )
    full_edge_index = torch.stack(
        torch.meshgrid(torch.arange(pos.shape[0]), torch.arange(pos.shape[0]), indexing="ij")
    ).reshape(2, -1)
    non_self = full_edge_index[0] != full_edge_index[1]
    filtered_edge_index = full_edge_index[:, non_self]
    atom_ind = torch.tensor([0, 1, 2], dtype=torch.long)
    module = GaussianLayer(dummy_basis_info, num_gaussians=8, normalized=True).to(torch.float64)

    full_attr, full_length = module(
        OFData(pos=pos, edge_index=full_edge_index, atom_ind=atom_ind)
    )
    filtered_attr, filtered_length = module(
        OFData(pos=pos, edge_index=filtered_edge_index, atom_ind=atom_ind)
    )

    assert torch.allclose(full_attr[non_self], filtered_attr)
    assert torch.allclose(full_length[non_self], filtered_length)
    assert torch.equal(full_length[~non_self], torch.zeros_like(full_length[~non_self]))


def test_gaussian_layer_preserves_float64_distance_resolution(dummy_basis_info) -> None:
    separation = 1.0e-9
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0 + separation, 0.0, 0.0]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    sample = OFData(
        pos=pos,
        edge_index=edge_index,
        atom_ind=torch.tensor([0, 1, 1], dtype=torch.long),
    )
    module = GaussianLayer(dummy_basis_info, num_gaussians=8, normalized=True).to(
        torch.float64
    )

    edge_attr, length = module(sample)

    torch.testing.assert_close(
        length[1] - length[0],
        torch.tensor([separation], dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-15,
    )
    assert torch.max(torch.abs(edge_attr[1] - edge_attr[0])) > 0.0
