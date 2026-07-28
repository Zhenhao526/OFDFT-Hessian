"""Run a tiny MLDFT force-supervision smoke test on generated QM9 labels."""

import argparse
import json
import pickle
from pathlib import Path

import torch

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import ToTorch
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    EnergyGradientLoss,
    EnergyLoss,
    ForceLoss,
    WeightedLoss,
    project_gradient_difference,
)
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.mldft_module import MLDFTLitModule


def _paths_from_split(dataset_dir: Path, split_name: str, max_paths: int) -> tuple[list[Path], list[int]]:
    with (dataset_dir / "split.pkl").open("rb") as f:
        split = pickle.load(f)
    entries = split[split_name][:max_paths]
    paths = [dataset_dir / "labels" / filename for _, filename, _ in entries]
    iterations = [int(n_iterations) for _, _, n_iterations in entries]
    return paths, iterations


def run_smoke(args: argparse.Namespace) -> dict:
    torch.set_default_dtype(torch.float64)
    dataset_dir = Path(args.data_root) / args.dataset_name
    basis_info = BasisInfo.from_dataset_info_yaml(
        (dataset_dir / "dataset_info.yaml").as_posix(),
        atomic_numbers=[1, 6, 7, 8, 9],
    )
    paths, iterations = _paths_from_split(dataset_dir, args.split, args.max_paths)
    dataset = OFDataset(
        paths=paths,
        basis_info=basis_info,
        transforms=ToTorch(float_dtype=torch.float64),
        num_scf_iterations_per_path=iterations,
        limit_scf_iterations=-1,
        keep_initial_guess=False,
        energy_key=args.energy_key,
        gradient_key=args.gradient_key,
        load_force_label=True,
    )
    loader = OFLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["coeffs", "atomic_numbers"],
        num_workers=0,
    )

    net = ToyNet(basis_info=basis_info, k_neighbors=args.k_neighbors).double()
    loss_function = WeightedLoss(
        energy_loss=dict(weight=0.1, loss=EnergyLoss()),
        gradient_loss=dict(weight=0.8, loss=EnergyGradientLoss()),
        force_loss=dict(weight=0.1, loss=ForceLoss()),
        coefficient_loss=dict(weight=0.0, loss=CoefficientLoss()),
    )
    module = MLDFTLitModule(
        net=net,
        basis_info=basis_info,
        optimizer=None,
        scheduler=None,
        target_key=args.target_key,
        loss_function=loss_function,
        compile=False,
        force_supervision=True,
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    losses = []
    force_losses = []
    force_norms = []
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        optimizer.zero_grad()
        pred_energy, pred_gradients, pred_diff, pred_forces = module.forward_predictions(batch)
        projected_gradient_difference = project_gradient_difference(pred_gradients, batch)
        weight_dict, loss_dict = loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
            pred_forces=pred_forces,
        )
        loss = sum(weight_dict[key] * loss_dict[key] for key in loss_dict)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        force_losses.append(float(loss_dict["force_loss"].detach()))
        force_norms.append(float(pred_forces.detach().norm(dim=1).max()))

    return {
        "batch_size": args.batch_size,
        "dataset": args.dataset_name,
        "force_loss_final": force_losses[-1] if force_losses else None,
        "force_loss_initial": force_losses[0] if force_losses else None,
        "force_norm_max": max(force_norms) if force_norms else None,
        "loss_final": losses[-1] if losses else None,
        "loss_initial": losses[0] if losses else None,
        "paths": len(paths),
        "samples": len(dataset),
        "steps": len(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/afs/home/xiazhenhao/dft/structures25/_runtime/qm9_p0"),
    )
    parser.add_argument("--dataset-name", default="QM9PBEForceSmoke")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--target-key", default="kin_plus_xc")
    parser.add_argument("--energy-key", default="e_kin_plus_xc")
    parser.add_argument("--gradient-key", default="grad_kin_plus_xc")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
