#!/usr/bin/env python3
"""Tiny force-supervision forward/backward smoke test for QM9 force labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr


class TinyEnergyModel(torch.nn.Module):
    """Minimal scalar energy model whose force is obtained by autograd."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        centered = positions - positions.mean(dim=0, keepdim=True)
        return self.scale * centered.square().sum() + self.bias


def load_sample(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    root = zarr.open(path, mode="r")
    positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    forces = np.asarray(root["metadata/pbe_derivatives/forces"], dtype=np.float64)
    if positions.shape != forces.shape:
        raise ValueError(f"{path}: positions shape {positions.shape} != force shape {forces.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(forces).all():
        raise ValueError(f"{path}: non-finite positions or forces")
    return torch.from_numpy(positions), torch.from_numpy(forces)


def expand_labels(paths: list[Path], max_files: int) -> list[Path]:
    labels = []
    for path in paths:
        if path.is_dir():
            labels.extend(path.glob("*.zarr.zip"))
        else:
            labels.append(path)
    labels = sorted(labels)
    if max_files > 0:
        labels = labels[:max_files]
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", type=Path, help="Label files or label directories.")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_paths = expand_labels(args.labels, args.max_files)
    if not label_paths:
        raise SystemExit("No labels found.")

    samples = [load_sample(path) for path in label_paths]
    model = TinyEnergyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    losses = []

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float64)
        for positions, target_forces in samples:
            positions = positions.clone().requires_grad_(True)
            energy = model(positions)
            (grad_positions,) = torch.autograd.grad(energy, positions, create_graph=True)
            pred_forces = -grad_positions
            total_loss = total_loss + torch.mean((pred_forces - target_forces) ** 2)
        loss = total_loss / len(samples)
        loss.backward()
        grad_norm = torch.sqrt(
            sum(
                torch.sum(param.grad.detach() ** 2)
                for param in model.parameters()
                if param.grad is not None
            )
        )
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite loss or gradient: loss={loss}, grad_norm={grad_norm}")
        optimizer.step()
        losses.append(float(loss.detach()))

    summary = {
        "samples": len(samples),
        "steps": args.steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "scale": float(model.scale.detach()),
        "bias": float(model.bias.detach()),
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_text + "\n")


if __name__ == "__main__":
    main()
