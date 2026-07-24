#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mpn_melting.suf_reference import KB_EV_PER_K
from run_pair_reference_md import wrap_positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mean lattice sites and a matched Einstein spring constant"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.trajectory.read_text(encoding="utf-8").splitlines()
    ]
    rows = [row for row in rows if "positions_angstrom" in row]
    start = int(len(rows) * args.discard_fraction)
    rows = rows[start:]
    if len(rows) < 10:
        raise ValueError("at least ten stored production frames are required")
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    lattice = torch.tensor(checkpoint["lattice_angstrom"], dtype=torch.float64)
    inverse = torch.linalg.inv(lattice)
    base = torch.tensor(rows[0]["positions_angstrom"], dtype=torch.float64)
    displacements = []
    for row in rows:
        positions = torch.tensor(row["positions_angstrom"], dtype=torch.float64)
        fractional = (positions - base) @ inverse
        fractional -= torch.round(fractional)
        displacements.append(fractional @ lattice)
    displacement_stack = torch.stack(displacements)
    mean_positions = wrap_positions(base + displacement_stack.mean(dim=0), lattice)

    squared_displacements = []
    for row in rows:
        positions = torch.tensor(row["positions_angstrom"], dtype=torch.float64)
        fractional = (positions - mean_positions) @ inverse
        fractional -= torch.round(fractional)
        displacement = fractional @ lattice
        squared_displacements.extend((displacement * displacement).sum(dim=1).tolist())
    msd = sum(squared_displacements) / len(squared_displacements)
    spring = 3.0 * KB_EV_PER_K * args.temperature / msd
    result = {
        "schema": "mpn-einstein-reference-v1",
        "trajectory": str(args.trajectory.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "temperature_k": args.temperature,
        "discard_fraction": args.discard_fraction,
        "frames": len(rows),
        "natoms": mean_positions.shape[0],
        "lattice_angstrom": lattice.tolist(),
        "anchor_positions_angstrom": mean_positions.tolist(),
        "mean_squared_displacement_angstrom2": msd,
        "spring_constant_ev_per_angstrom2": spring,
        "spring_definition": "U_E = 0.5 * k * sum_i |r_i-r_i0|^2",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "anchor_positions_angstrom"}, indent=2))


if __name__ == "__main__":
    main()
