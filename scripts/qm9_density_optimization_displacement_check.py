#!/usr/bin/env python3
"""Check density optimization convergence on original and +/- displaced QM9 geometries."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from pyscf import gto

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.optimizer import TorchOptimizer
from mldft.ofdft.run_density_optimization import SampleGenerator, density_optimization


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


@dataclass
class Trace:
    total_energies: list[float] = field(default_factory=list)
    electronic_energies: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)

    def __call__(self, minimization_locals: dict[str, Any]) -> None:
        energy = minimization_locals["energy"]
        self.total_energies.append(float(energy.total_energy))
        self.electronic_energies.append(float(energy.electronic_energy))
        self.gradient_norms.append(float(minimization_locals["gradient_norm"]))


@dataclass
class RunContext:
    spec: RunSpec
    model: MLDFTLitModule
    sample_generator: SampleGenerator
    functional_factory: FunctionalFactory


def _parse_run(item: str) -> RunSpec:
    name, run_dir, ckpt = item.split("=", maxsplit=2)
    return RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve())


def _load_context(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> RunContext:
    cfg = OmegaConf.load(spec.run_dir / "hparams.yaml")
    model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location=device)
    model.eval()
    model.to(device)
    model.to(torch.float64)
    sample_generator = SampleGenerator(
        cfg,
        model,
        negative_integrated_density_penalty_weight=args.negative_integrated_density_penalty_weight,
        transform_device=args.transform_device,
    )
    functional_factory = FunctionalFactory.from_module(
        model,
        negative_integrated_density_penalty_weight=args.negative_integrated_density_penalty_weight,
    )
    return RunContext(
        spec=spec,
        model=model,
        sample_generator=sample_generator,
        functional_factory=functional_factory,
    )


def _load_geometry(dataset_dir: Path, molecule_id: str, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    root = zarr.open(label_path, mode="r")
    return (
        np.asarray(root["geometry/atomic_numbers"], dtype=np.int64),
        np.asarray(root["geometry/atom_pos"], dtype=np.float64),
    )


def _make_mol(atomic_numbers: np.ndarray, positions_bohr: np.ndarray, charge: int) -> gto.Mole:
    atoms = [
        (int(atomic_number), tuple(float(x) for x in position))
        for atomic_number, position in zip(atomic_numbers, positions_bohr)
    ]
    nelectron = int(np.sum(atomic_numbers) - charge)
    spin = nelectron % 2
    return gto.M(
        atom=atoms,
        unit="Bohr",
        charge=charge,
        spin=spin,
        basis="sto-3g",
        verbose=0,
    )


def _first_cycle_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value < threshold:
            return idx
    return None


def _build_optimizer(args: argparse.Namespace):
    if args.optimizer == "sgd":
        return TorchOptimizer(
            torch.optim.SGD,
            lr=args.lr,
            momentum=args.momentum,
            max_cycle=args.max_cycle,
            convergence_tolerance=min(args.threshold),
        )
    if args.optimizer == "adam":
        return TorchOptimizer(
            torch.optim.Adam,
            lr=args.lr,
            max_cycle=args.max_cycle,
            convergence_tolerance=min(args.threshold),
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _optimize(
    context: RunContext,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Trace, Any, bool, float]:
    mol = _make_mol(atomic_numbers, positions_bohr, args.charge)
    sample = context.sample_generator.get_sample_from_mol(mol)
    optimizer = _build_optimizer(args)
    trace = Trace()
    t0 = time.time()
    final_energies, _, converged, _ = density_optimization(
        sample,
        sample.mol,
        optimizer,
        context.functional_factory,
        callback=trace,
        initialization=args.initialization,
        max_xc_memory=args.max_xc_memory,
        normalize_initial_guess=args.normalize_initial_guess,
        ks_basis=args.ks_basis,
        disable_printing=True,
        disable_pbar=True,
    )
    return trace, final_energies, bool(converged), time.time() - t0


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    args.threshold = args.threshold or [1e-2, 1e-3, 1e-4]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for context in contexts:
        for molecule_id in args.molecule_id:
            atomic_numbers, base_positions = _load_geometry(args.dataset_dir, molecule_id, args.sample_id)
            for side, offset in [("base", 0.0), ("plus", args.displacement), ("minus", -args.displacement)]:
                positions = base_positions.copy().reshape(-1)
                positions[args.coord_idx] += offset
                positions = positions.reshape(base_positions.shape)
                base_row = {
                    "run": context.spec.name,
                    "molecule_id": molecule_id,
                    "sample_id": args.sample_id,
                    "natoms": int(len(atomic_numbers)),
                    "coord_idx": args.coord_idx,
                    "side": side,
                    "displacement": offset,
                    "optimizer": args.optimizer,
                    "lr": args.lr,
                    "initialization": args.initialization,
                    "max_cycle": args.max_cycle,
                }
                try:
                    trace, final_energies, strict_converged, elapsed_s = _optimize(
                        context,
                        atomic_numbers,
                        positions,
                        args,
                    )
                    for idx, (grad, total_energy, electronic_energy) in enumerate(
                        zip(trace.gradient_norms, trace.total_energies, trace.electronic_energies),
                        start=1,
                    ):
                        curve_rows.append(
                            {
                                **base_row,
                                "cycle": idx,
                                "projected_gradient_norm": grad,
                                "total_energy": total_energy,
                                "electronic_energy": electronic_energy,
                            }
                        )
                    for threshold in args.threshold:
                        rows.append(
                            {
                                **base_row,
                                "threshold": threshold,
                                "converged": _first_cycle_below(trace.gradient_norms, threshold) is not None,
                                "cycles_to_threshold": _first_cycle_below(trace.gradient_norms, threshold),
                                "cycles_run": len(trace.gradient_norms),
                                "strict_optimizer_converged": strict_converged,
                                "initial_projected_gradient_norm": trace.gradient_norms[0],
                                "final_projected_gradient_norm": trace.gradient_norms[-1],
                                "best_projected_gradient_norm": min(trace.gradient_norms),
                                "final_total_energy": float(final_energies.total_energy),
                                "final_electronic_energy": float(final_energies.electronic_energy),
                                "elapsed_s": elapsed_s,
                                "error": None,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    for threshold in args.threshold:
                        rows.append({**base_row, "threshold": threshold, "error": repr(exc)})

    result = {
        "definition": "Original and +/- displaced geometry density optimization check.",
        "dataset_dir": args.dataset_dir.as_posix(),
        "sample_id": args.sample_id,
        "molecule_ids": args.molecule_id,
        "coord_idx": args.coord_idx,
        "displacement": args.displacement,
        "optimizer": {"name": args.optimizer, "lr": args.lr, "momentum": args.momentum},
        "initialization": args.initialization,
        "max_cycle": args.max_cycle,
        "thresholds": args.threshold,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
    if args.curves_csv is not None:
        _write_csv(curve_rows, args.curves_csv)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--molecule-id", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--curves-csv", type=Path, default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--coord-idx", type=int, default=0)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--threshold", action="append", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--max-cycle", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
