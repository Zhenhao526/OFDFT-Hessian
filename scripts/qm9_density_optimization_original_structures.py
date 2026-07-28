#!/usr/bin/env python3
"""Run OFDFT density optimization on selected original QM9 label geometries."""

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


DEFAULT_MOLECULE_IDS = ["0000010", "0000323", "0000109", "0000171", "0000062"]


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


@dataclass
class OptimizationTrace:
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
    optimizer: TorchOptimizer


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
    optimizer = TorchOptimizer(
        torch.optim.SGD,
        lr=args.lr,
        momentum=args.momentum,
        max_cycle=args.max_cycle,
        convergence_tolerance=args.convergence_tolerance,
    )
    return RunContext(
        spec=spec,
        model=model,
        sample_generator=sample_generator,
        functional_factory=functional_factory,
        optimizer=optimizer,
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


def _energy_components(energies: Any) -> dict[str, float]:
    return {f"energy_{key}": float(value) for key, value in energies.energies_dict.items()}


def _optimize_one(
    context: RunContext,
    molecule_id: str,
    sample_id: int,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mol = _make_mol(atomic_numbers, positions_bohr, args.charge)
    sample = context.sample_generator.get_sample_from_mol(mol)
    trace = OptimizationTrace()
    t0 = time.time()
    energies, _, converged, _ = density_optimization(
        sample,
        sample.mol,
        context.optimizer,
        context.functional_factory,
        callback=trace,
        initialization=args.initialization,
        max_xc_memory=args.max_xc_memory,
        normalize_initial_guess=args.normalize_initial_guess,
        ks_basis=args.ks_basis,
        disable_printing=True,
        disable_pbar=True,
    )
    elapsed_s = time.time() - t0
    final_density_loss = trace.gradient_norms[-1] if trace.gradient_norms else None
    initial_density_loss = trace.gradient_norms[0] if trace.gradient_norms else None
    return {
        "run": context.spec.name,
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "natoms": int(len(atomic_numbers)),
        "converged": bool(converged),
        "cycles": len(trace.gradient_norms),
        "initial_density_loss_projected_gradient_norm": initial_density_loss,
        "final_density_loss_projected_gradient_norm": final_density_loss,
        "final_total_energy": float(energies.total_energy),
        "final_electronic_energy": float(energies.electronic_energy),
        "elapsed_s": elapsed_s,
        "finite": bool(
            np.isfinite(
                [
                    value
                    for value in [
                        final_density_loss,
                        float(energies.total_energy),
                        float(energies.electronic_energy),
                    ]
                    if value is not None
                ]
            ).all()
        ),
        "error": None,
        **_energy_components(energies),
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fixed_fields = [
        "run",
        "molecule_id",
        "sample_id",
        "natoms",
        "converged",
        "cycles",
        "initial_density_loss_projected_gradient_norm",
        "final_density_loss_projected_gradient_norm",
        "final_total_energy",
        "final_electronic_energy",
        "elapsed_s",
        "finite",
        "error",
    ]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("energy_") and key not in fixed_fields
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fixed_fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    molecule_ids = args.molecule_id or DEFAULT_MOLECULE_IDS
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]

    rows: list[dict[str, Any]] = []
    for context in contexts:
        for molecule_id in molecule_ids:
            base_row = {
                "run": context.spec.name,
                "molecule_id": molecule_id,
                "sample_id": args.sample_id,
                "converged": False,
                "cycles": 0,
                "error": None,
            }
            try:
                atomic_numbers, positions_bohr = _load_geometry(
                    args.dataset_dir,
                    molecule_id,
                    args.sample_id,
                )
                row = _optimize_one(
                    context,
                    molecule_id,
                    args.sample_id,
                    atomic_numbers,
                    positions_bohr,
                    args,
                )
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append({**base_row, "error": repr(exc)})

    summaries = {}
    for context in contexts:
        run_rows = [row for row in rows if row["run"] == context.spec.name and row["error"] is None]
        summaries[context.spec.name] = {
            "run_dir": context.spec.run_dir.as_posix(),
            "ckpt": context.spec.ckpt.as_posix(),
            "n_success": len(run_rows),
            "n_failed": len([row for row in rows if row["run"] == context.spec.name and row["error"] is not None]),
            "n_converged": sum(1 for row in run_rows if row["converged"]),
            "mean_cycles": _mean([float(row["cycles"]) for row in run_rows]),
            "mean_final_density_loss_projected_gradient_norm": _mean(
                [row["final_density_loss_projected_gradient_norm"] for row in run_rows]
            ),
            "mean_final_total_energy": _mean([row["final_total_energy"] for row in run_rows]),
            "total_elapsed_s": float(sum(row["elapsed_s"] for row in run_rows)),
        }

    result = {
        "definition": (
            "single-point density optimization on original sample_id geometry, no displacement; "
            "density loss is recorded as the projected density-gradient norm used by the optimizer"
        ),
        "dataset_dir": args.dataset_dir.as_posix(),
        "molecule_ids": molecule_ids,
        "sample_id": args.sample_id,
        "device": str(device),
        "initialization": args.initialization,
        "optimizer": {
            "name": "torch.optim.SGD",
            "lr": args.lr,
            "momentum": args.momentum,
            "max_cycle": args.max_cycle,
            "convergence_tolerance": args.convergence_tolerance,
        },
        "runs": summaries,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--molecule-id", action="append", default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
