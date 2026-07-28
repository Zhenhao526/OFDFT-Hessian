#!/usr/bin/env python3
"""Sweep density optimization convergence on selected QM9 original geometries."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.optimizer import TorchOptimizer, VectorAdam
from mldft.ofdft.run_density_optimization import SampleGenerator, density_optimization


DEFAULT_MOLECULE_IDS = ["0000010", "0000323", "0000109", "0000171", "0000062"]
DEFAULT_MAX_CYCLE_BUDGETS = [50, 100, 200, 500]
DEFAULT_LRS = [1e-3, 3e-4, 1e-4]
DEFAULT_THRESHOLDS = [1e-2, 1e-3, 1e-4]


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


def _load_label_coeffs(dataset_dir: Path, molecule_id: str, sample_id: int) -> np.ndarray:
    label_path = dataset_dir / "labels" / f"{molecule_id}.{sample_id:07d}.zarr.zip"
    root = zarr.open(label_path, mode="r")
    return np.asarray(root["of_labels/spatial/coeffs"][-1], dtype=np.float64)


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


def _build_optimizer(name: str, lr: float, max_cycle: int, threshold: float, momentum: float):
    if name == "sgd":
        return TorchOptimizer(
            torch.optim.SGD,
            lr=lr,
            momentum=momentum,
            max_cycle=max_cycle,
            convergence_tolerance=threshold,
        )
    if name == "adam":
        return TorchOptimizer(
            torch.optim.Adam,
            lr=lr,
            max_cycle=max_cycle,
            convergence_tolerance=threshold,
        )
    if name == "vector_adam":
        return VectorAdam(
            learning_rate=lr,
            max_cycle=max_cycle,
            convergence_tolerance=threshold,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_initialization(
    init_name: str,
    sample: Any,
    dataset_dir: Path,
    molecule_id: str,
    sample_id: int,
) -> str | torch.Tensor:
    if init_name != "label":
        return init_name
    coeffs = torch.as_tensor(
        _load_label_coeffs(dataset_dir, molecule_id, sample_id),
        dtype=sample.coeffs.dtype,
        device=sample.coeffs.device,
    )
    return transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)


def _classify_curve(gradients: list[float], energies: list[float]) -> str:
    if not gradients or not np.isfinite(gradients).all() or not np.isfinite(energies).all():
        return "failed_or_nonfinite"
    grad = np.asarray(gradients, dtype=np.float64)
    energy = np.asarray(energies, dtype=np.float64)
    if grad[-1] > max(10.0 * max(grad[0], 1e-12), 10.0) or abs(energy[-1]) > 1e8:
        return "diverged"
    tail = grad[max(0, int(0.8 * len(grad))) :]
    if tail.size >= 5:
        tail_rel_change = abs(tail[-1] - tail[0]) / max(abs(tail[0]), 1e-12)
        if tail_rel_change < 0.05:
            return "plateau"
    diffs = np.diff(grad)
    if diffs.size >= 10:
        sign_changes = np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
        if sign_changes / max(diffs.size - 1, 1) > 0.35:
            return "oscillating"
    if grad[-1] < grad[0]:
        return "decreasing"
    return "not_improving"


def _first_cycle_below(values: list[float], threshold: float, limit: int) -> int | None:
    for idx, value in enumerate(values[:limit], start=1):
        if value < threshold:
            return idx
    return None


def _row_for_budget(
    base: dict[str, Any],
    trace: Trace,
    budget: int,
    threshold: float,
    final_energies: Any,
    strict_converged: bool,
    elapsed_s: float,
    error: str | None,
) -> dict[str, Any]:
    n = min(budget, len(trace.gradient_norms))
    if n == 0 or error is not None:
        return {
            **base,
            "max_cycle_budget": budget,
            "threshold": threshold,
            "converged": False,
            "cycles_to_threshold": None,
            "cycles_run": len(trace.gradient_norms),
            "final_projected_gradient_norm": None,
            "best_projected_gradient_norm": None,
            "final_total_energy": None,
            "final_electronic_energy": None,
            "strict_optimizer_converged": strict_converged,
            "elapsed_s": elapsed_s,
            "curve_status": "failed",
            "error": error,
        }
    window_grad = trace.gradient_norms[:n]
    window_energy = trace.total_energies[:n]
    cycles_to_threshold = _first_cycle_below(trace.gradient_norms, threshold, n)
    return {
        **base,
        "max_cycle_budget": budget,
        "threshold": threshold,
        "converged": cycles_to_threshold is not None,
        "cycles_to_threshold": cycles_to_threshold,
        "cycles_run": len(trace.gradient_norms),
        "final_projected_gradient_norm": float(window_grad[-1]),
        "best_projected_gradient_norm": float(min(window_grad)),
        "initial_projected_gradient_norm": float(trace.gradient_norms[0]),
        "final_total_energy": float(window_energy[-1]),
        "final_electronic_energy": float(trace.electronic_energies[n - 1]),
        "run_final_total_energy": float(final_energies.total_energy) if final_energies is not None else None,
        "strict_optimizer_converged": strict_converged,
        "elapsed_s": elapsed_s,
        "curve_status": _classify_curve(window_grad, window_energy),
        "error": error,
    }


def _curve_rows(base: dict[str, Any], trace: Trace) -> list[dict[str, Any]]:
    rows = []
    for idx, (grad, total_energy, electronic_energy) in enumerate(
        zip(trace.gradient_norms, trace.total_energies, trace.electronic_energies),
        start=1,
    ):
        rows.append(
            {
                **base,
                "cycle": idx,
                "projected_gradient_norm": grad,
                "total_energy": total_energy,
                "electronic_energy": electronic_energy,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("run", "optimizer", "lr", "initialization", "max_cycle_budget", "threshold")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        finite = [row for row in group if row["error"] is None]
        grads = [
            row["final_projected_gradient_norm"]
            for row in finite
            if row["final_projected_gradient_norm"] is not None
        ]
        energies = [row["final_total_energy"] for row in finite if row["final_total_energy"] is not None]
        status_counts = {}
        for row in finite:
            status_counts[row["curve_status"]] = status_counts.get(row["curve_status"], 0) + 1
        summary_rows.append(
            {
                **dict(zip(keys, key)),
                "n_success": len(finite),
                "n_failed": len(group) - len(finite),
                "n_converged": sum(1 for row in finite if row["converged"]),
                "mean_final_projected_gradient_norm": float(sum(grads) / len(grads)) if grads else None,
                "max_final_projected_gradient_norm": float(max(grads)) if grads else None,
                "mean_final_total_energy": float(sum(energies) / len(energies)) if energies else None,
                "status_counts": status_counts,
            }
        )
    return summary_rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    molecule_ids = args.molecule_id or DEFAULT_MOLECULE_IDS
    budgets = args.max_cycle_budget or DEFAULT_MAX_CYCLE_BUDGETS
    lrs = args.lr or DEFAULT_LRS
    thresholds = args.threshold or DEFAULT_THRESHOLDS
    optimizers = args.optimizer or ["sgd", "adam"]
    initializations = args.initialization or ["sad_default", "label"]
    run_max_cycle = max(budgets)
    strict_threshold = min(thresholds)

    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    result_rows: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []

    for context in contexts:
        for molecule_id in molecule_ids:
            atomic_numbers, positions_bohr = _load_geometry(args.dataset_dir, molecule_id, args.sample_id)
            mol = _make_mol(atomic_numbers, positions_bohr, args.charge)
            for init_name in initializations:
                for optimizer_name in optimizers:
                    for lr in lrs:
                        base = {
                            "run": context.spec.name,
                            "molecule_id": molecule_id,
                            "sample_id": args.sample_id,
                            "natoms": int(len(atomic_numbers)),
                            "optimizer": optimizer_name,
                            "lr": float(lr),
                            "initialization": init_name,
                            "run_max_cycle": run_max_cycle,
                            "strict_threshold_used": strict_threshold,
                        }
                        trace = Trace()
                        final_energies = None
                        strict_converged = False
                        elapsed_s = 0.0
                        error = None
                        try:
                            sample = context.sample_generator.get_sample_from_mol(mol)
                            initialization = _make_initialization(
                                init_name,
                                sample,
                                args.dataset_dir,
                                molecule_id,
                                args.sample_id,
                            )
                            optimizer = _build_optimizer(
                                optimizer_name,
                                float(lr),
                                run_max_cycle,
                                strict_threshold,
                                args.momentum,
                            )
                            t0 = time.time()
                            final_energies, _, strict_converged, _ = density_optimization(
                                sample,
                                sample.mol,
                                optimizer,
                                context.functional_factory,
                                callback=trace,
                                initialization=initialization,
                                max_xc_memory=args.max_xc_memory,
                                normalize_initial_guess=args.normalize_initial_guess,
                                ks_basis=args.ks_basis,
                                disable_printing=True,
                                disable_pbar=True,
                            )
                            elapsed_s = time.time() - t0
                        except Exception as exc:  # noqa: BLE001
                            error = repr(exc)
                        all_curve_rows.extend(_curve_rows(base, trace))
                        for budget in budgets:
                            for threshold in thresholds:
                                result_rows.append(
                                    _row_for_budget(
                                        base,
                                        trace,
                                        int(budget),
                                        float(threshold),
                                        final_energies,
                                        bool(strict_converged),
                                        elapsed_s,
                                        error,
                                    )
                                )

    summary_rows = _summaries(result_rows)
    result = {
        "definition": (
            "Density optimization convergence sweep. Each optimizer/lr/init/model/molecule "
            "is run once to the largest max_cycle budget using the strictest threshold; "
            "shorter max_cycle and relaxed thresholds are derived from the saved curve."
        ),
        "dataset_dir": args.dataset_dir.as_posix(),
        "sample_id": args.sample_id,
        "molecule_ids": molecule_ids,
        "runs": [
            {
                "name": context.spec.name,
                "run_dir": context.spec.run_dir.as_posix(),
                "ckpt": context.spec.ckpt.as_posix(),
            }
            for context in contexts
        ],
        "device": str(device),
        "optimizers": optimizers,
        "lrs": [float(lr) for lr in lrs],
        "initializations": initializations,
        "max_cycle_budgets": [int(value) for value in budgets],
        "thresholds": [float(value) for value in thresholds],
        "run_max_cycle": int(run_max_cycle),
        "strict_threshold_used": float(strict_threshold),
        "summary_rows": summary_rows,
        "rows": result_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        _write_csv(result_rows, args.output_csv)
    if args.curves_csv is not None:
        _write_csv(all_curve_rows, args.curves_csv)
    if args.summary_csv is not None:
        _write_csv(summary_rows, args.summary_csv)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--curves-csv", type=Path, default=None)
    parser.add_argument("--molecule-id", action="append", default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--optimizer", action="append", choices=["sgd", "adam", "vector_adam"], default=None)
    parser.add_argument("--lr", action="append", type=float, default=None)
    parser.add_argument("--max-cycle-budget", action="append", type=int, default=None)
    parser.add_argument("--threshold", action="append", type=float, default=None)
    parser.add_argument("--initialization", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
