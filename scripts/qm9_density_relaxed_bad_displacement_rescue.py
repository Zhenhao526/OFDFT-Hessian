#!/usr/bin/env python3
"""Rescue non-converged density-relaxed Hessian displacement optimizations.

The input is the per-displacement optimization CSV produced by
``qm9_hessian_density_relaxed_eval.py``.  Only rows with ``converged != True``
are rerun.  This script records every optimization cycle so slow-converging,
oscillating, and plateaued displacement points can be audited without rerunning
the whole Hessian.
"""

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
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.optimizer import SLSQP, TorchOptimizer, TrustRegionConstrained, VectorAdam
from mldft.ofdft.run_density_optimization import SampleGenerator, density_optimization


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


@dataclass(frozen=True)
class StageSpec:
    optimizer: str
    lr: float
    max_cycle: int


@dataclass(frozen=True)
class VariantSpec:
    name: str
    initialization: str
    stages: tuple[StageSpec, ...]


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


def _parse_variant(item: str) -> VariantSpec:
    parts = item.split(":")
    if len(parts) < 5 or (len(parts) - 2) % 3 != 0:
        raise ValueError(
            "Variant format must be name:initialization:optimizer:lr:max_cycle"
            "[:optimizer:lr:max_cycle...]"
        )
    name = parts[0]
    initialization = parts[1]
    stages = []
    for idx in range(2, len(parts), 3):
        stages.append(
            StageSpec(
                optimizer=parts[idx],
                lr=float(parts[idx + 1]),
                max_cycle=int(parts[idx + 2]),
            )
        )
    return VariantSpec(name=name, initialization=initialization, stages=tuple(stages))


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


def _build_optimizer(stage: StageSpec, args: argparse.Namespace):
    if stage.optimizer == "adam":
        return TorchOptimizer(
            torch.optim.Adam,
            lr=stage.lr,
            max_cycle=stage.max_cycle,
            convergence_tolerance=args.threshold,
        )
    if stage.optimizer == "sgd":
        return TorchOptimizer(
            torch.optim.SGD,
            lr=stage.lr,
            momentum=args.momentum,
            max_cycle=stage.max_cycle,
            convergence_tolerance=args.threshold,
        )
    if stage.optimizer == "vector_adam":
        return VectorAdam(
            learning_rate=stage.lr,
            max_cycle=stage.max_cycle,
            convergence_tolerance=args.threshold,
        )
    if stage.optimizer == "slsqp":
        return SLSQP(
            max_cycle=stage.max_cycle,
            convergence_tolerance=args.threshold,
            grad_scale=args.slsqp_grad_scale,
            use_projected_gradient=True,
        )
    if stage.optimizer == "trust_constr":
        return TrustRegionConstrained(
            max_cycle=stage.max_cycle,
            convergence_tolerance=args.threshold,
            initial_tr_radius=args.trust_initial_radius,
            initial_constr_penalty=args.trust_initial_constr_penalty,
            use_projected_gradient=True,
        )
    raise ValueError(f"Unsupported optimizer stage: {stage.optimizer}")


def _initialization(
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


def _read_bad_points(
    path: Path,
    run_names: set[str],
    source_displacement: float | None = None,
    source_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    points = []
    seen = set()
    for row in rows:
        if row.get("converged") == "True":
            continue
        if row["run"] not in run_names:
            continue
        row_displacement = float(row.get("displacement") or 0.0)
        row_tolerance = float(row.get("tolerance") or 0.0)
        if source_displacement is not None and not np.isclose(
            row_displacement, source_displacement, rtol=0.0, atol=1e-12
        ):
            continue
        if source_tolerance is not None and not np.isclose(
            row_tolerance, source_tolerance, rtol=0.0, atol=1e-12
        ):
            continue
        key = (
            row["run"],
            row["molecule_id"],
            int(row["sample_id"]),
            int(row["coord_idx"]),
            row["side"],
            row_displacement,
            row_tolerance,
        )
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "run": row["run"],
                "molecule_id": row["molecule_id"],
                "sample_id": int(row["sample_id"]),
                "coord_idx": int(row["coord_idx"]),
                "side": row["side"],
                "source_displacement": row_displacement,
                "source_tolerance": row_tolerance,
                "previous_cycles": int(row["cycles"]) if row.get("cycles") else None,
                "previous_final_gradient_norm": (
                    float(row["final_gradient_norm"]) if row.get("final_gradient_norm") else None
                ),
            }
        )
    return points


def _first_cycle_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value < threshold:
            return idx
    return None


def _classify_curve(values: list[float]) -> str:
    if not values or not np.isfinite(values).all():
        return "failed_or_nonfinite"
    grad = np.asarray(values, dtype=np.float64)
    if grad[-1] > max(10.0 * max(grad[0], 1e-12), 10.0):
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


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_curve_npz(curve_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_fields = (
        "stage_index",
        "stage_lr",
        "stage_max_cycle",
        "stage_cycle",
        "global_cycle",
        "projected_gradient_norm",
        "total_energy",
        "electronic_energy",
    )
    payload = {
        field: np.asarray([row[field] for row in curve_rows])
        for field in numeric_fields
    }
    payload["stage_optimizer"] = np.asarray(
        [row["stage_optimizer"] for row in curve_rows], dtype="U32"
    )
    np.savez_compressed(path, **payload)


def _optimize_point(
    context: RunContext,
    point: dict[str, Any],
    variant: VariantSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    atomic_numbers, base_positions = _load_geometry(
        args.dataset_dir,
        point["molecule_id"],
        point["sample_id"],
    )
    flat_positions = base_positions.copy().reshape(-1)
    sign = 1.0 if point["side"] == "plus" else -1.0
    displacement = float(point.get("source_displacement") or args.displacement)
    flat_positions[point["coord_idx"]] += sign * displacement
    positions = flat_positions.reshape(base_positions.shape)

    mol = _make_mol(atomic_numbers, positions, args.charge)
    sample = context.sample_generator.get_sample_from_mol(mol)
    initialization: str | torch.Tensor = _initialization(
        variant.initialization,
        sample,
        args.dataset_dir,
        point["molecule_id"],
        point["sample_id"],
    )
    base = {
        **point,
        "variant": variant.name,
        "initialization": variant.initialization,
        "natoms": int(len(atomic_numbers)),
        "displacement": displacement,
        "threshold": args.threshold,
    }
    curve_rows: list[dict[str, Any]] = []
    all_gradients: list[float] = []
    final_total_energy = None
    final_electronic_energy = None
    converged = False
    converged_stage = None
    elapsed_s = 0.0
    error = None
    global_cycle = 0

    try:
        for stage_index, stage in enumerate(variant.stages, start=1):
            trace = Trace()
            optimizer = _build_optimizer(stage, args)
            stage_initialization = initialization if stage_index == 1 else sample.coeffs.detach().clone()
            t0 = time.time()
            final_energies, _, stage_converged, _ = density_optimization(
                sample,
                sample.mol,
                optimizer,
                context.functional_factory,
                callback=trace,
                initialization=stage_initialization,
                max_xc_memory=args.max_xc_memory,
                normalize_initial_guess=args.normalize_initial_guess,
                ks_basis=args.ks_basis,
                disable_printing=True,
                disable_pbar=True,
            )
            elapsed_s += time.time() - t0
            final_total_energy = float(final_energies.total_energy)
            final_electronic_energy = float(final_energies.electronic_energy)
            for stage_cycle, (grad, total_energy, electronic_energy) in enumerate(
                zip(trace.gradient_norms, trace.total_energies, trace.electronic_energies),
                start=1,
            ):
                global_cycle += 1
                curve_rows.append(
                    {
                        **base,
                        "stage_index": stage_index,
                        "stage_optimizer": stage.optimizer,
                        "stage_lr": stage.lr,
                        "stage_max_cycle": stage.max_cycle,
                        "stage_cycle": stage_cycle,
                        "global_cycle": global_cycle,
                        "projected_gradient_norm": grad,
                        "total_energy": total_energy,
                        "electronic_energy": electronic_energy,
                    }
                )
            all_gradients.extend(trace.gradient_norms)
            if stage_converged or (trace.gradient_norms and trace.gradient_norms[-1] < args.threshold):
                converged = True
                converged_stage = stage_index
                break
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    result = {
        **base,
        "n_stages_configured": len(variant.stages),
        "n_stages_run": max((row["stage_index"] for row in curve_rows), default=0),
        "cycles_run": len(all_gradients),
        "cycles_to_threshold": _first_cycle_below(all_gradients, args.threshold),
        "converged": bool(converged),
        "converged_stage": converged_stage,
        "initial_projected_gradient_norm": all_gradients[0] if all_gradients else None,
        "final_projected_gradient_norm": all_gradients[-1] if all_gradients else None,
        "best_projected_gradient_norm": min(all_gradients) if all_gradients else None,
        "final_total_energy": final_total_energy,
        "final_electronic_energy": final_electronic_energy,
        "elapsed_s": elapsed_s,
        "curve_status": _classify_curve(all_gradients),
        "error": error,
    }
    return result, curve_rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["run"], row["variant"]), []).append(row)
    summaries = []
    for (run, variant), group in sorted(grouped.items()):
        finite = [row for row in group if row["error"] is None]
        grads = [
            row["final_projected_gradient_norm"]
            for row in finite
            if row["final_projected_gradient_norm"] is not None
        ]
        best_grads = [
            row["best_projected_gradient_norm"]
            for row in finite
            if row["best_projected_gradient_norm"] is not None
        ]
        status_counts: dict[str, int] = {}
        for row in finite:
            status_counts[row["curve_status"]] = status_counts.get(row["curve_status"], 0) + 1
        summaries.append(
            {
                "run": run,
                "variant": variant,
                "n_points": len(group),
                "n_success": len(finite),
                "n_failed": len(group) - len(finite),
                "n_converged": sum(1 for row in finite if row["converged"]),
                "mean_final_projected_gradient_norm": (
                    float(sum(grads) / len(grads)) if grads else None
                ),
                "max_final_projected_gradient_norm": float(max(grads)) if grads else None,
                "mean_best_projected_gradient_norm": (
                    float(sum(best_grads) / len(best_grads)) if best_grads else None
                ),
                "max_best_projected_gradient_norm": float(max(best_grads)) if best_grads else None,
                "total_elapsed_s": float(sum(row["elapsed_s"] for row in finite)),
                "status_counts": status_counts,
            }
        )
    return summaries


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    run_specs = [_parse_run(item) for item in args.run]
    variants = [_parse_variant(item) for item in args.variant]
    contexts = {
        spec.name: _load_context(spec, args, device)
        for spec in run_specs
    }
    points = _read_bad_points(
        args.previous_optimization_csv,
        set(contexts),
        source_displacement=args.source_displacement,
        source_tolerance=args.source_tolerance,
    )
    points.sort(
        key=lambda point: (
            -(point.get("previous_cycles") or 0),
            point["run"],
            point["molecule_id"],
            point["coord_idx"],
            point["side"],
        )
    )
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    points = points[args.shard_index :: args.num_shards]
    if args.max_points is not None:
        points = points[: args.max_points]

    metadata = {
        "definition": (
            "Rerun only non-strict density-relaxed Hessian displacement points and "
            "record every optimization cycle."
        ),
        "previous_optimization_csv": args.previous_optimization_csv.as_posix(),
        "dataset_dir": args.dataset_dir.as_posix(),
        "device": str(device),
        "displacement": args.displacement,
        "source_displacement_filter": args.source_displacement,
        "source_tolerance_filter": args.source_tolerance,
        "threshold": args.threshold,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "runs": [
            {
                "name": spec.name,
                "run_dir": spec.run_dir.as_posix(),
                "ckpt": spec.ckpt.as_posix(),
            }
            for spec in run_specs
        ],
        "variants": [
            {
                "name": variant.name,
                "initialization": variant.initialization,
                "stages": [
                    {
                        "optimizer": stage.optimizer,
                        "lr": stage.lr,
                        "max_cycle": stage.max_cycle,
                    }
                    for stage in variant.stages
                ],
            }
            for variant in variants
        ],
        "n_points": len(points),
    }

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for point in points:
        context = contexts[point["run"]]
        for variant in variants:
            row, curves = _optimize_point(context, point, variant, args)
            if args.curve_dir is not None:
                curve_name = (
                    f"{row['run']}_{row['molecule_id']}_{row['sample_id']:07d}_"
                    f"coord{row['coord_idx']:03d}_{row['side']}_{row['variant']}.npz"
                )
                curve_path = args.curve_dir / curve_name
                _write_curve_npz(curves, curve_path)
                row["curve_file"] = curve_path.resolve().as_posix()
            rows.append(row)
            if args.curve_dir is None:
                curve_rows.extend(curves)
            partial_result = {
                **metadata,
                "partial": True,
                "completed_variant_runs": len(rows),
                "summary_rows": _summaries(rows),
                "rows": rows,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(partial_result, indent=2, sort_keys=True) + "\n"
            )
            if args.output_csv is not None:
                _write_csv(rows, args.output_csv)
            if args.summary_csv is not None:
                _write_csv(partial_result["summary_rows"], args.summary_csv)
            print(
                "completed",
                row["run"],
                row["molecule_id"],
                row["coord_idx"],
                row["side"],
                row["variant"],
                f"converged={row['converged']}",
                f"cycles={row['cycles_run']}",
                f"final_grad={row['final_projected_gradient_norm']:.6g}"
                if row["final_projected_gradient_norm"] is not None
                else "final_grad=None",
                f"elapsed_s={row['elapsed_s']:.1f}",
                flush=True,
            )

    result = {
        **metadata,
        "partial": False,
        "completed_variant_runs": len(rows),
        "summary_rows": _summaries(rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
    if args.curves_csv is not None:
        _write_csv(curve_rows, args.curves_csv)
    if args.summary_csv is not None:
        _write_csv(result["summary_rows"], args.summary_csv)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-optimization-csv", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--curves-csv", type=Path, default=None)
    parser.add_argument("--curve-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--source-displacement", type=float, default=None)
    parser.add_argument("--source-tolerance", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--slsqp-grad-scale", type=float, default=1.0)
    parser.add_argument("--trust-initial-radius", type=float, default=1.0)
    parser.add_argument("--trust-initial-constr-penalty", type=float, default=1.0)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
