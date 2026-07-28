#!/usr/bin/env python3
"""Build symmetric relaxed scalar-energy Hessian blocks for force-Hessian auditing."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation

from qm9_hessian_density_relaxed_eval import (
    _load_context,
    _load_geometry,
    _make_mol,
    _optimize_density,
    _parse_run,
    _save_optimization_trace,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_force_and_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return (
        np.asarray(payload["density_relaxed_hessian"], dtype=np.float64),
        np.asarray(payload["pbe_hessian"], dtype=np.float64),
    )


def _largest_asymmetric_pair(hessian: np.ndarray) -> tuple[int, int, float]:
    asymmetry = np.abs(hessian - hessian.T)
    np.fill_diagonal(asymmetry, -np.inf)
    i, j = np.unravel_index(int(np.argmax(asymmetry)), asymmetry.shape)
    return int(i), int(j), float(asymmetry[i, j])


def _model_energy(context: Any, sample: Any) -> float:
    sample.coeffs = sample.coeffs.detach().clone()
    with torch.no_grad():
        pred_energy, _ = context.model.net(sample)
    return float(pred_energy.sum().detach().cpu())


def _optimize_energy_point(
    context: Any,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    base_coeffs: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any], Any]:
    mol = _make_mol(atomic_numbers, positions_bohr, args.charge)
    sample = context.sample_generator.get_sample_from_mol(mol)
    if base_coeffs is None:
        initialization: str | torch.Tensor = args.initialization
        initialization_mode = args.initialization
    else:
        coeffs = base_coeffs.to(device=sample.coeffs.device, dtype=sample.coeffs.dtype)
        initialization = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
        initialization_mode = "base_density_warm_start"
    total_energies, final_coeffs, metadata, trace = _optimize_density(
        context,
        sample,
        args,
        initialization,
        initialization_mode,
    )
    return (
        {
            "total_energy": float(total_energies.total_energy),
            "model_energy": _model_energy(context, sample),
        },
        {**metadata, "final_coeffs": final_coeffs},
        trace,
    )


def _energy_hessian_block(
    energies: dict[tuple[int, int], float], displacement: float
) -> np.ndarray:
    h2 = displacement**2
    e0 = energies[(0, 0)]
    h_ii = (energies[(1, 0)] - 2.0 * e0 + energies[(-1, 0)]) / h2
    h_jj = (energies[(0, 1)] - 2.0 * e0 + energies[(0, -1)]) / h2
    h_ij = (
        energies[(1, 1)]
        - energies[(1, -1)]
        - energies[(-1, 1)]
        + energies[(-1, -1)]
    ) / (4.0 * h2)
    return np.asarray([[h_ii, h_ij], [h_ij, h_jj]], dtype=np.float64)


def _block_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = candidate - reference
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_fro": (
            float(np.linalg.norm(diff) / np.linalg.norm(reference))
            if np.linalg.norm(reference)
            else float("nan")
        ),
        "symmetry_max_abs": float(np.max(np.abs(candidate - candidate.T))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    references = [
        row
        for row in json.loads(args.manifest_json.read_text())
        if row.get("success") and Path(row["cache_path"]).exists()
    ]
    if args.molecule_id:
        selected = set(args.molecule_id)
        references = [row for row in references if row["molecule_id"] in selected]
    references = references[: args.max_molecules]
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "blocks").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    signs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for context in contexts:
        for ref in references:
            t0 = time.time()
            molecule_id = ref["molecule_id"]
            sample_id = int(ref["sample_id"])
            atomic_numbers, positions_bohr = _load_geometry(
                args.dataset_dir, molecule_id, sample_id
            )
            force_path = args.force_hessian_dir / (
                f"{context.spec.name}_{molecule_id}_{sample_id:07d}_density_relaxed_hessian.npz"
            )
            force_hessian, pbe_hessian = _load_force_and_reference(force_path)
            coord_i, coord_j, max_asymmetry = _largest_asymmetric_pair(force_hessian)
            energy_values: dict[str, dict[tuple[int, int], float]] = {
                "total_energy": {},
                "model_energy": {},
            }
            base_coeffs = None
            for sign_i, sign_j in signs:
                flat = positions_bohr.reshape(-1).copy()
                flat[coord_i] += sign_i * args.displacement
                flat[coord_j] += sign_j * args.displacement
                energies, metadata, trace = _optimize_energy_point(
                    context,
                    atomic_numbers,
                    flat.reshape(positions_bohr.shape),
                    base_coeffs,
                    args,
                )
                if sign_i == 0 and sign_j == 0:
                    base_coeffs = metadata.pop("final_coeffs").detach().clone()
                else:
                    metadata.pop("final_coeffs")
                point_name = f"si_{sign_i:+d}_sj_{sign_j:+d}"
                metadata["trace_file"] = _save_optimization_trace(
                    args.optimization_trace_dir,
                    context,
                    molecule_id,
                    sample_id,
                    None,
                    f"energy_block_{point_name}",
                    trace,
                    metadata,
                )
                for energy_name, value in energies.items():
                    energy_values[energy_name][(sign_i, sign_j)] = value
                point_rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "sample_id": sample_id,
                        "natoms": int(ref["natoms"]),
                        "coord_i": coord_i,
                        "coord_j": coord_j,
                        "sign_i": sign_i,
                        "sign_j": sign_j,
                        "displacement": args.displacement,
                        **energies,
                        **metadata,
                    }
                )

            total_block = _energy_hessian_block(
                energy_values["total_energy"], args.displacement
            )
            model_block = _energy_hessian_block(
                energy_values["model_energy"], args.displacement
            )
            coordinate_indices = np.asarray([coord_i, coord_j], dtype=np.int64)
            force_block = force_hessian[np.ix_(coordinate_indices, coordinate_indices)]
            force_sym_block = 0.5 * (force_block + force_block.T)
            pbe_block = pbe_hessian[np.ix_(coordinate_indices, coordinate_indices)]
            block_path = args.output_dir / "blocks" / (
                f"{context.spec.name}_{molecule_id}_{sample_id:07d}_relaxed_energy_block.npz"
            )
            np.savez_compressed(
                block_path,
                coordinate_indices=coordinate_indices,
                total_energy_hessian_block=total_block,
                relaxed_model_energy_hessian_block=model_block,
                derived_force_hessian_block=force_block,
                symmetrized_derived_force_hessian_block=force_sym_block,
                pbe_hessian_block=pbe_block,
            )
            comparison_pairs = {
                "total_energy_vs_pbe": (total_block, pbe_block),
                "relaxed_model_energy_vs_pbe": (model_block, pbe_block),
                "derived_force_vs_pbe": (force_block, pbe_block),
                "sym_derived_force_vs_pbe": (force_sym_block, pbe_block),
                "derived_force_vs_total_energy": (force_block, total_block),
                "sym_derived_force_vs_total_energy": (force_sym_block, total_block),
                "derived_force_vs_relaxed_model_energy": (force_block, model_block),
            }
            molecule_points = [
                row
                for row in point_rows
                if row["run"] == context.spec.name and row["molecule_id"] == molecule_id
            ]
            for comparison, (candidate, reference) in comparison_pairs.items():
                rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "sample_id": sample_id,
                        "natoms": int(ref["natoms"]),
                        "coord_i": coord_i,
                        "coord_j": coord_j,
                        "baseline_force_hessian_max_asymmetry": max_asymmetry,
                        "displacement": args.displacement,
                        "comparison": comparison,
                        "strict_converged_points": sum(
                            1
                            for point in molecule_points
                            if point.get("final_gradient_norm") is not None
                            and float(point["final_gradient_norm"])
                            < args.fallback_convergence_tolerance
                        ),
                        "max_final_gradient_norm": max(
                            float(point["final_gradient_norm"])
                            for point in molecule_points
                            if point.get("final_gradient_norm") is not None
                        ),
                        "block_npz": block_path.as_posix(),
                        "elapsed_s": time.time() - t0,
                        **_block_metrics(candidate, reference),
                    }
                )
            print(
                "completed",
                context.spec.name,
                molecule_id,
                f"pair=({coord_i},{coord_j})",
                f"force_asym={max_asymmetry:.6e}",
                f"force_vs_total_mae={rows[-2]['mae']:.6e}",
                flush=True,
            )

    result = {
        "definition": (
            "A naturally symmetric 2x2 nuclear-coordinate Hessian block from central second "
            "differences of fully density-relaxed scalar energies. total_energy includes learned "
            "kin_plus_xc, Hartree, electron-nuclear attraction, and nuclear repulsion; model_energy "
            "tracks only the learned scalar along the relaxed density path."
        ),
        "manifest_json": args.manifest_json.as_posix(),
        "force_hessian_dir": args.force_hessian_dir.as_posix(),
        "displacement": args.displacement,
        "target_tolerance": args.fallback_convergence_tolerance,
        "rows": rows,
        "optimization_points": point_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "block_metrics.csv", rows)
    _write_csv(args.output_dir / "optimization_points.csv", point_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--force-hessian-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimization-trace-dir", type=Path, default=None)
    parser.add_argument("--molecule-id", action="append", default=[])
    parser.add_argument("--max-molecules", type=int, default=3)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument(
        "--normalize-initial-guess", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ks-basis", default="6-31G(2df,p)")
    parser.add_argument("--max-xc-memory", type=int, default=4000)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--max-cycle", type=int, default=1000)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fallback-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--fallback-max-cycle", type=int, default=10000)
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--fallback-lr", type=float, default=3e-4)
    parser.add_argument(
        "--fallback-always", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--base-density-warm-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
