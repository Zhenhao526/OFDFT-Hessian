#!/usr/bin/env python3
"""Audit fixed- and relaxed-density model-force conservativity on 2-D coordinate loops."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qm9_hessian_density_relaxed_eval import (
    _load_context,
    _load_geometry,
    _make_mol,
    _optimize_density,
    _parse_run,
    _save_optimization_trace,
    _density_relaxed_force,
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


def _baseline_hessian(path: Path) -> np.ndarray:
    payload = np.load(path)
    return np.asarray(payload["density_relaxed_hessian"], dtype=np.float64)


def _largest_asymmetric_pair(hessian: np.ndarray) -> tuple[int, int, float]:
    asymmetry = np.abs(hessian - hessian.T)
    np.fill_diagonal(asymmetry, -np.inf)
    i, j = np.unravel_index(int(np.argmax(asymmetry)), asymmetry.shape)
    return int(i), int(j), float(asymmetry[i, j])


def _fixed_density_force(context: Any, sample: Any, positions_bohr: np.ndarray) -> np.ndarray:
    sample.pos = torch.as_tensor(
        positions_bohr,
        dtype=sample.pos.dtype,
        device=sample.pos.device,
    ).detach().clone().requires_grad_(True)
    sample.coeffs = sample.coeffs.detach().clone()
    with torch.enable_grad():
        _, _, _, force = context.model.forward_predictions(
            sample,
            compute_density_gradients=False,
            compute_forces=True,
        )
    if force is None:
        raise RuntimeError("model.forward_predictions returned pred_forces=None")
    return force.detach().cpu().numpy()


def _loop_points(
    positions_bohr: np.ndarray, coord_i: int, coord_j: int, half_width: float
) -> list[tuple[str, np.ndarray]]:
    flat = positions_bohr.reshape(-1)
    corners = []
    for name, sign_i, sign_j in (
        ("minus_minus", -1.0, -1.0),
        ("plus_minus", 1.0, -1.0),
        ("plus_plus", 1.0, 1.0),
        ("minus_plus", -1.0, 1.0),
    ):
        point = flat.copy()
        point[coord_i] += sign_i * half_width
        point[coord_j] += sign_j * half_width
        corners.append((name, point.reshape(positions_bohr.shape)))
    return corners


def _loop_work(
    points: list[tuple[str, np.ndarray]], forces: dict[str, np.ndarray]
) -> tuple[float, list[float]]:
    edge_work = []
    for edge_idx in range(4):
        name_a, position_a = points[edge_idx]
        name_b, position_b = points[(edge_idx + 1) % 4]
        displacement = position_b.reshape(-1) - position_a.reshape(-1)
        average_force = 0.5 * (
            forces[name_a].reshape(-1) + forces[name_b].reshape(-1)
        )
        edge_work.append(float(np.dot(average_force, displacement)))
    return float(sum(edge_work)), edge_work


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    manifest = [
        row
        for row in json.loads(args.manifest_json.read_text())
        if row.get("success") and Path(row["cache_path"]).exists()
    ]
    if args.molecule_id:
        selected = set(args.molecule_id)
        manifest = [row for row in manifest if row["molecule_id"] in selected]
    manifest = manifest[: args.max_molecules]
    contexts = [_load_context(_parse_run(item), args, device) for item in args.run]
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "forces").mkdir(exist_ok=True)

    for context in contexts:
        for ref in manifest:
            molecule_id = ref["molecule_id"]
            sample_id = int(ref["sample_id"])
            t0 = time.time()
            atomic_numbers, positions_bohr = _load_geometry(
                args.dataset_dir, molecule_id, sample_id
            )
            baseline_path = args.baseline_hessian_dir / (
                f"{context.spec.name}_{molecule_id}_{sample_id:07d}_density_relaxed_hessian.npz"
            )
            if not baseline_path.exists():
                raise FileNotFoundError(baseline_path)
            coord_i, coord_j, baseline_max_asymmetry = _largest_asymmetric_pair(
                _baseline_hessian(baseline_path)
            )

            base_mol = _make_mol(atomic_numbers, positions_bohr, args.charge)
            base_sample = context.sample_generator.get_sample_from_mol(base_mol)
            _, base_coeffs, base_metadata, base_trace = _optimize_density(
                context,
                base_sample,
                args,
                args.initialization,
                args.initialization,
            )
            base_metadata["trace_file"] = _save_optimization_trace(
                args.optimization_trace_dir,
                context,
                molecule_id,
                sample_id,
                None,
                "base",
                base_trace,
                base_metadata,
            )
            points = _loop_points(positions_bohr, coord_i, coord_j, args.half_width)
            method_forces: dict[str, dict[str, np.ndarray]] = {
                "fixed_density": {},
                "relaxed_density": {},
            }
            for corner_name, corner_positions in points:
                fixed_force = _fixed_density_force(context, base_sample, corner_positions)
                method_forces["fixed_density"][corner_name] = fixed_force
                relaxed_force, metadata, trace = _density_relaxed_force(
                    context,
                    atomic_numbers,
                    corner_positions,
                    args,
                    base_coeffs_untransformed=base_coeffs,
                )
                metadata["trace_file"] = _save_optimization_trace(
                    args.optimization_trace_dir,
                    context,
                    molecule_id,
                    sample_id,
                    None,
                    f"loop_{corner_name}",
                    trace,
                    metadata,
                )
                method_forces["relaxed_density"][corner_name] = relaxed_force
                point_rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "sample_id": sample_id,
                        "natoms": int(ref["natoms"]),
                        "corner": corner_name,
                        "coord_i": coord_i,
                        "coord_j": coord_j,
                        "half_width": args.half_width,
                        **metadata,
                    }
                )

            force_path = args.output_dir / "forces" / (
                f"{context.spec.name}_{molecule_id}_{sample_id:07d}_loop_forces.npz"
            )
            np.savez_compressed(
                force_path,
                coord_i=np.asarray(coord_i),
                coord_j=np.asarray(coord_j),
                positions=np.stack([position for _, position in points]),
                fixed_forces=np.stack(
                    [method_forces["fixed_density"][name] for name, _ in points]
                ),
                relaxed_forces=np.stack(
                    [method_forces["relaxed_density"][name] for name, _ in points]
                ),
            )
            for method, forces in method_forces.items():
                loop_work, edge_work = _loop_work(points, forces)
                area = 4.0 * args.half_width**2
                relevant_points = [
                    row
                    for row in point_rows
                    if row["run"] == context.spec.name
                    and row["molecule_id"] == molecule_id
                ]
                rows.append(
                    {
                        "run": context.spec.name,
                        "molecule_id": molecule_id,
                        "sample_id": sample_id,
                        "natoms": int(ref["natoms"]),
                        "method": method,
                        "coord_i": coord_i,
                        "coord_j": coord_j,
                        "baseline_max_symmetry_abs": baseline_max_asymmetry,
                        "half_width": args.half_width,
                        "loop_area_bohr2": area,
                        "loop_work_hartree": loop_work,
                        "abs_loop_work_hartree": abs(loop_work),
                        "curl_estimate_hartree_per_bohr2": loop_work / area,
                        "edge_work_0": edge_work[0],
                        "edge_work_1": edge_work[1],
                        "edge_work_2": edge_work[2],
                        "edge_work_3": edge_work[3],
                        "strict_converged_corners": (
                            sum(
                                1
                                for point in relevant_points
                                if point.get("final_gradient_norm") is not None
                                and float(point["final_gradient_norm"])
                                < args.fallback_convergence_tolerance
                            )
                            if method == "relaxed_density"
                            else None
                        ),
                        "max_final_gradient_norm": (
                            max(
                                float(point["final_gradient_norm"])
                                for point in relevant_points
                                if point.get("final_gradient_norm") is not None
                            )
                            if method == "relaxed_density"
                            else None
                        ),
                        "base_converged": bool(base_metadata["converged"]),
                        "base_final_gradient_norm": base_metadata["final_gradient_norm"],
                        "force_npz": force_path.as_posix(),
                        "elapsed_s": time.time() - t0,
                    }
                )
            print(
                "completed",
                context.spec.name,
                molecule_id,
                f"pair=({coord_i},{coord_j})",
                f"fixed_work={rows[-2]['loop_work_hartree']:.6e}",
                f"relaxed_work={rows[-1]['loop_work_hartree']:.6e}",
                flush=True,
            )

    result = {
        "definition": (
            "Trapezoidal work around a square nuclear-coordinate loop. Fixed density holds the "
            "base optimized model state fixed; relaxed density rebuilds integrals and optimizes "
            "density at every corner before evaluating the incomplete model-only derived force."
        ),
        "manifest_json": args.manifest_json.as_posix(),
        "baseline_hessian_dir": args.baseline_hessian_dir.as_posix(),
        "half_width": args.half_width,
        "target_tolerance": args.fallback_convergence_tolerance,
        "rows": rows,
        "optimization_points": point_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "loop_metrics.csv", rows)
    _write_csv(args.output_dir / "optimization_points.csv", point_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-hessian-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimization-trace-dir", type=Path, default=None)
    parser.add_argument("--molecule-id", action="append", default=[])
    parser.add_argument("--max-molecules", type=int, default=3)
    parser.add_argument("--half-width", type=float, default=1e-3)
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
    parser.add_argument("--fallback-convergence-tolerance", type=float, default=1e-4)
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
