#!/usr/bin/env python3
"""Build one-direction HVP sidecars from PBE Hessians without using Test100.

The trainable target is a baseline-anchored fixed-density surrogate:

    H_model,target v = H_PBE,total v - (H_baseline,fixed-total v - H_baseline,trainer v)

This correction is an optional audit only. It is invalid when the label density is not stationary
for the model, in which case training must use ``correction_mode=none`` and call the objective a
learned-energy fixed-density HVP surrogate against a PBE analytic total-HVP target. Strict
density-relaxed complete-total HVP remains an independent validation metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from pyscf.data import elements
from torch_geometric.data import Batch

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from qm9_hessian_vibrational_metrics import _external_basis, _vibrational_eigensystem
from qm9_hessian_density_relaxed_eval import _load_context, _parse_run
from qm9_total_ofdft_hvp_audit import _force_at_fixed_coefficients


KIND_CODES = {
    "paired": 0,
    "random_internal": 1,
    "bond_stretch": 2,
    "low_frequency": 3,
    "angle_bend": 4,
}
COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
BOHR_PER_ANGSTROM = 1.8897261254578281


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_datamodule(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.split_file = args.split_file.as_posix()
        cfg.data.datamodule.data_dir = args.data_dir.as_posix()
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = 0
        cfg.data.datamodule.shuffle_train = False
        cfg.data.datamodule.shuffle_val = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
        cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = True
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("fit")
    return datamodule


def _base_samples(datamodule: Any, requested: set[str]) -> dict[str, tuple[Any, Path]]:
    result: dict[str, tuple[Any, Path]] = {}
    for dataset in (datamodule.train_set, datamodule.val_set):
        for path_index, path in enumerate(dataset.paths):
            parts = path.name.removesuffix(".zarr.zip").split(".")
            molecule_id, sample_id = parts[0].zfill(7), int(parts[1])
            if molecule_id not in requested or sample_id != 0:
                continue
            item = int(dataset.path_indices[path_index])
            sample = dataset[item]
            result.setdefault(molecule_id, (sample, path))
    missing = sorted(requested.difference(result))
    if missing:
        raise KeyError(f"Missing base samples for {missing[:10]}")
    return result


def _project_internal(
    direction: np.ndarray, atomic_numbers: np.ndarray, positions_bohr: np.ndarray
) -> np.ndarray:
    masses = np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    mass_sqrt = np.repeat(np.sqrt(masses), 3)
    external = _external_basis(atomic_numbers, positions_bohr)
    mass_weighted = direction.reshape(-1) * mass_sqrt
    mass_weighted -= external @ (external.T @ mass_weighted)
    projected = mass_weighted / mass_sqrt
    norm = np.linalg.norm(projected)
    if norm <= 1e-12:
        raise ValueError("Direction vanished after translation/rotation projection")
    return (projected / norm).reshape(direction.shape)


def _paired_direction(args: argparse.Namespace, molecule_id: str) -> np.ndarray:
    geometries: dict[int, np.ndarray] = {}
    for sample_id in (1, 2):
        path = args.paired_label_dir / f"{molecule_id}.{sample_id:07d}.zarr.zip"
        root = zarr.open(path, mode="r")
        sign = int(root["metadata/reference/perturbation_pair_sign"][()])
        geometries[sign] = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
    if set(geometries) != {-1, 1}:
        raise ValueError(f"Incomplete paired direction for {molecule_id}")
    return geometries[1] - geometries[-1]


def _bond_direction(atomic_numbers: np.ndarray, positions_bohr: np.ndarray) -> np.ndarray:
    candidates = []
    for i in range(len(atomic_numbers)):
        for j in range(i + 1, len(atomic_numbers)):
            if int(atomic_numbers[i]) == 1 and int(atomic_numbers[j]) == 1:
                continue
            distance = float(np.linalg.norm(positions_bohr[j] - positions_bohr[i]))
            radii = COVALENT_RADII_ANGSTROM[int(atomic_numbers[i])] + COVALENT_RADII_ANGSTROM[
                int(atomic_numbers[j])
            ]
            candidates.append((distance / radii, i, j))
    if not candidates:
        raise ValueError("No non-H-H pair available for bond direction")
    _, i, j = min(candidates)
    axis = positions_bohr[j] - positions_bohr[i]
    axis /= np.linalg.norm(axis)
    direction = np.zeros_like(positions_bohr)
    direction[i] = -axis
    direction[j] = axis
    return direction


def _bonded_neighbors(
    atomic_numbers: np.ndarray, positions_bohr: np.ndarray
) -> list[list[int]]:
    neighbors = [[] for _ in atomic_numbers]
    for i in range(len(atomic_numbers)):
        for j in range(i + 1, len(atomic_numbers)):
            cutoff = 1.25 * BOHR_PER_ANGSTROM * (
                COVALENT_RADII_ANGSTROM[int(atomic_numbers[i])]
                + COVALENT_RADII_ANGSTROM[int(atomic_numbers[j])]
            )
            if np.linalg.norm(positions_bohr[j] - positions_bohr[i]) <= cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def _angle_direction(
    atomic_numbers: np.ndarray, positions_bohr: np.ndarray
) -> np.ndarray:
    """Return the Cartesian gradient of a well-conditioned bonded angle."""
    neighbors = _bonded_neighbors(atomic_numbers, positions_bohr)
    candidates = []
    for center, bonded in enumerate(neighbors):
        for left_index in range(len(bonded)):
            for right_index in range(left_index + 1, len(bonded)):
                left = bonded[left_index]
                right = bonded[right_index]
                u_raw = positions_bohr[left] - positions_bohr[center]
                v_raw = positions_bohr[right] - positions_bohr[center]
                u_norm = np.linalg.norm(u_raw)
                v_norm = np.linalg.norm(v_raw)
                cosine = float(np.dot(u_raw, v_raw) / (u_norm * v_norm))
                sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
                if sine > 0.2:
                    candidates.append((sine, center, left, right, u_raw, v_raw))
    if not candidates:
        raise ValueError("No well-conditioned bonded angle available")
    _, center, left, right, u_raw, v_raw = max(candidates)
    u_norm = np.linalg.norm(u_raw)
    v_norm = np.linalg.norm(v_raw)
    u = u_raw / u_norm
    v = v_raw / v_norm
    cosine = float(np.dot(u, v))
    sine = float(np.sqrt(max(1e-16, 1.0 - cosine * cosine)))
    direction = np.zeros_like(positions_bohr)
    direction[left] = -(v - cosine * u) / (u_norm * sine)
    direction[right] = -(u - cosine * v) / (v_norm * sine)
    direction[center] = -direction[left] - direction[right]
    return direction


def _low_frequency_direction(
    hessian: np.ndarray, atomic_numbers: np.ndarray, positions_bohr: np.ndarray
) -> np.ndarray:
    external = _external_basis(atomic_numbers, positions_bohr)
    frequencies, modes = _vibrational_eigensystem(hessian, atomic_numbers, external)
    positive = np.where(frequencies > 20.0)[0]
    index = int(positive[np.argmin(frequencies[positive])]) if positive.size else int(np.argmin(np.abs(frequencies)))
    masses = np.repeat(
        np.sqrt(np.asarray([elements.MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)),
        3,
    )
    return (modes[:, index] / masses).reshape(-1, 3)


def _direction_kind(index: int, allow_paired: bool) -> str:
    cycle = index % 10
    if allow_paired and cycle < 4:
        return "paired"
    if cycle < 7:
        return "random_internal"
    if cycle < 9:
        return "bond_stretch"
    return "low_frequency"


def _direction(
    args: argparse.Namespace,
    index: int,
    molecule_id: str,
    atomic_numbers: np.ndarray,
    positions_bohr: np.ndarray,
    pbe_hessian: np.ndarray,
    allow_paired: bool,
    direction_slot: int = 0,
    directions_per_parent: int = 1,
) -> tuple[str, np.ndarray]:
    if directions_per_parent == 1:
        stable_index = int(hashlib.sha256(molecule_id.encode()).hexdigest()[:8], 16)
        kind = _direction_kind(stable_index, allow_paired)
    else:
        sequence = ("random_internal", "bond_stretch", "angle_bend", "low_frequency")
        kind = sequence[direction_slot % len(sequence)]
    if kind == "paired":
        raw = _paired_direction(args, molecule_id)
    elif kind == "random_internal":
        generator = np.random.default_rng(
            args.seed + 1009 * int(molecule_id) + 9176 * direction_slot
        )
        raw = generator.normal(size=positions_bohr.shape)
    elif kind == "bond_stretch":
        raw = _bond_direction(atomic_numbers, positions_bohr)
    elif kind == "angle_bend":
        try:
            raw = _angle_direction(atomic_numbers, positions_bohr)
        except ValueError:
            generator = np.random.default_rng(
                args.seed + 1009 * int(molecule_id) + 9176 * direction_slot
            )
            raw = generator.normal(size=positions_bohr.shape)
            kind = "random_internal"
    else:
        raw = _low_frequency_direction(pbe_hessian, atomic_numbers, positions_bohr)
    return kind, _project_internal(raw, atomic_numbers, positions_bohr)


def _trainer_hvp(model: Any, sample: Any, direction: np.ndarray, device: torch.device) -> np.ndarray:
    loader = OFLoader(
        [sample],
        batch_size=1,
        shuffle=False,
        follow_batch=["coeffs", "atomic_numbers"],
        list_keys=["overlap_matrix"],
    )
    batch: Batch = next(iter(loader)).to(device)
    batch.pos.requires_grad_(True)
    energy, _ = model.net(batch)
    gradient = torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
    vector = torch.as_tensor(direction, dtype=gradient.dtype, device=gradient.device)
    hvp = torch.autograd.grad(torch.sum(gradient * vector), batch.pos)[0]
    return hvp.detach().cpu().numpy()


def build(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    references = [row for row in json.loads(args.reference_manifest.read_text()) if row.get("success")]
    if args.molecules:
        wanted = {item.strip().zfill(7) for item in args.molecules.split(",") if item.strip()}
        references = [row for row in references if row["molecule_id"] in wanted]
    references = sorted(references, key=lambda row: row["molecule_id"])
    requested = {row["molecule_id"] for row in references}
    datamodule = _load_datamodule(args)
    samples = _base_samples(datamodule, requested)
    context = _load_context(
        _parse_run(f"Baseline={args.run_dir}={args.checkpoint}"), args, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    train_ids = set(args.train_parent_ids.read_text().split()) if args.train_parent_ids else set()

    for index, reference in enumerate(references):
        molecule_id = reference["molecule_id"]
        sample, label_path = samples[molecule_id]
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        coefficients = torch.as_tensor(
            np.asarray(root["of_labels/spatial/coeffs"][-1], dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        pbe_hessian = np.asarray(np.load(reference["cache_path"])["pbe_hessian"], dtype=np.float64)
        direction_rows = []
        directions = []
        pbe_hvps = []
        trainer_hvps = []
        fixed_total_hvps = []
        corrections = []
        model_targets = []
        kinds = []
        point_metadata_by_direction = []
        for direction_slot in range(args.directions_per_parent):
            kind, direction = _direction(
                args,
                index,
                molecule_id,
                atomic_numbers,
                positions,
                pbe_hessian,
                allow_paired=molecule_id in train_ids,
                direction_slot=direction_slot,
                directions_per_parent=args.directions_per_parent,
            )
            pbe_hvp = (pbe_hessian @ direction.reshape(-1)).reshape(direction.shape)
            baseline_trainer_hvp = _trainer_hvp(context.model, sample, direction, device)

            if args.correction_mode == "baseline_anchored_fixed_total":
                forces = {}
                point_metadata = {}
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    force, metadata = _force_at_fixed_coefficients(
                        context,
                        atomic_numbers,
                        positions + sign * args.hvp_step * direction,
                        coefficients,
                        args,
                    )
                    forces[side] = force
                    point_metadata[side] = metadata
                fixed_total_hvp = -(
                    forces["plus"] - forces["minus"]
                ) / (2.0 * args.hvp_step)
                correction = fixed_total_hvp - baseline_trainer_hvp
                correction_ratio = float(
                    np.linalg.norm(correction)
                    / max(np.linalg.norm(pbe_hvp), np.finfo(float).tiny)
                )
                if correction_ratio > args.max_correction_to_reference_ratio:
                    raise RuntimeError(
                        f"Rejected nonstationary fixed-total correction for {molecule_id}: "
                        f"direction={direction_slot}, ratio={correction_ratio:.3e}, "
                        f"max={args.max_correction_to_reference_ratio:.3e}"
                    )
            else:
                fixed_total_hvp = baseline_trainer_hvp.copy()
                correction = np.zeros_like(baseline_trainer_hvp)
                point_metadata = {}
            model_target = pbe_hvp - correction
            directions.append(direction)
            pbe_hvps.append(pbe_hvp)
            trainer_hvps.append(baseline_trainer_hvp)
            fixed_total_hvps.append(fixed_total_hvp)
            corrections.append(correction)
            model_targets.append(model_target)
            kinds.append(kind)
            point_metadata_by_direction.append(point_metadata)
            direction_rows.append(
                {
                    "direction_index": direction_slot,
                    "direction_kind": kind,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "pbe_hvp_rms": float(np.sqrt(np.mean(pbe_hvp**2))),
                    "baseline_trainer_hvp_rms": float(
                        np.sqrt(np.mean(baseline_trainer_hvp**2))
                    ),
                    "baseline_fixed_total_hvp_rms": float(
                        np.sqrt(np.mean(fixed_total_hvp**2))
                    ),
                    "correction_hvp_rms": float(np.sqrt(np.mean(correction**2))),
                    "model_target_hvp_rms": float(np.sqrt(np.mean(model_target**2))),
                }
            )
        directions_array = np.stack(directions)
        pbe_hvp_array = np.stack(pbe_hvps)
        trainer_hvp_array = np.stack(trainer_hvps)
        fixed_total_hvp_array = np.stack(fixed_total_hvps)
        correction_array = np.stack(corrections)
        model_target_array = np.stack(model_targets)
        kind_array = np.asarray(kinds)
        kind_code_array = np.asarray([KIND_CODES[kind] for kind in kinds], dtype=np.int64)
        if args.directions_per_parent == 1:
            directions_array = directions_array[0]
            pbe_hvp_array = pbe_hvp_array[0]
            trainer_hvp_array = trainer_hvp_array[0]
            fixed_total_hvp_array = fixed_total_hvp_array[0]
            correction_array = correction_array[0]
            model_target_array = model_target_array[0]
            kind_array = kind_array[0]
            kind_code_array = kind_code_array[0]
        sidecar = args.output_dir / f"{molecule_id}.0000000.npz"
        np.savez_compressed(
            sidecar,
            atomic_numbers=atomic_numbers,
            positions_bohr=positions,
            direction=directions_array,
            direction_kind=kind_array,
            direction_kind_code=kind_code_array,
            complete_total_reference_hvp=pbe_hvp_array,
            baseline_trainer_hvp=trainer_hvp_array,
            baseline_fixed_total_hvp=fixed_total_hvp_array,
            fixed_density_correction_hvp=correction_array,
            model_hvp_target=model_target_array,
            stability_mask=np.ones(args.directions_per_parent, dtype=np.bool_),
            pbe_hessian_path=np.asarray(reference["cache_path"]),
        )
        row = {
            "molecule_id": molecule_id,
            "natoms": int(atomic_numbers.size),
            "direction_count": args.directions_per_parent,
            "directions": direction_rows,
            "sidecar": sidecar.as_posix(),
            "sidecar_sha256": _sha256(sidecar),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    manifest = {
        "definition": __doc__.strip(),
        "training_label_definition": (
            "learned-energy fixed-density direct HVP matched to PBE analytic total H.v"
            if args.correction_mode == "none"
            else "baseline-anchored fixed-density complete-total residual surrogate"
        ),
        "correction_mode": args.correction_mode,
        "reference_manifest": args.reference_manifest.as_posix(),
        "reference_manifest_sha256": _sha256(args.reference_manifest),
        "split_file": args.split_file.as_posix(),
        "split_file_sha256": _sha256(args.split_file),
        "baseline_run_dir": args.run_dir.as_posix(),
        "baseline_checkpoint": args.checkpoint.as_posix(),
        "hvp_step_bohr": args.hvp_step,
        "integral_derivative_step_bohr": args.integral_derivative_step,
        "seed": args.seed,
        "directions_per_parent": args.directions_per_parent,
        "count": len(rows),
        "test_accessed": False,
        "rows": rows,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "rows"}, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--paired-label-dir", type=Path, required=True)
    parser.add_argument("--train-parent-ids", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecules", default=None)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--directions-per-parent", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hvp-step", type=float, default=3e-5)
    parser.add_argument("--integral-derivative-step", type=float, default=1e-4)
    parser.add_argument("--integral-derivative-workers", type=int, default=4)
    parser.add_argument("--model-geometry-derivative", choices=["autograd", "numerical"], default="autograd")
    parser.add_argument("--model-geometry-fd-step", type=float, default=1e-6)
    parser.add_argument("--model-geometry-fd-richardson", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--correction-mode",
        choices=["baseline_anchored_fixed_total", "none"],
        default="baseline_anchored_fixed_total",
    )
    parser.add_argument("--max-correction-to-reference-ratio", type=float, default=100.0)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--negative-integrated-density-penalty-weight", type=float, default=0.0)
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-cycle", type=int, default=1)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
