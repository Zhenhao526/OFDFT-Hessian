#!/usr/bin/env python3
"""Train20 parent-CV pilot for a structured density-aware Hessian head."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.models.components.structured_density_hessian_head import (
    QM9_ELEMENTS,
    StructuredDensityHessianHead,
)
from mldft.ofdft.internal_directions import rigid_external_basis


@dataclass(frozen=True)
class Parent:
    molecule_id: str
    atomic_numbers: torch.Tensor
    positions_bohr: torch.Tensor
    density_raw: torch.Tensor
    density_features: torch.Tensor
    projector: torch.Tensor
    target_hessian: torch.Tensor

    @property
    def natoms(self) -> int:
        return int(self.atomic_numbers.numel())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _density_statistics(
    coefficients: np.ndarray,
    atomic_numbers: np.ndarray,
    basis_info: BasisInfo,
) -> np.ndarray:
    rows = []
    offset = 0
    for atomic_number in atomic_numbers:
        type_index = int(
            basis_info.atomic_number_to_atom_index[int(atomic_number)]
        )
        if type_index < 0:
            raise ValueError(f"unsupported atomic number {atomic_number}")
        count = int(basis_info.basis_dim_per_atom[type_index])
        values = np.asarray(
            coefficients[offset : offset + count], dtype=np.float64
        )
        if values.size != count:
            raise ValueError("density coefficient partition is incomplete")
        rows.append(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.sqrt(np.mean(values**2))),
                float(np.mean(np.abs(values))),
                float(np.max(np.abs(values))),
                float(np.sum(values) / np.sqrt(count)),
            ]
        )
        offset += count
    if offset != coefficients.size:
        raise ValueError(
            f"density coefficient partition mismatch: {offset} != {coefficients.size}"
        )
    return np.asarray(rows, dtype=np.float64)


def _load_parents(protocol: dict[str, Any]) -> list[Parent]:
    inputs = protocol["inputs"]
    dataset_root = Path(inputs["dataset_root"])
    hessian_manifest_path = Path(inputs["pbe_hessian_manifest"])
    if _sha256(hessian_manifest_path) != inputs["pbe_hessian_manifest_sha256"]:
        raise ValueError("PBE Hessian manifest hash drift")
    train_manifest = dataset_root / "provenance" / "train_only_dataset_manifest.json"
    if _sha256(train_manifest) != inputs["train_only_dataset_manifest_sha256"]:
        raise ValueError("clean train-only dataset manifest hash drift")
    manifest = json.loads(hessian_manifest_path.read_text())
    if (
        manifest["parent_set"] != "train20"
        or int(manifest["parent_count"]) != int(protocol["scope"]["parent_count"])
        or int(manifest["failed_count"]) != 0
    ):
        raise ValueError("PBE Hessian manifest boundary drift")
    basis_info = BasisInfo.from_dataset_info_yaml(
        str(dataset_root / "dataset_info.yaml"),
        np.asarray(QM9_ELEMENTS, dtype=np.uint8),
    )
    parents = []
    for row in manifest["parents"]:
        if not row["success"] or int(row["sample_id"]) != 0:
            raise ValueError("train20 Hessian row is not an accepted sample-0 result")
        molecule_id = str(row["molecule_id"])
        hessian_path = Path(row["cache_path"])
        if _sha256(hessian_path) != row["pbe_hessian_sha256"]:
            raise ValueError(f"PBE Hessian hash drift for {molecule_id}")
        label_path = (
            dataset_root
            / inputs["label_subdir"]
            / f"{molecule_id}.0000000.zarr.zip"
        )
        root = zarr.open(label_path, mode="r")
        atomic_numbers = np.asarray(
            root["geometry/atomic_numbers"], dtype=np.int64
        )
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        coefficients = np.asarray(
            root["of_labels/spatial/coeffs"][
                int(inputs["density_iteration"])
            ],
            dtype=np.float64,
        )
        density = _density_statistics(
            coefficients, atomic_numbers, basis_info
        )
        with np.load(hessian_path) as payload:
            hessian = np.asarray(payload["pbe_hessian"], dtype=np.float64)
        if hessian.shape != (positions.size, positions.size):
            raise ValueError(f"Hessian shape mismatch for {molecule_id}")
        external = rigid_external_basis(positions)
        projector = np.eye(positions.size) - external @ external.T
        target = projector @ (0.5 * (hessian + hessian.T)) @ projector
        parents.append(
            Parent(
                molecule_id=molecule_id,
                atomic_numbers=torch.from_numpy(atomic_numbers),
                positions_bohr=torch.from_numpy(positions),
                density_raw=torch.from_numpy(density),
                density_features=torch.from_numpy(density),
                projector=torch.from_numpy(projector),
                target_hessian=torch.from_numpy(target),
            )
        )
    if len(parents) != int(protocol["scope"]["parent_count"]):
        raise ValueError("loaded parent count differs from protocol")
    return parents


def _folds(parents: list[Parent], fold_count: int) -> list[list[int]]:
    order = sorted(
        range(len(parents)),
        key=lambda index: (
            -parents[index].natoms,
            tuple(
                -int(
                    torch.count_nonzero(
                        parents[index].atomic_numbers == element
                    )
                )
                for element in (9, 8, 7, 6, 1)
            ),
            parents[index].molecule_id,
        ),
    )
    folds = [[] for _ in range(fold_count)]
    for position, parent_index in enumerate(order):
        cycle, offset = divmod(position, fold_count)
        fold_index = offset if cycle % 2 == 0 else fold_count - 1 - offset
        folds[fold_index].append(parent_index)
    if sorted(index for fold in folds for index in fold) != list(range(len(parents))):
        raise RuntimeError("fold assignment is not a partition")
    return folds


def _normalize_density(
    train_parents: list[Parent], parents: list[Parent]
) -> tuple[list[Parent], dict[str, list[float]]]:
    train_values = torch.cat(
        [parent.density_raw for parent in train_parents], dim=0
    )
    mean = torch.mean(train_values, dim=0)
    scale = torch.std(train_values, dim=0, unbiased=False)
    scale = torch.where(scale < 1.0e-12, torch.ones_like(scale), scale)
    normalized = [
        replace(
            parent,
            density_features=torch.tanh(
                (parent.density_raw - mean) / (3.0 * scale)
            ),
        )
        for parent in parents
    ]
    return normalized, {"mean": mean.tolist(), "scale": scale.tolist()}


def _build_model(
    protocol: dict[str, Any], *, use_density: bool, device: torch.device
) -> StructuredDensityHessianHead:
    settings = protocol["model"]
    return StructuredDensityHessianHead(
        density_feature_dim=len(protocol["inputs"]["density_features"]),
        hidden_dim=int(settings["hidden_dim"]),
        radial_centers_bohr=tuple(settings["radial_centers_bohr"]),
        radial_width_bohr=float(settings["radial_width_bohr"]),
        environment_scale_bohr=float(settings["environment_scale_bohr"]),
        use_density=use_density,
        zero_initialize=True,
    ).to(dtype=torch.float64, device=device)


def _to_device(parent: Parent, device: torch.device) -> Parent:
    return replace(
        parent,
        atomic_numbers=parent.atomic_numbers.to(device=device),
        positions_bohr=parent.positions_bohr.to(device=device),
        density_raw=parent.density_raw.to(device=device),
        density_features=parent.density_features.to(device=device),
        projector=parent.projector.to(device=device),
        target_hessian=parent.target_hessian.to(device=device),
    )


def _relative_squared(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denominator = torch.sum(target**2).clamp_min(torch.finfo(target.dtype).tiny)
    return torch.sum((predicted - target) ** 2) / denominator


def _train(
    model: StructuredDensityHessianHead,
    parents: list[Parent],
    protocol: dict[str, Any],
) -> list[dict[str, float]]:
    settings = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history = []
    steps = int(settings["steps"])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for parent in parents:
            predicted = model(
                parent.atomic_numbers,
                parent.positions_bohr,
                parent.density_features,
                parent.projector,
            )
            losses.append(_relative_squared(predicted, parent.target_hessian))
        loss = torch.mean(torch.stack(losses))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite structured-head training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return history


def _metrics(
    model: StructuredDensityHessianHead,
    parent: Parent,
) -> dict[str, float]:
    with torch.no_grad():
        predicted = model(
            parent.atomic_numbers,
            parent.positions_bohr,
            parent.density_features,
            parent.projector,
        )
    target = parent.target_hessian
    relative = float(
        torch.linalg.matrix_norm(predicted - target)
        / torch.linalg.matrix_norm(target).clamp_min(
            torch.finfo(target.dtype).tiny
        )
    )
    external = torch.eye(
        parent.projector.shape[0],
        dtype=parent.projector.dtype,
        device=parent.projector.device,
    ) - parent.projector
    return {
        "relative_frobenius": relative,
        "mae_hartree_per_bohr2": float(torch.mean(torch.abs(predicted - target))),
        "symmetry_max_abs": float(torch.max(torch.abs(predicted - predicted.T))),
        "external_leakage_frobenius": float(
            torch.linalg.matrix_norm(predicted @ external)
        ),
        "predicted_frobenius": float(torch.linalg.matrix_norm(predicted)),
        "target_frobenius": float(torch.linalg.matrix_norm(target)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    relative = np.asarray(
        [float(row["relative_frobenius"]) for row in rows], dtype=np.float64
    )
    return {
        "count": int(relative.size),
        "mean_relative_frobenius": float(np.mean(relative)),
        "median_relative_frobenius": float(np.median(relative)),
        "p90_relative_frobenius": float(np.quantile(relative, 0.9)),
        "max_relative_frobenius": float(np.max(relative)),
        "fraction_better_than_zero_hessian": float(np.mean(relative < 1.0)),
        "max_symmetry_max_abs": float(
            max(float(row["symmetry_max_abs"]) for row in rows)
        ),
        "max_external_leakage_frobenius": float(
            max(float(row["external_leakage_frobenius"]) for row in rows)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol["protocol_id"] != "qm9_structured_density_hessian_head_pilot_v1":
        raise ValueError("unexpected pilot protocol")
    scope = protocol["scope"]
    if (
        scope["validation_access_allowed"] is not False
        or scope["test100_access_allowed"] is not False
        or int(scope["validation_evaluations_used"]) != 0
        or int(scope["test100_evaluations_used"]) != 0
    ):
        raise ValueError("pilot access boundary is open")
    if args.smoke:
        protocol = copy.deepcopy(protocol)
        protocol["training"]["steps"] = 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    parents = _load_parents(protocol)
    folds = _folds(parents, int(scope["fold_count"]))
    active_folds = folds[:1] if args.smoke else folds
    rows = []
    fold_summaries = []
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir()

    for variant_index, variant_name in enumerate(
        protocol["ablations"]["order"]
    ):
        use_density = bool(
            protocol["ablations"][variant_name]["use_density"]
        )
        for fold_index, held_indices in enumerate(active_folds):
            held_set = set(held_indices)
            train_original = [
                parent
                for index, parent in enumerate(parents)
                if index not in held_set
            ]
            normalized, density_normalization = _normalize_density(
                train_original, parents
            )
            train_parents = [
                _to_device(parent, device)
                for index, parent in enumerate(normalized)
                if index not in held_set
            ]
            held_parents = [
                _to_device(normalized[index], device) for index in held_indices
            ]
            torch.manual_seed(
                int(protocol["training"]["seed"])
                + 100 * variant_index
                + fold_index
            )
            model = _build_model(
                protocol, use_density=use_density, device=device
            )
            history = _train(model, train_parents, protocol)
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            checkpoint_path = (
                checkpoints / f"{variant_name}_fold{fold_index}.pt"
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": variant_name,
                    "fold": fold_index,
                    "held_parent_ids": [
                        parent.molecule_id for parent in held_parents
                    ],
                    "parameter_count": parameter_count,
                    "density_normalization": density_normalization,
                    "history": history,
                },
                checkpoint_path,
            )
            fold_rows = []
            for role, role_parents in (
                ("fit", train_parents),
                ("held", held_parents),
            ):
                for parent in role_parents:
                    row = {
                        "variant": variant_name,
                        "fold": fold_index,
                        "role": role,
                        "molecule_id": parent.molecule_id,
                        "natoms": parent.natoms,
                        "parameter_count": parameter_count,
                        **_metrics(model, parent),
                    }
                    rows.append(row)
                    fold_rows.append(row)
            fold_summaries.append(
                {
                    "variant": variant_name,
                    "fold": fold_index,
                    "parameter_count": parameter_count,
                    "checkpoint": checkpoint_path.as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "final_training_loss": history[-1]["loss"],
                    "fit": _aggregate(
                        [row for row in fold_rows if row["role"] == "fit"]
                    ),
                    "held": _aggregate(
                        [row for row in fold_rows if row["role"] == "held"]
                    ),
                }
            )

    _write_csv(args.output_dir / "per_parent_metrics.csv", rows)
    variant_summaries = {}
    for variant_name in protocol["ablations"]["order"]:
        fit_rows = [
            row
            for row in rows
            if row["variant"] == variant_name and row["role"] == "fit"
        ]
        held_rows = [
            row
            for row in rows
            if row["variant"] == variant_name and row["role"] == "held"
        ]
        variant_summaries[variant_name] = {
            "parameter_count": int(held_rows[0]["parameter_count"]),
            "fit": _aggregate(fit_rows),
            "held": _aggregate(held_rows),
        }
    geometry_median = variant_summaries["geometry_only"]["held"][
        "median_relative_frobenius"
    ]
    density_median = variant_summaries["geometry_plus_density"]["held"][
        "median_relative_frobenius"
    ]
    density_improvement = (geometry_median - density_median) / max(
        geometry_median, np.finfo(float).tiny
    )
    gates = protocol["gates"]
    density_summary = variant_summaries["geometry_plus_density"]
    gate_results = {
        "fit_median": (
            density_summary["fit"]["median_relative_frobenius"]
            <= float(gates["fit_median_relative_frobenius_max"])
        ),
        "held_median": (
            density_summary["held"]["median_relative_frobenius"]
            <= float(gates["held_median_relative_frobenius_max"])
        ),
        "held_p90": (
            density_summary["held"]["p90_relative_frobenius"]
            <= float(gates["held_p90_relative_frobenius_max"])
        ),
        "held_fraction_better_than_zero": (
            density_summary["held"]["fraction_better_than_zero_hessian"]
            >= float(gates["held_fraction_better_than_zero_baseline_min"])
        ),
        "density_improves_geometry": (
            density_improvement
            >= float(
                gates[
                    "density_relative_median_improvement_over_geometry_min"
                ]
            )
        ),
        "symmetry": (
            density_summary["held"]["max_symmetry_max_abs"]
            <= float(gates["symmetry_max_abs_max"])
        ),
        "external_leakage": (
            density_summary["held"]["max_external_leakage_frobenius"]
            <= float(gates["external_leakage_frobenius_max"])
        ),
    }
    gate_results["passed"] = all(gate_results.values()) and not args.smoke
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "parent_count": len(parents),
        "fold_count": len(active_folds),
        "smoke": bool(args.smoke),
        "device": str(device),
        "formal_gates_evaluated": not args.smoke,
        "variants": variant_summaries,
        "density_relative_median_improvement_over_geometry": density_improvement,
        "gates": gate_results,
        "folds": fold_summaries,
        "elapsed_s": time.perf_counter() - started,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    registration = {
        "summary": summary_path.as_posix(),
        "summary_sha256": _sha256(summary_path),
        "per_parent_metrics": (
            args.output_dir / "per_parent_metrics.csv"
        ).as_posix(),
        "per_parent_metrics_sha256": _sha256(
            args.output_dir / "per_parent_metrics.csv"
        ),
        "protocol_sha256": _sha256(args.protocol),
        "validation_accessed": False,
        "test100_accessed": False,
    }
    (args.output_dir / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
