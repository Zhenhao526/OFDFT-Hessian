#!/usr/bin/env python3
"""Train a structured Hessian attention readout on frozen Graphformer states."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.graphformer_structured_hessian_attention_head import (
    GraphformerStructuredHessianAttentionHead,
)
from mldft.ml.models.mldft_module import MLDFTLitModule
from qm9_structured_density_hessian_head_pilot import (
    Parent,
    _aggregate,
    _folds,
    _load_parents,
    _sha256,
    _to_device,
)


def _load_cfg(protocol: dict[str, Any]) -> Any:
    source_run = Path(protocol["inputs"]["source_run_dir"])
    dataset_root = Path(protocol["inputs"]["dataset_root"])
    os.environ.setdefault("DFT_DATA", dataset_root.parent.as_posix())
    cfg = OmegaConf.load(source_run / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = 0
        cfg.data.datamodule.shuffle_train = False
        cfg.data.datamodule.pair_grouped_train_batches = False
        cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]
        cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = False
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = False
        cfg.data.datamodule.dataset_kwargs.load_hvp_label = False
    return cfg


def _selected_train_indices(
    dataset: Any, parent_ids: set[str]
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    keys: list[str] = []
    for path_index, (path, scf_iterations) in enumerate(
        zip(dataset.paths, dataset.scf_iterations_per_path)
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_text = label_name.split(".")
        molecule_id = molecule_id.zfill(7)
        if molecule_id not in parent_ids or int(sample_text) != 0:
            continue
        path_start = int(dataset.path_indices[path_index])
        for local_index, scf_iteration in enumerate(scf_iterations):
            if int(scf_iteration) != -1:
                raise ValueError("feature extraction admitted a nonfinal density")
            indices.append(path_start + local_index)
            keys.append(molecule_id)
    if len(keys) != len(parent_ids) or set(keys) != parent_ids:
        raise RuntimeError(
            f"train parent feature selection mismatch: {len(keys)} != {len(parent_ids)}"
        )
    return indices, keys


def _extract_frozen_features(
    protocol: dict[str, Any],
    parents: list[Parent],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    inputs = protocol["inputs"]
    checkpoint = Path(inputs["source_checkpoint"])
    if _sha256(checkpoint) != inputs["source_checkpoint_sha256"]:
        raise ValueError("source Graphformer checkpoint hash drift")
    cfg = _load_cfg(protocol)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("fit")
    parent_by_id = {parent.molecule_id: parent for parent in parents}
    indices, keys = _selected_train_indices(
        datamodule.train_set, set(parent_by_id)
    )
    loader = OFLoader(
        Subset(datamodule.train_set, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=datamodule.worker_init_fn,
        **datamodule.dataloader_kwargs,
    )

    model = MLDFTLitModule.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    if model.net.__class__.__name__ != "Graphformer":
        raise TypeError("source checkpoint did not restore a Graphformer")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Graphformer freeze failed")

    captured: dict[str, torch.Tensor] = {}

    def capture_energy_input(
        _module: torch.nn.Module, arguments: tuple[Any, ...]
    ) -> None:
        states = arguments[0]
        captured["states"] = (
            states[-1] if states.ndim == 3 else states
        ).detach()

    hook = model.net.energy_mlp.register_forward_pre_hook(
        capture_energy_input
    )
    features: dict[str, torch.Tensor] = {}
    energies: dict[str, float] = {}
    position_max_abs = 0.0
    started = time.perf_counter()
    try:
        for molecule_id, batch in zip(keys, loader):
            batch = batch.to(device)
            captured.clear()
            with torch.no_grad():
                energy, _ = model.net(batch)
            if "states" not in captured:
                raise RuntimeError("failed to capture final Graphformer states")
            states = captured["states"]
            parent = parent_by_id[molecule_id]
            if states.shape != (
                parent.natoms,
                int(protocol["model"]["node_feature_dim"]),
            ):
                raise ValueError(
                    f"hidden state shape mismatch for {molecule_id}: {states.shape}"
                )
            position_error = torch.max(
                torch.abs(
                    batch.pos.detach().cpu().to(torch.float64)
                    - parent.positions_bohr
                )
            )
            position_max_abs = max(position_max_abs, float(position_error))
            features[molecule_id] = states.cpu().to(torch.float64)
            energies[molecule_id] = float(energy.detach().cpu().reshape(-1)[0])
    finally:
        hook.remove()

    if set(features) != set(parent_by_id):
        raise RuntimeError("frozen feature extraction is incomplete")
    metadata = {
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "backbone_class": model.net.__class__.__name__,
        "backbone_parameter_count": sum(
            parameter.numel() for parameter in model.net.parameters()
        ),
        "backbone_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.net.parameters()
            if parameter.requires_grad
        ),
        "feature_parent_count": len(features),
        "feature_dim": int(next(iter(features.values())).shape[1]),
        "position_max_abs_difference_bohr": position_max_abs,
        "energy_min": min(energies.values()),
        "energy_max": max(energies.values()),
        "elapsed_s": time.perf_counter() - started,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, metadata


def _build_head(
    protocol: dict[str, Any], device: torch.device
) -> GraphformerStructuredHessianAttentionHead:
    settings = protocol["model"]
    return GraphformerStructuredHessianAttentionHead(
        node_feature_dim=int(settings["node_feature_dim"]),
        attention_dim=int(settings["attention_dim"]),
        attention_heads=int(settings["attention_heads"]),
        latent_dim=int(settings["latent_dim"]),
        structured_hidden_dim=int(settings["structured_hidden_dim"]),
        radial_centers_bohr=tuple(settings["radial_centers_bohr"]),
        radial_width_bohr=float(settings["radial_width_bohr"]),
        environment_scale_bohr=float(settings["environment_scale_bohr"]),
        zero_initialize=bool(settings["zero_initialize"]),
    ).to(dtype=torch.float64, device=device)


def _relative_squared(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    denominator = torch.sum(target**2).clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return torch.sum((predicted - target) ** 2) / denominator


def _train(
    model: GraphformerStructuredHessianAttentionHead,
    parents: list[Parent],
    features: dict[str, torch.Tensor],
    protocol: dict[str, Any],
) -> list[dict[str, float]]:
    settings = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    steps = int(settings["steps"])
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for parent in parents:
            predicted = model(
                parent.atomic_numbers,
                parent.positions_bohr,
                features[parent.molecule_id],
                parent.projector,
            )
            losses.append(
                _relative_squared(predicted, parent.target_hessian)
            )
        loss = torch.mean(torch.stack(losses))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite attention-head training loss")
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
    model: GraphformerStructuredHessianAttentionHead,
    parent: Parent,
    features: dict[str, torch.Tensor],
) -> dict[str, float]:
    with torch.no_grad():
        predicted = model(
            parent.atomic_numbers,
            parent.positions_bohr,
            features[parent.molecule_id],
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
        "mae_hartree_per_bohr2": float(
            torch.mean(torch.abs(predicted - target))
        ),
        "symmetry_max_abs": float(
            torch.max(torch.abs(predicted - predicted.T))
        ),
        "external_leakage_frobenius": float(
            torch.linalg.matrix_norm(predicted @ external)
        ),
        "predicted_frobenius": float(torch.linalg.matrix_norm(predicted)),
        "target_frobenius": float(torch.linalg.matrix_norm(target)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol["protocol_id"] not in {
        "qm9_graphformer_frozen_hessian_attention_head_pilot_v1",
        "qm9_graphformer_frozen_hessian_attention_head_fold0_s800_v1",
    }:
        raise ValueError("unexpected frozen-attention protocol")
    scope = protocol["scope"]
    if (
        scope["backbone_trainable"] is not False
        or scope["energy_head_modified"] is not False
        or scope["validation_access_allowed"] is not False
        or scope["test100_access_allowed"] is not False
        or int(scope["validation_evaluations_used"]) != 0
        or int(scope["test100_evaluations_used"]) != 0
    ):
        raise ValueError("frozen-attention access boundary is open")
    if args.smoke:
        protocol = copy.deepcopy(protocol)
        protocol["training"]["steps"] = 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    parents_cpu = _load_parents(protocol)
    features_cpu, backbone_metadata = _extract_frozen_features(
        protocol, parents_cpu, device
    )
    parents = [_to_device(parent, device) for parent in parents_cpu]
    features = {
        molecule_id: values.to(device=device)
        for molecule_id, values in features_cpu.items()
    }
    folds = _folds(parents, int(scope["fold_count"]))
    active_fold_count = int(scope.get("active_fold_count", scope["fold_count"]))
    if active_fold_count <= 0 or active_fold_count > len(folds):
        raise ValueError("active_fold_count is outside the frozen fold partition")
    active_folds = folds[:1] if args.smoke else folds[:active_fold_count]
    formal_cv = not args.smoke and active_fold_count == len(folds)
    rows: list[dict[str, Any]] = []
    fold_summaries = []

    for fold_index, held_indices in enumerate(active_folds):
        held_set = set(held_indices)
        train_parents = [
            parent
            for index, parent in enumerate(parents)
            if index not in held_set
        ]
        held_parents = [parents[index] for index in held_indices]
        torch.manual_seed(int(protocol["training"]["seed"]) + fold_index)
        model = _build_head(protocol, device)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        history = _train(model, train_parents, features, protocol)
        checkpoint_path = checkpoints / f"attention_fold{fold_index}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold_index,
                "held_parent_ids": [
                    parent.molecule_id for parent in held_parents
                ],
                "head_parameter_count": parameter_count,
                "backbone_trainable": False,
                "source_checkpoint_sha256": protocol["inputs"][
                    "source_checkpoint_sha256"
                ],
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
                    "variant": "frozen_graphformer_attention",
                    "fold": fold_index,
                    "role": role,
                    "molecule_id": parent.molecule_id,
                    "natoms": parent.natoms,
                    "head_parameter_count": parameter_count,
                    **_metrics(model, parent, features),
                }
                rows.append(row)
                fold_rows.append(row)
        fold_summaries.append(
            {
                "fold": fold_index,
                "checkpoint": checkpoint_path.as_posix(),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "head_parameter_count": parameter_count,
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
    fit_summary = _aggregate(
        [row for row in rows if row["role"] == "fit"]
    )
    held_summary = _aggregate(
        [row for row in rows if row["role"] == "held"]
    )
    comparison = protocol["comparison"]
    baseline_summary_path = Path(comparison["baseline_summary"])
    if _sha256(baseline_summary_path) != comparison["baseline_summary_sha256"]:
        raise ValueError("geometry baseline summary hash drift")
    baseline_median = float(
        comparison["baseline_held_median_relative_frobenius"]
    )
    improvement = (
        baseline_median - held_summary["median_relative_frobenius"]
    ) / baseline_median
    gates = protocol["gates"]
    gate_results = {
        "fit_median": (
            fit_summary["median_relative_frobenius"]
            <= float(gates["fit_median_relative_frobenius_max"])
        ),
        "held_median": (
            held_summary["median_relative_frobenius"]
            <= float(gates["held_median_relative_frobenius_max"])
        ),
        "held_p90": (
            held_summary["p90_relative_frobenius"]
            <= float(gates["held_p90_relative_frobenius_max"])
        ),
        "held_fraction_better_than_zero": (
            held_summary["fraction_better_than_zero_hessian"]
            >= float(
                gates["held_fraction_better_than_zero_baseline_min"]
            )
        ),
        "improves_geometry": (
            improvement
            >= float(
                gates["relative_median_improvement_over_geometry_min"]
            )
        ),
        "symmetry": (
            held_summary["max_symmetry_max_abs"]
            <= float(gates["symmetry_max_abs_max"])
        ),
        "external_leakage": (
            held_summary["max_external_leakage_frobenius"]
            <= float(gates["external_leakage_frobenius_max"])
        ),
    }
    gate_results["passed"] = all(gate_results.values()) and formal_cv
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "smoke": bool(args.smoke),
        "device": str(device),
        "parent_count": len(parents),
        "configured_fold_count": len(folds),
        "fold_count": len(active_folds),
        "result_role": scope.get(
            "result_role", "formal_parent_cross_validation"
        ),
        "backbone": backbone_metadata,
        "head_parameter_count": int(rows[0]["head_parameter_count"]),
        "backbone_modified": False,
        "backbone_trainable": False,
        "energy_head_modified": False,
        "fit": fit_summary,
        "held": held_summary,
        "geometry_baseline_held_median_relative_frobenius": baseline_median,
        "relative_median_improvement_over_geometry": improvement,
        "gates": gate_results,
        "folds": fold_summaries,
        "formal_gates_evaluated": formal_cv,
        "elapsed_s": time.perf_counter() - started,
        "validation_accessed": False,
        "test100_accessed": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    registration = {
        "summary": summary_path.as_posix(),
        "summary_sha256": _sha256(summary_path),
        "per_parent_metrics": (
            args.output_dir / "per_parent_metrics.csv"
        ).as_posix(),
        "per_parent_metrics_sha256": _sha256(
            args.output_dir / "per_parent_metrics.csv"
        ),
        "source_checkpoint_sha256": protocol["inputs"][
            "source_checkpoint_sha256"
        ],
        "backbone_modified": False,
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
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
