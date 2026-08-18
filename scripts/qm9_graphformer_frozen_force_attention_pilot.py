#!/usr/bin/env python3
"""Train an equivariant force attention head on frozen Graphformer states."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
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
from mldft.ml.models.components.graphformer_structured_force_attention_head import (
    GraphformerStructuredForceAttentionHead,
)
from mldft.ml.models.mldft_module import MLDFTLitModule


@dataclass(frozen=True)
class ForceSample:
    key: str
    molecule_id: str
    sample_id: int
    role: str
    positions_bohr: torch.Tensor
    node_features: torch.Tensor
    target_force: torch.Tensor
    source_force: torch.Tensor

    @property
    def natoms(self) -> int:
        return int(self.positions_bohr.shape[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
        cfg.data.datamodule.dataset_kwargs.load_pair_metadata = False
        cfg.data.datamodule.dataset_kwargs.load_hvp_label = False
    return cfg


def _parent_rank(parent_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{parent_id}".encode()).hexdigest()


def _select_entries(
    dataset: Any, protocol: dict[str, Any]
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    available: dict[str, list[dict[str, Any]]] = {}
    for path_index, (path, scf_iterations) in enumerate(
        zip(dataset.paths, dataset.scf_iterations_per_path)
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_text = label_name.split(".")
        molecule_id = molecule_id.zfill(7)
        path_start = int(dataset.path_indices[path_index])
        for local_index, scf_iteration in enumerate(scf_iterations):
            if int(scf_iteration) != int(
                protocol["selection"]["scf_iteration"]
            ):
                raise ValueError("force pilot admitted a nonfinal density")
            available.setdefault(molecule_id, []).append(
                {
                    "dataset_index": path_start + local_index,
                    "molecule_id": molecule_id,
                    "sample_id": int(sample_text),
                    "scf_iteration": int(scf_iteration),
                    "label_name": label_name,
                }
            )
    selection_seed = int(protocol["selection"]["seed"])
    ranked = sorted(
        available, key=lambda parent_id: _parent_rank(parent_id, selection_seed)
    )
    selected_count = int(protocol["scope"]["selected_parent_count"])
    fit_count = int(protocol["scope"]["fit_parent_count"])
    held_count = int(protocol["scope"]["held_parent_count"])
    if selected_count != fit_count + held_count:
        raise ValueError("force parent role counts do not add up")
    if len(ranked) < selected_count:
        raise RuntimeError("not enough train parents for force pilot")
    selected = ranked[:selected_count]
    fit = set(selected[:fit_count])
    held = set(selected[fit_count:])
    entries = []
    for parent_id in selected:
        role = "fit" if parent_id in fit else "held"
        for entry in available[parent_id]:
            entries.append({**entry, "role": role})
    entries.sort(
        key=lambda row: (
            row["role"] != "fit",
            row["molecule_id"],
            row["sample_id"],
        )
    )
    indices = [int(row["dataset_index"]) for row in entries]
    metadata = {
        "available_train_parent_count": len(available),
        "selected_parent_ids": selected,
        "fit_parent_ids": sorted(fit),
        "held_parent_ids": sorted(held),
        "fit_sample_count": sum(row["role"] == "fit" for row in entries),
        "held_sample_count": sum(row["role"] == "held" for row in entries),
        "sample_ids": sorted({int(row["sample_id"]) for row in entries}),
    }
    return indices, entries, metadata


def _extract_samples(
    protocol: dict[str, Any], device: torch.device
) -> tuple[list[ForceSample], dict[str, Any]]:
    inputs = protocol["inputs"]
    dataset_root = Path(inputs["dataset_root"])
    train_manifest = (
        dataset_root / "provenance" / "train_only_dataset_manifest.json"
    )
    if _sha256(train_manifest) != inputs[
        "train_only_dataset_manifest_sha256"
    ]:
        raise ValueError("clean train-only dataset manifest hash drift")
    checkpoint = Path(inputs["source_checkpoint"])
    if _sha256(checkpoint) != inputs["source_checkpoint_sha256"]:
        raise ValueError("source Graphformer checkpoint hash drift")

    cfg = _load_cfg(protocol)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("fit")
    indices, entries, selection_metadata = _select_entries(
        datamodule.train_set, protocol
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
        )

    hook = model.net.energy_mlp.register_forward_pre_hook(
        capture_energy_input
    )
    samples: list[ForceSample] = []
    started = time.perf_counter()
    try:
        for entry, batch in zip(entries, loader):
            batch = batch.to(device)
            batch.pos = batch.pos.detach().clone().requires_grad_(True)
            captured.clear()
            with torch.enable_grad():
                energy, _ = model.net(batch)
                source_force = -torch.autograd.grad(
                    energy.sum(), batch.pos, create_graph=False
                )[0]
            if "states" not in captured:
                raise RuntimeError("failed to capture final Graphformer states")
            if not hasattr(batch, "force_label"):
                raise RuntimeError("clean force label is missing")
            target_force = batch.force_label
            if (
                target_force.shape != batch.pos.shape
                or source_force.shape != batch.pos.shape
            ):
                raise ValueError("force shape mismatch")
            if not (
                torch.isfinite(target_force).all()
                and torch.isfinite(source_force).all()
                and torch.isfinite(captured["states"]).all()
            ):
                raise RuntimeError("nonfinite force-pilot sample")
            samples.append(
                ForceSample(
                    key=(
                        f"{entry['label_name']}:"
                        f"scf={entry['scf_iteration']}"
                    ),
                    molecule_id=str(entry["molecule_id"]),
                    sample_id=int(entry["sample_id"]),
                    role=str(entry["role"]),
                    positions_bohr=batch.pos.detach().cpu(),
                    node_features=captured["states"].detach().cpu(),
                    target_force=target_force.detach().cpu(),
                    source_force=source_force.detach().cpu(),
                )
            )
    finally:
        hook.remove()
    if len(samples) != len(entries):
        raise RuntimeError("force feature extraction is incomplete")
    metadata = {
        **selection_metadata,
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
        "feature_dim": int(samples[0].node_features.shape[1]),
        "sample_count": len(samples),
        "elapsed_s": time.perf_counter() - started,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return samples, metadata


def _to_device(
    sample: ForceSample, device: torch.device
) -> ForceSample:
    return replace(
        sample,
        positions_bohr=sample.positions_bohr.to(device=device),
        node_features=sample.node_features.to(device=device),
        target_force=sample.target_force.to(device=device),
        source_force=sample.source_force.to(device=device),
    )


def _build_head(
    protocol: dict[str, Any], device: torch.device
) -> GraphformerStructuredForceAttentionHead:
    settings = protocol["model"]
    return GraphformerStructuredForceAttentionHead(
        node_feature_dim=int(settings["node_feature_dim"]),
        attention_dim=int(settings["attention_dim"]),
        attention_heads=int(settings["attention_heads"]),
        latent_dim=int(settings["latent_dim"]),
        pair_hidden_dim=int(settings["pair_hidden_dim"]),
        radial_centers_bohr=tuple(settings["radial_centers_bohr"]),
        radial_width_bohr=float(settings["radial_width_bohr"]),
        zero_initialize=bool(settings["zero_initialize"]),
    ).to(device=device, dtype=torch.float32)


def _train(
    model: GraphformerStructuredForceAttentionHead,
    fit_samples: list[ForceSample],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, float]], float]:
    settings = protocol["training"]
    target_values = torch.cat(
        [sample.target_force.reshape(-1) for sample in fit_samples]
    )
    force_rms = float(torch.sqrt(torch.mean(target_values**2)))
    if not np.isfinite(force_rms) or force_rms <= 0.0:
        raise RuntimeError("invalid fit-force RMS normalization")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(settings["seed"]))
    steps = int(settings["steps"])
    samples_per_step = int(settings["samples_per_step"])
    by_natoms: dict[int, list[int]] = {}
    for index, sample in enumerate(fit_samples):
        by_natoms.setdefault(sample.natoms, []).append(index)
    history = []
    for step in range(steps):
        anchor = int(
            torch.randint(
                len(fit_samples), (1,), generator=generator
            ).item()
        )
        candidates = by_natoms[fit_samples[anchor].natoms]
        selected_offsets = torch.randint(
            len(candidates),
            (samples_per_step,),
            generator=generator,
        ).tolist()
        chosen = [candidates[offset] for offset in selected_offsets]
        optimizer.zero_grad(set_to_none=True)
        node_features = torch.stack(
            [fit_samples[index].node_features for index in chosen]
        )
        positions = torch.stack(
            [fit_samples[index].positions_bohr for index in chosen]
        )
        target = torch.stack(
            [fit_samples[index].target_force for index in chosen]
        )
        predicted = model.forward_batch(node_features, positions)
        loss = torch.mean((predicted - target) ** 2) / (force_rms**2)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite force-head loss")
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
    return history, force_rms


def _force_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, float]:
    error = predicted - target
    target_norm = torch.linalg.vector_norm(target)
    return {
        "component_mae": float(torch.mean(torch.abs(error))),
        "component_rmse": float(torch.sqrt(torch.mean(error**2))),
        "atom_vector_mae": float(
            torch.mean(torch.linalg.vector_norm(error, dim=1))
        ),
        "relative_frobenius": float(
            torch.linalg.vector_norm(error)
            / target_norm.clamp_min(torch.finfo(target.dtype).tiny)
        ),
        "net_force_max_abs": float(torch.max(torch.abs(predicted.sum(0)))),
        "net_torque_max_abs": float(
            torch.max(
                torch.abs(
                    torch.linalg.cross(positions, predicted).sum(dim=0)
                )
            )
        ),
    }


def _evaluate(
    model: GraphformerStructuredForceAttentionHead,
    samples: list[ForceSample],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows = []
    predictions = {}
    with torch.no_grad():
        for sample in samples:
            predicted = model(
                sample.node_features, sample.positions_bohr
            )
            predictions[sample.key] = predicted
            rows.append(
                {
                    "key": sample.key,
                    "molecule_id": sample.molecule_id,
                    "sample_id": sample.sample_id,
                    "role": sample.role,
                    "natoms": sample.natoms,
                    **{
                        f"head_{key}": value
                        for key, value in _force_metrics(
                            predicted,
                            sample.target_force,
                            sample.positions_bohr,
                        ).items()
                    },
                    **{
                        f"source_{key}": value
                        for key, value in _force_metrics(
                            sample.source_force,
                            sample.target_force,
                            sample.positions_bohr,
                        ).items()
                    },
                    **{
                        f"zero_{key}": value
                        for key, value in _force_metrics(
                            torch.zeros_like(sample.target_force),
                            sample.target_force,
                            sample.positions_bohr,
                        ).items()
                    },
                }
            )

    def aggregate(role_samples: list[ForceSample]) -> dict[str, Any]:
        target = torch.cat(
            [sample.target_force.reshape(-1) for sample in role_samples]
        )
        head = torch.cat(
            [predictions[sample.key].reshape(-1) for sample in role_samples]
        )
        source = torch.cat(
            [sample.source_force.reshape(-1) for sample in role_samples]
        )
        zero = torch.zeros_like(target)

        def flat_metrics(predicted: torch.Tensor) -> dict[str, float]:
            error = predicted - target
            return {
                "component_mae": float(torch.mean(torch.abs(error))),
                "component_rmse": float(torch.sqrt(torch.mean(error**2))),
                "global_relative_frobenius": float(
                    torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(target).clamp_min(
                        torch.finfo(target.dtype).tiny
                    )
                ),
            }

        role_rows = [
            row for row in rows if row["role"] == role_samples[0].role
        ]
        return {
            "sample_count": len(role_samples),
            "parent_count": len(
                {sample.molecule_id for sample in role_samples}
            ),
            "component_count": int(target.numel()),
            "head": {
                **flat_metrics(head),
                "median_sample_relative_frobenius": float(
                    np.median(
                        [
                            row["head_relative_frobenius"]
                            for row in role_rows
                        ]
                    )
                ),
                "net_force_max_abs": max(
                    row["head_net_force_max_abs"] for row in role_rows
                ),
                "net_torque_max_abs": max(
                    row["head_net_torque_max_abs"] for row in role_rows
                ),
            },
            "source": flat_metrics(source),
            "zero": flat_metrics(zero),
        }

    aggregate_by_role = {
        role: aggregate(
            [sample for sample in samples if sample.role == role]
        )
        for role in ("fit", "held")
    }
    parent_rows = []
    for role in ("fit", "held"):
        parent_ids = sorted(
            {
                sample.molecule_id
                for sample in samples
                if sample.role == role
            }
        )
        for parent_id in parent_ids:
            selected = [
                sample
                for sample in samples
                if sample.role == role
                and sample.molecule_id == parent_id
            ]
            target = torch.cat(
                [sample.target_force.reshape(-1) for sample in selected]
            )
            head = torch.cat(
                [predictions[sample.key].reshape(-1) for sample in selected]
            )
            source = torch.cat(
                [sample.source_force.reshape(-1) for sample in selected]
            )
            parent_rows.append(
                {
                    "role": role,
                    "molecule_id": parent_id,
                    "sample_count": len(selected),
                    "head_component_mae": float(
                        torch.mean(torch.abs(head - target))
                    ),
                    "source_component_mae": float(
                        torch.mean(torch.abs(source - target))
                    ),
                    "zero_component_mae": float(
                        torch.mean(torch.abs(target))
                    ),
                }
            )
    return rows, aggregate_by_role, parent_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if (
        protocol["protocol_id"]
        != "qm9_graphformer_frozen_force_attention_pilot_v1"
    ):
        raise ValueError("unexpected force-attention protocol")
    scope = protocol["scope"]
    if (
        scope["backbone_trainable"] is not False
        or scope["energy_head_modified"] is not False
        or scope["validation_access_allowed"] is not False
        or scope["test100_access_allowed"] is not False
    ):
        raise ValueError("force-attention access boundary is open")
    if args.smoke:
        protocol = copy.deepcopy(protocol)
        protocol["scope"]["selected_parent_count"] = 8
        protocol["scope"]["fit_parent_count"] = 6
        protocol["scope"]["held_parent_count"] = 2
        protocol["training"]["steps"] = 20
        protocol["training"]["samples_per_step"] = 4
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")

    samples_cpu, extraction_metadata = _extract_samples(protocol, device)
    samples = [_to_device(sample, device) for sample in samples_cpu]
    fit_samples = [sample for sample in samples if sample.role == "fit"]
    model = _build_head(protocol, device)
    head_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    history, fit_force_rms = _train(
        model, fit_samples, protocol
    )
    sample_rows, metrics, parent_rows = _evaluate(model, samples)
    _write_csv(args.output_dir / "per_sample_metrics.csv", sample_rows)
    _write_csv(args.output_dir / "per_parent_metrics.csv", parent_rows)
    checkpoint_path = args.output_dir / "force_attention_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head_parameter_count": head_parameter_count,
            "fit_force_rms": fit_force_rms,
            "history": history,
            "selection": extraction_metadata,
            "source_checkpoint_sha256": protocol["inputs"][
                "source_checkpoint_sha256"
            ],
            "backbone_trainable": False,
        },
        checkpoint_path,
    )
    held = metrics["held"]
    source_mae = held["source"]["component_mae"]
    zero_mae = held["zero"]["component_mae"]
    head_mae = held["head"]["component_mae"]
    improvement_source = (source_mae - head_mae) / source_mae
    improvement_zero = (zero_mae - head_mae) / zero_mae
    gates = protocol["gates"]
    gate_results = {
        "fit_component_mae": (
            metrics["fit"]["head"]["component_mae"]
            <= float(gates["fit_component_mae_max"])
        ),
        "held_component_mae": (
            head_mae <= float(gates["held_component_mae_max"])
        ),
        "improves_source": (
            improvement_source
            >= float(
                gates[
                    "held_relative_mae_improvement_over_source_min"
                ]
            )
        ),
        "improves_zero": (
            improvement_zero
            >= float(gates["held_relative_mae_improvement_over_zero_min"])
        ),
        "net_force": (
            held["head"]["net_force_max_abs"]
            <= float(gates["net_force_max_abs_max"])
        ),
        "net_torque": (
            held["head"]["net_torque_max_abs"]
            <= float(gates["net_torque_max_abs_max"])
        ),
    }
    gate_results["passed"] = all(gate_results.values()) and not args.smoke
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol": args.protocol.resolve().as_posix(),
        "protocol_sha256": _sha256(args.protocol),
        "smoke": bool(args.smoke),
        "device": str(device),
        "backbone": extraction_metadata,
        "backbone_modified": False,
        "backbone_trainable": False,
        "energy_head_modified": False,
        "head_parameter_count": head_parameter_count,
        "fit_force_rms": fit_force_rms,
        "metrics": metrics,
        "held_relative_mae_improvement_over_source": improvement_source,
        "held_relative_mae_improvement_over_zero": improvement_zero,
        "gates": gate_results,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "history": history,
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
        "per_sample_metrics_sha256": _sha256(
            args.output_dir / "per_sample_metrics.csv"
        ),
        "per_parent_metrics_sha256": _sha256(
            args.output_dir / "per_parent_metrics.csv"
        ),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_sha256": protocol["inputs"][
            "source_checkpoint_sha256"
        ],
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
