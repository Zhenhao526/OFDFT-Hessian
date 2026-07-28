#!/usr/bin/env python3
"""Evaluate Hessian-vector secants from force changes across QM9 perturbations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


@dataclass
class SampleRecord:
    molecule_id: str
    sample_id: int
    sample_key: str
    pos: torch.Tensor
    ref_force: torch.Tensor
    pred_force: torch.Tensor


def _load_cfg(run_dir: Path, num_workers: int) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
        cfg.data.datamodule.dataset_kwargs.load_force_label = True
    return cfg


def _sample_keys(datamodule: Any) -> list[tuple[str, str, int, int]]:
    keys: list[tuple[str, str, int, int]] = []
    for path, scf_iterations in zip(
        datamodule.test_set.paths, datamodule.test_set.scf_iterations_per_path
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_id_text = label_name.split(".")
        for scf_iteration in scf_iterations:
            keys.append(
                (
                    f"{path.name}:scf={int(scf_iteration)}",
                    molecule_id,
                    int(sample_id_text),
                    int(scf_iteration),
                )
            )
    return keys


def _predict_run(
    spec: RunSpec,
    molecules: set[str],
    scf_iteration: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, list[SampleRecord]]:
    cfg = _load_cfg(spec.run_dir, num_workers)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    sample_keys = _sample_keys(datamodule)

    model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location=device)
    model.force_supervision = True
    model.to(device)
    model.eval()

    records: dict[str, list[SampleRecord]] = {molecule_id: [] for molecule_id in molecules}
    loader = datamodule.test_dataloader()
    for sample_index, batch in enumerate(loader):
        sample_key, molecule_id, sample_id, sample_scf_iteration = sample_keys[sample_index]
        if molecule_id not in molecules or sample_scf_iteration != scf_iteration:
            continue
        batch = batch.to(device)
        with torch.enable_grad():
            _, _, _, pred_forces = model.forward_predictions(batch)
        if pred_forces is None:
            raise RuntimeError(f"{spec.name}: pred_forces is None for {sample_key}")
        records[molecule_id].append(
            SampleRecord(
                molecule_id=molecule_id,
                sample_id=sample_id,
                sample_key=sample_key,
                pos=batch.pos.detach().cpu(),
                ref_force=batch.force_label.detach().cpu(),
                pred_force=pred_forces.detach().cpu(),
            )
        )

    for molecule_id in list(records):
        records[molecule_id] = sorted(records[molecule_id], key=lambda record: record.sample_id)
    return records


def _force_mae(records: list[SampleRecord]) -> float:
    errors = [torch.mean(torch.abs(record.pred_force - record.ref_force)) for record in records]
    return float(torch.stack(errors).mean())


def _pair_metrics(
    run_name: str, molecule_id: str, base: SampleRecord, perturbed: SampleRecord
) -> dict[str, Any]:
    delta_r = (perturbed.pos - base.pos).reshape(-1)
    delta_ref_force = (perturbed.ref_force - base.ref_force).reshape(-1)
    delta_pred_force = (perturbed.pred_force - base.pred_force).reshape(-1)
    delta_force_error = delta_pred_force - delta_ref_force

    displacement_norm = float(torch.linalg.vector_norm(delta_r))
    displacement_rms = float(torch.sqrt(torch.mean(delta_r * delta_r)))
    h_action_mae = float(torch.mean(torch.abs(delta_force_error)))
    h_action_rmse = float(torch.sqrt(torch.mean(delta_force_error * delta_force_error)))
    h_slope_mae = h_action_mae / displacement_rms if displacement_rms > 0 else None
    h_slope_rmse = h_action_rmse / displacement_rms if displacement_rms > 0 else None

    denom = float(torch.dot(delta_r, delta_r))
    if denom > 0:
        ref_curvature = float(-torch.dot(delta_ref_force, delta_r) / denom)
        pred_curvature = float(-torch.dot(delta_pred_force, delta_r) / denom)
        curvature_abs_error = abs(pred_curvature - ref_curvature)
    else:
        ref_curvature = None
        pred_curvature = None
        curvature_abs_error = None

    return {
        "run": run_name,
        "molecule_id": molecule_id,
        "base_sample": base.sample_key,
        "perturbed_sample": perturbed.sample_key,
        "sample_id": perturbed.sample_id,
        "natoms": int(base.pos.shape[0]),
        "displacement_norm": displacement_norm,
        "displacement_rms": displacement_rms,
        "h_action_mae": h_action_mae,
        "h_action_rmse": h_action_rmse,
        "h_slope_mae": h_slope_mae,
        "h_slope_rmse": h_slope_rmse,
        "ref_directional_curvature": ref_curvature,
        "pred_directional_curvature": pred_curvature,
        "curvature_abs_error": curvature_abs_error,
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    molecules = {item.strip().zfill(7) for item in args.molecules.split(",") if item.strip()}
    run_specs = []
    for item in args.run:
        name, run_dir, ckpt = item.split("=", maxsplit=2)
        run_specs.append(RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve()))

    all_pair_rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}
    for spec in run_specs:
        records_by_molecule = _predict_run(
            spec,
            molecules=molecules,
            scf_iteration=args.scf_iteration,
            num_workers=args.num_workers,
            device=device,
        )
        pair_rows = []
        missing = {}
        for molecule_id in sorted(molecules):
            records = records_by_molecule.get(molecule_id, [])
            sample_ids = {record.sample_id: record for record in records}
            if 0 not in sample_ids or len(sample_ids) < 2:
                missing[molecule_id] = sorted(sample_ids)
                continue
            base = sample_ids[0]
            for sample_id in sorted(sample_ids):
                if sample_id == 0:
                    continue
                row = _pair_metrics(spec.name, molecule_id, base, sample_ids[sample_id])
                pair_rows.append(row)
                all_pair_rows.append(row)
        force_records = [record for records in records_by_molecule.values() for record in records]
        run_summaries[spec.name] = {
            "run_dir": spec.run_dir.as_posix(),
            "ckpt": spec.ckpt.as_posix(),
            "molecules_requested": sorted(molecules),
            "force_component_mae_on_selected_samples": _force_mae(force_records)
            if force_records
            else None,
            "pairs": len(pair_rows),
            "missing_or_incomplete_molecules": missing,
            "mean_h_action_mae": _mean([row["h_action_mae"] for row in pair_rows]),
            "mean_h_action_rmse": _mean([row["h_action_rmse"] for row in pair_rows]),
            "mean_h_slope_mae": _mean([row["h_slope_mae"] for row in pair_rows]),
            "mean_h_slope_rmse": _mean([row["h_slope_rmse"] for row in pair_rows]),
            "mean_curvature_abs_error": _mean(
                [row["curvature_abs_error"] for row in pair_rows]
            ),
        }

    return {
        "definition": (
            "Directional Hessian/secant proxy using existing perturbation labels: "
            "H*dR ~= -dF. Lower h_action/slope/curvature errors are better."
        ),
        "device": str(device),
        "scf_iteration": args.scf_iteration,
        "runs": run_summaries,
        "pairs": all_pair_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=run_dir=ckpt. Can be repeated.",
    )
    parser.add_argument("--molecules", required=True, help="Comma-separated molecule ids.")
    parser.add_argument("--scf-iteration", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run",
            "molecule_id",
            "base_sample",
            "perturbed_sample",
            "sample_id",
            "natoms",
            "displacement_norm",
            "displacement_rms",
            "h_action_mae",
            "h_action_rmse",
            "h_slope_mae",
            "h_slope_rmse",
            "ref_directional_curvature",
            "pred_directional_curvature",
            "curvature_abs_error",
        ]
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary["pairs"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
