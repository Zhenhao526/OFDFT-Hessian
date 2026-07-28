#!/usr/bin/env python3
"""Evaluate model finite-difference Hessians against a cached PBE Hessian reference set."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    ckpt: Path


def _load_cfg(run_dir: Path, num_workers: int) -> Any:
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    with open_dict(cfg):
        cfg.data.datamodule.batch_size = 1
        cfg.data.datamodule.num_workers = num_workers
        cfg.data.datamodule.shuffle_test = False
    return cfg


def _sample_keys(datamodule: Any) -> list[tuple[str, str, int, int]]:
    keys: list[tuple[str, str, int, int]] = []
    for path, scf_iterations in zip(
        datamodule.test_set.paths, datamodule.test_set.scf_iterations_per_path
    ):
        label_name = path.name.removesuffix(".zarr.zip")
        molecule_id, sample_text = label_name.split(".")
        for scf_iteration in scf_iterations:
            keys.append((f"{path.name}:scf={int(scf_iteration)}", molecule_id, int(sample_text), int(scf_iteration)))
    return keys


def _collect_batches(
    run_dir: Path,
    targets: set[tuple[str, int, int]],
    num_workers: int,
    device: torch.device,
) -> dict[tuple[str, int, int], Any]:
    cfg = _load_cfg(run_dir, num_workers)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    sample_keys = _sample_keys(datamodule)
    batches = {}
    loader = datamodule.test_dataloader()
    for sample_index, batch in enumerate(loader):
        _, molecule_id, sample_id, scf_iteration = sample_keys[sample_index]
        key = (molecule_id, sample_id, scf_iteration)
        if key in targets:
            batches[key] = batch.to(device)
            if len(batches) == len(targets):
                break
    return batches


def _pred_forces(model: MLDFTLitModule, batch: Any, positions: torch.Tensor) -> torch.Tensor:
    batch.pos = positions.detach().clone().requires_grad_(True)
    pred_energy, _ = model.net(batch)
    pred_forces = -torch.autograd.grad(
        pred_energy.sum(),
        batch.pos,
        create_graph=False,
        retain_graph=False,
    )[0]
    return pred_forces.detach()


def _finite_difference_hessian(
    model: MLDFTLitModule, batch: Any, displacement: float
) -> np.ndarray:
    base_pos = batch.pos.detach().clone()
    n_coords = int(base_pos.numel())
    columns = []
    for coord_idx in range(n_coords):
        pos_plus = base_pos.clone().reshape(-1)
        pos_minus = base_pos.clone().reshape(-1)
        pos_plus[coord_idx] += displacement
        pos_minus[coord_idx] -= displacement
        f_plus = _pred_forces(model, batch, pos_plus.reshape_as(base_pos)).reshape(-1)
        f_minus = _pred_forces(model, batch, pos_minus.reshape_as(base_pos)).reshape(-1)
        dforce_dcoord = (f_plus - f_minus) / (2.0 * displacement)
        columns.append((-dforce_dcoord).detach().cpu())
    batch.pos = base_pos
    return torch.stack(columns, dim=1).numpy()


def _compare(model_hessian: np.ndarray, ref_hessian: np.ndarray) -> dict[str, Any]:
    diff = model_hessian - ref_hessian
    ref_norm = float(np.linalg.norm(ref_hessian))
    return {
        "finite": bool(np.isfinite(model_hessian).all()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_fro_error": float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else None,
        "model_symmetry_max_abs_error": float(np.max(np.abs(model_hessian - model_hessian.T))),
        "model_hessian_max_abs": float(np.max(np.abs(model_hessian))),
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run",
        "molecule_id",
        "sample_id",
        "scf_iteration",
        "natoms",
        "success",
        "mae",
        "rmse",
        "relative_fro_error",
        "max_abs_error",
        "model_symmetry_max_abs_error",
        "model_hessian_max_abs",
        "reference_cache",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    manifest = json.loads(args.manifest_json.read_text())
    references = [
        row for row in manifest if row.get("success") and Path(row["cache_path"]).exists()
    ]
    targets = {
        (row["molecule_id"], int(row["sample_id"]), args.scf_iteration)
        for row in references
    }

    run_specs = []
    for item in args.run:
        name, run_dir, ckpt = item.split("=", maxsplit=2)
        run_specs.append(RunSpec(name=name, run_dir=Path(run_dir).resolve(), ckpt=Path(ckpt).resolve()))

    rows: list[dict[str, Any]] = []
    summaries = {}
    for spec in run_specs:
        batches = _collect_batches(spec.run_dir, targets, args.num_workers, device)
        model = MLDFTLitModule.load_from_checkpoint(spec.ckpt, map_location="cpu")
        model.to(device)
        model.eval()
        for ref in references:
            key = (ref["molecule_id"], int(ref["sample_id"]), args.scf_iteration)
            base_row = {
                "run": spec.name,
                "molecule_id": ref["molecule_id"],
                "sample_id": int(ref["sample_id"]),
                "scf_iteration": args.scf_iteration,
                "natoms": int(ref["natoms"]),
                "reference_cache": ref["cache_path"],
                "success": False,
                "error": None,
            }
            try:
                if key not in batches:
                    raise KeyError(f"target not found in dataloader: {key}")
                ref_hessian = np.load(ref["cache_path"])["pbe_hessian"]
                with torch.enable_grad():
                    model_hessian = _finite_difference_hessian(model, batches[key], args.displacement)
                rows.append({**base_row, "success": True, **_compare(model_hessian, ref_hessian)})
            except Exception as exc:  # noqa: BLE001
                rows.append({**base_row, "error": repr(exc)})
        run_rows = [row for row in rows if row["run"] == spec.name and row["success"]]
        summaries[spec.name] = {
            "run_dir": spec.run_dir.as_posix(),
            "ckpt": spec.ckpt.as_posix(),
            "n_success": len(run_rows),
            "n_failed": len([row for row in rows if row["run"] == spec.name and not row["success"]]),
            "mean_mae": _mean([row["mae"] for row in run_rows]),
            "mean_rmse": _mean([row["rmse"] for row in run_rows]),
            "mean_relative_fro_error": _mean([row["relative_fro_error"] for row in run_rows]),
            "mean_model_symmetry_max_abs_error": _mean(
                [row["model_symmetry_max_abs_error"] for row in run_rows]
            ),
        }

    result = {
        "manifest_json": args.manifest_json.as_posix(),
        "device": str(device),
        "model_displacement": args.displacement,
        "scf_iteration": args.scf_iteration,
        "n_references": len(references),
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
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--scf-iteration", type=int, default=1)
    parser.add_argument("--displacement", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
