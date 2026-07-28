"""Representative validation-only density optimization for recovered legacy models.

This script deliberately operates on the validation representatives frozen by
``legacy_select_acceptance_samples.py``.  It reports optimization convergence and
the final density error against the archived ground-state density.  It does not
read a test split and it does not interpret the learned coordinate derivative as
a physical total-OFDFT force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from pyscf import gto

import mldft.utils.local_frames  # noqa: F401 -- required before legacy torch.load
import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.density_optimization import density_optimization
from mldft.ofdft.optimizer import TorchOptimizer
from mldft.ofdft.run_density_optimization import SampleGenerator


FORBIDDEN_PATH_PREFIX = "/export/scratch/ialgroup"


def _portable_path(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if str(path).startswith(FORBIDDEN_PATH_PREFIX):
        raise RuntimeError(f"Refusing forbidden effective {name} path: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class OptimizationTrace:
    total_energies: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    density_l2_norms: list[float] = field(default_factory=list)

    def __call__(self, values: dict[str, Any]) -> None:
        sample = values["sample"]
        coeffs = values["coeffs"].detach()
        delta = coeffs - sample.ground_state_coeffs
        density_l2 = torch.sqrt(torch.clamp(delta @ sample.overlap_matrix @ delta, min=0))
        self.total_energies.append(float(values["energy"].total_energy))
        self.gradient_norms.append(float(values["gradient_norm"]))
        self.density_l2_norms.append(float(density_l2.item()))


def _optimizer(lr: float, max_cycle: int, tolerance: float) -> TorchOptimizer:
    return TorchOptimizer(
        torch.optim.Adam,
        lr=lr,
        max_cycle=max_cycle,
        convergence_tolerance=tolerance,
    )


def _load_label(path: Path) -> dict[str, Any]:
    store = zarr.ZipStore(path, mode="r")
    try:
        root = zarr.group(store=store)
        atomic_numbers = np.asarray(root["geometry/atomic_numbers"], dtype=np.int64)
        positions = np.asarray(root["geometry/atom_pos"], dtype=np.float64)
        coefficients = np.asarray(root["of_labels/spatial/coeffs"], dtype=np.float64)
        dual_integrals = np.asarray(
            root["of_labels/spatial/dual_basis_integrals"], dtype=np.float64
        )
        has_energy = np.asarray(
            root["ks_labels/energies/has_energy_label"], dtype=bool
        )
        labeled = np.flatnonzero(has_energy)
        if not len(labeled):
            raise RuntimeError(f"No energy-labeled density in {path}")
        index = int(labeled[-1])
        physical_total_energy = float(
            np.asarray(root["of_labels/energies/e_tot"], dtype=np.float64)[index]
        )
    finally:
        store.close()
    expected_electrons = int(round(float(dual_integrals[index] @ coefficients[index])))
    charge = int(atomic_numbers.sum() - expected_electrons)
    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "coefficients": coefficients[index],
        "dual_integrals": dual_integrals[index],
        "expected_electrons": expected_electrons,
        "charge": charge,
        "scf_iteration": index,
        "physical_total_energy": physical_total_energy,
    }


def _molecule(label: dict[str, Any]) -> gto.Mole:
    nelectron = int(label["expected_electrons"])
    return gto.M(
        atom=[
            (int(z), tuple(float(x) for x in pos))
            for z, pos in zip(label["atomic_numbers"], label["positions"])
        ],
        unit="Bohr",
        charge=int(label["charge"]),
        spin=nelectron % 2,
        basis="sto-3g",
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--selection-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initialization", default="sad_default")
    parser.add_argument("--stage1-lr", type=float, default=1.0e-3)
    parser.add_argument("--stage1-max-cycle", type=int, default=1000)
    parser.add_argument("--stage1-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--stage2-lr", type=float, default=3.0e-4)
    parser.add_argument("--stage2-max-cycle", type=int, default=10000)
    parser.add_argument("--stage2-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    code_root = _portable_path("PROJECT_ROOT", args.code_root)
    data_root = _portable_path("DFT_DATA", args.data_root)
    models_root = _portable_path("DFT_MODELS", args.models_root)
    hparams = _portable_path("hparams", args.hparams)
    os.environ.update(
        PROJECT_ROOT=str(code_root), DFT_DATA=str(data_root), DFT_MODELS=str(models_root)
    )
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)

    selection_report = json.loads(args.selection_json.read_text())
    selected = selection_report["datasets"][args.dataset_name]["partitions"]["val"]
    metadata = selected[args.selection_index]
    label_path = data_root / metadata["source"] / "labels" / metadata["filename"]
    checkpoint = hparams.parent / "checkpoints" / "last.ckpt"
    cfg = OmegaConf.load(hparams)
    ignored_checkpoint_provenance = {}
    for key in ("ckpt_path", "weight_ckpt_path"):
        if cfg.get(key):
            ignored_checkpoint_provenance[key] = str(cfg.get(key))
        cfg[key] = None

    base = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_name": args.dataset_name,
        "hparams": str(hparams),
        "ignored_checkpoint_provenance": ignored_checkpoint_provenance,
        "label_path": str(label_path),
        "label_sha256": _sha256(label_path),
        "partition": "val",
        "selection": metadata,
        "selection_index": args.selection_index,
    }
    started = time.perf_counter()
    try:
        label = _load_label(label_path)
        device = torch.device(args.device)
        model = MLDFTLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
        model = model.to(device=device, dtype=torch.float64).eval()
        generator = SampleGenerator(cfg, model, transform_device="cpu")
        factory = generator.get_functional_factory("PBE")
        sample = generator.get_sample_from_mol(_molecule(label))
        label_coefficients = torch.as_tensor(
            label["coefficients"], dtype=sample.coeffs.dtype, device=sample.coeffs.device
        )
        transformed_label = transform_tensor_with_sample(
            sample, label_coefficients, Representation.VECTOR
        )
        sample.ground_state_coeffs = transformed_label.detach().clone()

        stage1_trace = OptimizationTrace()
        energies, final_coefficients, stage1_converged, functional = density_optimization(
            sample,
            sample.mol,
            _optimizer(args.stage1_lr, args.stage1_max_cycle, args.stage1_tolerance),
            factory,
            callback=stage1_trace,
            initialization=args.initialization,
            normalize_initial_guess=True,
            ks_basis="6-31G(2df,p)",
            disable_printing=True,
            disable_pbar=True,
        )
        stage2_trace = OptimizationTrace()
        energies, final_coefficients, stage2_converged, functional = density_optimization(
            sample,
            sample.mol,
            _optimizer(args.stage2_lr, args.stage2_max_cycle, args.stage2_tolerance),
            factory,
            callback=stage2_trace,
            initialization=sample.coeffs.detach().clone(),
            normalize_initial_guess=True,
            ks_basis="6-31G(2df,p)",
            disable_printing=True,
            disable_pbar=True,
        )

        optimized_transformed = sample.coeffs.detach().clone()
        optimized_total_energy = float(energies.total_energy)
        sample.coeffs = transformed_label.detach().clone()
        label_density_energies, label_density_gradient = functional(sample)
        label_density_projected_gradient = project_gradient(label_density_gradient, sample)
        label_density_model_energy = float(label_density_energies.total_energy)
        sample.coeffs = optimized_transformed

        final_array = final_coefficients.detach().cpu().numpy().astype(np.float64)
        delta = final_array - label["coefficients"]
        electron_count = float(label["dual_integrals"] @ final_array)
        traces = stage1_trace.gradient_norms + stage2_trace.gradient_norms
        density_l2 = stage1_trace.density_l2_norms + stage2_trace.density_l2_norms
        row = {
            **base,
            "status": "passed",
            "finite": bool(
                np.isfinite(final_array).all()
                and np.isfinite(
                    [optimized_total_energy, label_density_model_energy, electron_count]
                ).all()
            ),
            "n_atoms": int(len(label["atomic_numbers"])),
            "coefficients": int(len(final_array)),
            "charge": label["charge"],
            "expected_electrons": label["expected_electrons"],
            "optimized_electrons": electron_count,
            "optimized_electron_error": abs(electron_count - label["expected_electrons"]),
            "archived_scf_iteration": label["scf_iteration"],
            "physical_label_total_energy": label["physical_total_energy"],
            "label_density_model_total_energy": label_density_model_energy,
            "optimized_model_total_energy": optimized_total_energy,
            "label_density_model_energy_error": (
                label_density_model_energy - label["physical_total_energy"]
            ),
            "optimized_model_energy_error": (
                optimized_total_energy - label["physical_total_energy"]
            ),
            "optimization_energy_change_from_label_density": (
                optimized_total_energy - label_density_model_energy
            ),
            "optimized_coefficient_mae": float(np.mean(np.abs(delta))),
            "optimized_coefficient_rmse": float(np.sqrt(np.mean(delta * delta))),
            "optimized_coefficient_relative_l2": float(
                np.linalg.norm(delta) / max(np.linalg.norm(label["coefficients"]), np.finfo(float).tiny)
            ),
            "optimized_density_l2": density_l2[-1] if density_l2 else None,
            "minimum_density_l2": min(density_l2) if density_l2 else None,
            "label_density_projected_gradient_norm": float(
                torch.linalg.vector_norm(label_density_projected_gradient).item()
            ),
            "initial_projected_gradient_norm": traces[0] if traces else None,
            "final_projected_gradient_norm": traces[-1] if traces else None,
            "cycles": len(traces),
            "converged": bool(stage2_converged),
            "stage1": {
                "converged_at_1e-2": bool(stage1_converged),
                "cycles": len(stage1_trace.gradient_norms),
                "final_projected_gradient_norm": (
                    stage1_trace.gradient_norms[-1] if stage1_trace.gradient_norms else None
                ),
                "lr": args.stage1_lr,
                "max_cycle": args.stage1_max_cycle,
                "tolerance": args.stage1_tolerance,
            },
            "stage2": {
                "converged_at_1e-4": bool(stage2_converged),
                "cycles": len(stage2_trace.gradient_norms),
                "final_projected_gradient_norm": (
                    stage2_trace.gradient_norms[-1] if stage2_trace.gradient_norms else None
                ),
                "lr": args.stage2_lr,
                "max_cycle": args.stage2_max_cycle,
                "tolerance": args.stage2_tolerance,
            },
            "elapsed_s": time.perf_counter() - started,
            "error": None,
        }
        if not row["finite"]:
            row["status"] = "failed"
            row["error"] = "nonfinite optimization output"
    except Exception as exc:  # preserve one failure artifact per task
        row = {
            **base,
            "status": "failed",
            "finite": False,
            "converged": False,
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": row["status"], "converged": row.get("converged"), "error": row["error"]}))
    # Scientific nonconvergence is a recorded result.  Infrastructure/runtime
    # failures still make the task fail so Slurm cannot silently count them.
    if row["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
