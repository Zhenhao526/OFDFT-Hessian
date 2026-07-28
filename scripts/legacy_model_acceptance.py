"""Multi-sample, multi-precision recovery acceptance for historical Graphformer models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import mldft.utils.omegaconf_resolvers  # noqa: F401
import mldft.utils.local_frames  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.mldft_module import MLDFTLitModule


FORBIDDEN_PATH_PREFIX = "/export/scratch/ialgroup"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _portable_path(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if str(path).startswith(FORBIDDEN_PATH_PREFIX):
        raise RuntimeError(f"Refusing forbidden effective {name} path: {path}")
    return path


def _cast_batch(batch, device: torch.device, dtype: torch.dtype):
    batch = batch.to(device)
    for key, value in batch:
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            batch[key] = value.to(dtype=dtype)
    return batch


def _one_batch(sample, device: torch.device, dtype: torch.dtype):
    batch = next(iter(OFLoader([sample.clone()], batch_size=1, shuffle=False, num_workers=0)))
    return _cast_batch(batch, device, dtype)


def _noncollatable_tensor_keys(samples) -> list[str]:
    """Find variable square/intermediate tensors not consumed by the frozen net.

    Raw transforms retain overlap/preprocessing matrices whose two dimensions
    both vary by molecule. They cannot be concatenated by PyG and are not model
    inputs after the configured natural-representation transform.
    """
    keys = set.intersection(*(set(sample.keys()) for sample in samples))
    excluded = []
    for key in sorted(keys):
        values = [sample[key] for sample in samples]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        cat_dim = samples[0].__cat_dim__(key, values[0])
        try:
            torch.cat(values, dim=cat_dim or 0)
        except RuntimeError:
            excluded.append(key)
    return excluded


def _forward(model, batch) -> dict:
    started = time.perf_counter()
    with torch.enable_grad():
        energy, gradient, difference, _ = model.forward_predictions(
            batch, compute_forces=False
        )
    if batch.has_energy_label.bool().all():
        projected_gradient_error = project_gradient_difference(gradient, batch)
    else:
        projected_gradient_error = torch.full_like(gradient, torch.nan)
    difference_label = batch.coeffs - batch.ground_state_coeffs
    difference_error = difference - difference_label
    predicted_ground_state_coeffs = batch.coeffs - difference
    graphs = int(batch.num_graphs)
    predicted_electrons = torch.zeros(
        graphs, dtype=batch.coeffs.dtype, device=batch.coeffs.device
    )
    predicted_electrons.index_add_(
        0,
        batch.coeffs_batch,
        batch.dual_basis_integrals * predicted_ground_state_coeffs,
    )
    expected_electrons = torch.zeros(
        graphs, dtype=batch.coeffs.dtype, device=batch.coeffs.device
    )
    expected_electrons.index_add_(
        0, batch.atomic_numbers_batch, batch.atomic_numbers.to(batch.coeffs.dtype)
    )

    def scalar_or_list(value: torch.Tensor):
        value = value.detach().cpu().reshape(-1)
        return float(value.item()) if value.numel() == 1 else value.tolist()
    tensors = {
        "density_gradient": gradient.detach().cpu(),
        "difference": difference.detach().cpu(),
        "energy": energy.detach().cpu(),
    }
    return {
        "difference_error_mae_per_coefficient": float(difference_error.abs().mean().item()),
        "difference_error_rmse_per_coefficient": float(
            torch.sqrt(torch.mean(difference_error.square())).item()
        ),
        "difference_label": difference_label.detach().cpu().tolist(),
        "elapsed_s": time.perf_counter() - started,
        "energy_error_absolute": (
            scalar_or_list((energy - batch.energy_label).abs())
            if batch.has_energy_label.bool().all()
            else None
        ),
        "energy_label": batch.energy_label.detach().cpu().tolist(),
        "expected_electrons": scalar_or_list(expected_electrons),
        "outputs": {name: value.tolist() for name, value in tensors.items()},
        "output_sha256": {name: _sha256_tensor(value) for name, value in tensors.items()},
        "outputs_finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
        "predicted_ground_state_electron_error": scalar_or_list(
            (predicted_electrons - expected_electrons).abs()
        ),
        "projected_density_gradient_error_l1": (
            float(projected_gradient_error.abs().sum().item())
            if batch.has_energy_label.bool().all()
            else None
        ),
        "projected_density_gradient_error_mae_per_coefficient": (
            float(projected_gradient_error.abs().mean().item())
            if batch.has_energy_label.bool().all()
            else None
        ),
        "projected_density_gradient_error_rmse_per_coefficient": (
            float(torch.sqrt(torch.mean(projected_gradient_error.square())).item())
            if batch.has_energy_label.bool().all()
            else None
        ),
        "tensors": tensors,
    }


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _load_selected_samples(cfg, selection: dict, safe_split: Path):
    cfg.data.transforms.use_cached_data = False
    cfg.data.datamodule.split_file = str(safe_split)
    cfg.data.datamodule.batch_size = 1
    cfg.data.datamodule.num_workers = 0
    cfg.data.datamodule.shuffle_train = False
    cfg.data.datamodule.shuffle_val = False
    cfg.data.datamodule.shuffle_test = False
    cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = None
    cfg.data.datamodule.dataset_kwargs.keep_initial_guess = True
    cfg.data.datamodule.dataset_kwargs.cache_in_memory = False

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("fit")
    datamodule.setup("test")
    sets = {
        "train": datamodule.train_set,
        "val": datamodule.val_set,
        "test": datamodule.test_set,
    }
    selected = []
    for partition in ("train", "val", "test"):
        dataset = sets[partition]
        path_lookup = {
            (path.parent.parent.name, path.name): index
            for index, path in enumerate(dataset.paths)
        }
        for label in selection["partitions"][partition]:
            key = (label["source"], label["filename"])
            geometry_index = path_lookup[key]
            available_steps = dataset.scf_iterations_per_path[geometry_index]
            for step in label["steps"]:
                scf_iteration = int(step["scf_iteration"])
                local_indices = np.flatnonzero(available_steps == scf_iteration)
                if len(local_indices) != 1:
                    raise RuntimeError(
                        f"Expected one {key}/{scf_iteration} sample, found {len(local_indices)}"
                    )
                dataset_index = int(
                    dataset.path_indices[geometry_index] + local_indices[0]
                )
                selected.append(
                    (
                        {
                            **{key: value for key, value in label.items() if key != "steps"},
                            **step,
                        },
                        dataset[dataset_index],
                    )
                )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--safe-split", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-json-gz", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    args = parser.parse_args()

    code_root = _portable_path("PROJECT_ROOT", args.code_root)
    data_root = _portable_path("DFT_DATA", args.data_root)
    models_root = _portable_path("DFT_MODELS", args.models_root)
    safe_split = _portable_path("safe split", args.safe_split)
    os.environ.update(
        PROJECT_ROOT=str(code_root), DFT_DATA=str(data_root), DFT_MODELS=str(models_root)
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.hparams)
    ignored_checkpoint_provenance = {}
    for key in ("ckpt_path", "weight_ckpt_path"):
        if cfg.get(key):
            ignored_checkpoint_provenance[key] = str(cfg.get(key))
        cfg[key] = None
    dataset_name = str(cfg.data.dataset_name)
    selection_report = json.loads(args.selection_json.read_text())
    selection = selection_report["datasets"][dataset_name]

    checkpoint = args.hparams.parent / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = _load_selected_samples(cfg, selection, safe_split)
    model = MLDFTLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device=device, dtype=dtype).eval()

    results = []
    individual_outputs = {}
    for metadata, sample in selected:
        key = f"{metadata['partition']}:{metadata['source']}/{metadata['filename']}:{metadata['scf_iteration']}"
        first_batch = _one_batch(sample, device, dtype)
        first = _forward(model, first_batch)
        second = _forward(model, _one_batch(sample, device, dtype))
        repeat_max = {
            name: _maximum_difference(first["tensors"][name], second["tensors"][name])
            for name in first["tensors"]
        }
        individual_outputs[key] = first["tensors"]
        input_record = {
            "atomic_numbers": first_batch.atomic_numbers.detach().cpu().tolist(),
            "coefficients": int(first_batch.coeffs.numel()),
            "coefficients_sha256": _sha256_tensor(first_batch.coeffs),
            "dual_basis_integrals_sha256": _sha256_tensor(
                first_batch.dual_basis_integrals
            ),
            "n_basis_per_atom": first_batch.n_basis_per_atom.detach().cpu().tolist(),
            "positions_sha256": _sha256_tensor(first_batch.pos),
        }
        first.pop("tensors")
        second.pop("tensors")
        results.append(
            {
                **metadata,
                "determinism_repeat_max_abs": repeat_max,
                "first": first,
                "input": input_record,
                "second_summary": {
                    "elapsed_s": second["elapsed_s"],
                    "output_sha256": second["output_sha256"],
                    "outputs_finite": second["outputs_finite"],
                },
            }
        )

    # Compare batched inference against the already frozen single-sample outputs.
    batch_candidates = [item for item in selected if item[0]["role"] == "ground_state_final"]
    batch_candidates = batch_candidates[: min(3, len(batch_candidates))]
    batch_metadata = [item[0] for item in batch_candidates]
    batch_samples = [item[1].clone() for item in batch_candidates]
    batch_excluded_keys = _noncollatable_tensor_keys(batch_samples)
    loader = OFLoader(
        batch_samples,
        batch_size=len(batch_candidates),
        shuffle=False,
        num_workers=0,
        exclude_keys=batch_excluded_keys,
    )
    batch = _cast_batch(next(iter(loader)), device, dtype)
    batch_forward = _forward(model, batch)
    batch_tensors = batch_forward.pop("tensors")
    coefficient_counts = torch.bincount(batch.coeffs_batch).cpu().tolist()
    batch_gradients = torch.split(batch_tensors["density_gradient"], coefficient_counts)
    batch_differences = torch.split(batch_tensors["difference"], coefficient_counts)
    batch_consistency = []
    for index, metadata in enumerate(batch_metadata):
        key = f"{metadata['partition']}:{metadata['source']}/{metadata['filename']}:{metadata['scf_iteration']}"
        reference = individual_outputs[key]
        batch_consistency.append(
            {
                "key": key,
                "max_abs": {
                    "density_gradient": _maximum_difference(
                        reference["density_gradient"], batch_gradients[index]
                    ),
                    "difference": _maximum_difference(
                        reference["difference"], batch_differences[index]
                    ),
                    "energy": _maximum_difference(
                        reference["energy"][0:1], batch_tensors["energy"][index : index + 1]
                    ),
                },
            }
        )

    repeat_tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    batch_tolerance = 2.0e-5 if dtype == torch.float32 else 1.0e-10
    repeat_maximum = max(
        value
        for result in results
        for value in result["determinism_repeat_max_abs"].values()
    )
    batch_maximum = max(
        value for item in batch_consistency for value in item["max_abs"].values()
    )
    all_finite = all(result["first"]["outputs_finite"] for result in results)
    status = (
        "passed"
        if all_finite
        and repeat_maximum <= repeat_tolerance
        and batch_maximum <= batch_tolerance
        else "failed"
    )
    report = {
        "batch_consistency": batch_consistency,
        "batch_excluded_preprocessing_keys": batch_excluded_keys,
        "batch_forward_summary": batch_forward,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "command": [sys.executable, *sys.argv],
        "data_root": str(data_root),
        "dataset": dataset_name,
        "device": str(device),
        "dtype": str(dtype),
        "environment": {
            "cuda": torch.version.cuda,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
        },
        "hparams": str(args.hparams.resolve()),
        "hparams_sha256": _sha256_file(args.hparams),
        "ignored_checkpoint_provenance": ignored_checkpoint_provenance,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "results": results,
        "safe_split": str(safe_split),
        "safe_split_sha256": _sha256_file(safe_split),
        "selection_json": str(args.selection_json.resolve()),
        "status": status,
        "tolerances": {
            "batch_max_abs": batch_tolerance,
            "repeat_max_abs": repeat_tolerance,
        },
    }
    args.output_json_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_json_gz, "wt") as handle:
        json.dump(report, handle, sort_keys=True)
        handle.write("\n")
    summary = {
        "all_finite": all_finite,
        "batch_max_abs": batch_maximum,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "dataset": dataset_name,
        "device": str(device),
        "dtype": str(dtype),
        "output_json_gz": str(args.output_json_gz.resolve()),
        "output_json_gz_sha256": _sha256_file(args.output_json_gz),
        "repeat_max_abs": repeat_maximum,
        "samples": len(results),
        "status": status,
    }
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
