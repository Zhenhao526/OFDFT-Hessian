"""Evaluate one sharded validation/test baseline for a historical Graphformer model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import mldft.utils.omegaconf_resolvers  # noqa: F401
import mldft.utils.local_frames  # noqa: F401
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import Representation
from mldft.ml.data.components.convert_transforms import AddRadiusEdgeIndex
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.mldft_module import MLDFTLitModule


SOURCES = ("QM9_perturbed_fock", "QMUGS_perturbed_fock", "QMUGS")
SOURCE_IDS = {source: index for index, source in enumerate(SOURCES)}
FORBIDDEN_PATH_PREFIX = "/export/scratch/ialgroup"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(name: str, value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if str(path).startswith(FORBIDDEN_PATH_PREFIX):
        raise RuntimeError(f"Refusing forbidden effective {name} path: {path}")
    return path


class _IndexedSubset(Dataset):
    def __init__(self, dataset, records):
        self.dataset = dataset
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        dataset_index, source_id, filename_id, geometry_index, selector = self.records[index]
        sample = self.dataset[dataset_index]
        sample.add_item(
            "audit_source_id", torch.tensor([source_id], dtype=torch.int64), Representation.NONE
        )
        sample.add_item(
            "audit_filename_id",
            torch.tensor([filename_id], dtype=torch.int64),
            Representation.NONE,
        )
        sample.add_item(
            "audit_geometry_index",
            torch.tensor([geometry_index], dtype=torch.int64),
            Representation.NONE,
        )
        sample.add_item(
            "audit_scf_iteration",
            torch.tensor([int(sample.scf_iteration)], dtype=torch.int64),
            Representation.NONE,
        )
        sample.add_item(
            "audit_density_role",
            torch.tensor([0 if selector < 0 else 1], dtype=torch.int64),
            Representation.NONE,
        )
        return sample


class _CachedSubset(Dataset):
    def __init__(self, records, cache_root: Path, radius: float | None):
        self.records = records
        self.cache_root = cache_root
        self.radius_edge_transform = AddRadiusEdgeIndex(radius) if radius is not None else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source, filename, geometry_index, selector = self.records[index]
        cache_suffix = "" if selector == -1 else f".scf_{selector}"
        path = self.cache_root / source / f"{filename}{cache_suffix}.pt"
        sample = torch.load(path, map_location="cpu", weights_only=False)
        # The common cache is prepared with the QM9 full-edge transform.  The
        # historical QMUGS hparams instead require a 6-Bohr radius graph.  The
        # basis transformation is identical, so replace only edge_index here.
        if self.radius_edge_transform is not None:
            sample = self.radius_edge_transform(sample)
        source_id = SOURCE_IDS[source]
        filename_id = int(filename.split(".", 1)[0])
        sample.add_item(
            "audit_source_id", torch.tensor([source_id], dtype=torch.int64), Representation.NONE
        )
        sample.add_item(
            "audit_filename_id", torch.tensor([filename_id], dtype=torch.int64), Representation.NONE
        )
        sample.add_item(
            "audit_geometry_index", torch.tensor([geometry_index], dtype=torch.int64), Representation.NONE
        )
        sample.add_item(
            "audit_scf_iteration", torch.tensor([int(sample.scf_iteration)], dtype=torch.int64), Representation.NONE
        )
        sample.add_item(
            "audit_density_role",
            torch.tensor([0 if selector < 0 else 1], dtype=torch.int64),
            Representation.NONE,
        )
        return sample


def _cast_batch(batch, device: torch.device, dtype: torch.dtype):
    batch = batch.to(device)
    for key, value in batch:
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            batch[key] = value.to(dtype=dtype)
    return batch


def _pool(values: torch.Tensor, assignment: torch.Tensor, graphs: int) -> torch.Tensor:
    output = torch.zeros(graphs, dtype=values.dtype, device=values.device)
    output.index_add_(0, assignment, values)
    return output


def _sample_records(dataset, shard_index: int, num_shards: int):
    records = []
    labels = 0
    for geometry_index, path in enumerate(dataset.paths):
        numeric_id = int(path.name.split(".", 1)[0])
        if numeric_id % num_shards != shard_index:
            continue
        source = path.parent.parent.name
        source_id = SOURCE_IDS[source]
        start = int(dataset.path_indices[geometry_index])
        stop = int(dataset.path_indices[geometry_index + 1])
        records.extend(
            (
                sample_index,
                source_id,
                numeric_id,
                geometry_index,
                int(dataset.scf_iterations_per_path[geometry_index][sample_index - start]),
            )
            for sample_index in range(start, stop)
        )
        labels += 1
    return records, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocessed-cache-root", type=Path)
    parser.add_argument(
        "--scf-iterations",
        default="6,-1",
        help="Comma-separated fixed density points; default covers one trained perturbation and final ground state.",
    )
    parser.add_argument("--output-jsonl-gz", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    code_root = _portable("PROJECT_ROOT", args.code_root)
    data_root = _portable("DFT_DATA", args.data_root)
    models_root = _portable("DFT_MODELS", args.models_root)
    split_file = _portable("split file", args.split_file)
    os.environ.update(
        PROJECT_ROOT=str(code_root), DFT_DATA=str(data_root), DFT_MODELS=str(models_root)
    )
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cfg = OmegaConf.load(args.hparams)
    ignored_checkpoint_provenance = {}
    for key in ("ckpt_path", "weight_ckpt_path"):
        if cfg.get(key):
            ignored_checkpoint_provenance[key] = str(cfg.get(key))
        cfg[key] = None
    scf_iterations = [int(value) for value in args.scf_iterations.split(",")]
    radius_edges = [
        float(item.radius)
        for item in cfg.data.transforms.pre_transforms
        if str(item.get("_target_", "")).endswith("AddRadiusEdgeIndex")
    ]
    if len(radius_edges) > 1:
        raise RuntimeError(f"Multiple radius-edge transforms are unsupported: {radius_edges}")
    edge_radius = radius_edges[0] if radius_edges else None
    if args.preprocessed_cache_root is not None:
        cache_root = _portable("preprocessed cache", args.preprocessed_cache_root)
        with split_file.open("rb") as handle:
            split = pickle.load(handle)
        label_records = [
            (str(source), str(filename), geometry_index)
            for geometry_index, (source, filename, _) in enumerate(split[args.partition])
            if int(str(filename).split(".", 1)[0]) % args.num_shards == args.shard_index
        ]
        selected_labels = len(label_records)
        records = [
            (source, filename, geometry_index, selector)
            for source, filename, geometry_index in label_records
            for selector in scf_iterations
        ]
        subset = _CachedSubset(records, cache_root, edge_radius)
    else:
        cache_root = None
        cfg.data.dataset_name = args.dataset_name
        cfg.data.transforms.use_cached_data = False
        cfg.data.datamodule.split_file = str(split_file)
        cfg.data.datamodule.data_dir = str(data_root)
        cfg.data.datamodule.batch_size = args.batch_size
        cfg.data.datamodule.num_workers = args.num_workers
        cfg.data.datamodule.shuffle_val = False
        cfg.data.datamodule.shuffle_test = False
        cfg.data.datamodule.dataset_kwargs.cache_in_memory = False
        cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = scf_iterations
        cfg.data.datamodule.dataset_kwargs.keep_initial_guess = False
        datamodule = hydra.utils.instantiate(cfg.data.datamodule)
        datamodule.setup("validate" if args.partition == "val" else "test")
        dataset = datamodule.val_set if args.partition == "val" else datamodule.test_set
        records, selected_labels = _sample_records(
            dataset, args.shard_index, args.num_shards
        )
        subset = _IndexedSubset(dataset, records)
    loader = OFLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        follow_batch=("coeffs", "atomic_numbers"),
        # Raw-transform recovery retains the per-molecule square overlap matrix
        # after it has already been consumed by natural reparametrization. It is
        # not a frozen-net input and cannot be concatenated across sizes.
        exclude_keys=("overlap_matrix",),
    )

    checkpoint = args.hparams.parent / "checkpoints" / "last.ckpt"
    model = MLDFTLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device=device, dtype=dtype).eval()

    started = time.perf_counter()
    processed = 0
    batches = 0
    nonfinite_samples = 0
    args.output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_jsonl_gz.with_suffix(args.output_jsonl_gz.suffix + ".partial")
    with gzip.open(partial, "wt") as output:
        for batch in loader:
            batch = _cast_batch(batch, device, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            with torch.enable_grad():
                energy, gradient, difference, _ = model.forward_predictions(
                    batch, compute_forces=False
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_elapsed = time.perf_counter() - forward_started

            graphs = int(batch.num_graphs)
            coeff_assignment = batch.coeffs_batch
            atom_assignment = batch.atomic_numbers_batch
            coefficient_counts = torch.bincount(coeff_assignment, minlength=graphs)
            atom_counts = torch.bincount(atom_assignment, minlength=graphs)
            electron_counts = _pool(
                batch.atomic_numbers.to(dtype), atom_assignment, graphs
            )
            current_electrons = _pool(
                batch.coeffs * batch.dual_basis_integrals, coeff_assignment, graphs
            )
            ground_electrons = _pool(
                batch.ground_state_coeffs * batch.dual_basis_integrals,
                coeff_assignment,
                graphs,
            )
            predicted_ground_electrons = _pool(
                (batch.coeffs - difference) * batch.dual_basis_integrals,
                coeff_assignment,
                graphs,
            )
            gradient_error = project_gradient_difference(gradient, batch)
            difference_error = difference - (batch.coeffs - batch.ground_state_coeffs)
            energy_error = energy - batch.energy_label

            def error_metrics(error):
                return {
                    "abs_sum": _pool(error.abs(), coeff_assignment, graphs),
                    "maximum_absolute": _pool(
                        torch.zeros_like(error), coeff_assignment, graphs
                    ),
                    "squared_sum": _pool(error.square(), coeff_assignment, graphs),
                }

            gradient_metrics = error_metrics(gradient_error)
            difference_metrics = error_metrics(difference_error)
            # index_reduce is not required; split is cheap at these batch sizes.
            boundaries = coefficient_counts.detach().cpu().tolist()
            gradient_max = torch.stack(
                [chunk.abs().max() for chunk in torch.split(gradient_error, boundaries)]
            )
            difference_max = torch.stack(
                [chunk.abs().max() for chunk in torch.split(difference_error, boundaries)]
            )

            atomic_numbers = batch.atomic_numbers.detach().cpu()
            atom_boundaries = atom_counts.detach().cpu().tolist()
            compositions = [
                dict(
                    sorted(
                        (int(value), int(count))
                        for value, count in zip(*torch.unique(chunk, return_counts=True))
                    )
                )
                for chunk in torch.split(atomic_numbers, atom_boundaries)
            ]
            cpu = {
                "atom_counts": atom_counts.detach().cpu(),
                "coefficient_counts": coefficient_counts.detach().cpu(),
                "current_electrons": current_electrons.detach().cpu(),
                "difference_abs_sum": difference_metrics["abs_sum"].detach().cpu(),
                "density_roles": batch.audit_density_role.detach().cpu(),
                "difference_max": difference_max.detach().cpu(),
                "difference_squared_sum": difference_metrics["squared_sum"].detach().cpu(),
                "electron_counts": electron_counts.detach().cpu(),
                "energy": energy.detach().cpu(),
                "energy_error": energy_error.detach().cpu(),
                "energy_label": batch.energy_label.detach().cpu(),
                "filename_ids": batch.audit_filename_id.detach().cpu(),
                "geometry_indices": batch.audit_geometry_index.detach().cpu(),
                "gradient_abs_sum": gradient_metrics["abs_sum"].detach().cpu(),
                "gradient_max": gradient_max.detach().cpu(),
                "gradient_squared_sum": gradient_metrics["squared_sum"].detach().cpu(),
                "ground_electrons": ground_electrons.detach().cpu(),
                "has_energy": batch.has_energy_label.detach().cpu(),
                "predicted_ground_electrons": predicted_ground_electrons.detach().cpu(),
                "scf_iterations": batch.audit_scf_iteration.detach().cpu(),
                "source_ids": batch.audit_source_id.detach().cpu(),
            }
            for index in range(graphs):
                coefficients = int(cpu["coefficient_counts"][index])
                finite = all(
                    bool(torch.isfinite(cpu[name][index]).all())
                    for name in (
                        "energy",
                        "energy_error",
                        "gradient_abs_sum",
                        "gradient_squared_sum",
                        "difference_abs_sum",
                        "difference_squared_sum",
                    )
                )
                if not finite:
                    nonfinite_samples += 1
                source = SOURCES[int(cpu["source_ids"][index])]
                row = {
                    "composition": compositions[index],
                    "coefficients": coefficients,
                    "current_density_electron_error": float(
                        abs(cpu["current_electrons"][index] - cpu["electron_counts"][index])
                    ),
                    "density_role": (
                        "ground_state_final"
                        if int(cpu["density_roles"][index]) == 0
                        else "trained_perturbed_step"
                    ),
                    "difference_abs_sum": float(cpu["difference_abs_sum"][index]),
                    "difference_mae_per_coefficient": float(
                        cpu["difference_abs_sum"][index] / coefficients
                    ),
                    "difference_max_absolute": float(cpu["difference_max"][index]),
                    "difference_mse_per_coefficient": float(
                        cpu["difference_squared_sum"][index] / coefficients
                    ),
                    "energy_error": float(cpu["energy_error"][index]),
                    "energy_label": float(cpu["energy_label"][index]),
                    "energy_prediction": float(cpu["energy"][index]),
                    "filename": f"{int(cpu['filename_ids'][index]):07d}.zarr.zip",
                    "finite": finite,
                    "geometry_index": int(cpu["geometry_indices"][index]),
                    "gradient_abs_sum": float(cpu["gradient_abs_sum"][index]),
                    "gradient_mae_per_coefficient": float(
                        cpu["gradient_abs_sum"][index] / coefficients
                    ),
                    "gradient_max_absolute": float(cpu["gradient_max"][index]),
                    "gradient_mse_per_coefficient": float(
                        cpu["gradient_squared_sum"][index] / coefficients
                    ),
                    "ground_state_density_electron_error": float(
                        abs(cpu["ground_electrons"][index] - cpu["electron_counts"][index])
                    ),
                    "has_energy_label": bool(cpu["has_energy"][index]),
                    "n_atoms": int(cpu["atom_counts"][index]),
                    "n_electrons": float(cpu["electron_counts"][index]),
                    "partition": args.partition,
                    "predicted_ground_state_electron_error": float(
                        abs(
                            cpu["predicted_ground_electrons"][index]
                            - cpu["electron_counts"][index]
                        )
                    ),
                    "scf_iteration": int(cpu["scf_iterations"][index]),
                    "source": source,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
            processed += graphs
            batches += 1
            if args.progress_interval and batches % args.progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "batches": batches,
                            "elapsed_s": time.perf_counter() - started,
                            "forward_s": forward_elapsed,
                            "processed": processed,
                            "selected": len(records),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    partial.replace(args.output_jsonl_gz)

    summary = {
        "batch_size": args.batch_size,
        "batches": batches,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "command": [sys.executable, *sys.argv],
        "dataset": args.dataset_name,
        "device": str(device),
        "dtype": str(dtype),
        "edge_policy": (
            {"kind": "radius", "radius_bohr": edge_radius}
            if edge_radius is not None
            else {"kind": "full"}
        ),
        "elapsed_s": time.perf_counter() - started,
        "hostname": socket.gethostname(),
        "hparams": str(args.hparams.resolve()),
        "ignored_checkpoint_provenance": ignored_checkpoint_provenance,
        "nonfinite_samples": nonfinite_samples,
        "num_shards": args.num_shards,
        "preprocessed_cache_root": str(cache_root) if cache_root is not None else None,
        "output_jsonl_gz": str(args.output_jsonl_gz.resolve()),
        "output_jsonl_gz_sha256": _sha256(args.output_jsonl_gz),
        "partition": args.partition,
        "processed": processed,
        "selected": len(records),
        "selected_labels": selected_labels,
        "scf_iterations": scf_iterations,
        "shard_index": args.shard_index,
        "split_file": str(split_file),
        "split_file_sha256": _sha256(split_file),
        "status": (
            "passed" if processed == len(records) and not nonfinite_samples else "failed"
        ),
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
