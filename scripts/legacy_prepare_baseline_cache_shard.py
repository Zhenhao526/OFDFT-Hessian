"""Prepare one selected-density transformed sample in a read-only-source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import mldft.utils.local_frames  # noqa: F401
import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.dataset import OFDataset


FORBIDDEN_PATH_PREFIX = "/export/scratch/ialgroup"


def _portable(name: str, value: str | Path) -> Path:
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


def _worker_init(_: int) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)


def _finite_sample(sample) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for _, value in sample
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--transform-device", default="cpu")
    parser.add_argument(
        "--scf-selector",
        type=int,
        default=-1,
        help="Archived SCF selector to cache; -1 denotes the final ground-state density.",
    )
    parser.add_argument("--maximum-items", type=int)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    code_root = _portable("PROJECT_ROOT", args.code_root)
    data_root = _portable("DFT_DATA", args.data_root)
    models_root = _portable("DFT_MODELS", args.models_root)
    cache_root = _portable("derived cache", args.cache_root)
    os.environ.update(
        PROJECT_ROOT=str(code_root), DFT_DATA=str(data_root), DFT_MODELS=str(models_root)
    )
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    transform_device = torch.device(args.transform_device)
    if transform_device.type == "cuda":
        if args.num_workers:
            raise RuntimeError("CUDA transforms require --num-workers=0")
        torch.cuda.set_device(transform_device)
    manifest = json.loads(args.manifest_json.read_text())
    if manifest["num_shards"] != args.num_shards:
        raise RuntimeError("Manifest and requested shard counts differ")
    entries = [row for row in manifest["rows"] if row["shard"] == args.shard_index]
    if args.maximum_items is not None:
        entries = entries[: args.maximum_items]

    cfg = OmegaConf.load(args.hparams)
    cfg.data.transforms.use_cached_data = False
    transforms = hydra.utils.instantiate(cfg.data.transforms)
    transforms.set_transform_device(transform_device)
    basis_info = hydra.utils.instantiate(cfg.data.basis_info)
    dataset_kwargs = OmegaConf.to_container(
        cfg.data.datamodule.dataset_kwargs, resolve=True
    )
    dataset_kwargs.update(
        limit_scf_iterations=[args.scf_selector],
        keep_initial_guess=False,
        cache_in_memory=False,
    )
    paths = [data_root / row["source"] / "labels" / row["filename"] for row in entries]
    dataset = OFDataset(
        paths=paths,
        num_scf_iterations_per_path=[row["scf_steps"] for row in entries],
        basis_info=basis_info,
        transforms=transforms,
        **dataset_kwargs,
    )
    if len(dataset) != len(entries):
        raise RuntimeError(
            f"Expected one selected-density sample per label: {len(dataset)} vs {len(entries)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init,
        prefetch_factor=1 if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
    )
    started = time.perf_counter()
    rows = []
    errors = []
    cache_root.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(loader):
        metadata = entries[index]
        cache_suffix = "" if args.scf_selector == -1 else f".scf_{args.scf_selector}"
        output = (
            cache_root
            / metadata["source"]
            / f"{metadata['filename']}{cache_suffix}.pt"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(sample, "overlap_matrix"):
                sample.delete_item("overlap_matrix")
            if hasattr(sample, "transformation_matrix"):
                sample.delete_item("transformation_matrix")
            if hasattr(sample, "inv_transformation_matrix"):
                sample.delete_item("inv_transformation_matrix")
            sample = sample.to("cpu")
            finite = _finite_sample(sample)
            partial = output.with_suffix(output.suffix + ".partial")
            torch.save(sample, partial)
            os.replace(partial, output)
            row = {
                **metadata,
                "cache": str(output.resolve()),
                "cache_bytes": output.stat().st_size,
                "cache_sha256": _sha256(output),
                "coefficients": int(sample.coeffs.numel()),
                "finite": finite,
                "n_atoms": int(sample.atomic_numbers.numel()),
                "scf_selector": args.scf_selector,
                "scf_iteration": int(sample.scf_iteration),
                "status": "passed" if finite else "failed",
            }
            if not finite:
                errors.append(f"nonfinite transformed sample: {metadata}")
        except Exception as exc:
            row = {**metadata, "status": "failed", "error": repr(exc)}
            errors.append(f"{metadata}: {exc!r}")
        rows.append(row)
        if args.progress_interval and (index + 1) % args.progress_interval == 0:
            print(
                json.dumps(
                    {
                        "elapsed_s": time.perf_counter() - started,
                        "processed": index + 1,
                        "selected": len(entries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "cache_root": str(cache_root),
        "command": [sys.executable, *sys.argv],
        "elapsed_s": time.perf_counter() - started,
        "errors": errors,
        "hostname": socket.gethostname(),
        "manifest": str(args.manifest_json.resolve()),
        "manifest_sha256": _sha256(args.manifest_json),
        "basis_dataset_info": str(Path(cfg.data.basis_info.path_to_data_info).resolve()),
        "basis_dataset_info_sha256": _sha256(
            Path(cfg.data.basis_info.path_to_data_info).resolve()
        ),
        "cache_edge_policy": {"kind": "full"},
        "hparams": str(args.hparams.resolve()),
        "hparams_sha256": _sha256(args.hparams),
        "num_shards": args.num_shards,
        "transform_device": str(transform_device),
        "partition": manifest["partition"],
        "scf_selector": args.scf_selector,
        "processed": len(rows),
        "rows": rows,
        "selected": len(entries),
        "shard_index": args.shard_index,
        "status": "passed" if not errors and len(rows) == len(entries) else "failed",
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary["status"], "processed": len(rows), "errors": len(errors)}, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
