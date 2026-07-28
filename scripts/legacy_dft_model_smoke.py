"""Restore and smoke-test the legacy perturbed-Fock Graphformer checkpoints.

The archived runs expect cached transformed labels that are not part of the
restored data tree.  This script deliberately disables that cache and rebuilds
the same transforms from the original ``labels`` archives for one sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

# Register the project's OmegaConf resolvers before resolving hparams.yaml.
import mldft.utils.omegaconf_resolvers  # noqa: F401

# PyTorch 2.4's checkpoint loader is not re-entrant. Importing this module while
# a legacy checkpoint is being unpickled would load Jd.pt through a nested
# torch.load and remove the outer loader's thread-local map_location. Preloading
# it here makes standalone legacy checkpoint loading reliable.
import mldft.utils.local_frames  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule


FORBIDDEN_PATH_PREFIX = "/export/scratch/ialgroup"


def _require_portable_effective_path(name: str, value: str | Path) -> Path:
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


def _one_batch(cfg, split: str):
    cfg.data.transforms.use_cached_data = False
    cfg.data.datamodule.batch_size = 1
    cfg.data.datamodule.num_workers = 0
    cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]

    if split == "train":
        cfg.data.datamodule.shuffle_train = False
        stage = "fit"
        loader_name = "train_dataloader"
        dataset_name = "train_set"
    elif split == "val":
        cfg.data.datamodule.shuffle_val = False
        stage = "validate"
        loader_name = "val_dataloader"
        dataset_name = "val_set"
    elif split == "test":
        cfg.data.datamodule.shuffle_test = False
        stage = "test"
        loader_name = "test_dataloader"
        dataset_name = "test_set"
    else:  # pragma: no cover - argparse/configuration guard
        raise ValueError(f"Unsupported split: {split}")

    started = time.perf_counter()
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup(stage)
    batch = next(iter(getattr(datamodule, loader_name)()))
    elapsed = time.perf_counter() - started
    return batch, len(getattr(datamodule, dataset_name)), elapsed


def _smoke_run(
    hparams_path: Path,
    *,
    device: torch.device,
    qm9_split: str,
    qmugs_split: str,
) -> dict:
    cfg = OmegaConf.load(hparams_path)
    ignored_legacy_provenance = {}
    for key in ("ckpt_path", "weight_ckpt_path"):
        value = cfg.get(key)
        if value:
            ignored_legacy_provenance[key] = str(value)
        # Never let an archived parent/continuation checkpoint path participate
        # in recovery. The checkpoint below is selected explicitly.
        cfg[key] = None
    dataset_name = str(cfg.data.dataset_name)
    split = qm9_split if dataset_name == "QM9_perturbed_fock" else qmugs_split

    dataset_info = Path(os.environ["DFT_DATA"]) / dataset_name / "dataset_info.yaml"
    if not dataset_info.is_file():
        raise FileNotFoundError(f"Missing recovered dataset metadata: {dataset_info}")

    checkpoint = hparams_path.parent / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    batch, split_samples, data_seconds = _one_batch(cfg, split)
    batch = batch.to(device)

    model = MLDFTLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device).eval()
    state_tensors = list(model.state_dict().values())
    state_nonfinite = sum(
        int((~torch.isfinite(value)).sum().item())
        for value in state_tensors
        if value.is_floating_point() or value.is_complex()
    )

    started = time.perf_counter()
    with torch.enable_grad():
        energy, gradients, coefficient_difference, forces = model.forward_predictions(
            batch, compute_forces=False
        )
    forward_seconds = time.perf_counter() - started

    outputs_finite = bool(
        torch.isfinite(energy).all()
        and torch.isfinite(gradients).all()
        and torch.isfinite(coefficient_difference).all()
    )
    if state_nonfinite or not outputs_finite:
        raise RuntimeError(
            f"Non-finite values for {dataset_name}: "
            f"state_nonfinite={state_nonfinite}, outputs_finite={outputs_finite}"
        )

    return {
        "batch": {
            "atoms": int(batch.atomic_numbers.numel()),
            "coefficients": int(batch.coeffs.numel()),
            "edges": int(batch.edge_index.shape[1]),
            "graphs": int(batch.num_graphs),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_seconds": data_seconds,
        "dataset": dataset_name,
        "dataset_info": str(dataset_info),
        "energy": energy.detach().cpu().tolist(),
        "forward_seconds": forward_seconds,
        "gradient_shape": list(gradients.shape),
        "ignored_checkpoint_provenance": ignored_legacy_provenance,
        "outputs_finite": outputs_finite,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": split,
        "split_samples": split_samples,
        "state_nonfinite": state_nonfinite,
        "state_tensors": len(state_tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--qm9-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--qmugs-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    data_root = _require_portable_effective_path("DFT_DATA", args.data_root)
    models_root = _require_portable_effective_path("DFT_MODELS", args.models_root)
    os.environ["DFT_DATA"] = str(data_root)
    os.environ["DFT_MODELS"] = str(models_root)
    torch.set_num_threads(args.threads)

    hparams_paths = sorted((models_root / "train" / "runs").glob("*/hparams.yaml"))
    if not hparams_paths:
        raise FileNotFoundError(f"No hparams.yaml files below {models_root / 'train' / 'runs'}")

    # Resolve only the effective data fields. hparams_resolved.yaml is never
    # loaded, and any stale path reaching an effective field fails immediately.
    for hparams_path in hparams_paths:
        cfg = OmegaConf.load(hparams_path)
        effective_fields = {
            "data.datamodule.data_dir": cfg.data.datamodule.data_dir,
            "data.datamodule.split_file": cfg.data.datamodule.split_file,
            "paths.data_dir": cfg.paths.data_dir,
            "paths.log_dir": cfg.paths.log_dir,
        }
        for name, value in effective_fields.items():
            _require_portable_effective_path(name, str(value))

    device = torch.device(args.device)
    results = [
        _smoke_run(
            path,
            device=device,
            qm9_split=args.qm9_split,
            qmugs_split=args.qmugs_split,
        )
        for path in hparams_paths
    ]
    report = {
        "data_root": str(data_root),
        "device": str(device),
        "models_root": str(models_root),
        "runs": results,
        "status": "passed",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
