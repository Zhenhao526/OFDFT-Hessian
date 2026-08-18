#!/usr/bin/env python3
"""Preserve model/AdamW state while starting a low-LR cosine tail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

# Match the training entry point before unpickling the Lightning checkpoint.
from mldft.ml.models.components.graphformer import Graphformer  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule  # noqa: F401


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--lr", type=float, default=7.0e-6)
    parser.add_argument("--t-max-epochs", type=int, default=20)
    args = parser.parse_args()

    source_sha = sha256(args.source)
    if source_sha != args.expected_source_sha256:
        raise ValueError(
            f"source checkpoint hash mismatch: {source_sha} != "
            f"{args.expected_source_sha256}"
        )
    payload = torch.load(args.source, map_location="cpu")
    if int(payload.get("global_step", -1)) != args.expected_global_step:
        raise ValueError(
            f"expected global step {args.expected_global_step}, found "
            f"{payload.get('global_step')}"
        )

    optimizer_states = payload.get("optimizer_states")
    scheduler_states = payload.get("lr_schedulers")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise TypeError("expected exactly one optimizer state")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise TypeError("expected exactly one scheduler state")
    param_groups = optimizer_states[0].get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise TypeError("optimizer state has no parameter groups")

    old_optimizer_lrs = []
    for group in param_groups:
        old_optimizer_lrs.append(
            {
                "lr": float(group["lr"]),
                "initial_lr": (
                    None
                    if "initial_lr" not in group
                    else float(group["initial_lr"])
                ),
            }
        )
        group["lr"] = args.lr
        group["initial_lr"] = args.lr

    scheduler = scheduler_states[0]
    required_scheduler_keys = {
        "T_max",
        "eta_min",
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_last_lr",
    }
    missing = required_scheduler_keys - set(scheduler)
    if missing:
        raise KeyError(f"scheduler state lacks required keys: {sorted(missing)}")
    old_scheduler = {
        key: scheduler[key]
        for key in (
            "T_max",
            "eta_min",
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_last_lr",
        )
    }
    scheduler["T_max"] = args.t_max_epochs
    scheduler["eta_min"] = 0.0
    scheduler["base_lrs"] = [args.lr for _ in param_groups]
    scheduler["last_epoch"] = 0
    scheduler["_step_count"] = 1
    scheduler["_last_lr"] = [args.lr for _ in param_groups]

    payload["egfh_low_lr_cosine_tail_adapter"] = {
        "definition": (
            "Preserved model parameters, optimizer moments, trainer epoch, "
            "and global step; reset only optimizer learning rates and the "
            "CosineAnnealingLR state for a monotone low-LR convergence tail."
        ),
        "source_checkpoint": args.source.resolve().as_posix(),
        "source_checkpoint_sha256": source_sha,
        "global_step": args.expected_global_step,
        "new_lr": args.lr,
        "new_t_max_epochs": args.t_max_epochs,
        "old_optimizer_lrs": old_optimizer_lrs,
        "old_scheduler": old_scheduler,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    output_sha = sha256(args.output)
    manifest = {
        **payload["egfh_low_lr_cosine_tail_adapter"],
        "adapted_checkpoint": args.output.resolve().as_posix(),
        "adapted_checkpoint_sha256": output_sha,
        "model_state_changed": False,
        "optimizer_moments_changed": False,
        "trainer_progress_changed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
