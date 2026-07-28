#!/usr/bin/env python3
"""Add a function-preserving depth-two branch to a conservative scalar MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand(args: argparse.Namespace) -> dict[str, Any]:
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("test100_accessed") is not False:
        raise ValueError("source checkpoint does not freeze Test100")
    if int(source.get("test100_evaluations_used", 0)) != 0:
        raise ValueError("source checkpoint records nonzero Test100 evaluations")
    source_state = {
        key: value.detach().cpu().clone()
        for key, value in source["state_dict"].items()
    }
    required = {"linear", "weight", "bias", "output"}
    if set(source_state) != required:
        raise ValueError("source must be a shallow scalar checkpoint")
    hidden_size, feature_count = source_state["weight"].shape
    if source_state["linear"].shape != (feature_count,):
        raise ValueError("source feature dimensions are inconsistent")
    if source_state["bias"].shape != (hidden_size,) or source_state["output"].shape != (
        hidden_size,
    ):
        raise ValueError("source hidden dimensions are inconsistent")
    if args.deep_hidden_size <= 0 or args.deep_hidden_size % 2 != 0:
        raise ValueError("deep_hidden_size must be positive and even")
    if not math.isfinite(args.pair_output_magnitude) or args.pair_output_magnitude <= 0:
        raise ValueError("pair_output_magnitude must be finite and positive")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    independent = torch.randn(
        args.deep_hidden_size // 2,
        hidden_size,
        dtype=source_state["weight"].dtype,
        generator=generator,
    ) / math.sqrt(hidden_size)
    deep_weight = torch.repeat_interleave(independent, 2, dim=0)
    deep_bias = torch.zeros(
        args.deep_hidden_size, dtype=source_state["bias"].dtype
    )
    deep_output = torch.empty(
        args.deep_hidden_size, dtype=source_state["output"].dtype
    )
    deep_output[0::2] = args.pair_output_magnitude
    deep_output[1::2] = -args.pair_output_magnitude
    state = {
        **source_state,
        "deep_weight": deep_weight,
        "deep_bias": deep_bias,
        "deep_output": deep_output,
    }

    config = dict(source.get("config", {}))
    config.update(
        hidden_size=hidden_size,
        deep_hidden_size=args.deep_hidden_size,
        resume_optimizer=False,
    )
    definition = (
        "exact function-preserving depth-two expansion: the shallow scalar is copied; "
        "deep units have paired-identical incoming weights and equal/opposite outputs"
    )
    payload = {
        "step": 0,
        "state_dict": state,
        "config": config,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "feature_inventory_manifest_sha256": source.get(
            "feature_inventory_manifest_sha256"
        ),
        "active_schema_manifest_sha256": source.get(
            "active_schema_manifest_sha256"
        ),
        "replay_cache_manifest_sha256": source.get("replay_cache_manifest_sha256"),
        "initialization_definition": definition,
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_checkpoint_step": int(source.get("step", -1)),
        "hidden_size": hidden_size,
        "deep_hidden_size": args.deep_hidden_size,
        "expansion_seed": args.seed,
        "pair_output_magnitude": args.pair_output_magnitude,
        "optimizer_state_preserved": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "initial.ckpt"
    torch.save(payload, checkpoint)
    result = {
        "definition": definition,
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_checkpoint_step": payload["source_checkpoint_step"],
        "output_checkpoint": checkpoint.resolve().as_posix(),
        "output_checkpoint_sha256": _sha256(checkpoint),
        "feature_count": feature_count,
        "hidden_size": hidden_size,
        "deep_hidden_size": args.deep_hidden_size,
        "expansion_seed": args.seed,
        "pair_output_magnitude": args.pair_output_magnitude,
        "deep_pair_output_sum_max_abs": float(
            torch.max(torch.abs(deep_output.reshape(-1, 2).sum(dim=1)))
        ),
        "deep_pair_weight_max_abs_difference": float(
            torch.max(torch.abs(deep_weight[0::2] - deep_weight[1::2]))
        ),
        "optimizer_state_preserved": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest = args.output_dir / "deep_expansion_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--deep-hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--pair-output-magnitude", type=float, default=1.0e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    expand(parse_args())
