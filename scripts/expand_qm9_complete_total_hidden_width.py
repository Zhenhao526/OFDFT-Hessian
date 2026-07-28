#!/usr/bin/env python3
"""Function-preserving hidden-width expansion for the conservative scalar MLP."""

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


def _validate_state(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    required = {"linear", "weight", "bias", "output"}
    if set(state) != required:
        raise ValueError(f"expected scalar state keys {sorted(required)}, got {sorted(state)}")
    linear = state["linear"]
    weight = state["weight"]
    bias = state["bias"]
    output = state["output"]
    if linear.ndim != 1 or weight.ndim != 2 or bias.ndim != 1 or output.ndim != 1:
        raise ValueError("scalar checkpoint tensors have invalid ranks")
    hidden_size, feature_count = weight.shape
    if linear.numel() != feature_count:
        raise ValueError("linear and nonlinear feature dimensions differ")
    if bias.numel() != hidden_size or output.numel() != hidden_size:
        raise ValueError("hidden tensor dimensions differ")
    if any(tensor.dtype != linear.dtype for tensor in state.values()):
        raise ValueError("scalar checkpoint tensors have mixed dtypes")
    if any(not bool(torch.all(torch.isfinite(tensor))) for tensor in state.values()):
        raise ValueError("scalar checkpoint contains non-finite tensors")
    return hidden_size, feature_count


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
    source_hidden, feature_count = _validate_state(source_state)
    if args.target_hidden_size <= source_hidden:
        raise ValueError(
            f"target hidden size {args.target_hidden_size} must exceed source {source_hidden}"
        )

    new_hidden = args.target_hidden_size - source_hidden
    initialization = getattr(args, "new_unit_initialization", "dormant")
    pair_output_magnitude = float(getattr(args, "pair_output_magnitude", 1.0e-3))
    if initialization not in {"dormant", "canceling_pairs"}:
        raise ValueError(f"unsupported new-unit initialization: {initialization}")
    if initialization == "canceling_pairs" and new_hidden % 2 != 0:
        raise ValueError("canceling_pairs requires an even number of new hidden units")
    if not math.isfinite(pair_output_magnitude) or pair_output_magnitude <= 0.0:
        raise ValueError("pair_output_magnitude must be finite and positive")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    independent_new_hidden = (
        new_hidden // 2 if initialization == "canceling_pairs" else new_hidden
    )
    independent_weight = torch.randn(
        independent_new_hidden,
        feature_count,
        dtype=source_state["weight"].dtype,
        generator=generator,
    ) / math.sqrt(feature_count)
    extra_weight = (
        torch.repeat_interleave(independent_weight, 2, dim=0)
        if initialization == "canceling_pairs"
        else independent_weight
    )
    if initialization == "canceling_pairs":
        extra_output = torch.empty(new_hidden, dtype=source_state["output"].dtype)
        extra_output[0::2] = pair_output_magnitude
        extra_output[1::2] = -pair_output_magnitude
    else:
        extra_output = torch.zeros(new_hidden, dtype=source_state["output"].dtype)
    state = {
        "linear": source_state["linear"],
        "weight": torch.cat((source_state["weight"], extra_weight), dim=0),
        "bias": torch.cat(
            (
                source_state["bias"],
                torch.zeros(new_hidden, dtype=source_state["bias"].dtype),
            )
        ),
        "output": torch.cat(
            (
                source_state["output"],
                extra_output,
            )
        ),
    }
    _validate_state(state)

    config = dict(source.get("config", {}))
    config.update(hidden_size=args.target_hidden_size, resume_optimizer=False)
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
        "initialization_definition": (
            "exact function-preserving hidden expansion: source units copied in order; "
            + (
                "new units have paired-identical incoming weights and equal/opposite "
                "output weights"
                if initialization == "canceling_pairs"
                else "new incoming weights are seeded random values and new output weights are zero"
            )
        ),
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_checkpoint_step": int(source.get("step", -1)),
        "source_hidden_size": source_hidden,
        "target_hidden_size": args.target_hidden_size,
        "expansion_seed": args.seed,
        "new_unit_initialization": initialization,
        "pair_output_magnitude": (
            pair_output_magnitude if initialization == "canceling_pairs" else None
        ),
        "optimizer_state_preserved": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "initial.ckpt"
    torch.save(payload, checkpoint)
    result = {
        "definition": payload["initialization_definition"],
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_checkpoint_step": payload["source_checkpoint_step"],
        "output_checkpoint": checkpoint.resolve().as_posix(),
        "output_checkpoint_sha256": _sha256(checkpoint),
        "feature_count": feature_count,
        "source_hidden_size": source_hidden,
        "target_hidden_size": args.target_hidden_size,
        "new_hidden_size": new_hidden,
        "expansion_seed": args.seed,
        "new_unit_initialization": initialization,
        "pair_output_magnitude": (
            pair_output_magnitude if initialization == "canceling_pairs" else None
        ),
        "new_output_max_abs": float(torch.max(torch.abs(state["output"][source_hidden:]))),
        "new_pair_output_sum_max_abs": (
            float(
                torch.max(
                    torch.abs(
                        extra_output.reshape(-1, 2).sum(dim=1)
                    )
                )
            )
            if initialization == "canceling_pairs"
            else None
        ),
        "new_pair_weight_max_abs_difference": (
            float(
                torch.max(
                    torch.abs(
                        extra_weight[0::2] - extra_weight[1::2]
                    )
                )
            )
            if initialization == "canceling_pairs"
            else None
        ),
        "optimizer_state_preserved": False,
        "test100_accessed": False,
        "test100_evaluations_used": 0,
    }
    manifest = args.output_dir / "hidden_expansion_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--new-unit-initialization",
        choices=("dormant", "canceling_pairs"),
        default="dormant",
    )
    parser.add_argument("--pair-output-magnitude", type=float, default=1.0e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    expand(parse_args())
