#!/usr/bin/env python3
"""Bind a capacity checkpoint to frozen original-A and strict-HVP provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    except AttributeError as error:
        # PyTorch 2.4 can lose this thread-local when first-time Lightning imports
        # trigger a nested torch.load while unpickling a checkpoint.
        if "map_location" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False, mmap=False)


def bind(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_hash = _sha256(args.checkpoint)
    summary_hash = _sha256(args.summary)
    if checkpoint_hash != args.checkpoint_sha256:
        raise ValueError("input checkpoint hash mismatch")
    if summary_hash != args.summary_sha256:
        raise ValueError("input summary hash mismatch")
    summary = json.loads(args.summary.read_text())
    if (
        summary.get("validation_accessed") is not False
        or summary.get("test100_accessed") is not False
        or summary.get("proxy_hvp_fallback_allowed") is not False
        or summary.get("complete_total_relaxed_hvp_graph_required") is not True
        or summary.get("strict_active_density_refresh") is not True
        or summary.get("implicit_density_parameter_response") is not True
    ):
        raise ValueError("summary does not certify a strict complete-total run")
    if summary.get("source_checkpoint_sha256") != args.root_source_checkpoint_sha256:
        raise ValueError("summary does not start from the requested original-A checkpoint")

    payload = _load_checkpoint(args.checkpoint)
    state = payload.get("complete_total_capacity")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks capacity optimizer state")
    if int(state.get("step", -1)) != int(summary["final_step"]):
        raise ValueError("checkpoint/summary step mismatch")
    state.update(
        {
            "root_source_checkpoint_sha256": args.root_source_checkpoint_sha256,
            "root_summary_sha256": summary_hash,
            "protocol_id": summary["protocol_id"],
            "direction_manifest_sha256": summary["direction_manifest_sha256"],
            "direction_role": summary["direction_role"],
            "molecules": summary["molecules"],
            "strict_active_density_refresh": True,
            "implicit_density_parameter_response": True,
            "proxy_hvp_fallback_allowed": False,
            "validation_accessed": False,
            "test100_accessed": False,
            "root_initial_full_hessian_metrics": summary[
                "initial_full_hessian_metrics"
            ],
            "loss_weights": summary["loss_weights"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "input_checkpoint": str(args.checkpoint),
        "input_checkpoint_sha256": checkpoint_hash,
        "source_summary": str(args.summary),
        "source_summary_sha256": summary_hash,
        "output_checkpoint": str(args.output),
        "output_checkpoint_sha256": _sha256(args.output),
        "root_source_checkpoint_sha256": args.root_source_checkpoint_sha256,
        "step": int(state["step"]),
        "protocol_id": state["protocol_id"],
        "direction_manifest_sha256": state["direction_manifest_sha256"],
        "molecules": state["molecules"],
        "loss_weights": state["loss_weights"],
        "validation_accessed": False,
        "test100_accessed": False,
        "proxy_hvp_fallback_allowed": False,
    }
    args.output.with_suffix(args.output.suffix + ".binding.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--root-source-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(bind(parse_args()), indent=2, sort_keys=True))
