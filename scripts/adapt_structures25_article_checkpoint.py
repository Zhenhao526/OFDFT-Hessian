#!/usr/bin/env python3
"""Remove only obsolete TensorFrames buffers from a released checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

# Preload the current model modules before unpickling the legacy checkpoint.
# Their first import loads Torch resources internally; doing that recursively
# from inside torch.load corrupts map_location state in this pinned Torch build.
from mldft.ml.models.components.graphformer import Graphformer  # noqa: F401
from mldft.ml.models.mldft_module import MLDFTLitModule  # noqa: F401


EXPECTED_KEYS = {
    f"net.gnn_module.g3d_layers.{layer}.transform_dict.{role}.odd_tensor"
    for layer in range(4)
    for role in ("key", "value")
}
KEY_PATTERN = re.compile(
    r"^net\.gnn_module\.g3d_layers\.[0-3]\."
    r"transform_dict\.(key|value)\.odd_tensor$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    source_sha = sha256(args.source)
    if source_sha != args.expected_source_sha256:
        raise ValueError(
            f"source checkpoint hash mismatch: {source_sha} != "
            f"{args.expected_source_sha256}"
        )

    # Match the repository training entry point. This pinned torch build has a
    # cleanup bug when ``weights_only=False`` is passed explicitly.
    payload = torch.load(args.source, map_location="cpu")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("released checkpoint has no state_dict mapping")

    matching = {key for key in state_dict if KEY_PATTERN.fullmatch(key)}
    if matching != EXPECTED_KEYS:
        raise ValueError(
            "legacy TensorFrames compatibility-key set differs from the "
            f"expected exact eight keys: {sorted(matching)}"
        )
    removed = {}
    for key in sorted(EXPECTED_KEYS):
        value = state_dict.pop(key)
        removed[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    payload["article_weight_compatibility_adapter"] = {
        "definition": (
            "Removed exactly eight obsolete, non-trainable TensorFrames "
            "odd_tensor buffers so the current Graphformer can strictly load "
            "all remaining state."
        ),
        "source_checkpoint": args.source.resolve().as_posix(),
        "source_checkpoint_sha256": source_sha,
        "removed_keys": removed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    output_sha = sha256(args.output)
    manifest = {
        "source_checkpoint": args.source.resolve().as_posix(),
        "source_checkpoint_sha256": source_sha,
        "adapted_checkpoint": args.output.resolve().as_posix(),
        "adapted_checkpoint_sha256": output_sha,
        "removed_keys": removed,
        "trainable_parameter_keys_removed": [],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
