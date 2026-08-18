#!/usr/bin/env python3
"""Migrate only the protocol-hash metadata of a v10 capacity checkpoint.

The model, optimizer, density state, RNG schedule, and cumulative step are
loaded and saved unchanged.  This narrowly records the user-authorized change
from dense 10-step full-Hessian evaluation to a 100-step evaluation schedule,
with checkpoint persistence every 50 updates.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

# Required before torch.load for this pinned PyTorch build; see the v10 monitor.
import mldft.utils.local_frames  # noqa: F401


PROTOCOL_ID = "qm9_graphformer_egfh_torch_autograd_joint_egfh_v10"
SOURCE_PROTOCOL_SHA256 = (
    "3ec061c6fc7dc849141f7c29d944eb2423095520652255e572b8d65364e210ba"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=80)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    protocol = args.protocol.resolve()
    if destination.exists():
        raise SystemExit(f"refusing to overwrite migrated checkpoint: {destination}")

    source_sha = sha256(source)
    target_protocol_sha = sha256(protocol)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    capacity = payload.get("complete_total_capacity")
    if not isinstance(capacity, dict):
        raise SystemExit("source lacks complete_total_capacity state")
    if int(capacity.get("step", -1)) != args.expected_step:
        raise SystemExit("source cumulative step mismatch")
    if capacity.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("source protocol id mismatch")
    if capacity.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise SystemExit("source protocol SHA256 mismatch")
    if capacity.get("alternating_hvp_updates") is not False:
        raise SystemExit("source is not the shared-optimizer joint protocol")
    if capacity.get("validation_accessed") is not False:
        raise SystemExit("source records Validation access")
    if capacity.get("test100_accessed") is not False:
        raise SystemExit("source records Test100 access")

    capacity["protocol_sha256"] = target_protocol_sha
    capacity["schedule_migration"] = {
        "kind": "user_authorized_sparse_full_hessian_schedule_v1",
        "source_checkpoint_sha256": source_sha,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "target_protocol_sha256": target_protocol_sha,
        "source_step": args.expected_step,
        "target_step": 300,
        "old_full_hessian_interval": 10,
        "new_full_hessian_interval": 100,
        "checkpoint_interval": 50,
        "network_configuration_changed": False,
        "optimizer_or_loss_configuration_changed": False,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(f"source_checkpoint_sha256={source_sha}")
    print(f"target_protocol_sha256={target_protocol_sha}")
    print(f"migrated_checkpoint_sha256={sha256(destination)}")


if __name__ == "__main__":
    main()
