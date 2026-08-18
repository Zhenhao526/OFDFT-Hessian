#!/usr/bin/env python3
"""Rebind a v10 schedule-migrated checkpoint to an equivalent calibration.

Only loss/calibration provenance is updated. Model, optimizer, density, RNG,
direction cursor, and cumulative step remain untouched. The old and new
formal lambda_H values must agree within the explicitly authorized tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import mldft.utils.local_frames  # noqa: F401


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
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=80)
    parser.add_argument("--lambda-h-abs-tol", type=float, default=1.0e-9)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    calibration = args.calibration.resolve()
    if destination.exists():
        raise SystemExit(f"refusing to overwrite checkpoint: {destination}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    capacity = payload.get("complete_total_capacity")
    if not isinstance(capacity, dict):
        raise SystemExit("source lacks complete_total_capacity state")
    if int(capacity.get("step", -1)) != args.expected_step:
        raise SystemExit("source cumulative step mismatch")

    calibration_payload = json.loads(calibration.read_text())
    if calibration_payload.get("protocol_id") != capacity.get("protocol_id"):
        raise SystemExit("calibration protocol id mismatch")
    if calibration_payload.get("protocol_sha256") != capacity.get("protocol_sha256"):
        raise SystemExit("calibration protocol SHA mismatch")

    weights = capacity.get("loss_weights")
    semantic_weights = capacity.get("semantic_loss_weights")
    old_provenance = capacity.get("hvp_calibration")
    if not isinstance(weights, dict) or not isinstance(semantic_weights, dict):
        raise SystemExit("source lacks loss-weight metadata")
    if not isinstance(old_provenance, dict):
        raise SystemExit("source lacks calibration provenance")

    old_lambda_h = float(weights["lambda_H"])
    if float(semantic_weights["lambda_H"]) != old_lambda_h:
        raise SystemExit("source loss-weight aliases disagree")
    new_lambda_h = float(calibration_payload["formal_lambda_h"])
    delta = abs(new_lambda_h - old_lambda_h)
    if delta > args.lambda_h_abs_tol:
        raise SystemExit(
            f"lambda_H delta {delta:.17g} exceeds tolerance {args.lambda_h_abs_tol:.17g}"
        )

    training_curve = Path(str(calibration_payload["training_curve"])).resolve()
    training_summary = Path(str(calibration_payload["training_summary"])).resolve()
    if sha256(training_curve) != calibration_payload["training_curve_sha256"]:
        raise SystemExit("calibration training-curve SHA mismatch")
    if sha256(training_summary) != calibration_payload["training_summary_sha256"]:
        raise SystemExit("calibration training-summary SHA mismatch")

    source_sha = sha256(source)
    weights["lambda_H"] = new_lambda_h
    semantic_weights["lambda_H"] = new_lambda_h
    capacity["hvp_calibration"] = {
        "path": calibration.as_posix(),
        "sha256": sha256(calibration),
        "training_curve": training_curve.as_posix(),
        "training_curve_sha256": str(calibration_payload["training_curve_sha256"]),
        "training_summary": training_summary.as_posix(),
        "training_summary_sha256": str(calibration_payload["training_summary_sha256"]),
        "code_provenance": calibration_payload.get("code_provenance"),
        "formal_lambda_h": new_lambda_h,
    }
    capacity["schedule_calibration_rebind"] = {
        "kind": "user_authorized_equivalent_lambda_h_tolerance_v1",
        "source_checkpoint_sha256": source_sha,
        "old_calibration_sha256": old_provenance.get("sha256"),
        "new_calibration_sha256": sha256(calibration),
        "old_lambda_h": old_lambda_h,
        "new_lambda_h": new_lambda_h,
        "absolute_difference": delta,
        "authorized_absolute_tolerance": args.lambda_h_abs_tol,
        "model_optimizer_density_rng_unchanged": True,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    print(f"source_checkpoint_sha256={source_sha}")
    print(f"old_lambda_h={old_lambda_h:.17g}")
    print(f"new_lambda_h={new_lambda_h:.17g}")
    print(f"absolute_difference={delta:.17g}")
    print(f"rebound_checkpoint_sha256={sha256(destination)}")


if __name__ == "__main__":
    main()
