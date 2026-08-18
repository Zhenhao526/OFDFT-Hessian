#!/usr/bin/env python3
"""Compare serial-global-batch and DDP8 v11 float64 probe artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


COMPONENTS = ("energy", "gradient", "force", "hessian")
TOLERANCE = {
    "scalar_atol": 1.0e-11,
    "scalar_rtol": 1.0e-10,
    "gradient_max_abs": 1.0e-10,
    "gradient_relative_l2": 1.0e-9,
    "update_max_abs": 1.0e-11,
    "update_relative_l2": 1.0e-10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metrics(left: dict[str, torch.Tensor | None], right: dict[str, torch.Tensor | None]) -> dict[str, float | int]:
    if set(left) != set(right):
        raise RuntimeError("parameter key sets differ")
    maximum = 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    compared = 0
    for key in sorted(left):
        lhs, rhs = left[key], right[key]
        if lhs is None or rhs is None:
            if lhs is not None or rhs is not None:
                raise RuntimeError(f"unused-parameter mismatch: {key}")
            continue
        delta = lhs.to(torch.float64) - rhs.to(torch.float64)
        maximum = max(maximum, float(delta.abs().max()))
        squared_difference += float(torch.sum(delta * delta))
        squared_reference += float(torch.sum(lhs.to(torch.float64) ** 2))
        compared += lhs.numel()
    return {
        "max_abs": maximum,
        "relative_l2": math.sqrt(squared_difference) / max(math.sqrt(squared_reference), 1.0e-300),
        "elements": compared,
    }


def aggregate_gradients(probe: dict) -> dict[str, torch.Tensor | None]:
    keys = set(probe["gradients"][COMPONENTS[0]])
    result: dict[str, torch.Tensor | None] = {}
    for key in keys:
        terms = [probe["gradients"][name][key] for name in COMPONENTS]
        present = [term for term in terms if term is not None]
        result[key] = None if not present else sum(present, torch.zeros_like(present[0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ddp8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    ddp8 = torch.load(args.ddp8, map_location="cpu", weights_only=False)
    scalar_rows = {}
    passed = True
    for component in COMPONENTS:
        lhs = float(reference["losses"][component])
        rhs = float(ddp8["losses"][component])
        absolute = abs(lhs - rhs)
        allowed = TOLERANCE["scalar_atol"] + TOLERANCE["scalar_rtol"] * abs(lhs)
        okay = math.isfinite(lhs) and math.isfinite(rhs) and absolute <= allowed
        scalar_rows[component] = {"reference": lhs, "ddp8": rhs, "absolute_difference": absolute, "allowed": allowed, "passed": okay}
        passed = passed and okay
    gradient_rows = {}
    for component in COMPONENTS:
        row = tensor_metrics(reference["gradients"][component], ddp8["gradients"][component])
        row["passed"] = row["max_abs"] <= TOLERANCE["gradient_max_abs"] and row["relative_l2"] <= TOLERANCE["gradient_relative_l2"]
        gradient_rows[component] = row
        passed = passed and bool(row["passed"])
    aggregate = tensor_metrics(aggregate_gradients(reference), aggregate_gradients(ddp8))
    aggregate["passed"] = aggregate["max_abs"] <= TOLERANCE["gradient_max_abs"] and aggregate["relative_l2"] <= TOLERANCE["gradient_relative_l2"]
    passed = passed and bool(aggregate["passed"])
    update = tensor_metrics(reference["updated_parameters"], ddp8["updated_parameters"])
    update["passed"] = update["max_abs"] <= TOLERANCE["update_max_abs"] and update["relative_l2"] <= TOLERANCE["update_relative_l2"]
    passed = passed and bool(update["passed"])
    report = {
        "artifact_id": "qm9_v11_serial_vs_ddp8_consistency_v1",
        "status": "passed" if passed else "failed",
        "tolerance": TOLERANCE,
        "reference_probe_sha256": sha256(args.reference),
        "ddp8_probe_sha256": sha256(args.ddp8),
        "raw_component_losses": scalar_rows,
        "weighted_component_gradients": gradient_rows,
        "aggregate_gradient": aggregate,
        "one_adamw_update": update,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
