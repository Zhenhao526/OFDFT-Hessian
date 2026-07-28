#!/usr/bin/env python3
"""Verify that Stage-2 baseline checkpoints embed one identical frozen base model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Preload the trusted checkpoint runtime before torch unpickles Lightning/Hydra objects.
import qm9_complete_total_capacity_train as _capacity_runtime  # noqa: F401,E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"checkpoint lacks a state_dict: {path}")
    return payload


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _canonical_state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Hash tensor values after canonical dtype promotion for cross-runtime comparison."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            if tensor.is_complex():
                tensor = tensor.to(torch.complex128)
            elif tensor.is_floating_point():
                tensor = tensor.to(torch.float64)
            elif tensor.dtype == torch.bool:
                tensor = tensor.to(torch.uint8)
            else:
                tensor = tensor.to(torch.int64)
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _compare_state_dicts(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    missing = sorted(reference_keys - candidate_keys)
    unexpected = sorted(candidate_keys - reference_keys)
    mismatched = []
    dtype_differences = []
    max_abs_difference = 0.0
    for name in sorted(reference_keys & candidate_keys):
        expected = reference[name]
        actual = candidate[name]
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if expected.shape != actual.shape:
                mismatched.append(name)
                continue
            if expected.dtype != actual.dtype:
                dtype_differences.append(
                    f"{name}:{expected.dtype}->{actual.dtype}"
                )
            common_dtype = torch.promote_types(expected.dtype, actual.dtype)
            expected_cpu = expected.detach().cpu().to(common_dtype)
            actual_cpu = actual.detach().cpu().to(common_dtype)
            if not torch.equal(expected_cpu, actual_cpu):
                mismatched.append(name)
                if expected.is_floating_point() or expected.is_complex():
                    difference = torch.max(torch.abs(expected_cpu - actual_cpu))
                    max_abs_difference = max(max_abs_difference, float(difference))
        elif expected != actual:
            mismatched.append(name)
    return {
        "exact_match": not (missing or unexpected or mismatched),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "mismatched_key_count": len(mismatched),
        "dtype_difference_count": len(dtype_differences),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "dtype_differences": dtype_differences,
        "max_abs_parameter_difference": max_abs_difference,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_checkpoint(args.source_checkpoint)
    source_state = source["state_dict"]
    source_state_sha = _state_dict_sha256(source_state)
    run_dirs = list(args.run_dir)
    for root in args.run_root:
        run_dirs.extend(path for path in sorted(root.glob("*_job*")) if path.is_dir())
    run_dirs = sorted({path.resolve() for path in run_dirs})
    records = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "checkpoints" / "step_0000000.ckpt"
        record: dict[str, Any] = {
            "baseline_run_dir": run_dir.as_posix(),
            "summary_path": summary_path.as_posix(),
            "capacity_checkpoint": checkpoint_path.as_posix(),
            "status": "failed",
            "test100_accessed": None,
            "source_run_name": None,
            "base_state_exact_match": False,
        }
        reasons = []
        if not summary_path.is_file():
            reasons.append("missing_summary")
        else:
            summary = json.loads(summary_path.read_text())
            record["summary_sha256"] = _sha256(summary_path)
            record["test100_accessed"] = summary.get("test100_accessed")
            record["source_run_name"] = summary.get("source_run_name")
            molecules = summary.get("molecules", [])
            record["molecule_ids"] = ";".join(str(value) for value in molecules)
            if summary.get("test100_accessed") is not False:
                reasons.append("test100_not_frozen")
            if (
                args.expected_source_run_name
                and summary.get("source_run_name") != args.expected_source_run_name
            ):
                reasons.append("source_run_name_mismatch")
        if not checkpoint_path.is_file():
            reasons.append("missing_capacity_checkpoint")
        else:
            candidate = _load_checkpoint(checkpoint_path)
            comparison = _compare_state_dicts(source_state, candidate["state_dict"])
            record.update(
                {
                    "capacity_checkpoint_sha256": _sha256(checkpoint_path),
                    "capacity_state_dict_sha256": _state_dict_sha256(
                        candidate["state_dict"]
                    ),
                    "capacity_canonical_state_dict_sha256": (
                        _canonical_state_dict_sha256(candidate["state_dict"])
                    ),
                    "base_state_exact_match": comparison["exact_match"],
                    "missing_key_count": comparison["missing_key_count"],
                    "unexpected_key_count": comparison["unexpected_key_count"],
                    "mismatched_key_count": comparison["mismatched_key_count"],
                    "dtype_difference_count": comparison[
                        "dtype_difference_count"
                    ],
                    "max_abs_parameter_difference": comparison[
                        "max_abs_parameter_difference"
                    ],
                    "mismatched_keys": ";".join(comparison["mismatched_keys"]),
                    "dtype_differences": ";".join(
                        comparison["dtype_differences"]
                    ),
                }
            )
            if not comparison["exact_match"]:
                reasons.append("base_state_dict_mismatch")
        record["failure_reasons"] = ";".join(reasons)
        record["status"] = "pass" if not reasons else "failed"
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "baseline_checkpoint_provenance.csv"
    _write_csv(csv_path, records)
    result = {
        "definition": "Stage-2 uniform frozen-base checkpoint provenance audit",
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_state_dict_sha256": source_state_sha,
        "source_canonical_state_dict_sha256": _canonical_state_dict_sha256(
            source_state
        ),
        "expected_source_run_name": args.expected_source_run_name,
        "run_count": len(records),
        "pass_count": sum(row["status"] == "pass" for row in records),
        "failure_count": sum(row["status"] != "pass" for row in records),
        "all_base_states_exact_match": bool(records)
        and all(row["base_state_exact_match"] for row in records),
        "test100_accessed": False,
        "test100_evaluations_used": 0,
        "records": records,
        "csv": csv_path.resolve().as_posix(),
    }
    output = args.output_dir / "provenance_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if args.require_all_match and result["failure_count"]:
        raise RuntimeError(
            f"{result['failure_count']}/{result['run_count']} baseline checkpoints fail provenance"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-run-name")
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-all-match", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
