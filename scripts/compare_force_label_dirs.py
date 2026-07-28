#!/usr/bin/env python3
"""Compare PBE force labels between two label directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


FORCE_KEY = "metadata/pbe_derivatives/forces"


def _label_map(path: Path) -> dict[str, Path]:
    if path.is_file():
        return {path.name: path}
    return {item.name: item for item in path.glob("*.zarr.zip")}


def _force(path: Path) -> np.ndarray:
    root = zarr.open(path, mode="r")
    if FORCE_KEY not in root:
        raise KeyError(f"{path}: missing {FORCE_KEY}")
    return np.asarray(root[FORCE_KEY], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = _label_map(args.reference)
    candidate = _label_map(args.candidate)
    common = sorted(set(reference).intersection(candidate))
    missing_in_candidate = sorted(set(reference).difference(candidate))
    extra_in_candidate = sorted(set(candidate).difference(reference))
    failures = []

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    max_abs = 0.0
    compared_files = 0
    for name in common:
        try:
            ref_force = _force(reference[name])
            cand_force = _force(candidate[name])
            if ref_force.shape != cand_force.shape:
                failures.append(
                    {
                        "label": name,
                        "failure": f"shape mismatch {ref_force.shape} != {cand_force.shape}",
                    }
                )
                continue
            diff = cand_force - ref_force
            abs_sum += float(np.abs(diff).sum())
            sq_sum += float(np.square(diff).sum())
            count += int(diff.size)
            if diff.size:
                max_abs = max(max_abs, float(np.abs(diff).max()))
            compared_files += 1
        except Exception as exc:
            failures.append({"label": name, "failure": repr(exc)})

    summary = {
        "reference": args.reference.as_posix(),
        "candidate": args.candidate.as_posix(),
        "common_files": len(common),
        "compared_files": compared_files,
        "missing_in_candidate": missing_in_candidate,
        "extra_in_candidate": extra_in_candidate,
        "failures": failures,
        "force_component_mae": abs_sum / count if count else None,
        "force_component_rmse": (sq_sum / count) ** 0.5 if count else None,
        "force_component_max_abs": max_abs if count else None,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(summary_text + "\n")
    if failures or missing_in_candidate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
