#!/usr/bin/env python3
"""Merge independently passed solid/liquid zero-pressure confirmations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one confirmation summary is required")
    documents = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]
    target_kedfs = {
        str(document.get("target_kedf", "")).lower()
        for _, document in documents
    }
    if len(target_kedfs) != 1:
        raise ValueError("confirmation summaries use different KEDFs")
    temperatures = {
        float(document["temperature_K"]) for _, document in documents
    }
    if len(temperatures) != 1:
        raise ValueError("confirmation summaries use different temperatures")

    selected = {}
    for _, document in documents:
        for item in document["phase_results"]:
            if item.get("status") == "confirmation_passed":
                selected[str(item["phase"])] = item
    checks = {
        "solid_confirmation_available": "solid" in selected,
        "liquid_confirmation_available": "liquid" in selected,
    }
    provenance = [
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "status": document.get("status"),
        }
        for path, document in documents
    ]
    all_passed = all(checks.values())
    return {
        "schema": "kedf-zero-pressure-confirmation-merged-v1",
        "target_kedf": target_kedfs.pop(),
        "temperature_K": temperatures.pop(),
        "target_pressure_kbar": 0.0,
        "phase_results": [
            selected[phase]
            for phase in ("solid", "liquid")
            if phase in selected
        ],
        "checks": checks,
        "provenance": provenance,
        "status": (
            "all_confirmations_passed"
            if all_passed
            else "confirmation_incomplete"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = merge([path.resolve() for path in args.summaries])
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
