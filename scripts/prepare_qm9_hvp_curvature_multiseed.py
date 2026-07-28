#!/usr/bin/env python3
"""Freeze one A-E configuration across three seeds after the validation-only screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tag(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text())
    summary = json.loads(args.screen_summary.read_text())
    if protocol.get("test100_access_allowed") is not False:
        raise ValueError("Protocol must prohibit Test100 access")
    if summary.get("test100_accessed") is not False:
        raise ValueError("Screen summary does not certify frozen Test100")
    selected = summary["selected_config_by_variant"]
    seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    max_steps = int(protocol["training"]["max_steps"])
    rows = []
    for variant in ("A", "B", "C", "D", "E"):
        config = selected[variant]
        weight = float(config["curvature_weight"])
        floor = float(config["reference_floor"])
        for seed in seeds:
            run_name = (
                f"qm9_hvp_curvature_v1_{variant}_w{_tag(weight)}_f{_tag(floor)}_"
                f"seed{seed}_s{max_steps}"
            )
            rows.append(
                {
                    "task_index": len(rows),
                    "variant": variant,
                    "curvature_weight": f"{weight:.12g}",
                    "reference_floor": f"{floor:.12g}",
                    "seed": seed,
                    "max_steps": max_steps,
                    "run_name": run_name,
                    "screen_selection_status": config["selection_status"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "definition": __doc__,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "screen_summary": str(args.screen_summary.resolve()),
        "screen_summary_sha256": _sha256(args.screen_summary),
        "task_count": len(rows),
        "seeds": seeds,
        "selected_config_by_variant": selected,
        "test100_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
