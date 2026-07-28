#!/usr/bin/env python3
"""Freeze A/B plus validation-shortlisted C/D runs for strict complete-total HVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (676368232, 20260716, 314159)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--run-suffix", default="gated25v2_20260716")
    args = parser.parse_args()
    summary = json.loads(args.stage1_summary.read_text())
    names = [("A", "force", 0.0), ("B", "force_secant", 0.0)]
    for row in summary["shortlist"]:
        weight_tag = {1e-5: "1e5", 1e-4: "1e4", 1e-3: "1e3"}[float(row["hvp_weight"])]
        if row["variant"] == "C":
            name = f"force_hvp_w{weight_tag}"
        elif row["variant"] == "D":
            name = f"force_secant_hvp_w{weight_tag}"
        else:
            raise ValueError(row)
        names.append((row["variant"], name, float(row["hvp_weight"])))
    if len(names) < 3:
        raise RuntimeError("No direct-HVP variant passed stage-1 gates")
    lines = ["variant\thvp_weight\tseed\trun_name"]
    for variant, name, weight in names:
        for seed in SEEDS:
            run_name = f"qm9_hvp100_{name}_seed{seed}_s600_{args.run_suffix}"
            lines.append(f"{variant}\t{weight:.0e}\t{seed}\t{run_name}")
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(lines) - 1, "output": str(args.output_tsv)}, indent=2))


if __name__ == "__main__":
    main()
