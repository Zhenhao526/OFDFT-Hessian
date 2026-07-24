#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from mpn_melting.free_energy import write_dataset


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must have the form PHASE=PATH")
    phase, path = value.split("=", 1)
    if not phase or not path:
        raise argparse.ArgumentTypeError("run must have the form PHASE=PATH")
    return phase, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an auditable MPN free-energy reference dataset")
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="PHASE=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--discard-fraction", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    manifest = write_dataset(args.run, args.out, args.discard_fraction, args.stride)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

