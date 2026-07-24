#!/usr/bin/env python3
"""Export the guarded JSON pair reference to ABACUS's compact TI format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_model(source: Path, output: Path) -> None:
    document = json.loads(source.read_text())
    if not document.get("reference_gate_passed") or not document.get("short_range_guard_passed"):
        raise ValueError("pair reference has not passed its validation gates")
    model = document["model"]
    centers = model["centers_angstrom"]
    coefficients = model["coefficients_ev"]
    if len(coefficients) != len(centers) + 1:
        raise ValueError("expected one constant plus one coefficient per radial center")
    core = model["repulsive_core"]
    lines = [
        "MPN_PAIR_REFERENCE_V1",
        f"n_centers {len(centers)}",
        f"sigma_angstrom {model['sigma_angstrom']:.17g}",
        f"cutoff_angstrom {model['cutoff_angstrom']:.17g}",
        f"constant_ev_per_atom {coefficients[0]:.17g}",
        f"core_amplitude_ev {core['amplitude_ev']:.17g}",
        f"core_cutoff_angstrom {core['cutoff_angstrom']:.17g}",
        f"core_power {int(core['power'])}",
        "centers_angstrom " + " ".join(f"{value:.17g}" for value in centers),
        "coefficients_ev " + " ".join(f"{value:.17g}" for value in coefficients[1:]),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_model(args.source, args.output)


if __name__ == "__main__":
    main()
