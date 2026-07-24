#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpn_melting.cli import load_atom_source
from mpn_melting.structures import lattice_volume


def interface_minimum(
    solid_boundary: list[tuple[float, float, float]],
    liquid_boundary: list[tuple[float, float, float]],
    lengths: tuple[float, float, float, float],
    cutoff: float,
    side: str,
) -> float:
    lx, ly, solid_z, liquid_z = lengths
    bins_x = max(1, int(lx / cutoff))
    bins_y = max(1, int(ly / cutoff))
    bins: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for x, y, z in liquid_boundary:
        key = (int(x / lx * bins_x) % bins_x, int(y / ly * bins_y) % bins_y)
        bins.setdefault(key, []).append((x, y, z))

    best = cutoff
    for x, y, z in solid_boundary:
        bin_x = int(x / lx * bins_x) % bins_x
        bin_y = int(y / ly * bins_y) % bins_y
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                key = ((bin_x + offset_x) % bins_x, (bin_y + offset_y) % bins_y)
                for other_x, other_y, other_z in bins.get(key, ()):
                    delta_x = min(abs(x - other_x), lx - abs(x - other_x))
                    delta_y = min(abs(y - other_y), ly - abs(y - other_y))
                    if side == "internal":
                        delta_z = solid_z - z + other_z
                    else:
                        delta_z = z + liquid_z - other_z
                    distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                    best = min(best, distance)
    return best


def score_shift(solid, liquid, shift: tuple[float, float, float], cutoff: float) -> float:
    lx = solid.lattice_vectors[0][0]
    ly = solid.lattice_vectors[1][1]
    area = lx * ly
    solid_z = lattice_volume(solid.lattice_vectors) / area
    liquid_z = lattice_volume(liquid.lattice_vectors) / area
    lengths = (lx, ly, solid_z, liquid_z)
    solid_cartesian = [(p[0] * lx, p[1] * ly, p[2] * solid_z) for p in solid.scaled_positions]
    solid_low = [p for p in solid_cartesian if p[2] < cutoff]
    solid_high = [p for p in solid_cartesian if p[2] > solid_z - cutoff]
    liquid_low = []
    liquid_high = []
    for position in liquid.scaled_positions:
        fractional = tuple((position[axis] + shift[axis]) % 1.0 for axis in range(3))
        cartesian = (fractional[0] * lx, fractional[1] * ly, fractional[2] * liquid_z)
        if cartesian[2] < cutoff:
            liquid_low.append(cartesian)
        if cartesian[2] > liquid_z - cutoff:
            liquid_high.append(cartesian)
    return min(
        interface_minimum(solid_high, liquid_low, lengths, cutoff, "internal"),
        interface_minimum(solid_low, liquid_high, lengths, cutoff, "periodic"),
    )


def exact_cross_minimum(solid, liquid, shift: tuple[float, float, float]) -> float:
    lx = solid.lattice_vectors[0][0]
    ly = solid.lattice_vectors[1][1]
    area = lx * ly
    solid_z = lattice_volume(solid.lattice_vectors) / area
    liquid_z = lattice_volume(liquid.lattice_vectors) / area
    total_z = solid_z + liquid_z
    best = math.inf
    for solid_position in solid.scaled_positions:
        solid_cartesian = (solid_position[0] * lx, solid_position[1] * ly, solid_position[2] * solid_z)
        for liquid_position in liquid.scaled_positions:
            shifted = tuple((liquid_position[axis] + shift[axis]) % 1.0 for axis in range(3))
            liquid_cartesian = (shifted[0] * lx, shifted[1] * ly, solid_z + shifted[2] * liquid_z)
            deltas = []
            for first, second, length in zip(solid_cartesian, liquid_cartesian, (lx, ly, total_z)):
                delta = abs(first - second)
                deltas.append(min(delta, length - delta))
            best = min(best, math.sqrt(sum(delta**2 for delta in deltas)))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solid-source", required=True)
    parser.add_argument("--liquid-source", required=True)
    parser.add_argument("--solid-frame", default="last")
    parser.add_argument("--liquid-frame", default="last")
    parser.add_argument("--element", default="Al")
    parser.add_argument("--trials", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--cutoff", type=float, default=2.7)
    args = parser.parse_args()

    solid = load_atom_source(args.solid_source, args.solid_frame, args.element)["atoms"]
    liquid = load_atom_source(args.liquid_source, args.liquid_frame, args.element)["atoms"]
    generator = random.Random(args.seed)
    best_score = -math.inf
    best_shift = (0.0, 0.0, 0.0)
    for _ in range(args.trials):
        shift = (generator.random(), generator.random(), generator.random())
        score = score_shift(solid, liquid, shift, args.cutoff)
        if score > best_score:
            best_score, best_shift = score, shift
    for radius in (0.08, 0.025, 0.008, 0.0025):
        base = best_shift
        for _ in range(max(1000, args.trials // 4)):
            shift = tuple((base[axis] + generator.uniform(-radius, radius)) % 1.0 for axis in range(3))
            score = score_shift(solid, liquid, shift, args.cutoff)
            if score > best_score:
                best_score, best_shift = score, shift

    print(
        json.dumps(
            {
                "liquid_shift": best_shift,
                "search_minimum_A": best_score,
                "exact_cross_minimum_A": exact_cross_minimum(solid, liquid, best_shift),
                "trials": args.trials,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
