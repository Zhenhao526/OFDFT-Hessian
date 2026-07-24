from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
from typing import List

from .trajectory import invert_3x3, matmul_row


@dataclass(frozen=True)
class ObservablePlan:
    name: str
    purpose: str


DEFAULT_OBSERVABLES: List[ObservablePlan] = [
    ObservablePlan("rdf", "Track liquid short-range order and compare against reference data."),
    ObservablePlan("msd", "Distinguish diffusive liquid-like atoms from solid-like atoms."),
    ObservablePlan("q4_q6", "Estimate local crystalline order in coexistence simulations."),
    ObservablePlan("density_profile", "Track interface motion along the coexistence direction."),
    ObservablePlan("solid_fraction", "Bracket melting point from solid/liquid growth direction."),
]


def observable_checklist() -> str:
    return "\n".join(f"- {item.name}: {item.purpose}" for item in DEFAULT_OBSERVABLES)


def centrosymmetry_parameters(
    positions,
    lattice,
    neighbor_count: int = 12,
) -> List[float]:
    """Return the minimum centrosymmetry pairing score for each atom."""
    if neighbor_count <= 0 or neighbor_count % 2:
        raise ValueError("neighbor_count must be a positive even integer")
    inv_lattice = invert_3x3(lattice)
    values: List[float] = []
    for index, center in enumerate(positions):
        nearest = []
        for other_index, other in enumerate(positions):
            if index == other_index:
                continue
            delta = tuple(other[axis] - center[axis] for axis in range(3))
            fractional = matmul_row(delta, inv_lattice)
            wrapped = tuple(component - round(component) for component in fractional)
            vector = matmul_row(wrapped, lattice)
            distance_sq = sum(component * component for component in vector)
            item = (-distance_sq, vector)
            if len(nearest) < neighbor_count:
                heapq.heappush(nearest, item)
            elif distance_sq < -nearest[0][0]:
                heapq.heapreplace(nearest, item)
        if len(nearest) != neighbor_count:
            raise ValueError("not enough atoms for the requested neighbor count")
        vectors = [item[1] for item in nearest]
        pair_cost = [
            [
                sum((vectors[i][axis] + vectors[j][axis]) ** 2 for axis in range(3))
                for j in range(neighbor_count)
            ]
            for i in range(neighbor_count)
        ]

        @lru_cache(maxsize=None)
        def minimum_matching(mask: int) -> float:
            if mask == 0:
                return 0.0
            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            remaining = mask ^ first_bit
            best = float("inf")
            candidates = remaining
            while candidates:
                partner_bit = candidates & -candidates
                partner = partner_bit.bit_length() - 1
                best = min(
                    best,
                    pair_cost[first][partner] + minimum_matching(remaining ^ partner_bit),
                )
                candidates ^= partner_bit
            return best

        values.append(minimum_matching((1 << neighbor_count) - 1))
    return values
