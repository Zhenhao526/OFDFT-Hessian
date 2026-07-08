from __future__ import annotations

from dataclasses import dataclass
from typing import List


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

