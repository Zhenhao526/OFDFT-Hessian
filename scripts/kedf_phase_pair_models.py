from __future__ import annotations

from pathlib import Path


def resolve_phase_pair_models(
    *,
    pair_model: Path | None,
    solid_pair_model: Path | None,
    liquid_pair_model: Path | None,
) -> dict[str, Path]:
    phase_specific = solid_pair_model is not None or liquid_pair_model is not None
    if pair_model is not None and phase_specific:
        raise ValueError(
            "use either --pair-model or both phase-specific pair-model options"
        )
    if pair_model is not None:
        return {"solid": pair_model, "liquid": pair_model}
    if solid_pair_model is None or liquid_pair_model is None:
        raise ValueError(
            "both --solid-pair-model and --liquid-pair-model are required"
        )
    return {"solid": solid_pair_model, "liquid": liquid_pair_model}
