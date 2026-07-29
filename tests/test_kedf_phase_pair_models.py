from pathlib import Path

import pytest

from scripts.kedf_phase_pair_models import resolve_phase_pair_models


def test_common_pair_model_is_used_for_both_phases():
    common = Path("/tmp/common.json")

    assert resolve_phase_pair_models(
        pair_model=common,
        solid_pair_model=None,
        liquid_pair_model=None,
    ) == {"solid": common, "liquid": common}


def test_phase_specific_pair_models_are_preserved():
    solid = Path("/tmp/solid.json")
    liquid = Path("/tmp/liquid.json")

    assert resolve_phase_pair_models(
        pair_model=None,
        solid_pair_model=solid,
        liquid_pair_model=liquid,
    ) == {"solid": solid, "liquid": liquid}


@pytest.mark.parametrize(
    ("common", "solid", "liquid"),
    [
        (Path("/tmp/common.json"), Path("/tmp/solid.json"), Path("/tmp/liquid.json")),
        (None, Path("/tmp/solid.json"), None),
        (None, None, Path("/tmp/liquid.json")),
        (None, None, None),
    ],
)
def test_invalid_pair_model_combinations_fail(common, solid, liquid):
    with pytest.raises(ValueError):
        resolve_phase_pair_models(
            pair_model=common,
            solid_pair_model=solid,
            liquid_pair_model=liquid,
        )
