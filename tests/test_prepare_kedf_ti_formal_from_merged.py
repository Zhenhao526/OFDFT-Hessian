import pytest

from scripts.prepare_kedf_ti_formal_from_merged import (
    parse_tau_overrides,
    selected_rows,
)


def test_parse_phase_specific_tau_overrides():
    assert parse_tau_overrides(
        ["solid:lambda_0p250=2", "liquid:lambda_0p125=1"]
    ) == {
        ("solid", "lambda_0p250"): 2.0,
        ("liquid", "lambda_0p125"): 1.0,
    }


def test_reject_duplicate_tau_override():
    with pytest.raises(ValueError, match="duplicate"):
        parse_tau_overrides(
            ["liquid:lambda_0p125=1", "liquid:lambda_0p125=2"]
        )


def test_selects_nine_verified_rows_in_lambda_order():
    rows = [
        {
            "label": f"lambda_{index:03d}",
            "lambda": index / 8,
            "status": "verified",
        }
        for index in reversed(range(9))
    ]
    merged = {"phase_results": {"solid": {"status": "verified", "window_results": rows}}}

    selected = selected_rows(merged, "solid")

    assert [row["lambda"] for row in selected] == sorted(
        row["lambda"] for row in rows
    )
