import pytest

from scripts.calibrate_qm9_implicit_hvp_weight import calibrate


def test_calibration_matches_frozen_gradient_ratio_rule():
    protocol = {
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        }
    }
    row = {
        "gradient_norm/energy": "3",
        "gradient_norm/density": "4",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.5",
    }

    result = calibrate(protocol, row)

    assert result["aggregate_weighted_egf_gradient_norm"] == pytest.approx(5.0)
    assert result["formal_lambda_h"] == pytest.approx(0.025)


def test_calibration_clips_extreme_hvp_weight():
    protocol = {
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        }
    }
    row = {
        "gradient_norm/energy": "100",
        "gradient_norm/density": "0",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.001",
    }

    assert calibrate(protocol, row)["formal_lambda_h"] == 1.0


def test_canonical_calibration_uses_actual_aggregate_egf_gradient():
    protocol = {
        "definitions": {
            "physical_definition_id": "qm9_complete_total_relaxed_egfh_v1"
        },
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        },
    }
    row = {
        "physical_definition_id": "qm9_complete_total_relaxed_egfh_v1",
        "computed_components": "E;G;F;H",
        "gradient_norm/energy": "3",
        "gradient_norm/density": "4",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.5",
        # Antiparallel component gradients can make the actual aggregate much
        # smaller than their quadrature (which is 5 in this synthetic row).
        "gradient_norm/aggregate_egf": "1",
    }

    result = calibrate(protocol, row)

    assert result["component_gradient_norm_quadrature"] == pytest.approx(5.0)
    assert result["aggregate_weighted_egf_gradient_norm"] == pytest.approx(1.0)
    assert result["formal_lambda_h"] == pytest.approx(0.005)
    assert result["aggregate_egf_gradient_definition"] == (
        "norm_of_gradient_of_weighted_E_plus_G_plus_F"
    )


def test_canonical_calibration_rejects_mixed_physical_semantics():
    protocol = {
        "definitions": {
            "physical_definition_id": "qm9_complete_total_relaxed_egfh_v1"
        },
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        },
    }
    row = {
        "physical_definition_id": "legacy_hybrid_egfh_v0",
        "computed_components": "E;G;F;H",
        "gradient_norm/energy": "3",
        "gradient_norm/density": "4",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.5",
    }

    with pytest.raises(ValueError, match="wrong physical definition"):
        calibrate(protocol, row)


def test_canonical_calibration_requires_all_four_canonical_components():
    protocol = {
        "definitions": {
            "physical_definition_id": "qm9_complete_total_relaxed_egfh_v1"
        },
        "gradient_calibration": {
            "smoke": {"lambda_H": 0.01},
            "target_hvp_to_egf_gradient_ratio": 0.25,
            "lambda_h_min": 1.0e-4,
            "lambda_h_max": 1.0,
        },
    }
    row = {
        "physical_definition_id": "qm9_complete_total_relaxed_egfh_v1",
        "computed_components": "G;F;H",
        "gradient_norm/energy": "3",
        "gradient_norm/density": "4",
        "gradient_norm/force": "0",
        "gradient_norm/hvp": "0.5",
    }

    with pytest.raises(ValueError, match="exactly E/G/F/H"):
        calibrate(protocol, row)
