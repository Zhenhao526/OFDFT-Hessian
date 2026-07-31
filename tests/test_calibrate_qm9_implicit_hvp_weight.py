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
