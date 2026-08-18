from scripts.analyze_pair_reference_md import linear_slope


def test_linear_slope_detects_bounded_solid_motion():
    assert abs(linear_slope([0.0, 1.0, 2.0], [0.2, 0.2, 0.2])) < 1.0e-12


def test_linear_slope_detects_liquid_diffusion():
    assert linear_slope([0.0, 100.0, 200.0], [0.0, 1.0, 2.0]) > 1.0e-4
