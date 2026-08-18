from scripts.fit_suf_reference import select_phase_frames


def test_select_phase_frames_excludes_other_phases():
    frames = [
        {"phase": "solid", "id": 1},
        {"phase": "liquid", "id": 2},
        {"phase": "liquid", "id": 3},
    ]

    assert select_phase_frames(frames, "liquid") == frames[1:]


def test_select_phase_frames_rejects_missing_phase():
    frames = [{"phase": "solid"}]

    try:
        select_phase_frames(frames, "liquid")
    except ValueError as error:
        assert "available phases" in str(error)
    else:
        raise AssertionError("missing phase was not rejected")
