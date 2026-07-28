import json
from types import SimpleNamespace

import numpy as np
import yaml

from scripts.qm9_complete_total_direction_confirmation import (
    _jitter_structured_directions,
    evaluate_confirmation,
)


def test_structured_confirmation_jitter_is_internal_normalized_and_new():
    directions = np.eye(3)
    kinds = np.asarray(["bond", "angle", "random_internal"])
    jittered = _jitter_structured_directions(
        directions,
        kinds,
        np.empty((3, 0)),
        fraction=0.25,
        seed=19,
    )

    np.testing.assert_allclose(np.linalg.norm(jittered, axis=1), 1.0)
    assert abs(np.dot(jittered[0], directions[0])) < 1.0
    assert abs(np.dot(jittered[1], directions[1])) < 1.0
    np.testing.assert_array_equal(jittered[2], directions[2])


def test_frozen_confirmation_reports_distribution_gate_without_training(tmp_path):
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "metrics": {"hvp_reference_floor": 0.05},
                "gates": {
                    "median_parent_hvp_relative_l2_max": 0.15,
                    "p90_parent_hvp_relative_l2_max": 0.20,
                    "fraction_parent_hvp_relative_l2_below_0p15_min": 0.80,
                },
            }
        )
    )
    direction_path = tmp_path / "directions.npz"
    np.savez_compressed(
        direction_path,
        directions=np.eye(2),
        kinds=np.asarray(["bond", "random_internal"]),
    )
    manifest_path = tmp_path / "confirmation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "confirmation_results_may_not_return_to_training": True,
                "parents": [
                    {
                        "molecule_id": "0000001",
                        "natoms": 1,
                        "direction_path": str(direction_path),
                    }
                ],
            }
        )
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"test100_accessed": False})
    )
    np.savez_compressed(
        run_dir / "0000001_result.npz",
        predicted_hessian=np.diag([1.1, 2.2]),
        pbe_hessian=np.diag([1.0, 2.0]),
    )
    output_dir = tmp_path / "output"

    result = evaluate_confirmation(
        SimpleNamespace(
            protocol=protocol_path,
            confirmation_manifest=manifest_path,
            run=[f"candidate={run_dir}"],
            output_dir=output_dir,
        )
    )

    summary = result["summaries"][0]
    np.testing.assert_allclose(
        summary["median_parent_hvp_relative_l2"], 0.1, rtol=1.0e-12
    )
    assert summary["confirmation_gate_passed"] is True
    assert result["confirmation_results_may_not_return_to_training"] is True
    assert (output_dir / "per_direction.csv").is_file()
