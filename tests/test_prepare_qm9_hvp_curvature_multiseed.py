import csv
import json
import subprocess
from pathlib import Path

import yaml


def test_multiseed_table_freezes_one_configuration_per_variant(tmp_path):
    protocol = {
        "test100_access_allowed": False,
        "training": {"seeds": [1, 2, 3], "max_steps": 12},
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    selected = {
        variant: {
            "curvature_weight": 0.0 if variant == "A" else 1e-4,
            "reference_floor": 0.05,
            "selection_status": "tier1_eligible",
        }
        for variant in "ABCDE"
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"test100_accessed": False, "selected_config_by_variant": selected}
        )
    )
    output = tmp_path / "tasks.tsv"

    subprocess.run(
        [
            "python",
            "scripts/prepare_qm9_hvp_curvature_multiseed.py",
            "--protocol",
            str(protocol_path),
            "--screen-summary",
            str(summary_path),
            "--output",
            str(output),
        ],
        check=True,
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 15
    assert {int(row["seed"]) for row in rows} == {1, 2, 3}
    assert {row["variant"] for row in rows} == set("ABCDE")
    assert all("seed" in row["run_name"] for row in rows)
    assert json.loads(output.with_suffix(".json").read_text())["test100_accessed"] is False
