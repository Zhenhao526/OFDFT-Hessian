from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.qm9_complete_total_validation_baseline_merge import merge


def test_merge_writes_missing_parent_audit_before_failing(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        """
scope:
  test100_accessed: false
stability:
  strict_density_gradient_max: 1.0e-8
  baseline_asym_over_pbe_frobenius_max: 5.0e-3
""".lstrip()
    )
    capacity = tmp_path / "capacity.json"
    capacity.write_text(
        json.dumps(
            {
                "test100_accessed": False,
                "parent_count": 1,
                "parents": [{"molecule_id": "0000001", "natoms": 2}],
            }
        )
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"test100_accessed": False, "records": []})
    )
    output = tmp_path / "output"
    source_split = tmp_path / "split.pkl"
    source_split.write_bytes(b"frozen-split")
    args = argparse.Namespace(
        protocol=protocol,
        capacity_manifest=capacity,
        source_split=source_split,
        run_root=tmp_path / "runs",
        provenance_audit=provenance,
        output_dir=output,
        require_complete=True,
    )

    with pytest.raises(RuntimeError, match="0/1"):
        merge(args)

    result = json.loads((output / "validation_baseline_manifest.json").read_text())
    assert result["test100_accessed"] is False
    assert result["complete"] is False
    assert result["source_split_sha256"]
    assert result["audit_rows"][0]["failure_reasons"] == "missing_complete_total_baseline"
